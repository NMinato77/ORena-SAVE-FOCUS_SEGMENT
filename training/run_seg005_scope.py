#!/usr/bin/env python3
"""Train and scope-screen one SEG005 SFT condition.

Each process owns one GPU and one condition.  The shared manifest and schedule
make the four processes comparable while avoiding DDP communication.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import PyNvVideoCodec as nvc  # must precede torch/CUDA.
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from orena_procedure.format_reducer import format_instruction, reduce_answer
from orena_procedure.seg005 import (
    CANDIDATES,
    CHECKPOINT_STEPS,
    GRADIENT_ACCUMULATION,
    LANGUAGE_LR,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MAX_UPDATES,
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    VISION_LR,
    WEIGHT_DECAY,
    candidate_parameter_norms,
    configure_train_modes,
    install_candidate,
    stable_hash,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + "\n", encoding="utf-8")


def validate_rows(rows: list[dict[str, Any]], role: str) -> None:
    if not rows:
        raise ValueError(f"empty {role} manifest")
    seen: set[tuple[str, str]] = set()
    forbidden = {"reference", "reference_answer", "target_seconds", "official_tolerance_seconds", "event_kind", "fo_class", "primary", "ood", "answer_format"}
    for row in rows:
        key = (str(row["dataset"]), str(row["qID"]))
        if key in seen:
            raise ValueError(f"duplicate {role} qID: {key}")
        seen.add(key)
        required = {"dataset", "qID", "videoID", "procedure_type", "question", "start_time", "end_time", "video_path", "frame_indices", "frame_timestamps_seconds", "frame_count"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{role} row missing {sorted(missing)}: {key}")
        if role == "training":
            if not str(row.get("answer", "")).strip():
                raise ValueError(f"training answer missing: {key}")
        elif forbidden.intersection(row):
            raise ValueError(f"answer-bearing field in {role} manifest: {key}: {sorted(forbidden.intersection(row))}")
        indices = [int(value) for value in row["frame_indices"]]
        timestamps = [float(value) for value in row["frame_timestamps_seconds"]]
        if len(indices) != int(row["frame_count"]) or len(indices) != len(timestamps) or len(indices) != len(set(indices)) or indices != sorted(indices):
            raise ValueError(f"invalid fixed-1fps frame manifest: {key}")


def load_model(
    device: torch.device,
    max_pixels: int = MAX_PIXELS,
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        max_pixels=max_pixels,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        local_files_only=True,
    )
    return model, processor


def install_and_prepare_model(
    device: torch.device,
    candidate: str,
    module_audit: Path,
    load_model: bool = True,
) -> tuple[torch.nn.Module, Any, dict[str, Any], torch.optim.Optimizer]:
    if not load_model:
        raise ValueError("install_and_prepare_model requires load_model=True")
    model, processor = globals()["load_model"](device)
    installation = install_candidate(model, candidate)
    audit = json.loads(module_audit.read_text(encoding="utf-8"))
    expected = set(audit["candidates"][candidate]["trainable_parameter_names"])
    actual = set(installation["trainable_parameter_names"])
    if actual != expected:
        raise ValueError(f"runtime module scope differs from module audit for {candidate}")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    groups: list[dict[str, Any]] = []
    language = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and ".language_model." in name]
    merger = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and (".visual.merger." in name or ".visual.deepstack_merger_list." in name)]
    vision = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and ".visual.blocks." in name]
    if language:
        groups.append({"params": language, "lr": LANGUAGE_LR, "weight_decay": WEIGHT_DECAY, "scope": "language_lora"})
    if merger:
        groups.append({"params": merger, "lr": LANGUAGE_LR, "weight_decay": WEIGHT_DECAY, "scope": "merger_full"})
    if vision:
        groups.append({"params": vision, "lr": VISION_LR, "weight_decay": WEIGHT_DECAY, "scope": "vision_lora"})
    if not groups:
        raise RuntimeError(f"no optimizer parameters for {candidate}")
    optimizer = torch.optim.AdamW(groups)
    return model, processor, installation, optimizer


def temporal_messages(
    row: dict[str, Any],
    images: list[Image.Image],
    *,
    evidence_instruction: str = "Use only the sampled frames as evidence.",
    timestamp_frame_template: str = "Frame timestamp (absolute source-procedure timeline): {timestamp:.1f} seconds.",
    timestamp_context: str = "Sampled-frame timestamps are absolute source-procedure timeline timestamps.",
) -> list[dict[str, Any]]:
    timestamps = [float(value) for value in row["frame_timestamps_seconds"]]
    if len(timestamps) != len(images):
        raise ValueError("timestamp/image count mismatch")
    content: list[dict[str, Any]] = []
    for timestamp, image in zip(timestamps, images, strict=True):
        content.extend(
            [
                {
                    "type": "text",
                    "text": timestamp_frame_template.format(timestamp=timestamp),
                },
                {"type": "image", "image": image},
            ]
        )
    instruction = format_instruction(
        str(row["question"]),
        request_start_seconds=float(row["start_time"]),
        request_end_seconds=float(row["end_time"]),
    )
    content.append(
        {
            "type": "text",
            "text": (
                "You are assisting with laparoscopic surgery. "
                f"{evidence_instruction} "
                f"Procedure type: {row['procedure_type']}. The request window is from {float(row['start_time']):.1f} "
                f"to {float(row['end_time']):.1f} seconds on the original source-procedure timeline. "
                f"{timestamp_context} "
                f"{instruction}\nQuestion: {row['question']}"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def decode_indices(row: dict[str, Any], gpu: int) -> tuple[list[Image.Image], dict[str, float]]:
    indices = [int(value) for value in row["frame_indices"]]
    started = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        str(row["video_path"]),
        gpu_id=gpu,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
        bWaitForSessionWarmUp=True,
    )
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    surfaces = decoder.get_batch_frames_by_index(indices)
    arrays = [torch.utils.dlpack.from_dlpack(surface).cpu().numpy() for surface in surfaces]
    torch.cuda.synchronize(torch.device(f"cuda:{gpu}"))
    decoded = time.perf_counter() - started
    del surfaces, decoder
    started = time.perf_counter()
    images = [Image.fromarray(frame) for frame in arrays]
    converted = time.perf_counter() - started
    if len(images) != len(indices):
        raise ValueError(f"decoded frame count mismatch for {row['dataset']}/{row['qID']}")
    return images, {"video_loading": loaded, "frame_decoding": decoded, "image_conversion": converted}


def encode_row(
    row: dict[str, Any],
    processor: Any,
    device: torch.device,
    with_answer: bool,
    *,
    evidence_instruction: str = "Use only the sampled frames as evidence.",
    timestamp_frame_template: str = "Frame timestamp (absolute source-procedure timeline): {timestamp:.1f} seconds.",
    timestamp_context: str = "Sampled-frame timestamps are absolute source-procedure timeline timestamps.",
) -> tuple[Any, dict[str, Any]]:
    images, decode_timings = decode_indices(row, int(device.index or 0))
    messages = temporal_messages(
        row,
        images,
        evidence_instruction=evidence_instruction,
        timestamp_frame_template=timestamp_frame_template,
        timestamp_context=timestamp_context,
    )
    conversation = messages
    answer_start = answer_end = target_end = None
    if with_answer:
        conversation = [*messages, {"role": "assistant", "content": str(row["answer"])}]
    encoded = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=not with_answer,
        return_dict=True,
        return_tensors="pt",
    )
    if with_answer:
        answer_ids = processor.tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"]
        values = encoded.input_ids[0].tolist()
        start_search = max(0, len(values) - len(answer_ids) - 64)
        candidates = [index for index in range(start_search, len(values) - len(answer_ids) + 1) if values[index : index + len(answer_ids)] == answer_ids]
        if not candidates:
            raise ValueError(f"assistant answer span not found: {row['dataset']}/{row['qID']}")
        answer_start = candidates[-1]
        answer_end = answer_start + len(answer_ids)
        eot_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if answer_end >= len(values) or values[answer_end] != eot_id:
            raise ValueError(f"answer not followed by EOT: {row['dataset']}/{row['qID']}")
        target_end = answer_end + 1
    encoded = encoded.to(device)
    return encoded, {
        "dataset": str(row["dataset"]),
        "qID": str(row["qID"]),
        "frame_count": len(images),
        "frame_timestamps_seconds": [float(value) for value in row["frame_timestamps_seconds"]],
        "answer_start": answer_start,
        "answer_end": answer_end,
        "target_end": target_end,
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "decode_timings_seconds": decode_timings,
    }


def loss_for_encoded(model: torch.nn.Module, encoded: Any, audit: dict[str, Any]) -> torch.Tensor:
    start = int(audit["answer_start"])
    end = int(audit["target_end"])
    keep = torch.arange(start - 1, end - 1, device=encoded.input_ids.device)
    output = model(**encoded, logits_to_keep=keep)
    loss = F.cross_entropy(output.logits[0].float(), encoded.input_ids[0, start:end])
    if not torch.isfinite(loss):
        raise ValueError("non-finite SFT loss")
    return loss


def gradient_norm(model: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def generation_rows(
    model: torch.nn.Module,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    model.eval()
    model.config.use_cache = True
    eos_token_id = getattr(model.generation_config, "eos_token_id", None) or processor.tokenizer.eos_token_id
    pad_token_id = getattr(model.generation_config, "pad_token_id", None) or processor.tokenizer.pad_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("generation EOS and PAD token IDs must be defined")
    eos_ids = [int(value) for value in eos_token_id] if isinstance(eos_token_id, (list, tuple)) else [int(eos_token_id)]
    results: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result: dict[str, Any] = {
            "dataset": str(row["dataset"]), "qID": str(row["qID"]), "videoID": str(row["videoID"]),
            "status": "error", "failure_kind": "runtime_exception", "raw_prediction": "", "prediction": "",
            "sampling_strategy": str(row["sampling_strategy"]), "frame_count_requested": int(row["frame_count"]),
        }
        try:
            torch.cuda.reset_peak_memory_stats(device)
            encoded, audit = encode_row(row, processor, device, with_answer=False)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_ids, pad_token_id=int(pad_token_id))
            new_tokens = generated[:, encoded.input_ids.shape[1] :]
            raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            reduced = reduce_answer(
                str(row["question"]), raw,
                request_start_seconds=float(row["start_time"]),
                request_end_seconds=float(row["end_time"]),
            )
            last_id = int(new_tokens[0, -1]) if new_tokens.shape[1] else None
            result.update(
                {
                    "status": "ok" if raw.strip() else "error",
                    "failure_kind": None if raw.strip() else "parse_failure",
                    "raw_prediction": raw,
                    "prediction": reduced.content,
                    "reduced_format_valid": bool(reduced.format_valid),
                    "reduced_answer_format": reduced.answer_format,
                    "reduced_error": reduced.error,
                    "request_window_valid": reduced.request_window_valid,
                    "request_window_violation": reduced.request_window_violation,
                    "sampled_frames": audit["frame_count"],
                    "sampled_timestamps_seconds": audit["frame_timestamps_seconds"],
                    "prompt_tokens": audit["prompt_tokens"],
                    "generated_tokens": int(new_tokens.shape[1]),
                    "generated_token_ids": [int(value) for value in new_tokens[0].tolist()],
                    "last_generated_token_id": last_id,
                    "stopped_with_eos": last_id in eos_ids if last_id is not None else False,
                    "stopped_with_eot": last_id == 151645 if last_id is not None else False,
                    "max_new_tokens": max_new_tokens,
                    "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                }
            )
            del encoded, generated, new_tokens
        except Exception as exc:
            result["error"] = repr(exc)
            torch.cuda.empty_cache()
        result["runtime_total_seconds"] = time.perf_counter() - started
        result["over_15_seconds"] = result["runtime_total_seconds"] > 15.0
        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[generation {number}/{len(rows)}] {row['dataset']}/{row['qID']} status={result['status']} runtime={result['runtime_total_seconds']:.3f}s", flush=True)
    model.config.use_cache = False
    return results


def save_trainable_checkpoint(model: torch.nn.Module, path: Path) -> None:
    torch.save({name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}, path)


def save_optimizer_checkpoint(optimizer: torch.optim.Optimizer, path: Path, update: int) -> None:
    torch.save({"update": update, "optimizer_state_dict": optimizer.state_dict()}, path)


def validate_eot_contract(processor: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for row in rows:
        # The full visual encoding is intentionally omitted from this audit;
        # the same chat template and answer target are validated in encode_row.
        answer_ids = processor.tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"]
        samples.append({"dataset": row["dataset"], "qID": row["qID"], "answer_token_count": len(answer_ids)})
    return {"rows": len(samples), "target": "answer tokens plus exactly one assistant <|im_end|>", "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--module-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--checkpoint-steps", default=",".join(str(value) for value in CHECKPOINT_STEPS))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_updates != MAX_UPDATES or args.gradient_accumulation != GRADIENT_ACCUMULATION:
        raise ValueError("SEG005 training contract is fixed at 100 updates and accumulation 8")
    checkpoints = tuple(sorted({int(value) for value in args.checkpoint_steps.split(",") if value.strip()}))
    if checkpoints != CHECKPOINT_STEPS:
        raise ValueError(f"SEG005 checkpoints must be exactly {CHECKPOINT_STEPS}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SEG005 training requires CUDA")
    torch.cuda.set_device(device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_jsonl(args.manifest_dir / "training.jsonl")
    scope_rows = load_jsonl(args.manifest_dir / "scope_screen.jsonl")
    schedule = json.loads((args.manifest_dir / "schedule.json").read_text(encoding="utf-8"))
    validate_rows(train_rows, "training")
    validate_rows(scope_rows, "scope_screen")
    if len(train_rows) != 665 or len(scope_rows) != 135 or len(schedule) != args.max_updates * args.gradient_accumulation:
        raise ValueError(f"unexpected SEG005 manifest counts: train={len(train_rows)}, scope={len(scope_rows)}, schedule={len(schedule)}")
    if any(int(index) < 0 or int(index) >= len(train_rows) for index in schedule):
        raise ValueError("schedule index out of range")

    args.output_dir.mkdir(parents=True)
    model_started = time.perf_counter()
    model, processor, installation, optimizer = install_and_prepare_model(device, args.candidate, args.module_audit)
    model_load_seconds = time.perf_counter() - model_started
    write_json(args.output_dir / "target_contract.json", validate_eot_contract(processor, train_rows))
    write_json(args.output_dir / "config.json", {
        "experiment": "SEG005", "candidate": args.candidate, "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "frame_sampling": "fixed_1fps_request_start_half_open_nearest_5fps", "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "microbatch": 1, "gradient_accumulation": args.gradient_accumulation, "max_updates": args.max_updates,
        "checkpoint_steps": checkpoints, "optimizer": "AdamW", "learning_rates": {"language": LANGUAGE_LR, "vision": VISION_LR}, "weight_decay": WEIGHT_DECAY,
        "seed": SEED, "manifest_dir": str(args.manifest_dir), "module_audit": str(args.module_audit), "schedule_sha256": stable_hash(args.manifest_dir / "schedule.json"),
    })
    write_json(args.output_dir / "installation.json", installation)
    save_trainable_checkpoint(model, args.output_dir / "checkpoint_update_000.pt")
    save_optimizer_checkpoint(optimizer, args.output_dir / "optimizer_update_000.pt", 0)

    history: list[dict[str, Any]] = []
    eval_dir = args.output_dir / "scope_screen"
    eval_dir.mkdir()
    configure_train_modes(model, args.candidate, training=False)
    initial_started = time.perf_counter()
    initial_predictions = generation_rows(model, processor, scope_rows, device)
    write_jsonl(eval_dir / "update_000.jsonl", initial_predictions)
    history.append({
        "update": 0, "training_loss": None, "gradient_norm": None, "seconds_per_update": None,
        "generation_wall_seconds": time.perf_counter() - initial_started, "train_wall_seconds_cumulative": 0.0,
        "parameter_norms": candidate_parameter_norms(model), "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    })
    write_json(args.output_dir / "progress.json", {"status": "running", "history": history})

    training_started = time.perf_counter()
    for update in range(1, args.max_updates + 1):
        configure_train_modes(model, args.candidate, training=True)
        model.config.use_cache = False
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for index in schedule[(update - 1) * args.gradient_accumulation : update * args.gradient_accumulation]:
            row = train_rows[int(index)]
            encoded, audit = encode_row(row, processor, device, with_answer=True)
            loss = loss_for_encoded(model, encoded, audit) / args.gradient_accumulation
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            loss.backward()
            del encoded, loss
            gc.collect()
        norm = gradient_norm(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed_update = time.perf_counter() - update_started
        train_loss = float(np.mean(losses))
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"[update {update}/{args.max_updates}] loss={train_loss:.6f} grad_norm={norm:.6f} seconds={elapsed_update:.3f}", flush=True)
        if update in checkpoints:
            checkpoint = args.output_dir / f"checkpoint_update_{update:03d}.pt"
            optimizer_checkpoint = args.output_dir / f"optimizer_update_{update:03d}.pt"
            save_trainable_checkpoint(model, checkpoint)
            save_optimizer_checkpoint(optimizer, optimizer_checkpoint, update)
            eval_started = time.perf_counter()
            predictions = generation_rows(model, processor, scope_rows, device)
            write_jsonl(eval_dir / f"update_{update:03d}.jsonl", predictions)
            history.append({
                "update": update, "training_loss": train_loss, "gradient_norm": norm, "seconds_per_update": elapsed_update,
                "generation_wall_seconds": time.perf_counter() - eval_started, "train_wall_seconds_cumulative": time.perf_counter() - training_started,
                "parameter_norms": candidate_parameter_norms(model), "peak_vram_allocated_gib": max(peak_allocated, torch.cuda.max_memory_allocated(device) / 1024**3),
                "peak_vram_reserved_gib": max(peak_reserved, torch.cuda.max_memory_reserved(device) / 1024**3),
                "checkpoint": str(checkpoint), "checkpoint_sha256": stable_hash(checkpoint), "optimizer_checkpoint": str(optimizer_checkpoint), "optimizer_checkpoint_sha256": stable_hash(optimizer_checkpoint),
            })
            write_json(args.output_dir / "progress.json", {"status": "running", "candidate": args.candidate, "completed_update": update, "history": history})
        gc.collect()
        torch.cuda.empty_cache()

    metadata = {
        "experiment": "SEG005", "candidate": args.candidate, "status": "complete",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "candidate_spec": CANDIDATES[args.candidate], "installation": installation,
        "module_audit_sha256": stable_hash(args.module_audit), "manifest_dir": str(args.manifest_dir),
        "training": {"rows": len(train_rows), "microbatch": 1, "gradient_accumulation": args.gradient_accumulation, "updates": args.max_updates, "schedule_sha256": stable_hash(args.manifest_dir / "schedule.json"), "loss": "assistant answer tokens plus exactly one EOT", "seed": SEED},
        "evaluation": {"scope_rows": len(scope_rows), "scope_qids_sha256": stable_hash(args.manifest_dir / "scope_screen.jsonl"), "selection_rule": "official time accuracy, then event-instance macro accuracy, then median absolute error, then +/-5s, then +/-10s"},
        "history": history, "model_load_seconds": model_load_seconds, "train_wall_seconds": time.perf_counter() - training_started,
        "gpu": {"device": str(device), "name": torch.cuda.get_device_name(device), "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3},
        "platform": platform.platform(), "torch": torch.__version__, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prohibited_model_inputs": ["answer", "reference", "target_seconds", "official_tolerance_seconds", "event_kind", "fo_class", "primary", "ood", "answer_format"],
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "progress.json", {"status": "complete", "candidate": args.candidate, "history": history})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
