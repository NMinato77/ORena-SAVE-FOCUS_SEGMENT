"""Contracts and model surgery for the SEG006 FRAME-to-SEGMENT pilot.

SEG006 uses direct official FRAME foreign-object identification answers as a
single-frame visual teacher, then transfers only the explicitly adapted visual
pathway into the SEGMENT temporal SFT stage.  The module keeps the stage
boundary and trainable scopes auditable; it never creates boxes, masks,
tracks, temporal labels, or synthetic negatives.

PyNvVideoCodec must be imported before torch for every GPU experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import PyNvVideoCodec as nvc
import torch
from PIL import Image

from orena_procedure.exp094 import LoRALinear
from orena_procedure.exp102 import install_candidate as install_exp102_candidate
from orena_procedure.format_reducer import format_instruction
from orena_procedure.seg005 import merger_modules

EXPERIMENT = "SEG006"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_PIXELS = 50176
FRAME_UPDATES = 100
FRAME_CHECKPOINT_STEPS = (0, 25, 50, 100)
FRAME_GRADIENT_ACCUMULATION = 8
FRAME_TRAIN_ROWS = 800
FRAME_DEV_ROWS = 256
FRAME_SANITY_ROWS = 64
SEGMENT_GRADIENT_ACCUMULATION = 8
SEGMENT_UPDATES = 100
SEGMENT_A_UPDATES = 200
SEGMENT_CHECKPOINT_STEPS = (0, 10, 25, 50, 100)
SEGMENT_A_CHECKPOINT_STEPS = (100, 110, 125, 150, 200)
SEED = 6006
CACHE_FPS = 5.0
LANGUAGE_RANK = 16
LANGUAGE_ALPHA = 32.0
LANGUAGE_DROPOUT = 0.05
LANGUAGE_LR = 5e-5
VISION_RANK = 16
VISION_ALPHA = 32.0
VISION_DROPOUT = 0.05
VISION_LR = 1e-5
WEIGHT_DECAY = 0.01
FRAME_MAX_NEW_TOKENS = 32


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_frame_index(timestamp_seconds: float, cache_fps: float = CACHE_FPS) -> int:
    timestamp = float(timestamp_seconds)
    if not math.isfinite(timestamp) or timestamp < 0 or cache_fps <= 0:
        raise ValueError(f"invalid source frame timestamp: {timestamp_seconds}")
    return math.floor(timestamp * cache_fps + 0.5)


def _parent_and_child(root: torch.nn.Module, relative_name: str) -> tuple[torch.nn.Module, str]:
    parts = relative_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _vision_last_blocks_modules(model: torch.nn.Module, block_count: int) -> tuple[list[int], list[tuple[str, torch.nn.Linear]]]:
    blocks = model.model.visual.blocks
    indices = list(range(len(blocks) - block_count, len(blocks)))
    modules: list[tuple[str, torch.nn.Linear]] = []
    for index in indices:
        for name, module in blocks[index].named_modules():
            if isinstance(module, torch.nn.Linear):
                suffix = f"blocks.{index}.{name}" if name else f"blocks.{index}"
                modules.append((f"model.visual.{suffix}", module))
    return indices, modules


def _install_vision_lora(model: torch.nn.Module) -> list[str]:
    _indices, modules = _vision_last_blocks_modules(model, block_count=4)
    installed: list[str] = []
    for full_name, base in modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(parent, child, LoRALinear(base, VISION_RANK, VISION_ALPHA, VISION_DROPOUT))
        installed.append(full_name)
    return installed


def install_frame_scope(model: torch.nn.Module, mode: str) -> dict[str, Any]:
    """Install FRAME-V or FRAME-MV without a language adapter."""

    if mode not in {"V", "MV"}:
        raise ValueError(f"unknown FRAME mode: {mode}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    vision_modules = _install_vision_lora(model)
    merger_names: list[str] = []
    if mode == "MV":
        for name, module in merger_modules(model):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                parameter.requires_grad = True
                merger_names.append(f"{name}.{parameter_name}")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _parameter in trainable]
    expected_blocks = set(range(len(model.model.visual.blocks) - 4, len(model.model.visual.blocks)))
    actual_blocks = {
        int(name.split(".visual.blocks.", 1)[1].split(".", 1)[0])
        for name in names
        if ".visual.blocks." in name
    }
    if actual_blocks != expected_blocks:
        raise RuntimeError(f"FRAME vision scope mismatch: {actual_blocks} != {expected_blocks}")
    return {
        "stage": "FRAME",
        "mode": mode,
        "vision_modules": vision_modules,
        "merger_trainable_modules": merger_names,
        "trainable_parameter_names": names,
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
        "trainable_parameter_count_by_scope": {
            "vision_lora": int(sum(parameter.numel() for name, parameter in trainable if name in names and ".visual.blocks." in name)),
            "merger_full": int(sum(parameter.numel() for name, parameter in trainable if name in merger_names)),
        },
        "vision_block_indices": sorted(actual_blocks),
    }


def install_segment_scope(model: torch.nn.Module, candidate: str) -> dict[str, Any]:
    """Install the exact SEG005 L or LM scope after optional FRAME adapters."""

    if candidate not in {"L", "LM"}:
        raise ValueError(f"unknown SEGMENT candidate: {candidate}")
    installed = install_exp102_candidate(model, candidate)
    installed = dict(installed)
    installed["stage"] = "SEGMENT"
    installed["candidate"] = candidate
    return installed


def frame_adapted_state_names(model: torch.nn.Module, mode: str) -> list[str]:
    if mode not in {"V", "MV"}:
        raise ValueError(mode)
    names = [
        name
        for name, _parameter in model.named_parameters()
        if ".visual.blocks." in name and "lora_" in name
    ]
    if mode == "MV":
        names.extend(
            name
            for name, _parameter in model.named_parameters()
            if ".visual.merger." in name or ".visual.deepstack_merger_list." in name
        )
    return sorted(set(names))


def save_state(model: torch.nn.Module, path: Path, names: set[str] | None = None) -> None:
    selected = names or {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save(
        {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if name in selected},
        path,
    )


def load_state(model: torch.nn.Module, path: Path, names: set[str]) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != names:
        raise ValueError(
            f"state mismatch for {path}: missing={sorted(names - set(state))[:3]}, "
            f"unexpected={sorted(set(state) - names)[:3]}"
        )
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters:
            raise ValueError(f"state parameter absent from model: {name}")
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    return stable_hash(path)


def configure_train_mode(model: torch.nn.Module, stage: str, training: bool) -> None:
    model.train(training)
    model.model.visual.eval()
    if training and stage == "FRAME":
        for index in range(len(model.model.visual.blocks) - 4, len(model.model.visual.blocks)):
            model.model.visual.blocks[index].train()


def optimizer_for(model: torch.nn.Module, require_language: bool) -> tuple[torch.optim.Optimizer, dict[str, list[str]]]:
    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "language_lora": [],
        "merger_full": [],
        "vision_lora": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".language_model." in name and "lora_" in name:
            scope = "language_lora"
        elif ".visual.blocks." in name and "lora_" in name:
            scope = "vision_lora"
        elif ".visual.merger." in name or ".visual.deepstack_merger_list." in name:
            scope = "merger_full"
        else:
            raise ValueError(f"unexpected trainable parameter: {name}")
        buckets[scope].append((name, parameter))
    groups: list[dict[str, Any]] = []
    names: dict[str, list[str]] = {}
    for scope, values in buckets.items():
        if not values:
            continue
        names[scope] = [name for name, _parameter in values]
        groups.append(
            {
                "params": [parameter for _name, parameter in values],
                "lr": VISION_LR if scope == "vision_lora" else LANGUAGE_LR,
                "weight_decay": WEIGHT_DECAY,
                "scope": scope,
            }
        )
    if require_language and "language_lora" not in names:
        raise RuntimeError("SEGMENT stage has no language LoRA group")
    if not groups:
        raise RuntimeError("no trainable parameters")
    return torch.optim.AdamW(groups), names


def frame_messages(
    row: dict[str, Any],
    image: Image.Image,
    *,
    evidence_instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Use exactly the official question and one image; no metadata oracle."""

    text = (
        str(row["question"])
        if evidence_instruction is None
        else f"{evidence_instruction}\nQuestion: {row['question']}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]


def temporal_messages(row: dict[str, Any], images: list[Image.Image]) -> list[dict[str, Any]]:
    timestamps = [float(value) for value in row["frame_timestamps_seconds"]]
    if len(timestamps) != len(images):
        raise ValueError("timestamp/image count mismatch")
    content: list[dict[str, Any]] = []
    for timestamp, image in zip(timestamps, images, strict=True):
        content.extend(
            [
                {"type": "text", "text": f"Frame timestamp (absolute source-procedure timeline): {timestamp:0.1f} seconds."},
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
                "You are assisting with laparoscopic surgery. Use only the sampled frames as evidence. "
                f"Procedure type: {row['procedure_type']}. The request window is from {float(row['start_time']):.1f} "
                f"to {float(row['end_time']):.1f} seconds on the original source-procedure timeline. "
                "Sampled-frame timestamps are absolute source-procedure timeline timestamps. "
                f"{instruction}\nQuestion: {row['question']}"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def decode_indices(video_path: str, indices: list[int], gpu: int) -> tuple[list[Image.Image], dict[str, float]]:
    import time

    started = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        video_path,
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
        raise ValueError(f"decoded frame count mismatch: {video_path}")
    return images, {"video_loading": loaded, "frame_decoding": decoded, "image_conversion": converted}


def encode_frame_row(
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
    with_answer: bool,
    *,
    evidence_instruction: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    images, timings = decode_indices(str(row["video_path"]), [int(row["frame_index"])], int(device.index or 0))
    conversation: list[dict[str, Any]] = frame_messages(
        row,
        images[0],
        evidence_instruction=evidence_instruction,
    )
    if with_answer:
        conversation = [*conversation, {"role": "assistant", "content": str(row["answer"])}]
    encoded = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=not with_answer,
        return_dict=True,
        return_tensors="pt",
    )
    audit = {
        "dataset": str(row["dataset"]),
        "qID": str(row["qID"]),
        "frame_index": int(row["frame_index"]),
        "source_timestamp_seconds": float(row["source_timestamp_seconds"]),
        "decode_timings_seconds": timings,
        "input_tokens": int(encoded.input_ids.shape[1]),
    }
    if with_answer:
        answer_ids = processor.tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"]
        values = encoded.input_ids[0].tolist()
        candidates = [
            index
            for index in range(max(0, len(values) - len(answer_ids) - 64), len(values) - len(answer_ids) + 1)
            if values[index : index + len(answer_ids)] == answer_ids
        ]
        if not candidates:
            raise ValueError(f"FRAME answer span not found: {row['dataset']}/{row['qID']}")
        answer_start = candidates[-1]
        answer_end = answer_start + len(answer_ids)
        eot_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if answer_end >= len(values) or int(values[answer_end]) != eot_id:
            raise ValueError(f"FRAME answer is not followed by one EOT: {row['dataset']}/{row['qID']}")
        target_end = answer_end + 1
        labels = torch.full_like(encoded.input_ids[0], -100)
        labels[answer_start:target_end] = encoded.input_ids[0, answer_start:target_end]
        positions = torch.nonzero(labels != -100, as_tuple=False).flatten().tolist()
        if positions != list(range(answer_start, target_end)):
            raise ValueError(f"FRAME target mask mismatch: {row['dataset']}/{row['qID']}")
        audit.update(
            {
                "answer_token_count": len(answer_ids),
                "answer_token_start": answer_start,
                "target_end_exclusive": target_end,
                "eot_token_id": eot_id,
                "finite_label_count": len(positions),
                "masked_prefix_token_count": answer_start,
                "masked_suffix_token_count": len(values) - target_end,
            }
        )
    return encoded.to(device), audit


def encode_temporal_row(processor: Any, row: dict[str, Any], device: torch.device, with_answer: bool) -> tuple[Any, dict[str, Any]]:
    images, timings = decode_indices(
        str(row["video_path"]), [int(value) for value in row["frame_indices"]], int(device.index or 0)
    )
    conversation = temporal_messages(row, images)
    if with_answer:
        conversation = [*conversation, {"role": "assistant", "content": str(row["answer"])}]
    encoded = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=not with_answer,
        return_dict=True,
        return_tensors="pt",
    )
    audit = {
        "dataset": str(row["dataset"]),
        "qID": str(row["qID"]),
        "frame_count": len(images),
        "frame_timestamps_seconds": [float(value) for value in row["frame_timestamps_seconds"]],
        "decode_timings_seconds": timings,
        "input_tokens": int(encoded.input_ids.shape[1]),
    }
    if with_answer:
        answer_ids = processor.tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"]
        values = encoded.input_ids[0].tolist()
        candidates = [
            index
            for index in range(max(0, len(values) - len(answer_ids) - 64), len(values) - len(answer_ids) + 1)
            if values[index : index + len(answer_ids)] == answer_ids
        ]
        if not candidates:
            raise ValueError(f"SEGMENT answer span not found: {row['dataset']}/{row['qID']}")
        answer_start = candidates[-1]
        answer_end = answer_start + len(answer_ids)
        eot_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if answer_end >= len(values) or int(values[answer_end]) != eot_id:
            raise ValueError(f"SEGMENT answer is not followed by one EOT: {row['dataset']}/{row['qID']}")
        target_end = answer_end + 1
        audit.update(
            {
                "answer_token_count": len(answer_ids),
                "answer_token_start": answer_start,
                "target_end_exclusive": target_end,
                "eot_token_id": eot_id,
            }
        )
    return encoded.to(device), audit


def target_positions(encoded: Any, audit: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(audit["answer_token_start"])
    end = int(audit["target_end_exclusive"])
    positions = torch.arange(start, end, device=encoded.input_ids.device)
    return positions - 1, encoded.input_ids[0, start:end]


def candidate_parameter_norms(model: torch.nn.Module) -> dict[str, float]:
    values = [parameter.detach().float().square().sum() for parameter in model.parameters() if parameter.requires_grad]
    lora = [parameter.detach().float().square().sum() for name, parameter in model.named_parameters() if "lora_" in name]
    return {
        "trainable_parameter_l2": float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0,
        "lora_parameter_l2": float(torch.sqrt(torch.stack(lora).sum()).cpu()) if lora else 0.0,
    }
