#!/usr/bin/env python3
"""Run the frozen SEG010 final recipes with four-rank DDP.

This runner is intentionally separate from the historical two-GPU SEG010
screen trainer.  It implements the final-training contract only:

* fresh Qwen3-VL base plus the canonical zero-output SEG010 union;
* PF-C main-thread decode with a depth-one background PIL/processor worker;
* four ranks, global batch eight, two microbatches per rank;
* explicit General F,F,S and Aggregation stage schedules;
* complete resume checkpoints every 250 optimizer updates, plus immutable
  stage-boundary and FINAL checkpoints.

PyNvVideoCodec must be imported before torch or CUDA model initialization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
TRAINING_SRC = TRAINING_ROOT / "src"
for _import_root in (TRAINING_ROOT, TRAINING_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

MODEL_CACHE_ROOT = Path(os.environ.get("SEGMENT_CACHE_ROOT", "/cache")).expanduser().resolve()
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import PyNvVideoCodec as nvc  # mandatory import order: before torch/CUDA.
import torch
import torch.distributed as dist
from PIL import Image
from run_seg005_scope import loss_for_encoded, temporal_messages
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from train_seg010_ddp import load_canonical_initialization, load_model

from orena_procedure.seg006 import frame_messages
from orena_procedure.seg010 import (
    FRAME_EVIDENCE_INSTRUCTION,
    GLOBAL_BATCH,
    LANGUAGE_LR,
    MERGER_LR,
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    SEGMENT_EVIDENCE_INSTRUCTION,
    SEGMENT_TIMESTAMP_CONTEXT,
    SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
    VISION_LR,
    WEIGHT_DECAY,
    checkpoint_union_state,
    expected_union_names,
    install_union,
    make_optimizer,
    optimizer_group_names,
    optimizer_state_hash,
    parameter_group_norms,
    parameter_group_snapshot,
    parameter_group_update_norms,
    prompt_contract,
    read_json,
    select_task,
    set_train_modes,
    sha256_file,
    tensor_hash,
    validate_union,
)

WORLD_SIZE_REQUIRED = 4
MICROBATCH_PER_GPU = 1
GRADIENT_ACCUMULATION_FINAL = 2
CHECKPOINT_INTERVAL = 250
RECOVERY_RETENTION = 3
GENERAL_CONDITION = "PS_JOINT_LMV"
DEFAULT_ARTIFACT_ROOT = TRAINING_ROOT / "artifacts/final_manifests"
DEFAULT_SCOPE_ROOT = TRAINING_ROOT / "artifacts/scope"
DEFAULT_INIT_DIR = TRAINING_ROOT / "weights/canonical_union"
DEFAULT_OUTPUT_ROOT = PUBLIC_ROOT / "outputs/SEG010_FINAL"
FOCUS_DATA_ROOT = Path(os.environ.get("FOCUS_DATA_ROOT", "/data/focus")).expanduser().resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_path = str(row.get("video_path", ""))
        if video_path.startswith("${FOCUS_DATA_ROOT}/"):
            row["video_path"] = str(FOCUS_DATA_ROOT / video_path.removeprefix("${FOCUS_DATA_ROOT}/"))
        elif video_path.startswith("/data/focus/"):
            # Accept historical manifests while keeping the public manifests portable.
            row["video_path"] = str(FOCUS_DATA_ROOT / video_path.removeprefix("/data/focus/"))
        rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PUBLIC_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def gpu_snapshot() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            result.append(
                {
                    "index": values[0],
                    "name": values[1],
                    "memory_total_mib": values[2],
                    "memory_used_mib": values[3],
                    "utilization_gpu_percent": values[4],
                }
            )
    return result


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def local_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(value: Any, world_size: int) -> list[Any]:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def initialize_distributed() -> tuple[int, int, torch.device]:
    required = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
    if not required.issubset(os.environ):
        raise RuntimeError("final SEG010 training must be launched with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"final SEG010 training requires WORLD_SIZE=4, got {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("final SEG010 training requires four visible CUDA devices")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, device


def qa_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}:{row['qID']}"


def validate_manifest(
    rows: list[dict[str, Any]],
    kind: str,
    expected_count: int,
    *,
    expected_unique_qa: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{kind} manifest count {len(rows)} != expected {expected_count}")
    seen: set[str] = set()
    for row in rows:
        key = qa_key(row)
        seen.add(key)
        for field in ("dataset", "qID", "question", "answer", "video_path"):
            if field not in row:
                raise ValueError(f"{kind} row missing {field}: {key}")
        if not str(row["answer"]).strip():
            raise ValueError(f"empty {kind} answer: {key}")
        if not Path(str(row["video_path"])).is_file():
            raise FileNotFoundError(f"missing {kind} video: {row['video_path']}")
    if len(seen) != expected_unique_qa:
        raise AssertionError(f"{kind} unique QA count {len(seen)} != expected {expected_unique_qa}")


def load_final_manifests(artifact_root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = {
        "GENERAL_FRAME": artifact_root / "general_ps_frame_stream.jsonl",
        "GENERAL_SEGMENT_STAGE1": artifact_root / "general_stage1_seg_shuffle.jsonl",
        "GENERAL_SEGMENT_STAGE2": artifact_root / "general_stage2_seg_shuffle.jsonl",
        "AGG_FRAME": artifact_root / "aggregation_rand_frame_2048.jsonl",
        "AGG_SEGMENT_TRANSFER": artifact_root / "aggregation_general_seg_2048.jsonl",
        "AGG_SEGMENT_FINAL": artifact_root / "aggregation_fullpublic_shuffle.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    validate_manifest(rows["GENERAL_FRAME"], "GENERAL_FRAME", 36_464, expected_unique_qa=10_073)
    validate_manifest(rows["GENERAL_SEGMENT_STAGE1"], "GENERAL_SEGMENT_STAGE1", 18_232, expected_unique_qa=18_227)
    validate_manifest(rows["GENERAL_SEGMENT_STAGE2"], "GENERAL_SEGMENT_STAGE2", 18_232, expected_unique_qa=18_227)
    validate_manifest(rows["AGG_FRAME"], "AGG_FRAME", 2_048, expected_unique_qa=2_048)
    validate_manifest(rows["AGG_SEGMENT_TRANSFER"], "AGG_SEGMENT_TRANSFER", 2_048, expected_unique_qa=1_024)
    validate_manifest(rows["AGG_SEGMENT_FINAL"], "AGG_SEGMENT_FINAL", 1_776, expected_unique_qa=1_773)
    return rows


def make_update(
    *,
    global_step: int,
    stage: str,
    task: str,
    task_step: int,
    manifest_key: str,
    qa_start: int,
    qa_end_exclusive: int,
    cycle_index: int | None = None,
    cycle_position: int | None = None,
    optimizer_reset_before: bool = False,
) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "stage": stage,
        "task": task,
        "task_step": task_step,
        "manifest_key": manifest_key,
        "qa_start": qa_start,
        "qa_end_exclusive": qa_end_exclusive,
        "cycle_index": cycle_index,
        "cycle_position": cycle_position,
        "optimizer_reset_before": optimizer_reset_before,
    }


def build_schedule(recipe: str) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    global_step = 0
    if recipe == "general":
        frame_step = segment_step = 0
        for cycle in range(2_279):
            for position, task in enumerate(("F", "F", "S")):
                global_step += 1
                if task == "F":
                    frame_step += 1
                    task_step = frame_step
                    manifest_key = "GENERAL_FRAME"
                else:
                    segment_step += 1
                    task_step = segment_step
                    manifest_key = "GENERAL_SEGMENT_STAGE1"
                schedule.append(
                    make_update(
                        global_step=global_step,
                        stage="GENERAL_STAGE1",
                        task=task,
                        task_step=task_step,
                        manifest_key=manifest_key,
                        qa_start=(task_step - 1) * GLOBAL_BATCH,
                        qa_end_exclusive=task_step * GLOBAL_BATCH,
                        cycle_index=cycle + 1,
                        cycle_position=position + 1,
                    )
                )
        if (frame_step, segment_step, global_step) != (4_558, 2_279, 6_837):
            raise AssertionError("General Stage 1 schedule count mismatch")
        for task_step in range(1, 2_280):
            global_step += 1
            schedule.append(
                make_update(
                    global_step=global_step,
                    stage="GENERAL_STAGE2",
                    task="S",
                    task_step=task_step,
                    manifest_key="GENERAL_SEGMENT_STAGE2",
                    qa_start=(task_step - 1) * GLOBAL_BATCH,
                    qa_end_exclusive=task_step * GLOBAL_BATCH,
                    cycle_index=None,
                    cycle_position=None,
                )
            )
        if len(schedule) != 9_116:
            raise AssertionError("General full schedule count mismatch")
        return schedule
    if recipe == "aggregation":
        global_step = 0
        for task_step in range(1, 257):
            global_step += 1
            schedule.append(
                make_update(
                    global_step=global_step,
                    stage="AGG_STAGE1_FRAME",
                    task="F",
                    task_step=task_step,
                    manifest_key="AGG_FRAME",
                    qa_start=(task_step - 1) * GLOBAL_BATCH,
                    qa_end_exclusive=task_step * GLOBAL_BATCH,
                )
            )
        for task_step in range(1, 257):
            global_step += 1
            schedule.append(
                make_update(
                    global_step=global_step,
                    stage="AGG_STAGE2_TRANSFER",
                    task="S",
                    task_step=task_step,
                    manifest_key="AGG_SEGMENT_TRANSFER",
                    qa_start=(task_step - 1) * GLOBAL_BATCH,
                    qa_end_exclusive=task_step * GLOBAL_BATCH,
                )
            )
        for task_step in range(1, 223):
            global_step += 1
            schedule.append(
                make_update(
                    global_step=global_step,
                    stage="AGG_STAGE3_FULL_PUBLIC",
                    task="S",
                    task_step=task_step,
                    manifest_key="AGG_SEGMENT_FINAL",
                    qa_start=(task_step - 1) * GLOBAL_BATCH,
                    qa_end_exclusive=task_step * GLOBAL_BATCH,
                    optimizer_reset_before=task_step == 1,
                )
            )
        if len(schedule) != 734:
            raise AssertionError("Aggregation full schedule count mismatch")
        return schedule
    raise ValueError(recipe)


def schedule_sha256(schedule: list[dict[str, Any]]) -> str:
    return json_sha256(schedule)


def decode_arrays(row: dict[str, Any], device: torch.device) -> tuple[list[np.ndarray], dict[str, float]]:
    if "frame_index" in row:
        indices = [int(row["frame_index"])]
    else:
        indices = [int(value) for value in row["frame_indices"]]
    started = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        str(row["video_path"]),
        gpu_id=int(device.index or 0),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
        bWaitForSessionWarmUp=True,
    )
    loading = time.perf_counter() - started
    started = time.perf_counter()
    surfaces = decoder.get_batch_frames_by_index(indices)
    arrays = [torch.utils.dlpack.from_dlpack(surface).cpu().numpy() for surface in surfaces]
    torch.cuda.synchronize(device)
    decoding = time.perf_counter() - started
    del surfaces, decoder
    if len(arrays) != len(indices):
        raise ValueError(f"decoded frame count mismatch: {qa_key(row)}")
    return arrays, {"video_loading": loading, "frame_decoding": decoding}


def answer_span(processor: Any, input_ids: list[int], answer: str) -> tuple[int, int, int]:
    answer_ids = processor.tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    candidates = [
        index
        for index in range(max(0, len(input_ids) - len(answer_ids) - 64), len(input_ids) - len(answer_ids) + 1)
        if input_ids[index : index + len(answer_ids)] == answer_ids
    ]
    if not candidates:
        raise ValueError("assistant answer span not found")
    start = candidates[-1]
    end = start + len(answer_ids)
    eot_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if end >= len(input_ids) or int(input_ids[end]) != eot_id:
        raise ValueError("answer is not followed by exactly one EOT")
    return start, end, end + 1


def encode_cpu(processor: Any, row: dict[str, Any], arrays: list[np.ndarray], task: str) -> tuple[Any, dict[str, Any]]:
    images = [Image.fromarray(array) for array in arrays]
    if task == "F":
        if len(images) != 1:
            raise ValueError(f"FRAME requires one image: {qa_key(row)}")
        messages = frame_messages(
            row,
            images[0],
            evidence_instruction=FRAME_EVIDENCE_INSTRUCTION,
        )
    elif task == "S":
        messages = temporal_messages(
            row,
            images,
            evidence_instruction=SEGMENT_EVIDENCE_INSTRUCTION,
            timestamp_frame_template=SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
            timestamp_context=SEGMENT_TIMESTAMP_CONTEXT,
        )
    else:
        raise ValueError(task)
    conversation = [*messages, {"role": "assistant", "content": str(row["answer"])}]
    encoded = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    start, end, target_end = answer_span(processor, encoded.input_ids[0].tolist(), str(row["answer"]))
    audit = {
        "dataset": str(row["dataset"]),
        "qID": str(row["qID"]),
        "answer_start": start,
        "answer_end": end,
        "target_end": target_end,
        "answer_token_count": end - start,
        "frame_count": len(images),
        "prompt_tokens": int(encoded.input_ids.shape[1]),
    }
    return encoded, audit


@dataclass
class DecodedItem:
    row: dict[str, Any]
    arrays: list[np.ndarray]
    timings: dict[str, float]


@dataclass
class PreparedItem:
    row: dict[str, Any]
    encoded_cpu: Any
    audit: dict[str, Any]
    timings: dict[str, float]
    ready_at: float


class PFCProducer:
    """PF-C: main-thread decode, depth-one background PIL+processor."""

    def __init__(self, processor: Any, depth: int = 1) -> None:
        self.processor = processor
        self.input_queue: queue.Queue[DecodedItem | None] = queue.Queue(maxsize=depth)
        self.output_queue: queue.Queue[PreparedItem | None] = queue.Queue(maxsize=depth)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="seg010-pfc-producer", daemon=True)
        self.rows_produced = 0
        self.processor_seconds: list[float] = []

    def _run(self) -> None:
        try:
            while True:
                item = self.input_queue.get()
                if item is None:
                    return
                started = time.perf_counter()
                encoded, audit = encode_cpu(self.processor, item.row, item.arrays, item.row["_task"])
                self.processor_seconds.append(time.perf_counter() - started)
                self.rows_produced += 1
                self.output_queue.put(
                    PreparedItem(
                        row=item.row,
                        encoded_cpu=encoded,
                        audit=audit,
                        timings={**item.timings, "processor": self.processor_seconds[-1]},
                        ready_at=time.perf_counter(),
                    )
                )
        except BaseException as exc:  # propagate producer failures to consumer
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def submit(self, item: DecodedItem) -> None:
        if self.error is not None:
            raise RuntimeError("PF-C producer failed") from self.error
        self.input_queue.put(item)

    def get(self, timeout_seconds: float = 600.0) -> PreparedItem:
        started = time.perf_counter()
        while True:
            if self.error is not None:
                raise RuntimeError("PF-C producer failed") from self.error
            try:
                item = self.output_queue.get(timeout=1.0)
                if item is None:
                    raise RuntimeError("PF-C producer ended before producing requested item")
                item.timings["producer_wait"] = time.perf_counter() - started
                return item
            except queue.Empty:
                if time.perf_counter() - started >= timeout_seconds:
                    raise TimeoutError("PF-C producer output timeout") from None

    def close(self) -> None:
        self.input_queue.put(None)
        self.thread.join(timeout=600.0)
        if self.thread.is_alive():
            raise TimeoutError("PF-C producer did not terminate")
        if self.error is not None:
            raise RuntimeError("PF-C producer failed") from self.error


def finite_gradients(model: torch.nn.Module, selected_names: set[str]) -> None:
    parameters = dict(model.named_parameters())
    for name in selected_names:
        parameter = parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"active parameter has no gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def optimizer_step_summary(optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    if optimizer is None:
        return {"present": False, "min": 0, "max": 0, "unique": []}
    values: list[int] = []
    for state in optimizer.state.values():
        value = state.get("step", 0)
        if isinstance(value, torch.Tensor):
            value = int(value.item())
        values.append(int(value))
    return {
        "present": True,
        "min": min(values, default=0),
        "max": max(values, default=0),
        "unique": sorted(set(values)),
        "parameter_states": len(values),
    }


def checkpoint_state_path(checkpoint_dir: Path, name: str) -> Path:
    return checkpoint_dir / name


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_kind: str,
    recipe: str,
    model: torch.nn.Module,
    scope_root: Path,
    manifest_root: Path,
    init_metadata: dict[str, Any],
    schedule: list[dict[str, Any]],
    global_step: int,
    stage: str,
    current_item: dict[str, Any] | None,
    manifests: dict[str, list[dict[str, Any]]],
    optimizer_f: torch.optim.Optimizer | None,
    optimizer_s: torch.optim.Optimizer | None,
    optimizer_generations: dict[str, int],
    device: torch.device,
    rank: int,
    world_size: int,
    run_started_at: str,
    runtime_seconds: float,
) -> Path:
    rng_states = all_gather_object(local_rng_state(device), world_size)
    rank_hashes = all_gather_object(tensor_hash(checkpoint_union_state(model, scope_root)), world_size)
    if rank == 0:
        if checkpoint_dir.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
        temp_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.tmp-{os.getpid()}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        weights = checkpoint_union_state(model, scope_root)
        union_digest = tensor_hash(weights)
        save_file(weights, str(temp_dir / "union_adapted_weights.safetensors"))
        for name, optimizer in (("F", optimizer_f), ("S", optimizer_s)):
            torch.save(
                {
                    "present": optimizer is not None,
                    "state_dict": None if optimizer is None else optimizer.state_dict(),
                },
                temp_dir / f"optimizer_{name}.pt",
            )
            # The final recipe intentionally has no scheduler.  Persisting an
            # explicit absent state makes the resume contract unambiguous.
            torch.save({"present": False, "state_dict": None}, temp_dir / f"scheduler_{name}.pt")
        for rng_rank, state in enumerate(rng_states):
            torch.save(state, temp_dir / f"rng_rank_{rng_rank}.pt")
        next_item = schedule[global_step] if global_step < len(schedule) else None
        cursors: dict[str, dict[str, Any]] = {}
        for key, rows in manifests.items():
            future_item = next(
                (item for item in schedule[global_step:] if item["manifest_key"] == key),
                None,
            )
            cursor = len(rows) if future_item is None else int(future_item["qa_start"])
            cursors[key] = {
                "next_qa_cursor": cursor,
                "manifest_rows": len(rows),
                "manifest_sha256": sha256_file(
                    manifest_root
                    / {
                        "GENERAL_FRAME": "general_ps_frame_stream.jsonl",
                        "GENERAL_SEGMENT_STAGE1": "general_stage1_seg_shuffle.jsonl",
                        "GENERAL_SEGMENT_STAGE2": "general_stage2_seg_shuffle.jsonl",
                        "AGG_FRAME": "aggregation_rand_frame_2048.jsonl",
                        "AGG_SEGMENT_TRANSFER": "aggregation_general_seg_2048.jsonl",
                        "AGG_SEGMENT_FINAL": "aggregation_fullpublic_shuffle.jsonl",
                    }[key]
                ),
            }
        metadata = {
            "schema_version": 1,
            "experiment": "SEG010_FINAL_v2",
            "recipe": recipe,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_cadence": "recovery every 250 optimizer updates; stage boundary and FINAL independent",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
            "prompt_contract": prompt_contract(),
            "training_loss": "W0 unweighted answer+EOT mean NLL; no loss weighting",
            "evaluation_primary": "answer-content NLL; capability x dataset 10-cell equal macro",
            "git_commit": git_commit(),
            "initialization": {
                "weights_sha256": init_metadata["weights_sha256"],
                "tensor_sha256": init_metadata["tensor_sha256"],
            },
            "global_optimizer_update": global_step,
            "stage": stage,
            "schedule_length": len(schedule),
            "schedule_sha256": schedule_sha256(schedule),
            "schedule_position_next": global_step,
            "current_update": current_item,
            "next_update": next_item,
            "manifest_cursors": cursors,
            "task_updates_completed": {
                "F": sum(item["task"] == "F" for item in schedule[:global_step]),
                "S": sum(item["task"] == "S" for item in schedule[:global_step]),
            },
            "ff_s_phase": (
                {"mode": "F,F,S", "cycle_index": int(next_item["cycle_index"]), "cycle_position": int(next_item["cycle_position"])}
                if next_item is not None and next_item["stage"] == "GENERAL_STAGE1"
                else {"mode": "SEGMENT_ONLY" if next_item is not None else "COMPLETE"}
            ),
            "optimizer_generations": optimizer_generations,
            "optimizer_step_counters": {
                "F": optimizer_step_summary(optimizer_f),
                "S": optimizer_step_summary(optimizer_s),
            },
            "optimizer_state_sha256": {
                "F": optimizer_state_hash(optimizer_f),
                "S": optimizer_state_hash(optimizer_s),
            },
            "scheduler": {
                "F": {"present": False, "state_file": "scheduler_F.pt"},
                "S": {"present": False, "state_file": "scheduler_S.pt"},
            },
            "union_tensor_count": len(expected_union_names(scope_root)),
            "union_parameter_count": int(sum(value.numel() for value in weights.values())),
            "union_tensor_sha256": union_digest,
            "rank_model_hashes": rank_hashes,
            "world_size": world_size,
            "global_batch": GLOBAL_BATCH,
            "microbatch_per_gpu": MICROBATCH_PER_GPU,
            "gradient_accumulation": GRADIENT_ACCUMULATION_FINAL,
            "rng_files": [f"rng_rank_{index}.pt" for index in range(world_size)],
            "runtime": {"run_started_at_utc": run_started_at, "runtime_seconds": runtime_seconds},
            "gpu": gpu_snapshot(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(temp_dir / "cursor.json", {"global_optimizer_update": global_step, "stage": stage, "current_update": current_item, "next_update": next_item, "manifest_cursors": cursors, "ff_s_phase": metadata["ff_s_phase"]})
        write_json(temp_dir / "metadata.json", metadata)
        temp_dir.rename(checkpoint_dir)
    barrier()
    return checkpoint_dir


def retain_recovery_checkpoints(checkpoint_root: Path, keep: int = RECOVERY_RETENTION) -> None:
    recovery = sorted(checkpoint_root.glob("recovery_step_*"))
    for path in recovery[:-keep]:
        shutil.rmtree(path)


def load_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    scope_root: Path,
    init_metadata: dict[str, Any],
    optimizer_f: torch.optim.Optimizer | None,
    optimizer_s: torch.optim.Optimizer | None,
    device: torch.device,
    rank: int,
    world_size: int,
    schedule: list[dict[str, Any]],
    recipe: str,
) -> tuple[int, str, dict[str, int]]:
    metadata = read_json(checkpoint_dir / "metadata.json")
    if metadata["recipe"] != recipe:
        raise ValueError("resume recipe mismatch")
    if metadata["model"]["id"] != MODEL_ID or metadata["model"]["revision"] != MODEL_REVISION:
        raise ValueError("resume model identity mismatch")
    if metadata["initialization"]["tensor_sha256"] != init_metadata["tensor_sha256"]:
        raise ValueError("resume canonical initialization mismatch")
    if metadata["schedule_sha256"] != schedule_sha256(schedule):
        raise ValueError("resume schedule mismatch")
    global_step = int(metadata["global_optimizer_update"])
    if int(metadata["schedule_position_next"]) != global_step:
        raise ValueError("resume schedule position mismatch")
    expected_next = schedule[global_step] if global_step < len(schedule) else None
    if metadata.get("next_update") != expected_next:
        raise ValueError("resume next-update cursor mismatch")
    expected_current = schedule[global_step - 1] if global_step else None
    if metadata.get("current_update") != expected_current:
        raise ValueError("resume current-update cursor mismatch")
    if int(metadata["world_size"]) != world_size:
        raise ValueError("resume world size mismatch")
    weights = load_file(str(checkpoint_dir / "union_adapted_weights.safetensors"), device="cpu")
    if set(weights) != expected_union_names(scope_root):
        raise ValueError("resume union parameter set mismatch")
    if tensor_hash(weights) != metadata["union_tensor_sha256"]:
        raise ValueError("resume union tensor hash mismatch")
    parameters = dict(model.named_parameters())
    for name, value in weights.items():
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    for name, optimizer in (("F", optimizer_f), ("S", optimizer_s)):
        payload = torch.load(checkpoint_dir / f"optimizer_{name}.pt", map_location="cpu", weights_only=False)
        if bool(payload["present"]) != (optimizer is not None):
            raise ValueError(f"resume optimizer presence mismatch: {name}")
        if optimizer is not None:
            optimizer.load_state_dict(payload["state_dict"])
        scheduler = torch.load(checkpoint_dir / f"scheduler_{name}.pt", map_location="cpu", weights_only=False)
        if bool(scheduler["present"]):
            raise ValueError("final recipe unexpectedly contains a scheduler state")
    rng_path = checkpoint_dir / f"rng_rank_{rank}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(rng_path)
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False), device)
    return global_step, str(metadata["stage"]), {key: int(value) for key, value in metadata["optimizer_generations"].items()}


def checkpoint_label(recipe: str, stage: str) -> str:
    if recipe == "general" and stage == "GENERAL_STAGE1":
        return "GENERAL_STAGE1"
    if recipe == "general" and stage == "GENERAL_STAGE2":
        return "GENERAL_FINAL"
    if recipe == "aggregation" and stage == "AGG_STAGE1_FRAME":
        return "AGG_FRAME_STAGE1"
    if recipe == "aggregation" and stage == "AGG_STAGE2_TRANSFER":
        return "AGG_TRANSFER"
    if recipe == "aggregation" and stage == "AGG_STAGE3_FULL_PUBLIC":
        return "AGG_FINAL"
    raise ValueError((recipe, stage))


def expected_boundary(recipe: str, global_step: int, schedule: list[dict[str, Any]]) -> str | None:
    if recipe == "general" and global_step == 6_837:
        return "GENERAL_STAGE1"
    if recipe == "aggregation" and global_step in {256, 512}:
        return "AGG_FRAME_STAGE1" if global_step == 256 else "AGG_TRANSFER"
    if global_step == len(schedule):
        return checkpoint_label(recipe, schedule[-1]["stage"])
    return None


def prepare_model(
    scope_root: Path,
    init_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    model, processor = load_model(device)
    installation = install_union(model)
    validate_union(model, scope_root)
    init_metadata = load_canonical_initialization(model, scope_root, init_dir)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    union_names = union_parameter_names_for_model(model, scope_root)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in union_names)
    return model, processor, {"installation": installation, **init_metadata}


def union_parameter_names_for_model(model: torch.nn.Module, artifact_root: Path) -> set[str]:
    expected = expected_union_names(artifact_root)
    actual = {
        name
        for name, _parameter in model.named_parameters()
        if ("lora_" in name and (".language_model." in name or ".visual.blocks." in name))
        or ".visual.merger." in name
        or ".visual.deepstack_merger_list." in name
    }
    if actual != expected:
        raise ValueError("final model union scope mismatch")
    return actual


def run(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root.resolve()
    scope_root = args.scope_root.resolve()
    output_dir = args.output_dir.resolve()
    rank, local_rank, device = initialize_distributed()
    seed_everything()
    manifests = load_final_manifests(artifact_root)
    schedule = build_schedule(args.recipe)
    target_updates = len(schedule) if args.max_updates is None else int(args.max_updates)
    if target_updates < 1 or target_updates > len(schedule):
        raise ValueError(f"invalid target update count: {target_updates}")
    if rank == 0:
        if args.resume is None:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty final output: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            if not output_dir.is_dir():
                raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
    barrier()
    model, processor, init_metadata = prepare_model(scope_root, args.init_dir.resolve(), device)
    optimizer_f = make_optimizer(model, scope_root, GENERAL_CONDITION, "F")
    optimizer_s = make_optimizer(model, scope_root, GENERAL_CONDITION, "S")
    training_model: torch.nn.Module = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        static_graph=False,
    )
    optimizer_generations = {"F": 0, "S": 0}
    start_index = 0
    resumed_stage = "INITIAL"
    if args.resume is not None:
        start_index, resumed_stage, optimizer_generations = load_checkpoint(
            checkpoint_dir=args.resume.resolve(),
            model=model,
            scope_root=scope_root,
            init_metadata=init_metadata,
            optimizer_f=optimizer_f,
            optimizer_s=optimizer_s,
            device=device,
            rank=rank,
            world_size=WORLD_SIZE_REQUIRED,
            schedule=schedule,
            recipe=args.recipe,
        )
        if start_index >= len(schedule):
            raise ValueError("resume checkpoint is already complete")
    run_started_at = datetime.now(timezone.utc).isoformat()
    if rank == 0:
        config = {
            "experiment": "SEG010_FINAL_v2",
            "recipe": args.recipe,
            "artifact_root": str(artifact_root),
            "scope_root": str(scope_root),
            "artifact_manifest_sha256": read_json(artifact_root / "manifest_sha.json"),
            "schedule_sha256": schedule_sha256(schedule),
            "schedule_length": len(schedule),
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
            "prompt_contract": prompt_contract(),
            "training_loss": "W0 unweighted answer+EOT mean NLL; no loss weighting",
            "evaluation_primary": "answer-content NLL; capability x dataset 10-cell equal macro",
            "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY, "scheduler": None, "learning_rates": {"language_lora": LANGUAGE_LR, "vision_lora": VISION_LR, "merger_full": MERGER_LR}},
            "batch": {"global_batch": GLOBAL_BATCH, "world_size": WORLD_SIZE_REQUIRED, "microbatch_per_gpu": MICROBATCH_PER_GPU, "gradient_accumulation": GRADIENT_ACCUMULATION_FINAL},
            "prefetch": {"selected": "PF-C_main_decode", "queue_depth": 1, "producer": "main-thread PyNvVideoCodec decode; background PIL+processor", "consumer": "main-thread H2D+forward/backward+optimizer+DDP", "processor_on_consumer": False},
            "checkpoint": {"recovery_interval_optimizer_updates": CHECKPOINT_INTERVAL, "recovery_retention": RECOVERY_RETENTION, "stage_boundary_and_final_permanent": True, "resume_fields": ["model", "optimizer", "scheduler", "global_optimizer_update", "stage", "manifest_cursor", "F:F:S phase", "per-rank RNG"]},
            "init_metadata": init_metadata,
            "git_commit": git_commit(),
            "git_status": subprocess.check_output(
                ["git", "-C", str(PUBLIC_ROOT), "status", "--porcelain=v1"], text=True
            ),
            "gpu": gpu_snapshot(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "resumed_from": None if args.resume is None else str(args.resume.resolve()),
            "resumed_global_step": start_index,
            "resumed_stage": resumed_stage,
        }
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "schedule.json", schedule)
        write_json(output_dir / "status.json", {"status": "running", "global_optimizer_update": start_index, "target_updates": target_updates, "stage": resumed_stage, "started_at_utc": run_started_at})
    barrier()
    checkpoint_root = output_dir / "checkpoints"
    if args.resume is None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            checkpoint_dir=checkpoint_root / "initial",
            checkpoint_kind="initial",
            recipe=args.recipe,
            model=model,
            scope_root=scope_root,
            manifest_root=artifact_root,
            init_metadata=init_metadata,
            schedule=schedule,
            global_step=0,
            stage="INITIAL",
            current_item=None,
            manifests=manifests,
            optimizer_f=optimizer_f,
            optimizer_s=optimizer_s,
            optimizer_generations=optimizer_generations,
            device=device,
            rank=rank,
            world_size=WORLD_SIZE_REQUIRED,
            run_started_at=run_started_at,
            runtime_seconds=0.0,
        )
    producer = PFCProducer(processor, depth=1)
    producer.start()
    metrics_path = output_dir / "training_metrics.jsonl"
    started = time.perf_counter()
    try:
        for schedule_index in range(start_index, target_updates):
            item = schedule[schedule_index]
            if item["optimizer_reset_before"]:
                if args.recipe != "aggregation" or item["stage"] != "AGG_STAGE3_FULL_PUBLIC":
                    raise AssertionError("unexpected optimizer reset marker")
                if optimizer_generations["S"] != 0:
                    raise RuntimeError("Aggregation optimizer reset state is inconsistent")
                optimizer_s = make_optimizer(model, scope_root, GENERAL_CONDITION, "S")
                optimizer_generations["S"] = 1
            task = str(item["task"])
            rows = manifests[str(item["manifest_key"])][int(item["qa_start"]) : int(item["qa_end_exclusive"])]
            if len(rows) != GLOBAL_BATCH:
                raise ValueError(f"batch row count mismatch at step {item['global_step']}: {len(rows)}")
            local_rows = rows[rank::WORLD_SIZE_REQUIRED]
            if len(local_rows) != GRADIENT_ACCUMULATION_FINAL:
                raise ValueError(f"rank-local batch mismatch at step {item['global_step']}: {len(local_rows)}")
            active_optimizer = optimizer_f if task == "F" else optimizer_s
            if active_optimizer is None:
                raise ValueError(f"missing optimizer for task {task}")
            selected_names = set(select_task(model, scope_root, GENERAL_CONDITION, task))
            groups = optimizer_group_names(scope_root, GENERAL_CONDITION, task)
            if groups is None or set().union(*groups.values()) != selected_names:
                raise ValueError(f"optimizer scope mismatch at step {item['global_step']}")
            set_train_modes(model, task, selected_names)
            model.zero_grad(set_to_none=True)
            # Mark task only on the local copies consumed by the producer.
            task_rows = []
            decode_started = time.perf_counter()
            for original_row in local_rows:
                row = dict(original_row)
                row["_task"] = task
                arrays, timings = decode_arrays(row, device)
                task_rows.append((row, timings))
                producer.submit(DecodedItem(row=row, arrays=arrays, timings=timings))
            decode_and_submit_seconds = time.perf_counter() - decode_started
            losses: list[float] = []
            answer_tokens = 0
            update_started = time.perf_counter()
            before = parameter_group_snapshot(model, groups)
            for micro_index, (expected_row, _timings) in enumerate(task_rows):
                prepared = producer.get()
                if qa_key(prepared.row) != qa_key(expected_row) or prepared.row["_task"] != task:
                    raise RuntimeError(f"PF-C QA/task order mismatch at step {item['global_step']}")
                encoded = prepared.encoded_cpu.to(device)
                sync_context = training_model.no_sync() if micro_index < GRADIENT_ACCUMULATION_FINAL - 1 else nullcontext()
                with sync_context:
                    loss = loss_for_encoded(training_model, encoded, prepared.audit) / GRADIENT_ACCUMULATION_FINAL
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {item['global_step']}")
                    loss.backward()
                losses.append(float(loss.detach().cpu()) * GRADIENT_ACCUMULATION_FINAL)
                answer_tokens += int(prepared.audit["answer_token_count"])
                del encoded, loss, prepared
            finite_gradients(model, selected_names)
            gradient_norms = parameter_group_norms(model, groups)
            active_optimizer.step()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - update_started
            update_norms = parameter_group_update_norms(model, groups, before)
            local_stats = torch.tensor([sum(losses), len(losses), answer_tokens], dtype=torch.float64, device=device)
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
            mean_loss = float((local_stats[0] / local_stats[1]).item())
            rank_record = {
                "rank": rank,
                "global_step": int(item["global_step"]),
                "stage": item["stage"],
                "task": task,
                "mean_loss_local": float(np.mean(losses)),
                "seconds_per_update_local": elapsed,
                "decode_submit_seconds": decode_and_submit_seconds,
                "answer_token_count_local": answer_tokens,
                "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "gradient_norm_by_group": gradient_norms,
                "parameter_update_norm_by_group": update_norms,
            }
            gathered = all_gather_object(rank_record, WORLD_SIZE_REQUIRED)
            global_step = schedule_index + 1
            if rank == 0:
                record = {
                    "global_optimizer_update": global_step,
                    "stage": item["stage"],
                    "task": task,
                    "task_step": item["task_step"],
                    "cycle_index": item["cycle_index"],
                    "cycle_position": item["cycle_position"],
                    "qa_start": item["qa_start"],
                    "qa_end_exclusive": item["qa_end_exclusive"],
                    "manifest_key": item["manifest_key"],
                    "mean_loss": mean_loss,
                    "answer_token_count": int(local_stats[2].item()),
                    "seconds_per_update_max_rank": max(value["seconds_per_update_local"] for value in gathered),
                    "rank_records": gathered,
                    "optimizer_generations": dict(optimizer_generations),
                }
                append_jsonl(metrics_path, record)
                write_json(output_dir / "status.json", {"status": "running", "global_optimizer_update": global_step, "target_updates": target_updates, "stage": item["stage"], "task": task, "elapsed_seconds": time.perf_counter() - started, "last_mean_loss": mean_loss})
            checkpoint_kind: str | None = None
            boundary = expected_boundary(args.recipe, global_step, schedule)
            if global_step % CHECKPOINT_INTERVAL == 0:
                checkpoint_kind = "recovery"
            if boundary is not None:
                checkpoint_kind = "stage_boundary" if global_step != len(schedule) else "final"
            if checkpoint_kind is not None:
                if checkpoint_kind == "recovery":
                    checkpoint_dir = checkpoint_root / f"recovery_step_{global_step:05d}"
                else:
                    checkpoint_dir = checkpoint_root / boundary  # type: ignore[arg-type]
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_kind=checkpoint_kind,
                    recipe=args.recipe,
                    model=model,
                    scope_root=scope_root,
                    manifest_root=artifact_root,
                    init_metadata=init_metadata,
                    schedule=schedule,
                    global_step=global_step,
                    stage=str(item["stage"]),
                    current_item=item,
                    manifests=manifests,
                    optimizer_f=optimizer_f,
                    optimizer_s=optimizer_s,
                    optimizer_generations=optimizer_generations,
                    device=device,
                    rank=rank,
                    world_size=WORLD_SIZE_REQUIRED,
                    run_started_at=run_started_at,
                    runtime_seconds=time.perf_counter() - started,
                )
                if rank == 0 and checkpoint_kind == "recovery":
                    retain_recovery_checkpoints(checkpoint_root)
            print(
                f"[SEG010 FINAL {args.recipe} rank={rank} step={global_step}/{len(schedule)}] "
                f"stage={item['stage']} task={task} loss={mean_loss:.6f} sec={elapsed:.3f}",
                flush=True,
            )
            del before, task_rows, local_rows, rows
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        producer.close()
    if args.max_updates is not None and target_updates < len(schedule):
        smoke_item = schedule[target_updates - 1]
        save_checkpoint(
            checkpoint_dir=checkpoint_root / f"smoke_step_{target_updates:05d}",
            checkpoint_kind="smoke",
            recipe=args.recipe,
            model=model,
            scope_root=scope_root,
            manifest_root=artifact_root,
            init_metadata=init_metadata,
            schedule=schedule,
            global_step=target_updates,
            stage=str(smoke_item["stage"]),
            current_item=smoke_item,
            manifests=manifests,
            optimizer_f=optimizer_f,
            optimizer_s=optimizer_s,
            optimizer_generations=optimizer_generations,
            device=device,
            rank=rank,
            world_size=WORLD_SIZE_REQUIRED,
            run_started_at=run_started_at,
            runtime_seconds=time.perf_counter() - started,
        )
    barrier()
    if rank == 0:
        complete = target_updates == len(schedule)
        final_stage = schedule[target_updates - 1]["stage"]
        write_json(output_dir / "status.json", {"status": "complete" if complete else "smoke_complete", "global_optimizer_update": target_updates, "target_updates": target_updates, "stage": final_stage, "elapsed_seconds": time.perf_counter() - started, "end_time_utc": datetime.now(timezone.utc).isoformat(), "final_checkpoint": str(checkpoint_root / checkpoint_label(args.recipe, final_stage)) if complete else None})
    barrier()
    dist.destroy_process_group()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("general", "aggregation"), required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--scope-root", type=Path, default=DEFAULT_SCOPE_ROOT)
    parser.add_argument("--init-dir", type=Path, default=DEFAULT_INIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int, help="short GPU smoke limit; omit for the frozen full recipe")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
