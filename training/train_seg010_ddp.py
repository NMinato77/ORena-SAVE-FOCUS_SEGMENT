#!/usr/bin/env python3
"""Train one SEG010 condition with a shared L+V union and two task optimizers.

The command supports both canonical initialization creation and the actual
two-GPU DDP trainer.  It is intentionally SEG010-specific so SEG005/006/009
artifacts and trainers remain untouched.

Import order is part of the repository contract: PyNvVideoCodec is imported
before torch or any model/CUDA initialization.
"""

# ruff: noqa: I001 -- PyNvVideoCodec must stay before torch/CUDA imports.

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import PyNvVideoCodec as nvc  # noqa: F401 -- mandatory before torch/CUDA.
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from orena_procedure.seg010 import (
    FRAME_EVIDENCE_INSTRUCTION,
    GLOBAL_BATCH,
    GRADIENT_ACCUMULATION,
    LANGUAGE_LR,
    MAX_PIXELS,
    MERGER_LR,
    MICROBATCH_PER_GPU,
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    SEGMENT_EVIDENCE_INSTRUCTION,
    SEGMENT_TIMESTAMP_CONTEXT,
    SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
    VISION_LR,
    WEIGHT_DECAY,
    checkpoint_union_state,
    condition_spec,
    condition_specs,
    expected_union_names,
    make_optimizer,
    optimizer_group_names,
    optimizer_group_learning_rates,
    optimizer_state_hash,
    parameter_group_norms,
    parameter_group_snapshot,
    parameter_group_update_norms,
    prompt_contract,
    read_json,
    select_task,
    set_train_modes,
    sha256_file,
    task_scope,
    task_sequence,
    phase2_s_only_schedule,
    tensor_hash,
    union_parameter_names,
    install_union,
    validate_union,
)
from orena_procedure.seg006 import encode_frame_row as _encode_frame_row
from run_seg005_scope import encode_row as _encode_segment_row
from run_seg005_scope import loss_for_encoded


DEFAULT_ARTIFACT_ROOT = Path("outputs/CROSS_TIMESCALE_PREFLIGHT")
DEFAULT_OUTPUT_ROOT = Path("outputs/SEG010_cross_timescale_21way_v1")
DEFAULT_INIT_DIR = Path("/cache/models/SEG010_cross_timescale_21way_v1/canonical_union")


def encode_segment_row(
    row: dict[str, Any],
    processor: Any,
    device: torch.device,
    with_answer: bool,
) -> tuple[Any, dict[str, Any]]:
    """Encode SEGMENT with the immutable SEG010 P2/T0 contract."""

    return _encode_segment_row(
        row,
        processor,
        device,
        with_answer,
        evidence_instruction=SEGMENT_EVIDENCE_INSTRUCTION,
        timestamp_frame_template=SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
        timestamp_context=SEGMENT_TIMESTAMP_CONTEXT,
    )


def encode_frame_row(
    processor: Any,
    row: dict[str, Any],
    device: torch.device,
    with_answer: bool,
) -> tuple[Any, dict[str, Any]]:
    """Encode FRAME with the single-frame P2 semantic counterpart."""

    return _encode_frame_row(
        processor,
        row,
        device,
        with_answer,
        evidence_instruction=FRAME_EVIDENCE_INSTRUCTION,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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
    result = []
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


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        max_pixels=MAX_PIXELS,
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


def save_canonical_initialization(
    model: torch.nn.Module,
    artifact_root: Path,
    init_dir: Path,
    output_dir: Path,
    installation: dict[str, Any],
) -> dict[str, Any]:
    if init_dir.exists() and any(init_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite canonical initialization: {init_dir}")
    init_dir.mkdir(parents=True, exist_ok=False)
    state = checkpoint_union_state(model, artifact_root)
    if set(state) != expected_union_names(artifact_root):
        raise ValueError("canonical state is not the complete SEG010 union")
    tensor_digest = tensor_hash(state)
    weights_path = init_dir / "union_adapted_weights.safetensors"
    save_file(
        state,
        str(weights_path),
        metadata={
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": str(SEED),
            "tensor_sha256": tensor_digest,
        },
    )
    metadata = {
        "schema_version": 1,
        "experiment": "SEG010",
        "kind": "canonical_union_initialization",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "seed": SEED,
        "challenge_finetuning": "none",
        "prompt_contract": prompt_contract(),
        "artifact_root": str(artifact_root),
        "artifact_conditions_sha256": sha256_file(artifact_root / "conditions.json"),
        "artifact_scopes_sha256": sha256_file(artifact_root / "adaptation_scopes.json"),
        "installation": installation,
        "union_parameter_count": int(sum(value.numel() for value in state.values())),
        "union_parameter_names": sorted(state),
        "tensor_sha256": tensor_digest,
        "weights_file": weights_path.name,
        "weights_sha256": sha256_file(weights_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "gpu": gpu_snapshot(),
    }
    write_json(init_dir / "metadata.json", metadata)
    write_json(output_dir / "canonical_initialization.json", metadata)
    return metadata


def load_canonical_initialization(
    model: torch.nn.Module, artifact_root: Path, init_dir: Path
) -> dict[str, Any]:
    metadata_path = init_dir / "metadata.json"
    weights_path = init_dir / "union_adapted_weights.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete canonical initialization: {init_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"]["id"] != MODEL_ID or metadata["model"]["revision"] != MODEL_REVISION:
        raise ValueError("canonical model identity mismatch")
    if metadata["seed"] != SEED:
        raise ValueError("canonical seed mismatch")
    if sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError("canonical weights file hash mismatch")
    state = load_file(str(weights_path), device="cpu")
    if set(state) != expected_union_names(artifact_root):
        raise ValueError("canonical union tensor names mismatch")
    if tensor_hash(state) != metadata["tensor_sha256"]:
        raise ValueError("canonical union tensor hash mismatch")
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters:
            raise ValueError(f"canonical parameter is absent from model: {name}")
        if tuple(value.shape) != tuple(parameters[name].shape):
            raise ValueError(f"canonical shape mismatch for {name}")
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    return metadata


def load_checkpoint_initialization(
    model: torch.nn.Module, artifact_root: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    """Install a completed SEG010 union checkpoint as a fresh-optimizer init.

    This is deliberately different from ``--resume``: optimizer and RNG state
    are not restored.  The checkpoint supplies model weights only, which is
    the contract needed for the Stage 1 recipe pilot's initialization screen.
    """

    metadata_path = checkpoint_dir / "metadata.json"
    weights_path = checkpoint_dir / "union_adapted_weights.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete source checkpoint: {checkpoint_dir}")
    source_metadata = read_json(metadata_path)
    source_model = source_metadata.get("model", {})
    if source_model.get("id") != MODEL_ID or source_model.get("revision") != MODEL_REVISION:
        raise ValueError("source checkpoint model identity mismatch")
    expected = expected_union_names(artifact_root)
    state = load_file(str(weights_path), device="cpu")
    if set(state) != expected:
        raise ValueError("source checkpoint union tensor names mismatch")
    actual_weights_sha256 = sha256_file(weights_path)
    recorded_weights_sha256 = source_metadata.get("weights_sha256")
    if recorded_weights_sha256 and actual_weights_sha256 != recorded_weights_sha256:
        raise ValueError("source checkpoint weights file hash mismatch")
    tensor_digest = tensor_hash(state)
    recorded_tensor_sha256 = source_metadata.get("union_tensor_sha256")
    if recorded_tensor_sha256 and tensor_digest != recorded_tensor_sha256:
        raise ValueError("source checkpoint tensor hash mismatch")
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters or tuple(value.shape) != tuple(parameters[name].shape):
            raise ValueError(f"source checkpoint shape mismatch for {name}")
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    return {
        "schema_version": 1,
        "kind": "seg010_checkpoint_initialization",
        "model": source_model,
        "seed": SEED,
        "source_checkpoint": str(checkpoint_dir.resolve()),
        "source_checkpoint_metadata_sha256": sha256_file(metadata_path),
        "source_checkpoint_global_step": source_metadata.get("global_step"),
        "tensor_sha256": tensor_digest,
        "weights_sha256": actual_weights_sha256,
        "union_parameter_count": int(sum(value.numel() for value in state.values())),
    }


def validate_frame_rows(rows: list[dict[str, Any]], role: str) -> None:
    seen: set[tuple[str, str]] = set()
    required = {"dataset", "qID", "question", "answer", "video_path", "frame_index"}
    for row in rows:
        key = (str(row["dataset"]), str(row["qID"]))
        if key in seen:
            raise ValueError(f"duplicate FRAME {role} row: {key}")
        seen.add(key)
        missing = required - set(row)
        if missing:
            raise ValueError(f"FRAME {role} row missing {sorted(missing)}: {key}")
        if not str(row["answer"]).strip():
            raise ValueError(f"empty FRAME answer: {key}")
        if not Path(str(row["video_path"])).is_file():
            raise FileNotFoundError(f"missing FRAME video: {row['video_path']}")


def frame_row_for_encoder(row: dict[str, Any]) -> dict[str, Any]:
    """Add the audit-only timestamp expected by the shared FRAME encoder.

    The SEG010 preflight manifests intentionally record the native frame index
    rather than adding a second timestamp field.  The timestamp is not included
    in the prompt; it is only retained in the decode audit.
    """

    if "source_timestamp_seconds" in row:
        return row
    native_fps = float(row.get("native_fps", 0.0))
    if native_fps <= 0.0:
        raise ValueError(f"FRAME row has no positive native_fps: {row['dataset']}/{row['qID']}")
    enriched = dict(row)
    enriched["source_timestamp_seconds"] = float(row["frame_index"]) / native_fps
    return enriched


def frame_audit_for_loss(audit: dict[str, Any]) -> dict[str, Any]:
    """Adapt SEG006's audit names to the SEG005 loss helper's interface."""

    adapted = dict(audit)
    adapted["answer_start"] = int(audit["answer_token_start"])
    adapted["target_end"] = int(audit["target_end_exclusive"])
    return adapted


def validate_segment_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    required = {
        "dataset",
        "qID",
        "question",
        "answer",
        "video_path",
        "frame_indices",
        "frame_timestamps_seconds",
        "frame_count",
    }
    for row in rows:
        key = (str(row["dataset"]), str(row["qID"]))
        if key in seen:
            raise ValueError(f"duplicate SEGMENT row: {key}")
        seen.add(key)
        missing = required - set(row)
        if missing:
            raise ValueError(f"SEGMENT row missing {sorted(missing)}: {key}")
        if not str(row["answer"]).strip():
            raise ValueError(f"empty SEGMENT answer: {key}")
        indices = [int(value) for value in row["frame_indices"]]
        timestamps = [float(value) for value in row["frame_timestamps_seconds"]]
        if len(indices) != int(row["frame_count"]) or len(indices) != len(timestamps):
            raise ValueError(f"SEGMENT frame count mismatch: {key}")
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError(f"SEGMENT frame indices are not ordered/unique: {key}")
        if not Path(str(row["video_path"])).is_file():
            raise FileNotFoundError(f"missing SEGMENT video: {row['video_path']}")


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in {1, 2}:
        raise RuntimeError(f"SEG010 requires one or two processes, got WORLD_SIZE={world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("SEG010 requires enough CUDA devices")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_rng_states(state: dict[str, Any], world_size: int, rank: int) -> list[dict[str, Any]] | None:
    if world_size == 1:
        return [state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, state)
    return [value for value in gathered if value is not None] if rank == 0 else None


def save_checkpoint(
    *,
    checkpoint_root: Path,
    model: torch.nn.Module,
    artifact_root: Path,
    condition: str,
    init_metadata: dict[str, Any],
    schedule: list[dict[str, Any]],
    global_step: int,
    optimizer_f: torch.optim.Optimizer | None,
    optimizer_s: torch.optim.Optimizer | None,
    device: torch.device,
    world_size: int,
    rank: int,
    output_state: dict[str, Any],
) -> Path:
    rng_states = gather_rng_states(local_rng_state(device), world_size, rank)
    checkpoint_dir = checkpoint_root / f"step_{global_step:04d}"
    if rank == 0:
        if checkpoint_dir.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True)
        weights = checkpoint_union_state(model, artifact_root)
        union_digest = tensor_hash(weights)
        save_file(weights, str(checkpoint_dir / "union_adapted_weights.safetensors"))
        torch.save(
            {"present": optimizer_f is not None, "state_dict": None if optimizer_f is None else optimizer_f.state_dict()},
            checkpoint_dir / "optimizer_F.pt",
        )
        torch.save(
            {"present": optimizer_s is not None, "state_dict": None if optimizer_s is None else optimizer_s.state_dict()},
            checkpoint_dir / "optimizer_S.pt",
        )
        assert rng_states is not None
        for rng_rank, rng_state in enumerate(rng_states):
            torch.save(rng_state, checkpoint_dir / f"rng_rank_{rng_rank}.pt")
        f_updates = sum(item["task"] == "F" for item in schedule[:global_step])
        s_updates = sum(item["task"] == "S" for item in schedule[:global_step])
        metadata = {
            "schema_version": 3,
            "experiment": "SEG010",
            "condition": condition,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
            "prompt_contract": prompt_contract(),
            "git_commit": git_commit(),
            "initialization": {
                "weights_sha256": init_metadata["weights_sha256"],
                "tensor_sha256": init_metadata["tensor_sha256"],
            },
            "global_step": global_step,
            "schedule_length": len(schedule),
            "schedule_position_next": global_step,
            "task_updates": {"F": int(f_updates), "S": int(s_updates)},
            "scope_by_task": {
                "F": None if task_scope(artifact_root, condition, "F") is None else task_scope(artifact_root, condition, "F")["canonical_name"],
                "S": task_scope(artifact_root, condition, "S")["canonical_name"],
            },
            "optimizer_groups": {
                "F": optimizer_group_names(artifact_root, condition, "F"),
                "S": optimizer_group_names(artifact_root, condition, "S"),
            },
            "optimizer_learning_rates": {
                "F": optimizer_group_learning_rates(artifact_root, condition, "F"),
                "S": optimizer_group_learning_rates(artifact_root, condition, "S"),
            },
            "optimizer_state_sha256": {
                "F": optimizer_state_hash(optimizer_f),
                "S": optimizer_state_hash(optimizer_s),
            },
            "union_tensor_count": len(expected_union_names(artifact_root)),
            "union_parameter_count": int(sum(value.numel() for value in weights.values())),
            "union_tensor_sha256": union_digest,
            "weights_file": "union_adapted_weights.safetensors",
            "optimizer_files": {"F": "optimizer_F.pt", "S": "optimizer_S.pt"},
            "rng_files": [f"rng_rank_{index}.pt" for index in range(world_size)],
            "world_size": world_size,
            "rank_model_hashes": output_state.get("rank_model_hashes"),
            "runtime": output_state,
            "gpu": gpu_snapshot(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(checkpoint_dir / "metadata.json", metadata)
    barrier()
    return checkpoint_dir


def load_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    artifact_root: Path,
    condition: str,
    init_metadata: dict[str, Any],
    optimizer_f: torch.optim.Optimizer | None,
    optimizer_s: torch.optim.Optimizer | None,
    device: torch.device,
    rank: int,
    world_size: int,
) -> int:
    metadata = read_json(checkpoint_dir / "metadata.json")
    if metadata["condition"] != condition:
        raise ValueError("resume condition mismatch")
    if metadata["initialization"]["tensor_sha256"] != init_metadata["tensor_sha256"]:
        raise ValueError("resume canonical initialization mismatch")
    weights = load_file(str(checkpoint_dir / "union_adapted_weights.safetensors"), device="cpu")
    if set(weights) != expected_union_names(artifact_root):
        raise ValueError("resume union names mismatch")
    parameters = dict(model.named_parameters())
    for name, value in weights.items():
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    for task, optimizer in (("F", optimizer_f), ("S", optimizer_s)):
        payload = torch.load(checkpoint_dir / f"optimizer_{task}.pt", map_location="cpu", weights_only=False)
        if bool(payload["present"]) != (optimizer is not None):
            raise ValueError(f"resume optimizer presence mismatch: {task}")
        if optimizer is not None:
            optimizer.load_state_dict(payload["state_dict"])
    rng_path = checkpoint_dir / f"rng_rank_{rank}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"missing per-rank RNG state: {rng_path}")
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False), device)
    if int(metadata["world_size"]) != world_size:
        raise ValueError("resume world size mismatch")
    return int(metadata["global_step"])


def prepare_manifest_rows(
    artifact_root: Path, condition: str, segment_manifest: Path | None = None
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    spec = condition_spec(artifact_root, condition)
    segment_path = (
        segment_manifest.resolve()
        if segment_manifest is not None
        else artifact_root / str(spec["SEGMENT_manifest"])
    )
    segment_rows = read_jsonl(segment_path)
    validate_segment_rows(segment_rows)
    frame_rows = None
    if spec["FRAME_manifest"]:
        frame_rows = read_jsonl(artifact_root / str(spec["FRAME_manifest"]))
        validate_frame_rows(frame_rows, "training")
    return frame_rows, segment_rows


def segment_only_schedule(update_count: int) -> list[dict[str, Any]]:
    if update_count < 1:
        raise ValueError("segment-only update count must be positive")
    return [
        {
            "global_step": step,
            "task": "S",
            "task_step": step,
            "qa_start": (step - 1) * GLOBAL_BATCH,
            "qa_end_exclusive": step * GLOBAL_BATCH,
        }
        for step in range(1, update_count + 1)
    ]


def schedule_rows(
    artifact_root: Path,
    condition: str,
    schedule: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]] | None,
    segment_rows: list[dict[str, Any]],
    global_step: int,
) -> tuple[str, list[dict[str, Any]]]:
    item = schedule[global_step]
    rows = frame_rows if item["task"] == "F" else segment_rows
    if rows is None:
        raise ValueError(f"condition {condition} schedule requests FRAME without FRAME rows")
    start = int(item["qa_start"])
    end = int(item["qa_end_exclusive"])
    selected = rows[start:end]
    expected = GLOBAL_BATCH // (MICROBATCH_PER_GPU * 1)
    if len(selected) != expected:
        raise ValueError(f"schedule batch has {len(selected)} rows, expected {expected}")
    return str(item["task"]), selected


def finite_gradients(model: torch.nn.Module, selected_names: set[str]) -> None:
    parameters = dict(model.named_parameters())
    for name in selected_names:
        parameter = parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"active parameter has no gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def optimizer_step_summary(optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    """Summarize AdamW update counters without summing per-parameter states."""

    if optimizer is None:
        return {"max": 0, "min": 0, "unique": [], "parameter_states": 0}
    values: list[int] = []
    for state in optimizer.state.values():
        step = state.get("step", 0)
        if isinstance(step, torch.Tensor):
            step = int(step.item())
        values.append(int(step))
    unique = sorted(set(values))
    return {
        "max": max(values, default=0),
        "min": min(values, default=0),
        "unique": unique,
        "parameter_states": len(values),
    }


def validate_phase2_resume(
    *,
    model: torch.nn.Module,
    training_model: torch.nn.Module,
    processor: Any,
    artifact_root: Path,
    condition: str,
    schedule: list[dict[str, Any]],
    start_index: int,
    optimizer_f: torch.optim.Optimizer | None,
    optimizer_s: torch.optim.Optimizer | None,
    device: torch.device,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> int:
    """Exercise one S batch after resume without updating or writing a checkpoint."""

    if any(item["task"] != "S" for item in schedule[start_index:]):
        raise ValueError("resume validation found a non-SEGMENT task in the Phase 2 tail")
    task, rows = schedule_rows(
        artifact_root,
        condition,
        schedule,
        None,
        read_jsonl(artifact_root / str(condition_spec(artifact_root, condition)["SEGMENT_manifest"])),
        start_index,
    )
    if task != "S":
        raise ValueError(f"resume validation expected S, got {task}")

    before_model_hashes = model_hashes_for_artifact(model, artifact_root, world_size, rank)
    before_optimizer_hashes = {
        "F": optimizer_state_hash(optimizer_f),
        "S": optimizer_state_hash(optimizer_s),
    }
    before_counters = {
        "F": optimizer_step_summary(optimizer_f),
        "S": optimizer_step_summary(optimizer_s),
    }
    selected_names = set(select_task(model, artifact_root, condition, "S"))
    active_group_names = optimizer_group_names(artifact_root, condition, "S")
    set_train_modes(model, "S", selected_names)
    model.zero_grad(set_to_none=True)
    for micro_index in range(GRADIENT_ACCUMULATION):
        local_rows = rows[rank::world_size]
        row = local_rows[micro_index]
        encoded, audit = encode_segment_row(row, processor, device, with_answer=True)
        sync_context = (
            training_model.no_sync()
            if world_size == 2 and micro_index < GRADIENT_ACCUMULATION - 1
            else nullcontext()
        )
        with sync_context:
            loss = loss_for_encoded(training_model, encoded, audit) / GRADIENT_ACCUMULATION
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite resume validation loss at {start_index + 1}")
            loss.backward()
        del encoded, loss
        gc.collect()
    finite_gradients(model, selected_names)
    if active_group_names is None:
        raise ValueError("resume validation has no active SEGMENT optimizer groups")
    inactive_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if name not in selected_names and parameter.grad is not None
    ]
    if inactive_with_grad:
        raise RuntimeError(f"resume validation found inactive gradients: {inactive_with_grad[:5]}")
    model.zero_grad(set_to_none=True)
    after_model_hashes = model_hashes_for_artifact(model, artifact_root, world_size, rank)
    after_optimizer_hashes = {
        "F": optimizer_state_hash(optimizer_f),
        "S": optimizer_state_hash(optimizer_s),
    }
    after_counters = {
        "F": optimizer_step_summary(optimizer_f),
        "S": optimizer_step_summary(optimizer_s),
    }
    if before_model_hashes != after_model_hashes:
        raise RuntimeError("resume validation changed model parameters without optimizer.step")
    if before_optimizer_hashes != after_optimizer_hashes:
        raise RuntimeError("resume validation changed optimizer state without optimizer.step")
    if before_counters != after_counters:
        raise RuntimeError("resume validation changed optimizer counters without optimizer.step")
    if rank == 0:
        write_json(
            output_dir / "resume_validation.json",
            {
                "status": "passed",
                "condition": condition,
                "resume_global_step": start_index,
                "validated_task": "S",
                "tail_tasks_all_segment": True,
                "optimizer_step_called": False,
                "model_unchanged": True,
                "optimizer_state_unchanged": True,
                "inactive_gradients": inactive_with_grad,
                "selected_scope_groups": active_group_names,
                "optimizer_step_counters": after_counters,
                "global_batch": GLOBAL_BATCH,
                "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        )
        write_json(
            output_dir / "status.json",
            {"status": "validated", "global_step": start_index, "target_updates": len(schedule)},
        )
    barrier()
    if world_size == 2 and dist.is_initialized():
        dist.destroy_process_group()
    return 0


def model_hashes_for_artifact(
    model: torch.nn.Module, artifact_root: Path, world_size: int, rank: int
) -> list[str] | None:
    digest = tensor_hash(checkpoint_union_state(model, artifact_root))
    if world_size == 1:
        return [digest]
    values: list[str | None] = [None] * world_size
    dist.all_gather_object(values, digest)
    return [value for value in values if value is not None] if rank == 0 else None


def train(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    rank, _local_rank, world_size, device = initialize_distributed()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    barrier()
    seed_everything()
    if world_size != 2:
        raise RuntimeError("SEG010 training/pilot must use exactly two GPUs per condition")
    spec = condition_spec(artifact_root, args.condition)
    frame_rows, segment_rows = prepare_manifest_rows(
        artifact_root, args.condition, args.segment_manifest
    )
    phase1_schedule = task_sequence(artifact_root, str(spec["schedule"]))
    schedule = phase1_schedule
    resume_metadata: dict[str, Any] | None = None
    if args.resume:
        resume_metadata = read_json(args.resume.resolve() / "metadata.json")
    if args.phase2_s_only:
        if not args.resume:
            raise ValueError("Phase 2 S-only continuation requires --resume")
        if args.task_sequence:
            raise ValueError("Phase 2 S-only continuation cannot use --task-sequence")
        schedule = phase2_s_only_schedule(artifact_root, args.condition)
    if args.segment_only:
        if args.max_updates is None:
            raise ValueError("--segment-only requires --max-updates")
        schedule = segment_only_schedule(args.max_updates)
    if args.task_sequence:
        requested = [value.strip() for value in args.task_sequence.split(",") if value.strip()]
        if any(value not in {"F", "S"} for value in requested):
            raise ValueError("--task-sequence accepts only F and S")
        counters = {"F": 0, "S": 0}
        schedule = []
        for step, task in enumerate(requested, start=1):
            start = counters[task] * GLOBAL_BATCH
            counters[task] += 1
            schedule.append(
                {
                    "global_step": step,
                    "task": task,
                    "task_step": counters[task],
                    "qa_start": start,
                    "qa_end_exclusive": start + GLOBAL_BATCH,
                }
            )
    if not schedule:
        raise ValueError("empty SEG010 schedule")
    requested_updates = len(schedule) if args.max_updates is None else args.max_updates
    if args.phase2_s_only and requested_updates != len(schedule):
        raise ValueError("Phase 2 continuation must run the complete 128-update S-only tail")
    if requested_updates < 1 or requested_updates > len(schedule):
        raise ValueError(f"invalid max updates: {requested_updates}")

    model, processor = load_model(device)
    installation = install_union(model)
    validate_union(model, artifact_root)
    if args.source_checkpoint is not None:
        init_metadata = load_checkpoint_initialization(
            model, artifact_root, args.source_checkpoint.resolve()
        )
    else:
        init_metadata = load_canonical_initialization(model, artifact_root, args.init_dir.resolve())
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # DDP must register hooks for the complete union before dynamic task masks
    # are applied.  find_unused_parameters handles the inactive task branches.
    union_names = union_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in union_names)
    optimizer_f = make_optimizer(model, artifact_root, args.condition, "F")
    optimizer_s = make_optimizer(model, artifact_root, args.condition, "S")
    training_model: torch.nn.Module = model
    if world_size == 2:
        training_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
            static_graph=False,
        )
    if rank == 0:
        write_json(
            output_dir / "config.json",
            {
                "experiment": "SEG010",
                "condition": args.condition,
                "condition_spec": spec,
                "prompt_contract": prompt_contract(),
                "artifact_root": str(artifact_root),
                "conditions_sha256": sha256_file(artifact_root / "conditions.json"),
                "schedule_sha256": hashlib.sha256(
                    json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "schedule_source": "segment_only_generated" if args.segment_only else "artifact_task_schedules",
                "segment_manifest": str(
                    args.segment_manifest.resolve()
                    if args.segment_manifest is not None
                    else artifact_root / str(spec["SEGMENT_manifest"])
                ),
                "canonical_initialization": init_metadata,
                "installation": installation,
                "world_size": world_size,
                "devices": [int(index) for index in range(world_size)],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "global_batch": GLOBAL_BATCH,
                "microbatch_per_gpu": MICROBATCH_PER_GPU,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "learning_rates": {
                    "language_lora": LANGUAGE_LR,
                    "vision_lora": VISION_LR,
                    "merger_full": MERGER_LR,
                },
                "weight_decay": WEIGHT_DECAY,
                "loss": "unweighted answer+EOT mean NLL",
                "git_commit": git_commit(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gpu": gpu_snapshot(),
                "start_time_utc": datetime.now(timezone.utc).isoformat(),
                "pilot": bool(args.task_sequence or args.segment_only),
                "phase": "stage1_recipe_pilot" if args.segment_only else ("phase2" if args.phase2_s_only else "phase1"),
                "phase2_s_only": bool(args.phase2_s_only),
                "segment_only": bool(args.segment_only),
                "phase1_schedule_length": len(phase1_schedule),
                "phase1_global_step": None
                if resume_metadata is None
                else int(resume_metadata["global_step"]),
                "phase1_segment_updates": None
                if resume_metadata is None
                else int(resume_metadata["task_updates"]["S"]),
                "phase2_additional_segment_updates": 128 if args.phase2_s_only else 0,
                "total_segment_updates": sum(item["task"] == "S" for item in schedule),
                "resume_validation": bool(args.validate_resume),
                "source_checkpoint": None
                if args.source_checkpoint is None
                else str(args.source_checkpoint.resolve()),
                "source_checkpoint_metadata_sha256": None
                if args.source_checkpoint is None
                else sha256_file(args.source_checkpoint.resolve() / "metadata.json"),
                "resume_checkpoint": None
                if args.resume is None
                else str(args.resume.resolve()),
            },
        )
        write_json(output_dir / "status.json", {"status": "running", "global_step": 0})
    barrier()

    start_index = 0
    if args.resume:
        start_index = load_checkpoint(
            checkpoint_dir=args.resume.resolve(),
            model=model,
            artifact_root=artifact_root,
            condition=args.condition,
            init_metadata=init_metadata,
            optimizer_f=optimizer_f,
            optimizer_s=optimizer_s,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if start_index >= requested_updates:
            raise ValueError("resume checkpoint is not before requested max_updates")
        if args.phase2_s_only:
            if start_index != len(phase1_schedule):
                raise ValueError(
                    "Phase 2 resume point does not match the Phase 1 schedule length: "
                    f"checkpoint={start_index}, expected={len(phase1_schedule)}"
                )
            if any(item["task"] != "S" for item in schedule[start_index:]):
                raise ValueError("Phase 2 continuation contains a non-SEGMENT update")
            if len(schedule) - start_index != 128:
                raise ValueError("Phase 2 continuation must contain exactly 128 SEGMENT updates")
        if args.validate_resume:
            return validate_phase2_resume(
                model=model,
                training_model=training_model,
                processor=processor,
                artifact_root=artifact_root,
                condition=args.condition,
                schedule=schedule,
                start_index=start_index,
                optimizer_f=optimizer_f,
                optimizer_s=optimizer_s,
                device=device,
                rank=rank,
                world_size=world_size,
                output_dir=output_dir,
            )
    else:
        initial_rank_hashes = model_hashes_for_artifact(model, artifact_root, world_size, rank)
        save_checkpoint(
            checkpoint_root=output_dir / "checkpoints",
            model=model,
            artifact_root=artifact_root,
            condition=args.condition,
            init_metadata=init_metadata,
            schedule=schedule,
            global_step=0,
            optimizer_f=optimizer_f,
            optimizer_s=optimizer_s,
            device=device,
            world_size=world_size,
            rank=rank,
            output_state={"phase": "initial", "rank_model_hashes": initial_rank_hashes},
        )

    metrics_path = output_dir / "training_metrics.jsonl"
    started = time.perf_counter()
    for schedule_index in range(start_index, requested_updates):
        task, rows = schedule_rows(
            artifact_root, args.condition, schedule, frame_rows, segment_rows, schedule_index
        )
        if task == "F" and optimizer_f is None:
            raise ValueError(f"{args.condition} schedule contains F but has no FRAME optimizer")
        active_optimizer = optimizer_f if task == "F" else optimizer_s
        active_group_names = optimizer_group_names(artifact_root, args.condition, task)
        if active_optimizer is None or active_group_names is None:
            raise ValueError(f"missing optimizer/scope for task {task}")
        selected_names = set(select_task(model, artifact_root, args.condition, task))
        if set().union(*active_group_names.values()) != selected_names:
            raise ValueError("active optimizer scope does not equal task mask")
        set_train_modes(model, task, selected_names)
        model.zero_grad(set_to_none=True)
        before = parameter_group_snapshot(model, active_group_names)
        losses: list[float] = []
        answer_tokens = 0
        update_started = time.perf_counter()
        for micro_index in range(GRADIENT_ACCUMULATION):
            local_rows = rows[micro_index // 1 :: GRADIENT_ACCUMULATION]
            # The six-row expression above is intentionally replaced below by
            # the rank-strided four-row view; each rank owns four microbatches.
            local_rows = rows[rank::world_size]
            row = local_rows[micro_index]
            if task == "S":
                encoded, audit = encode_segment_row(row, processor, device, with_answer=True)
            else:
                encoded, audit = encode_frame_row(
                    processor, frame_row_for_encoder(row), device, with_answer=True
                )
                audit = frame_audit_for_loss(audit)
            answer_tokens += int(audit.get("answer_token_count", len(str(row["answer"]))))
            sync_context = (
                training_model.no_sync()
                if world_size == 2 and micro_index < GRADIENT_ACCUMULATION - 1
                else nullcontext()
            )
            with sync_context:
                loss = loss_for_encoded(training_model, encoded, audit) / GRADIENT_ACCUMULATION
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {schedule_index + 1}")
                loss.backward()
            losses.append(float(loss.detach().cpu()) * GRADIENT_ACCUMULATION)
            del encoded, loss
            gc.collect()
        finite_gradients(model, selected_names)
        gradient_norms = parameter_group_norms(model, active_group_names)
        active_optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - update_started
        update_norms = parameter_group_update_norms(model, active_group_names, before)
        local_loss_sum = float(np.sum(losses))
        local_loss_count = float(len(losses))
        local_answer_tokens = float(answer_tokens)
        global_stats = torch.tensor(
            [local_loss_sum, local_loss_count, local_answer_tokens],
            device=device,
            dtype=torch.float64,
        )
        if world_size == 2:
            dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        global_mean_loss = float((global_stats[0] / global_stats[1]).item())
        global_answer_tokens = int(global_stats[2].item())
        record = {
            "global_step": schedule_index + 1,
            "task": task,
            "task_step": int(schedule[schedule_index]["task_step"]),
            "frame_updates": sum(item["task"] == "F" for item in schedule[: schedule_index + 1]),
            "segment_updates": sum(item["task"] == "S" for item in schedule[: schedule_index + 1]),
            "mean_loss": global_mean_loss,
            "answer_token_count": global_answer_tokens,
            "seconds_per_update": elapsed,
            "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "gradient_norm_by_group": gradient_norms,
            "parameter_update_norm_by_group": update_norms,
            "optimizer_step_counters": {
                "F": optimizer_step_summary(optimizer_f),
                "S": optimizer_step_summary(optimizer_s),
            },
            "rank": rank,
        }
        if world_size == 2:
            gathered_records: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered_records, record)
            if rank == 0:
                record["rank_records"] = gathered_records
        if rank == 0:
            append_jsonl(metrics_path, record)
            write_json(
                output_dir / "status.json",
                {
                    "status": "running",
                    "global_step": schedule_index + 1,
                    "target_updates": requested_updates,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        checkpoint_due = (
            (schedule_index + 1) % args.checkpoint_interval == 0
            or schedule_index + 1 == requested_updates
            or schedule_index + 1 in {192, 256}
        )
        if checkpoint_due:
            rank_hashes = model_hashes_for_artifact(model, artifact_root, world_size, rank)
            save_checkpoint(
                checkpoint_root=output_dir / "checkpoints",
                model=model,
                artifact_root=artifact_root,
                condition=args.condition,
                init_metadata=init_metadata,
                schedule=schedule,
                global_step=schedule_index + 1,
                optimizer_f=optimizer_f,
                optimizer_s=optimizer_s,
                device=device,
                world_size=world_size,
                rank=rank,
                output_state={
                    "phase": "training",
                    "seconds_per_update": elapsed,
                    "rank_model_hashes": rank_hashes,
                },
            )
        print(
            f"[SEG010 {args.condition} rank={rank} step={schedule_index + 1}/{requested_updates}] "
            f"task={task} loss={global_mean_loss:.6f} sec={elapsed:.3f}",
            flush=True,
        )
        del before
        gc.collect()
        torch.cuda.empty_cache()
    barrier()
    if rank == 0:
        write_json(
            output_dir / "status.json",
            {
                "status": "complete",
                "global_step": requested_updates,
                "target_updates": requested_updates,
                "elapsed_seconds": time.perf_counter() - started,
                "end_time_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    if world_size == 2 and dist.is_initialized():
        dist.destroy_process_group()
    return 0


def init_mode(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("canonical initialization requires one CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything()
    model, _processor = load_model(device)
    installation = install_union(model)
    validate_union(model, artifact_root)
    metadata = save_canonical_initialization(
        model, artifact_root, args.init_dir.resolve(), output_dir, installation
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("init", "train"), default="train")
    parser.add_argument("--condition", choices=[item["condition"] for item in condition_specs(DEFAULT_ARTIFACT_ROOT)], default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--init-dir", type=Path, default=DEFAULT_INIT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--task-sequence", help="pilot-only comma-separated task sequence, e.g. F,F,S,S,F,S")
    parser.add_argument(
        "--segment-only",
        action="store_true",
        help="generate a fixed SEGMENT-only schedule of --max-updates updates",
    )
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        help="optional SEGMENT manifest override for a pilot; rows retain source order",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help="install union weights from a completed checkpoint with fresh optimizers",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--phase2-s-only",
        action="store_true",
        help="resume a Phase 1 checkpoint and append exactly 128 SEGMENT-only updates",
    )
    parser.add_argument(
        "--validate-resume",
        action="store_true",
        help="run one no-step SEGMENT resume validation batch and exit",
    )
    args = parser.parse_args()
    if args.mode == "init":
        if (
            args.condition
            or args.resume
            or args.task_sequence
            or args.segment_only
            or args.segment_manifest
            or args.source_checkpoint
            or args.phase2_s_only
            or args.validate_resume
        ):
            parser.error("--mode init does not accept train-only options")
    elif not args.condition:
        parser.error("--condition is required in train mode")
    if args.segment_only and (args.task_sequence or args.phase2_s_only or args.resume is not None):
        parser.error("--segment-only cannot be combined with --task-sequence, --phase2-s-only, or --resume")
    if args.segment_only and args.max_updates is None:
        parser.error("--segment-only requires --max-updates")
    if args.source_checkpoint is not None and args.resume is not None:
        parser.error("--source-checkpoint cannot be combined with --resume")
    if args.validate_resume and (not args.phase2_s_only or args.resume is None):
        parser.error("--validate-resume requires --phase2-s-only and --resume")
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(init_mode(arguments) if arguments.mode == "init" else train(arguments))
