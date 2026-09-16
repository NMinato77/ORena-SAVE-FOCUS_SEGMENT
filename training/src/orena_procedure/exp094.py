"""Contracts and small utilities for EXP094 PROCEDURE-specific Qwen SFT."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import PyNvVideoCodec as nvc  # import before torch/CUDA.
import torch
from PIL import Image

from orena_procedure.qwen_baseline import FO_CLASS_NAMES, uniform_indices

EXPERIMENT = "EXP094"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_PIXELS = 50176
FRAME_COUNT = 640
MAX_NEW_TOKENS = 32
VIDEO_ROOT = os.environ.get("FOCUS_DATA_ROOT", "/data/focus") + "/derived/procedure_5fps_cache"
STRATA = ("count_free", "count_sensitive", "lifecycle_or_semantic")
FO_CLASS_GUIDANCE = True

_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
_LIFECYCLE_RE = re.compile(
    r"\b(?:insert\w*|creat\w*|retriev\w*|remov\w*|re-?appear\w*|"
    r"leave\w*|placed\w*|appl(?:y|ied)\w*|deploy\w*|left behind)\b",
    re.IGNORECASE,
)


def stable_rank(value: str) -> str:
    """Return a stable ordering key independent of Python hash randomization."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify_stratum(question: str, answer_format: str, primary: str) -> str:
    """Reproduce EXP090's three audit strata from question metadata only."""

    if _LIFECYCLE_RE.search(question) and any(
        token in question.lower()
        for token in ("in this video", "during the video", "throughout", "at the end", "before")
    ):
        return "lifecycle_or_semantic"
    if answer_format in {"number", "percentage"} or primary in {
        "object_aggregation",
        "event_aggregation",
    }:
        return "count_sensitive"
    if _TIME_RE.search(question) and any(
        token in question.lower() for token in ("insert", "retriev", "first", "last")
    ):
        return "lifecycle_or_semantic"
    return "count_free"


def cache_path(video_root: Path, dataset: str, video_id: str) -> Path:
    """Resolve a normalized 5-fps cache video using the repository convention."""

    direct = video_root / dataset / "videos" / video_id
    if direct.exists():
        return direct
    return video_root / dataset / "videos" / f"{Path(video_id).stem}.mp4"


def assert_video_disjoint(split_rows: dict[str, Iterable[dict[str, Any]]]) -> None:
    """Reject any video appearing in more than one of train/calibration/evaluation."""

    videos = {name: {str(row["videoID"]) for row in rows} for name, rows in split_rows.items()}
    names = tuple(videos)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = videos[left] & videos[right]
            if overlap:
                raise ValueError(f"video leakage between {left} and {right}: {sorted(overlap)}")


def _check_common_row(row: dict[str, Any]) -> None:
    required = {
        "qID",
        "dataset",
        "videoID",
        "video_path",
        "question",
        "answer_format",
        "primary",
        "stratum",
        "start_time",
        "end_time",
        "frame_count",
        "source_track",
        "source_split",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"EXP094 row missing fields: {sorted(missing)}")
    if row["source_track"] != "procedure" or row["source_split"] != "train":
        raise ValueError(f"EXP094 row is not official PROCEDURE train: {row['qID']}")
    if int(row["frame_count"]) != FRAME_COUNT:
        raise ValueError(f"EXP094 frame contract mismatch: {row['qID']}")
    if float(row["end_time"]) <= float(row["start_time"]):
        raise ValueError(f"non-positive request window: {row['qID']}")
    if row["stratum"] not in STRATA:
        raise ValueError(f"unknown EXP094 stratum: {row['qID']}")


def assert_training_manifest(rows: Iterable[dict[str, Any]], role: str) -> None:
    """Validate answer-bearing training/calibration rows before loading a model."""

    if role not in {"train", "calibration"}:
        raise ValueError(f"invalid answer-bearing role: {role}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        _check_common_row(row)
        qid = str(row["qID"])
        key = (str(row["dataset"]), qid)
        if key in seen:
            raise ValueError(f"duplicate dataset/qID in {role} manifest: {key}")
        seen.add(key)
        if str(row.get("role")) != role:
            raise ValueError(f"unexpected role in {role} manifest: {qid}")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"missing official answer in {role} manifest: {qid}")
        forbidden = {
            "reference",
            "reference_answer",
            "reference_for_audit",
            "teacher",
            "teacher_label",
            "pseudo_label",
            "oracle_timestamp",
            "oracle_frame",
        }
        if forbidden.intersection(row):
            raise ValueError(f"forbidden audit field in {role} manifest: {qid}")


def assert_inference_manifest(rows: Iterable[dict[str, Any]], role: str = "evaluation") -> None:
    """Validate an answer-free inference manifest."""

    if role not in {"evaluation", "legacy_evaluation"}:
        raise ValueError(f"invalid inference role: {role}")
    seen: set[tuple[str, str]] = set()
    forbidden = {
        "answer",
        "reference",
        "reference_answer",
        "reference_for_audit",
        "teacher",
        "teacher_label",
        "pseudo_label",
        "oracle_timestamp",
        "oracle_frame",
    }
    for row in rows:
        _check_common_row(row)
        qid = str(row["qID"])
        key = (str(row["dataset"]), qid)
        if key in seen:
            raise ValueError(f"duplicate dataset/qID in inference manifest: {key}")
        seen.add(key)
        if str(row.get("role")) != role:
            raise ValueError(f"unexpected inference role: {qid}")
        leaked = forbidden.intersection(row)
        if leaked:
            raise ValueError(f"answer/teacher fields in inference manifest {qid}: {sorted(leaked)}")
        if not str(row.get("question", "")).strip():
            raise ValueError(f"empty question in inference manifest: {qid}")


def decode_uniform_frames(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    device: torch.device,
    fps: float = 5.0,
) -> tuple[list[Image.Image], list[float], dict[str, float]]:
    """Decode the EXP092-compatible uniform source timestamps with NVDEC."""

    if device.type != "cuda":
        raise ValueError("EXP094 requires a CUDA device for PyNvVideoCodec")
    if frame_count != FRAME_COUNT:
        raise ValueError(f"EXP094 requires {FRAME_COUNT} frames")
    timings: dict[str, float] = {}
    start = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        str(video_path),
        gpu_id=device.index if device.index is not None else torch.cuda.current_device(),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
        bWaitForSessionWarmUp=True,
    )
    timings["video_loading"] = time.perf_counter() - start
    total_frames = len(decoder)
    window_start = max(0, min(total_frames - 1, int(np.floor(start_seconds * fps))))
    window_end = max(window_start + 1, min(total_frames, int(np.ceil(end_seconds * fps))))
    indices = uniform_indices(window_start, window_end, frame_count)
    timestamps = [index / fps for index in indices]
    start = time.perf_counter()
    surfaces = decoder.get_batch_frames_by_index(indices)
    arrays = [torch.utils.dlpack.from_dlpack(surface).cpu().numpy() for surface in surfaces]
    torch.cuda.synchronize(device)
    timings["frame_decoding"] = time.perf_counter() - start
    del surfaces, decoder
    start = time.perf_counter()
    images = [Image.fromarray(frame) for frame in arrays]
    timings["image_conversion"] = time.perf_counter() - start
    if len(images) != frame_count or len(timestamps) != frame_count:
        raise ValueError("decoded frame/timestamp count mismatch")
    return images, timestamps, timings


class LoRALinear(torch.nn.Module):
    """A dependency-free LoRA wrapper for one frozen linear layer."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.scaling = float(alpha) / rank
        self.lora_A = torch.nn.Linear(
            base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        self.lora_B = torch.nn.Linear(
            rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_B(self.lora_A(self.dropout(value))) * self.scaling


def install_language_lora(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    layer_start: int = 28,
) -> list[str]:
    """Install q/v LoRA only in the final language-model layers."""

    language_model = model.model.language_model
    layers = language_model.layers
    if not 0 <= layer_start < len(layers):
        raise ValueError(f"invalid language LoRA layer_start={layer_start}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced: list[str] = []
    for layer_index in range(layer_start, len(layers)):
        attention = layers[layer_index].self_attn
        for name in ("q_proj", "v_proj"):
            base = getattr(attention, name)
            if not isinstance(base, torch.nn.Linear):
                raise TypeError(f"unexpected language module: layers.{layer_index}.self_attn.{name}")
            setattr(attention, name, LoRALinear(base, rank, alpha, dropout))
            replaced.append(f"model.language_model.layers.{layer_index}.self_attn.{name}")
    return replaced


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable LoRA parameters, detached on CPU."""

    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("no trainable LoRA parameters found")
    return state


def load_lora_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a saved LoRA-only state after the same module injection."""

    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        raise ValueError(f"LoRA state mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}")
    for name, parameter in current.items():
        parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def answer_format_guidance(answer_format: str) -> str:
    """Return the EXP092 prompt suffix used for FO class answers."""

    if FO_CLASS_GUIDANCE and answer_format == "fo_class":
        return (
            " For foreign-object class answers, use only comma-separated names from: "
            f"{', '.join(FO_CLASS_NAMES)}; use none only when no class applies."
        )
    return ""


def answer_token_span(input_ids: torch.Tensor, answer_ids: list[int]) -> tuple[int, int]:
    """Locate only the assistant answer content in a rendered conversation."""

    if input_ids.ndim != 1:
        raise ValueError("expected one-dimensional input IDs")
    values = input_ids.tolist()
    candidates = []
    first = max(0, len(values) - len(answer_ids) - 16)
    last = len(values) - len(answer_ids)
    for start in range(first, last + 1):
        if values[start : start + len(answer_ids)] == answer_ids:
            candidates.append(start)
    if len(candidates) != 1:
        raise ValueError(
            f"could not uniquely locate assistant answer tokens: candidates={candidates}, "
            f"sequence_length={len(values)}, answer_length={len(answer_ids)}"
        )
    return candidates[0], candidates[0] + len(answer_ids)
