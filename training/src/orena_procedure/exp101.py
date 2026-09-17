"""Contracts and shared utilities for EXP101.

The module keeps the EXP101 boundary explicit: the task schema is derived only
from the question and official answer-format name, while the M selector uses
only answer-free video pixels and deterministic state traces.

PyNvVideoCodec is intentionally imported before torch.  This import order is a
runtime contract for the GPU video experiments in this repository.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import PyNvVideoCodec as nvc  # import before torch/CUDA.
import torch
from PIL import Image

from orena_procedure.exp094 import (
    FRAME_COUNT,
    assert_inference_manifest,
    assert_training_manifest,
)
from orena_procedure.exp099 import (
    FRAME_FPS,
    expand_trace_states,
    invalid_static_intervals,
    select_state_aware_indices,
)
from orena_procedure.multiple_choice import multiple_choice_prompt_instruction
from orena_procedure.qwen_baseline import (
    FO_CLASS_NAMES,
    uniform_indices,
)

EXPERIMENT = "EXP101"
PROMPT_SCHEMA_VERSIONS = ("B0", "B1")
SCAN_CONDITION = "M_invalid_static_exclusion"
SCAN_ROOT_NAME = "m_scan"
# These values are part of the EXP101 fixed model contract.
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_PIXELS = 50176
MAX_NEW_TOKENS = 32

_QUESTION_TIME_RE = re.compile(r"\b(?:when|what time|time point|timepoint|timestamp)\b", re.I)
_LIFECYCLE_RE = re.compile(
    r"\b(?:insert\w*|creat\w*|retriev\w*|remov\w*|re-?appear\w*|"
    r"leave\w*|placed\w*|appl(?:y|ied)\w*|deploy\w*|remain\w*|"
    r"first\s+appear\w*|last\s+visible\w*)\b",
    re.I,
)
_AGGREGATION_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|count|total|maximum|percentage|percent|"
    r"how\s+much|different\s+foreign\s+object)\b",
    re.I,
)
_SPATIAL_RE = re.compile(
    r"\b(?:quadrant|left|right|proximal|distal|anatomical\s+location|"
    r"where\s+was|where\s+is|position|central\s+position)\b",
    re.I,
)
_COOCCURRENCE_RE = re.compile(
    r"\b(?:co-?occur|both\s+appear|same\s+frame|present|visible|"
    r"appear(?:s|ed)?\s+at|also\s+appear)\b",
    re.I,
)
_OBJECT_RE = re.compile(
    r"\b(?:foreign\s+object|sponge|clip|needle|specimen|drain|"
    r"gallstone|mesh|loop|object\s+class)\b",
    re.I,
)


def stable_hash(path: Path) -> str:
    """Return SHA256 for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    """Hash relative paths and contents for an immutable experiment tree."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_schema(question: str, answer_format: str) -> dict[str, str]:
    """Derive a deterministic task schema from question text and format only.

    ``primary`` is deliberately not an argument.  It is useful for reporting
    after evaluation but is not guaranteed to be available at inference time.
    """

    question = str(question)
    answer_format = str(answer_format)
    lowered = question.lower()
    if answer_format == "time" or (
        _QUESTION_TIME_RE.search(question) and _LIFECYCLE_RE.search(question)
    ):
        family = "temporal_lifecycle"
    elif answer_format in {"number", "percentage"} or _AGGREGATION_RE.search(question):
        family = "inventory_aggregation"
    elif answer_format == "multiple_choice" and _SPATIAL_RE.search(question):
        family = "spatial_relation"
    elif answer_format == "binary" and _COOCCURRENCE_RE.search(question):
        family = "presence_or_cooccurrence"
    elif answer_format in {"fo_class", "binary"} and _OBJECT_RE.search(question):
        family = "object_identity"
    elif _SPATIAL_RE.search(question):
        family = "spatial_relation"
    elif "order" in lowered or "consequence" in lowered or "reason" in lowered:
        family = "general_reasoning"
    else:
        family = "general_reasoning"

    contracts = {
        "binary": "Output exactly yes or no.",
        "number": "Output one non-negative integer only.",
        "percentage": "Output one numeric percentage only.",
        "time": "Output timestamp(s) as HH:MM:SS; separate multiple timestamps with commas.",
        "fo_class": (
            "Output registered foreign-object class name(s), comma-separated, or none. "
            f"Allowed names: {', '.join(FO_CLASS_NAMES)}."
        ),
        "multiple_choice": "Output exactly one option listed in the question.",
        "open_ended": "Output only the concise answer required by the question.",
    }
    output_contract = contracts.get(answer_format, contracts["open_ended"])
    if answer_format == "multiple_choice":
        output_contract = multiple_choice_prompt_instruction(question)
        if output_contract is None:
            raise ValueError("multiple-choice question has no supported option list")
    return {
        "task_family": family,
        "answer_format": answer_format,
        "output_contract": output_contract,
    }


def parser_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize parser output without using answer, reference, or primary."""

    counts: dict[str, int] = {}
    formats: dict[str, int] = {}
    for row in rows:
        schema = task_schema(row["question"], row["answer_format"])
        counts[schema["task_family"]] = counts.get(schema["task_family"], 0) + 1
        formats[schema["answer_format"]] = formats.get(schema["answer_format"], 0) + 1
    total = sum(counts.values())
    fallback = counts.get("general_reasoning", 0)
    return {
        "rows": total,
        "task_family_counts": dict(sorted(counts.items())),
        "answer_format_counts": dict(sorted(formats.items())),
        "fallback_family": "general_reasoning",
        "fallback_rows": fallback,
        "fallback_rate": fallback / total if total else 0.0,
        "inputs": ["question", "answer_format"],
        "prohibited_inputs": [
            "answer",
            "reference",
            "primary",
            "FRAME",
            "SEGMENT",
            "oracle timestamp",
        ],
    }


def prompt_with_schema(
    question: str,
    procedure_type: str,
    timestamps: list[float],
    images: list[Image.Image],
    answer_format: str,
    schema_version: str,
    fo_class_guidance: bool = True,
) -> list[dict[str, Any]]:
    """Build B0 or B1 while keeping B0 byte-for-byte compatible with baseline."""

    if schema_version not in PROMPT_SCHEMA_VERSIONS:
        raise ValueError(f"unknown prompt schema version: {schema_version}")
    from orena_procedure.qwen_baseline import procedure_prompt

    messages = procedure_prompt(
        question,
        procedure_type,
        timestamps,
        images,
        answer_format,
        fo_class_guidance,
    )
    if schema_version == "B0":
        return messages
    schema = task_schema(question, answer_format)
    text = str(messages[0]["content"][-1]["text"])
    marker = "\nQuestion:"
    if marker not in text:
        raise ValueError("baseline prompt no longer has the expected Question marker")
    prefix, suffix = text.split(marker, 1)
    schema_text = (
        "\nTask schema: "
        f"family={schema['task_family']}; format={schema['answer_format']}. "
        f"{schema['output_contract']}"
    )
    messages[0]["content"][-1]["text"] = prefix + schema_text + marker + suffix
    return messages


def answer_token_span(input_ids: torch.Tensor, answer_ids: list[int]) -> tuple[int, int]:
    """Locate the final assistant answer content span.

    The answer text can legitimately occur in the question (for example, a
    binary question may end with the word ``yes``).  The final occurrence is
    the assistant target because the rendered training conversation ends with
    that target.
    """

    if input_ids.ndim != 1 or not answer_ids:
        raise ValueError("expected one-dimensional input IDs and non-empty answer IDs")
    values = input_ids.tolist()
    first = max(0, len(values) - len(answer_ids) - 32)
    last = len(values) - len(answer_ids)
    candidates = [
        start
        for start in range(first, last + 1)
        if values[start : start + len(answer_ids)] == answer_ids
    ]
    if not candidates:
        raise ValueError(
            f"could not locate assistant answer tokens: candidates={candidates}, "
            f"sequence_length={len(values)}, answer_length={len(answer_ids)}"
        )
    start = candidates[-1]
    return start, start + len(answer_ids)


def trace_path(scan_root: Path, dataset: str, video_id: str) -> Path:
    path = scan_root / "traces" / dataset / f"{Path(video_id).stem}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing M trace: {path}")
    return path


def load_scan_context(scan_root: Path, dataset: str, video_id: str) -> dict[str, Any]:
    import pandas as pd

    trace = pd.read_parquet(trace_path(scan_root, dataset, video_id))
    summary_path = scan_root / "video_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows = summary[(summary.dataset == dataset) & (summary.videoID == video_id)]
    if len(rows) != 1:
        raise ValueError(f"M summary mismatch for {dataset}/{video_id}")
    total_frames = int(rows.iloc[0].total_frames)
    states = expand_trace_states(total_frames, trace.source_frame_index, trace.state)
    excluded = invalid_static_intervals(states)
    return {
        "total_frames": total_frames,
        "source_states": states,
        "excluded_ranges": excluded,
    }


def question_indices(
    context: dict[str, Any],
    start_seconds: float,
    end_seconds: float,
    sampling: str,
) -> list[int]:
    """Select exactly 640 unique source indices for one request window."""

    total_frames = int(context["total_frames"])
    window_start = max(0, min(total_frames - 1, int(np.floor(start_seconds * FRAME_FPS))))
    window_end = max(window_start + 1, min(total_frames, int(np.ceil(end_seconds * FRAME_FPS))))
    if window_end - window_start < FRAME_COUNT:
        raise ValueError(
            f"request window has fewer than {FRAME_COUNT} normalized frames: "
            f"{window_end - window_start}"
        )
    if sampling == "uniform":
        indices = uniform_indices(window_start, window_end, FRAME_COUNT)
    elif sampling == SCAN_CONDITION:
        states = context["source_states"][window_start:window_end]
        relative_excluded = tuple(
            (
                max(start, window_start) - window_start,
                min(end, window_end) - window_start,
            )
            for start, end in context["excluded_ranges"]
            if end > window_start and start < window_end
        )
        selection = select_state_aware_indices(
            len(states), states, sampling, relative_excluded
        )
        indices = [window_start + index for index in selection.indices]
    else:
        raise ValueError(f"unsupported EXP101 sampling condition: {sampling}")
    if len(indices) != FRAME_COUNT or len(set(indices)) != FRAME_COUNT:
        raise ValueError(f"frame contract failed: {len(indices)} unique indices")
    if indices != sorted(indices):
        raise ValueError("selected frame indices are not sorted")
    return indices


def decode_indices(
    video_path: Path, indices: list[int], gpu: int, expected_count: int = FRAME_COUNT
) -> tuple[list[Image.Image], dict[str, float]]:
    """Decode selected normalized-cache frames with NVDEC."""

    if len(indices) != expected_count or len(set(indices)) != expected_count:
        raise ValueError(
            "decode_indices frame contract failed: "
            f"{len(indices)} unique frames, expected {expected_count}"
        )
    device = torch.device(f"cuda:{gpu}")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        str(video_path),
        gpu_id=gpu,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
        bWaitForSessionWarmUp=True,
    )
    timings["video_loading"] = time.perf_counter() - started
    started = time.perf_counter()
    surfaces = decoder.get_batch_frames_by_index(indices)
    arrays = [torch.utils.dlpack.from_dlpack(surface).cpu().numpy() for surface in surfaces]
    torch.cuda.synchronize(device)
    timings["frame_decoding"] = time.perf_counter() - started
    del surfaces, decoder
    started = time.perf_counter()
    images = [Image.fromarray(frame) for frame in arrays]
    timings["image_conversion"] = time.perf_counter() - started
    if len(images) != expected_count:
        raise ValueError(
            f"decoded frame count mismatch: {len(images)}, expected {expected_count}"
        )
    return images, timings


def assert_answer_free_rows(rows: Iterable[dict[str, Any]], label: str) -> None:
    """Reject annotations before a scan/inference process starts."""

    forbidden = {
        "answer",
        "reference",
        "reference_answer",
        "evaluation_target",
        "teacher",
        "teacher_label",
        "pseudo_label",
        "oracle_timestamp",
        "oracle_frame",
        "observer_score",
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        leaked = forbidden.intersection(row)
        if leaked:
            raise ValueError(f"{label} contains forbidden fields: {sorted(leaked)}")
        key = (str(row["dataset"]), str(row["qID"])) if "qID" in row else (
            str(row["dataset"]), str(row["videoID"])
        )
        if key in seen:
            raise ValueError(f"duplicate key in {label}: {key}")
        seen.add(key)


def validate_source_manifests(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> None:
    """Validate the inherited EXP094 split before EXP101 artifacts are made."""

    assert_training_manifest(train_rows, "train")
    assert_training_manifest(calibration_rows, "calibration")
    assert_inference_manifest(evaluation_rows, "evaluation")
    splits = {
        "train": {(str(row["dataset"]), str(row["videoID"])) for row in train_rows},
        "calibration": {(str(row["dataset"]), str(row["videoID"])) for row in calibration_rows},
        "evaluation": {(str(row["dataset"]), str(row["videoID"])) for row in evaluation_rows},
    }
    names = tuple(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(f"video leakage between {left} and {right}: {sorted(overlap)}")
