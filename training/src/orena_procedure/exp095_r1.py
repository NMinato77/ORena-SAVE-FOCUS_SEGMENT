"""Pure contracts for EXP095-R1 generic foreign-object ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from focus import FOClass

EXPERIMENT = "EXP095-R1"
MODEL_ID = "google/siglip-base-patch16-224"
FRAME_FPS = 5.0
SCAN_FPS = 0.2
SCAN_STEP = round(FRAME_FPS / SCAN_FPS)
FRAME_COUNT = 640
ANCHOR_COUNT = 320
GUIDED_COUNT = 320
GUIDED_MIN_SPACING_SECONDS = 10.0
SIGLIP_DIMENSION = 768
HEAD_DIMENSION = 256
SEED = 17

OFFICIAL_FO_CLASSES = tuple(FOClass().valid_names)
DOMAIN_SUPPORTED_CLASSES = {
    "heico": frozenset({"Sponge", "Clip", "Silicone Loop", "External Drain", "Needle"}),
    "lapchole": frozenset(
        {"Sponge", "Clip", "Specimen Bag", "External Drain", "Gallstone", "Specimen"}
    ),
}


@dataclass(frozen=True)
class HybridSelection:
    """A fixed-size selection with provenance for every budget component."""

    indices: tuple[int, ...]
    anchor_indices: tuple[int, ...]
    guided_indices: tuple[int, ...]
    backfill_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(set(self.indices)):
            raise ValueError("hybrid selection contains duplicate frame indices")
        if set(self.anchor_indices) | set(self.guided_indices) != set(self.indices):
            raise ValueError("hybrid selection provenance does not cover selected indices")
        if not set(self.backfill_indices) <= set(self.guided_indices):
            raise ValueError("backfill indices must be a subset of guided indices")


def class_targets(label_text: str, class_names: Sequence[str] = OFFICIAL_FO_CLASSES) -> np.ndarray:
    """Convert a canonical comma-separated complete label set to binary targets."""

    labels = {part.strip() for part in str(label_text).split(",") if part.strip()}
    if not labels <= set(class_names):
        raise ValueError(f"unknown class in label text: {sorted(labels - set(class_names))}")
    return np.asarray([int(name in labels) for name in class_names], dtype=np.float32)


def supported_class_indices(dataset: str) -> tuple[int, ...]:
    """Return only class heads with a registered domain meaning."""

    try:
        supported = DOMAIN_SUPPORTED_CLASSES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {dataset}") from exc
    return tuple(index for index, name in enumerate(OFFICIAL_FO_CLASSES) if name in supported)


def generic_score(
    probabilities: np.ndarray, dataset: str, class_names: Sequence[str] = OFFICIAL_FO_CLASSES
) -> np.ndarray:
    """Return a ranking score, never an Any-FO probability."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(class_names):
        raise ValueError("probabilities must have shape [rows, class_count]")
    supported = DOMAIN_SUPPORTED_CLASSES.get(dataset)
    if supported is None:
        raise ValueError(f"unknown dataset: {dataset}")
    indices = [index for index, name in enumerate(class_names) if name in supported]
    if not indices:
        raise ValueError(f"no supported classes for dataset: {dataset}")
    return values[:, indices].max(axis=1)


def uniform_indices(total_frames: int, count: int = FRAME_COUNT) -> tuple[int, ...]:
    """Return exactly count unique source indices including both endpoints."""

    if total_frames < count or count < 1:
        raise ValueError(f"cannot select {count} unique frames from {total_frames}")
    indices = np.linspace(0, total_frames - 1, count, dtype=np.int64)
    result = tuple(int(value) for value in indices)
    if len(set(result)) != count:
        raise AssertionError("uniform allocator produced duplicate indices")
    return result


def _uniform_excluding(total_frames: int, count: int, excluded: set[int]) -> list[int]:
    candidates = np.linspace(0, total_frames - 1, max(count * 4, count), dtype=np.int64)
    result: list[int] = []
    seen: set[int] = set()
    for value in candidates.tolist():
        index = int(value)
        if index not in excluded and index not in seen:
            result.append(index)
            seen.add(index)
            if len(result) == count:
                return result
    for index in range(total_frames):
        if index not in excluded and index not in seen:
            result.append(index)
            seen.add(index)
            if len(result) == count:
                return result
    return result


def select_hybrid_indices(
    total_frames: int,
    trace_indices: Sequence[int],
    trace_scores: Sequence[float],
    anchor_count: int = ANCHOR_COUNT,
    guided_count: int = GUIDED_COUNT,
    min_spacing_seconds: float = GUIDED_MIN_SPACING_SECONDS,
    frame_fps: float = FRAME_FPS,
) -> HybridSelection:
    """Select uniform anchors plus spaced candidates and uniform backfill."""

    if total_frames < anchor_count + guided_count:
        raise ValueError("video has fewer frames than the fixed 640-frame budget")
    if len(trace_indices) != len(trace_scores):
        raise ValueError("trace indices and scores must have equal length")
    if min_spacing_seconds < 0 or frame_fps <= 0:
        raise ValueError("invalid temporal spacing or frame rate")

    anchors = tuple(sorted(uniform_indices(total_frames, anchor_count)))
    anchor_set = set(anchors)
    trace = sorted(
        (
            (int(index), float(score))
            for index, score in zip(trace_indices, trace_scores, strict=True)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    guided: list[int] = []
    selected_times: list[float] = []
    min_spacing = float(min_spacing_seconds)
    for index, score in trace:
        del score
        if not 0 <= index < total_frames or index in anchor_set or index in guided:
            continue
        timestamp = index / frame_fps
        if all(abs(timestamp - previous) >= min_spacing for previous in selected_times):
            guided.append(index)
            selected_times.append(timestamp)
            if len(guided) == guided_count:
                break

    backfill: list[int] = []
    if len(guided) < guided_count:
        selected = anchor_set | set(guided)
        for index in _uniform_excluding(total_frames, guided_count - len(guided), selected):
            guided.append(index)
            backfill.append(index)
            selected.add(index)
            if len(guided) == guided_count:
                break
    if len(guided) != guided_count:
        raise AssertionError("hybrid allocator could not fill guided budget")
    indices = tuple(sorted(anchor_set | set(guided)))
    return HybridSelection(
        indices=indices,
        anchor_indices=anchors,
        guided_indices=tuple(guided),
        backfill_indices=tuple(backfill),
    )


def timestamp_hit_rates(
    selected_indices: Sequence[int],
    positive_timestamps: Sequence[float],
    tolerances_seconds: Sequence[float] = (5.0, 10.0, 30.0),
    frame_fps: float = FRAME_FPS,
) -> dict[str, float | int | None]:
    """Measure sparse known-positive hits without calling it Any-FO recall."""

    positive = np.asarray(positive_timestamps, dtype=float)
    selected = np.asarray(selected_indices, dtype=float) / frame_fps
    result: dict[str, float | int | None] = {"positive_timestamps": len(positive)}
    if len(positive) == 0:
        for tolerance in tolerances_seconds:
            result[f"hit_count_pm{int(tolerance)}s"] = 0
            result[f"hit_rate_pm{int(tolerance)}s"] = None
        return result
    distance = np.min(np.abs(positive[:, None] - selected[None, :]), axis=1)
    for tolerance in tolerances_seconds:
        result[f"hit_count_pm{int(tolerance)}s"] = int(np.sum(distance <= tolerance))
        result[f"hit_rate_pm{int(tolerance)}s"] = float(np.mean(distance <= tolerance))
    return result


def score_summary(scores: Sequence[float]) -> dict[str, float | int]:
    """Store distribution diagnostics for a non-probabilistic ranking score."""

    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("score trace must be non-empty and finite")
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return {
        "points": int(values.size),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
