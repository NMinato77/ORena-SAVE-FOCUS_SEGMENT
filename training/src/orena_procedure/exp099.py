"""Pure contracts for EXP099 state-aware static exclusion sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from orena_procedure.exp095_r1 import FRAME_COUNT, FRAME_FPS, uniform_indices

EXPERIMENT = "EXP099"
COARSE_STEP = 25
CHANGE_THRESHOLD = 0.02
INFORMATION_THRESHOLD = 0.10
MIN_INVALID_RUN_SECONDS = 30.0
BOUNDARY_BUFFER_SECONDS = 10.0
STATES = ("invalid_static", "stable_informative", "dynamic")
CONDITIONS = (
    "U_uniform_control",
    "M_invalid_static_exclusion",
    "C_conservative_three_state",
    "A_aggressive_three_state",
)
ADAPTIVE_QUOTAS = {
    "C_conservative_three_state": {
        "anchor": 480,
        "boundary": 40,
        "dynamic": 80,
        "stable": 40,
    },
    "A_aggressive_three_state": {
        "anchor": 320,
        "boundary": 80,
        "dynamic": 160,
        "stable": 80,
    },
}


@dataclass(frozen=True)
class StateAwareSelection:
    """Fixed-budget selection with explicit state provenance."""

    indices: tuple[int, ...]
    uniform_indices: tuple[int, ...]
    boundary_indices: tuple[int, ...]
    dynamic_indices: tuple[int, ...]
    stable_indices: tuple[int, ...]
    backfill_indices: tuple[int, ...]
    excluded_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        selected = set(self.indices)
        role_sets = [
            set(self.uniform_indices),
            set(self.boundary_indices),
            set(self.dynamic_indices),
            set(self.stable_indices),
            set(self.backfill_indices),
        ]
        if len(self.indices) != len(selected):
            raise ValueError("state-aware selection contains duplicate frame indices")
        if any(left & right for number, left in enumerate(role_sets) for right in role_sets[number + 1 :]):
            raise ValueError("state-aware selection roles must be disjoint")
        if set().union(*role_sets) != selected:
            raise ValueError("state-aware selection roles do not cover selected indices")
        if len(selected) != FRAME_COUNT:
            raise ValueError(f"state-aware selection must contain {FRAME_COUNT} frames")
        previous_end = -1
        for start, end in self.excluded_ranges:
            if start < 0 or end <= start or start < previous_end:
                raise ValueError("excluded ranges must be sorted half-open intervals")
            previous_end = end


def classify_states(
    change: Sequence[float],
    information: Sequence[float],
    change_threshold: float = CHANGE_THRESHOLD,
    information_threshold: float = INFORMATION_THRESHOLD,
) -> np.ndarray:
    """Classify each coarse point without labels or learned scores."""

    change_array = np.asarray(change, dtype=float)
    information_array = np.asarray(information, dtype=float)
    if change_array.ndim != 1 or information_array.ndim != 1:
        raise ValueError("change and information must be one-dimensional")
    if len(change_array) != len(information_array):
        raise ValueError("change and information must have equal length")
    if not np.isfinite(change_array).all() or not np.isfinite(information_array).all():
        raise ValueError("change and information must be finite")
    if change_threshold < 0 or information_threshold < 0:
        raise ValueError("state thresholds must be non-negative")
    states = np.full(len(change_array), "dynamic", dtype=object)
    stable = change_array <= change_threshold
    states[stable & (information_array <= information_threshold)] = "invalid_static"
    states[stable & (information_array > information_threshold)] = "stable_informative"
    return states


def expand_trace_states(
    total_frames: int,
    trace_indices: Sequence[int],
    states: Sequence[str],
) -> np.ndarray:
    """Map regularly sampled states to half-open source-frame intervals."""

    indices = np.asarray(trace_indices, dtype=np.int64)
    state_array = np.asarray(states, dtype=object)
    if total_frames < 1 or len(indices) == 0 or len(indices) != len(state_array):
        raise ValueError("invalid trace or frame count")
    if indices[0] != 0 or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("trace indices must be strictly increasing and start at zero")
    if indices[-1] >= total_frames or not set(state_array) <= set(STATES):
        raise ValueError("trace indices or states are invalid")
    edges = np.empty(len(indices) + 1, dtype=np.int64)
    edges[0] = 0
    edges[1:-1] = (indices[:-1] + indices[1:]) // 2 + 1
    edges[-1] = total_frames
    result = np.empty(total_frames, dtype=object)
    for start, end, state in zip(edges[:-1], edges[1:], state_array, strict=True):
        result[start:end] = state
    return result


def contiguous_ranges(mask: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    """Return sorted half-open ranges for true values."""

    values = np.asarray(mask, dtype=bool)
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        start = index
        index += 1
        while index < len(values) and values[index]:
            index += 1
        ranges.append((start, index))
    return tuple(ranges)


def invalid_static_intervals(
    source_states: Sequence[str],
    frame_fps: float = FRAME_FPS,
    min_run_seconds: float = MIN_INVALID_RUN_SECONDS,
    boundary_buffer_seconds: float = BOUNDARY_BUFFER_SECONDS,
) -> tuple[tuple[int, int], ...]:
    """Return only the interior of sufficiently long invalid-static runs."""

    values = np.asarray(source_states, dtype=object)
    if frame_fps <= 0 or min_run_seconds <= 0 or boundary_buffer_seconds < 0:
        raise ValueError("invalid interval parameters")
    if not set(values) <= set(STATES):
        raise ValueError("unknown source state")
    margin = round(boundary_buffer_seconds * frame_fps)
    intervals: list[tuple[int, int]] = []
    for start, end in contiguous_ranges(values == "invalid_static"):
        if (end - start) / frame_fps < min_run_seconds:
            continue
        core_start = start + margin
        core_end = end - margin
        if core_start < core_end:
            intervals.append((core_start, core_end))
    return tuple(intervals)


def interval_mask(total_frames: int, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    """Convert sorted half-open intervals into a boolean frame mask."""

    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    mask = np.zeros(total_frames, dtype=bool)
    previous_end = -1
    for start, end in intervals:
        if start < 0 or end <= start or end > total_frames or start < previous_end:
            raise ValueError("invalid or overlapping intervals")
        mask[start:end] = True
        previous_end = end
    return mask


def _sample_mask(mask: np.ndarray, count: int, selected: set[int]) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("sample count must be non-negative")
    candidates = np.flatnonzero(mask)
    candidates = candidates[~np.isin(candidates, np.fromiter(selected, dtype=np.int64))]
    if count == 0 or len(candidates) == 0:
        return ()
    if count >= len(candidates):
        return tuple(int(value) for value in candidates)
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    chosen = candidates[np.unique(positions)]
    if len(chosen) < count:
        chosen_set = set(int(value) for value in chosen)
        chosen = np.asarray(
            list(chosen) + [value for value in candidates if int(value) not in chosen_set][: count - len(chosen)],
            dtype=np.int64,
        )
    return tuple(int(value) for value in chosen[:count])


def _selection_masks(
    source_states: np.ndarray,
    excluded_ranges: tuple[tuple[int, int], ...],
) -> dict[str, np.ndarray]:
    excluded = interval_mask(len(source_states), excluded_ranges)
    qualified_invalid = np.zeros(len(source_states), dtype=bool)
    boundary = np.zeros(len(source_states), dtype=bool)
    for start, end in contiguous_ranges(source_states == "invalid_static"):
        qualified = (end - start) / FRAME_FPS >= MIN_INVALID_RUN_SECONDS
        if not qualified:
            continue
        core = interval_mask(len(source_states), ((start + round(BOUNDARY_BUFFER_SECONDS * FRAME_FPS), end - round(BOUNDARY_BUFFER_SECONDS * FRAME_FPS)),))
        qualified_invalid[start:end] = True
        boundary[start:end] = ~core[start:end]
    valid = ~excluded
    return {
        "valid": valid,
        "boundary": boundary & valid,
        "dynamic": (source_states == "dynamic") & valid,
        "stable": (source_states == "stable_informative") & valid,
        "qualified_invalid": qualified_invalid,
    }


def select_state_aware_indices(
    total_frames: int,
    source_states: Sequence[str],
    condition: str,
    excluded_ranges: Sequence[tuple[int, int]] | None = None,
) -> StateAwareSelection:
    """Select a deterministic 640-frame digest for one registered condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    values = np.asarray(source_states, dtype=object)
    if len(values) != total_frames or total_frames < FRAME_COUNT:
        raise ValueError("source states must cover a sufficiently long video")
    if not set(values) <= set(STATES):
        raise ValueError("unknown source state")
    excluded = tuple(excluded_ranges or ())
    masks = _selection_masks(values, excluded)
    if condition == "U_uniform_control":
        indices = uniform_indices(total_frames)
        return StateAwareSelection(indices, indices, (), (), (), (), excluded)

    selected: set[int] = set()
    if condition == "M_invalid_static_exclusion":
        uniform_valid = _sample_mask(masks["valid"], FRAME_COUNT, selected)
        selected.update(uniform_valid)
        backfill = ()
        return StateAwareSelection(
            tuple(sorted(selected)),
            uniform_valid,
            (),
            (),
            (),
            backfill,
            excluded,
        )

    quota = ADAPTIVE_QUOTAS[condition]
    uniform = _sample_mask(masks["valid"], quota["anchor"], selected)
    selected.update(uniform)
    boundary = _sample_mask(masks["boundary"], quota["boundary"], selected)
    selected.update(boundary)
    dynamic = _sample_mask(masks["dynamic"], quota["dynamic"], selected)
    selected.update(dynamic)
    stable = _sample_mask(masks["stable"], quota["stable"], selected)
    selected.update(stable)
    missing = FRAME_COUNT - len(selected)
    backfill = _sample_mask(masks["valid"], missing, selected)
    selected.update(backfill)
    if len(selected) != FRAME_COUNT:
        raise AssertionError("state-aware sampler could not fill fixed budget")
    return StateAwareSelection(
        tuple(sorted(selected)),
        uniform,
        boundary,
        dynamic,
        stable,
        backfill,
        excluded,
    )


def selection_roles(selection: StateAwareSelection) -> dict[int, str]:
    """Return one provenance role for every selected frame."""

    roles: dict[int, str] = {}
    for name, indices in (
        ("uniform", selection.uniform_indices),
        ("boundary", selection.boundary_indices),
        ("dynamic", selection.dynamic_indices),
        ("stable", selection.stable_indices),
        ("backfill", selection.backfill_indices),
    ):
        for index in indices:
            roles[int(index)] = name
    if set(roles) != set(selection.indices):
        raise AssertionError("selection role map is incomplete")
    return roles
