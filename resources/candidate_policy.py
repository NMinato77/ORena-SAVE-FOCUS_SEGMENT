"""Pure, offline policy helpers for the SEG010 final candidate.

This module does not load a model, tokenizer, encoder, classifier, or
specialist checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("final_candidate_config.json")
CONFIG: dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

DESIGN_VERSION = str(CONFIG["design_version"])
SEGMENT_EVIDENCE_INSTRUCTION = str(CONFIG["prompt"]["segment_evidence_instruction"])
FRAME_EVIDENCE_INSTRUCTION = str(CONFIG["prompt"]["frame_evidence_instruction"])
TIMESTAMP_VARIANT = str(CONFIG["prompt"]["timestamp_variant"])


@dataclass(frozen=True)
class CandidateRoute:
    """A complete route decision, without any model state binding."""

    route_id: str
    name: str
    specialist: str
    resolution_policy: str
    sampling_policy: str
    max_frames: int
    max_pixels: int


def _capability(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    allowed = {
        "aggregation",
        "temporal_grounding",
        "temporal",
        "object_recognition",
        "object",
        "event_understanding",
        "event",
        "complex_reasoning",
        "complex",
    }
    if normalized not in allowed:
        raise ValueError(f"{label} has unsupported capability: {value!r}")
    return normalized


def _route(route_id: str) -> CandidateRoute:
    spec = CONFIG["routes"][route_id]
    return CandidateRoute(
        route_id=route_id,
        name=str(spec["name"]),
        specialist=str(spec["specialist"]),
        resolution_policy=str(spec["resolution_policy"]),
        sampling_policy=str(spec["sampling_policy"]),
        max_frames=int(spec["max_frames"]),
        max_pixels=int(spec["max_pixels"]),
    )


def route_from_predictions(q0: str, q1: str, q2: str) -> CandidateRoute:
    """Apply H2's question-only precedence exactly.

    Q2 Aggregation has priority over the temporal rescue rule.  All three
    labels are validated even when an earlier rule determines the route, so a
    malformed upstream prediction cannot silently pass into production.
    """

    q0_value = _capability(q0, "Q0")
    q1_value = _capability(q1, "Q1")
    q2_value = _capability(q2, "Q2")
    if q2_value == "aggregation":
        return _route("A")
    if q0_value in {"temporal", "temporal_grounding"} or q1_value in {
        "temporal",
        "temporal_grounding",
    }:
        return _route("B")
    return _route("C")


def specialist_for_route(route_id: str) -> str:
    """Return only the symbolic specialist name; no weights are loaded."""

    return _route(route_id).specialist
