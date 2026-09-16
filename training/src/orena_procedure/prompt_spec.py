"""Shared machine-readable prompt and timestamp defaults.

SEG010 and future relevant SEGMENT experiments import these values rather than
copying prompt wording into individual trainers. Historical experiment
encoders retain the specification they actually used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parents[2] / "configs/segment_prompt_defaults.json"
# Public name used in experiment metadata and trainer configs.
PROMPT_SPEC_PATH = SPEC_PATH


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError(f"unsupported prompt spec schema: {spec.get('schema_version')}")
    segment = spec.get("segment", {})
    frame = spec.get("frame", {})
    if segment.get("evidence_instruction_id") != "P2":
        raise ValueError("SEGMENT default must be P2")
    if segment.get("timestamp_variant") != "T0":
        raise ValueError("SEGMENT default timestamp must be T0")
    if frame.get("evidence_instruction_id") != "P2_FRAME":
        raise ValueError("FRAME default must use the P2 semantic counterpart")
    required = {
        "evidence_instruction",
        "timestamp_frame_template",
        "timestamp_context",
    }
    if required - set(segment):
        raise ValueError("SEGMENT prompt spec is missing required fields")
    if "evidence_instruction" not in frame:
        raise ValueError("FRAME prompt spec is missing evidence_instruction")
    return spec


PROMPT_SPEC = load_prompt_spec()
DESIGN_VERSION = str(PROMPT_SPEC["design_version"])
PROMPT_SPEC_SHA256 = file_sha256(SPEC_PATH)
SEGMENT_EVIDENCE_INSTRUCTION = str(PROMPT_SPEC["segment"]["evidence_instruction"])
SEGMENT_TIMESTAMP_VARIANT = str(PROMPT_SPEC["segment"]["timestamp_variant"])
SEGMENT_TIMESTAMP_FRAME_TEMPLATE = str(
    PROMPT_SPEC["segment"]["timestamp_frame_template"]
)
SEGMENT_TIMESTAMP_CONTEXT = str(PROMPT_SPEC["segment"]["timestamp_context"])
FRAME_EVIDENCE_INSTRUCTION = str(PROMPT_SPEC["frame"]["evidence_instruction"])


def segment_timestamp_text(timestamp: float) -> str:
    """Render the canonical T0 per-frame timestamp line."""

    return SEGMENT_TIMESTAMP_FRAME_TEMPLATE.format(timestamp=float(timestamp))
