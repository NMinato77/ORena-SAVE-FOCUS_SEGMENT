"""Generic CPU-side state-pack validation for future specialist binding.

This deliberately stops before loading or applying model weights.  A final
checkpoint must supply the exact expected key set and metadata at integration
time; missing or unexpected tensors are hard failures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any


class SpecialistAssetError(RuntimeError):
    """Raised when final specialist assets are absent or fail validation."""


def require_final_specialist_bindings(config_path: Path) -> dict[str, Path]:
    """Require both final LMV bindings before any model forward can occur."""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not bool(config.get("training_protection", {}).get("final_candidate_mode", True)):
        raise SpecialistAssetError("FINAL_CANDIDATE_MODE_NOT_ENABLED")
    specialists = config.get("specialists", {})
    missing: list[str] = []
    bindings: dict[str, Path] = {}
    for name in ("general", "agg"):
        label = name.upper()
        value = specialists.get(name, {}) if isinstance(specialists, dict) else {}
        checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
        if isinstance(value, dict) and (
            value.get("scope") != "LMV"
            or int(value.get("trainable_tensor_count", -1)) != 560
            or int(value.get("trainable_parameter_count", -1)) != 205284608
            or set(value.get("required_components", ())) != {"language_lora", "vision_lora", "merger"}
        ):
            missing.append(f"FINAL_{label}_SPECIALIST_METADATA_INVALID")
            continue
        if not checkpoint:
            missing.append(f"FINAL_{label}_SPECIALIST_NOT_BOUND")
            continue
        path = Path(str(checkpoint))
        if not path.is_absolute():
            path = Path(config_path).parent / path
        if not path.is_file():
            missing.append(f"FINAL_{label}_SPECIALIST_MISSING:{path}")
            continue
        expected_sha = value.get("sha256")
        if expected_sha:
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha != str(expected_sha):
                missing.append(f"FINAL_{label}_SPECIALIST_SHA256_MISMATCH:{actual_sha}")
                continue
        bindings[name] = path
    if missing:
        raise SpecialistAssetError("; ".join(missing))
    return bindings


def validate_state_pack(
    state: Mapping[str, Any],
    expected_keys: Collection[str],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
    expected_dtypes: Mapping[str, str] | None = None,
    expected_key_count: int | None = None,
) -> dict[str, Any]:
    """Validate exact keys and optional shape/dtype metadata without a forward."""

    expected = set(expected_keys)
    actual = set(state)
    if expected_key_count is not None and len(expected) != int(expected_key_count):
        raise ValueError(
            f"state-pack specification key count mismatch: expected={expected_key_count}, specified={len(expected)}"
        )
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"state-pack key mismatch: missing={missing}, unexpected={unexpected}")

    shape_errors: dict[str, dict[str, Any]] = {}
    dtype_errors: dict[str, dict[str, Any]] = {}
    for key in sorted(expected):
        value = state[key]
        if expected_shapes is not None and key in expected_shapes:
            actual_shape = tuple(int(item) for item in getattr(value, "shape", ()))
            wanted_shape = tuple(int(item) for item in expected_shapes[key])
            if actual_shape != wanted_shape:
                shape_errors[key] = {"expected": wanted_shape, "actual": actual_shape}
        if expected_dtypes is not None and key in expected_dtypes:
            actual_dtype = str(getattr(value, "dtype", ""))
            if actual_dtype != str(expected_dtypes[key]):
                dtype_errors[key] = {"expected": str(expected_dtypes[key]), "actual": actual_dtype}
    if shape_errors or dtype_errors:
        raise ValueError(f"state-pack metadata mismatch: shapes={shape_errors}, dtypes={dtype_errors}")
    return {
        "valid": True,
        "key_count": len(actual),
        "shape_checked": bool(expected_shapes),
        "dtype_checked": bool(expected_dtypes),
        "expected_key_count": expected_key_count,
        "forward_applied": False,
    }
