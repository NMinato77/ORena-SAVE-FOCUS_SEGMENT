"""Contracts and module surgery for EXP102 micro-overfit experiments.

The experiment intentionally keeps the data boundary explicit.  Selection is
performed on answer-free projections of the existing EXP101 training
partition, while official answers are used only by the SFT manifest and the
post-inference scorer.

PyNvVideoCodec is imported before torch by the repository GPU contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import PyNvVideoCodec as nvc  # noqa: F401 -- initialize NVDEC before torch/CUDA.
import torch

from orena_procedure.exp094 import LoRALinear
from orena_procedure.exp101 import (
    FRAME_COUNT,
    SCAN_CONDITION,
    load_scan_context,
    question_indices,
)

EXPERIMENT = "EXP102"
PROMPT_SCHEMA = "B0"
LANGUAGE_RANK = 16
LANGUAGE_ALPHA = 32.0
LANGUAGE_DROPOUT = 0.05
LANGUAGE_LR = 5e-5
VISION_RANK = 4
VISION_ALPHA = 8.0
VISION_DROPOUT = 0.05
VISION_LR = 1e-5
WEIGHT_DECAY = 0.01
MAX_UPDATES = 500
CHECKPOINT_STEPS = (0, 50, 100, 200, 300, 500)

CANDIDATES: dict[str, dict[str, Any]] = {
    "L": {
        "language": {
            "scope": "all_linear",
            "rank": LANGUAGE_RANK,
            "alpha": LANGUAGE_ALPHA,
            "dropout": LANGUAGE_DROPOUT,
            "learning_rate": LANGUAGE_LR,
        },
        "merger": {"scope": "frozen"},
        "vision": {"scope": "frozen"},
    },
    "LM": {
        "language": {
            "scope": "all_linear",
            "rank": LANGUAGE_RANK,
            "alpha": LANGUAGE_ALPHA,
            "dropout": LANGUAGE_DROPOUT,
            "learning_rate": LANGUAGE_LR,
        },
        "merger": {"scope": "full", "learning_rate": LANGUAGE_LR},
        "vision": {"scope": "frozen"},
    },
    "LMV": {
        "language": {
            "scope": "all_linear",
            "rank": LANGUAGE_RANK,
            "alpha": LANGUAGE_ALPHA,
            "dropout": LANGUAGE_DROPOUT,
            "learning_rate": LANGUAGE_LR,
        },
        "merger": {"scope": "full", "learning_rate": LANGUAGE_LR},
        "vision": {
            "scope": "final_block_all_linear_lora",
            "rank": VISION_RANK,
            "alpha": VISION_ALPHA,
            "dropout": VISION_DROPOUT,
            "learning_rate": VISION_LR,
        },
    },
}


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trace_path(scan_root: Path, dataset: str, video_id: str) -> Path:
    path = scan_root / "traces" / dataset / f"{Path(video_id).stem}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing M trace: {path}")
    return path


def answer_free_projection(row: dict[str, Any], role: str = "micro_inference") -> dict[str, Any]:
    """Drop all answer/reference fields before deterministic selection or inference."""

    forbidden = {
        "answer",
        "reference",
        "reference_answer",
        "teacher",
        "teacher_label",
        "pseudo_label",
        "oracle_timestamp",
        "oracle_frame",
        "observer_score",
    }
    result = {key: value for key, value in row.items() if key not in forbidden}
    result["role"] = role
    return result


def _parent_and_child(root: torch.nn.Module, relative_name: str) -> tuple[torch.nn.Module, str]:
    parts = relative_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _module_path(prefix: str, relative_name: str) -> str:
    return f"{prefix}.{relative_name}" if relative_name else prefix


def language_linear_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    language_model = model.model.language_model
    return [
        (_module_path("model.language_model", name), module)
        for name, module in language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def merger_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    visual = model.model.visual
    result: list[tuple[str, torch.nn.Module]] = []
    for root_name in ("merger", "deepstack_merger_list"):
        root = getattr(visual, root_name)
        for name, module in root.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                relative = f"{root_name}.{name}" if name else root_name
                result.append((_module_path("model.visual", relative), module))
    return result


def vision_final_block_modules(model: torch.nn.Module) -> tuple[int, list[tuple[str, torch.nn.Linear]]]:
    blocks = model.model.visual.blocks
    final_index = len(blocks) - 1
    final_block = blocks[final_index]
    result = [
        (_module_path("model.visual", f"blocks.{final_index}.{name}"), module)
        for name, module in final_block.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    return final_index, result


def module_tree_summary(model: torch.nn.Module) -> dict[str, Any]:
    final_index, vision = vision_final_block_modules(model)
    language = language_linear_modules(model)
    merger = merger_modules(model)
    return {
        "language_model": {
            "linear_count": len(language),
            "linear_modules": [
                {"name": name, "shape": list(module.weight.shape)} for name, module in language
            ],
        },
        "multimodal_merger": {
            "modules": [
                {
                    "name": name,
                    "type": type(module).__name__,
                    "shape": list(module.weight.shape) if hasattr(module, "weight") else None,
                }
                for name, module in merger
            ]
        },
        "vision_encoder": {
            "final_block_index": final_index,
            "final_block_linear_modules": [
                {"name": name, "shape": list(module.weight.shape)} for name, module in vision
            ],
        },
    }


def install_candidate(model: torch.nn.Module, candidate_name: str) -> dict[str, Any]:
    """Freeze the base and install the exact candidate scope on actual modules."""

    if candidate_name not in CANDIDATES:
        raise ValueError(f"unknown EXP102 candidate: {candidate_name}")
    spec = CANDIDATES[candidate_name]
    for parameter in model.parameters():
        parameter.requires_grad = False

    language_replaced: list[str] = []
    for full_name, base in language_linear_modules(model):
        relative = full_name.removeprefix("model.")
        parent, child = _parent_and_child(model.model, relative)
        setattr(
            parent,
            child,
            LoRALinear(
                base,
                rank=int(spec["language"]["rank"]),
                alpha=float(spec["language"]["alpha"]),
                dropout=float(spec["language"]["dropout"]),
            ),
        )
        language_replaced.append(full_name)

    merger_trainable: list[str] = []
    if spec["merger"]["scope"] == "full":
        for name, module in merger_modules(model):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                parameter.requires_grad = True
                merger_trainable.append(f"{name}.{parameter_name}")

    vision_replaced: list[str] = []
    if spec["vision"]["scope"] == "final_block_all_linear_lora":
        _final_index, modules = vision_final_block_modules(model)
        for full_name, base in modules:
            relative = full_name.removeprefix("model.")
            parent, child = _parent_and_child(model.model, relative)
            setattr(
                parent,
                child,
                LoRALinear(
                    base,
                    rank=int(spec["vision"]["rank"]),
                    alpha=float(spec["vision"]["alpha"]),
                    dropout=float(spec["vision"]["dropout"]),
                ),
            )
            vision_replaced.append(full_name)

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError(f"candidate {candidate_name} has no trainable parameters")
    trainable_names = [name for name, _parameter in trainable]
    language_names = [name for name in trainable_names if ".language_model." in name]
    merger_names = [
        name
        for name in trainable_names
        if ".visual.merger." in name or ".visual.deepstack_merger_list." in name
    ]
    vision_names = [name for name in trainable_names if ".visual.blocks." in name]
    if set(language_names) != {
        name
        for name in trainable_names
        if ".language_model." in name and "lora_" in name
    }:
        raise RuntimeError("unexpected non-LoRA language parameters are trainable")
    return {
        "candidate": candidate_name,
        "language_modules": language_replaced,
        "merger_trainable_modules": merger_trainable,
        "vision_modules": vision_replaced,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
        "trainable_parameter_count_by_scope": {
            "language_lora": int(sum(parameter.numel() for name, parameter in trainable if name in language_names)),
            "merger_full": int(sum(parameter.numel() for name, parameter in trainable if name in merger_names)),
            "vision_lora": int(sum(parameter.numel() for name, parameter in trainable if name in vision_names)),
        },
        "language_linear_count": len(language_replaced),
        "vision_linear_count": len(vision_replaced),
    }


def configure_train_modes(model: torch.nn.Module, candidate_name: str, training: bool) -> None:
    model.train(training)
    model.model.visual.eval()
    if training and CANDIDATES[candidate_name]["vision"]["scope"] != "frozen":
        final_index = len(model.model.visual.blocks) - 1
        model.model.visual.blocks[final_index].train()


def parameter_l2(model: torch.nn.Module, predicate: Any = None) -> float:
    values = []
    for name, parameter in model.named_parameters():
        if predicate is None or predicate(name, parameter):
            values.append(torch.sum(parameter.detach().float() ** 2))
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def candidate_parameter_norms(model: torch.nn.Module) -> dict[str, float]:
    return {
        "trainable_parameter_l2": parameter_l2(
            model, lambda _name, parameter: parameter.requires_grad
        ),
        "lora_parameter_l2": parameter_l2(
            model, lambda name, _parameter: "lora_" in name
        ),
    }


def selected_indices_for_row(
    row: dict[str, Any], scan_root: Path, contexts: dict[tuple[str, str], dict[str, Any]]
) -> list[int]:
    key = (str(row["dataset"]), str(row["videoID"]))
    context = contexts.setdefault(key, load_scan_context(scan_root, *key))
    return question_indices(
        context,
        float(row["start_time"]),
        float(row["end_time"]),
        SCAN_CONDITION,
    )


def frame_contract_summary(indices: list[int], expected_count: int = FRAME_COUNT) -> dict[str, Any]:
    if len(indices) != expected_count or len(set(indices)) != expected_count:
        raise ValueError(
            "EXP102 M frame contract failed: "
            f"{len(indices)} unique frames, expected {expected_count}"
        )
    return {
        "count": len(indices),
        "unique": len(set(indices)),
        "ordered": indices == sorted(indices),
        "index_sha256": json_hash(indices),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {value}")
    return float(value)
