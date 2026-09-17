"""SEG010 union-adapter and task-switch contracts.

The experiment keeps one physical model tree containing the complete
Language/Vision/Merger union.  Conditions only change the active gradient mask;
they never reinstall or reset inherited visual weights at the FRAME/SEGMENT
boundary.

GPU callers must import :mod:`PyNvVideoCodec` before importing ``torch``.  This
module preserves that repository-wide import order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import PyNvVideoCodec as nvc  # noqa: F401 -- mandatory before torch/CUDA.
import torch

from orena_procedure.exp094 import LoRALinear
from orena_procedure.exp102 import (
    _parent_and_child,
    language_linear_modules,
)
from orena_procedure.prompt_spec import (
    DESIGN_VERSION,
    FRAME_EVIDENCE_INSTRUCTION,
    PROMPT_SPEC,
    PROMPT_SPEC_PATH,
    PROMPT_SPEC_SHA256,
    SEGMENT_EVIDENCE_INSTRUCTION,
    SEGMENT_TIMESTAMP_CONTEXT,
    SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
    SEGMENT_TIMESTAMP_VARIANT,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SEED = 11011
MAX_PIXELS = 50176
GLOBAL_BATCH = 8
MICROBATCH_PER_GPU = 1
GRADIENT_ACCUMULATION = 4
LANGUAGE_RANK = 16
LANGUAGE_ALPHA = 32.0
LANGUAGE_DROPOUT = 0.05
VISION_RANK = 16
VISION_ALPHA = 32.0
VISION_DROPOUT = 0.05
LANGUAGE_LR = 1e-4
VISION_LR = 1e-5
MERGER_LR = 5e-5
WEIGHT_DECAY = 0.01
OPTIMIZER_LEARNING_RATES = {
    "language_lora": LANGUAGE_LR,
    "vision_lora": VISION_LR,
    "merger_full": MERGER_LR,
}

# The prompt/timestamp contract is shared by every SEG010 condition.  These
# aliases make the decision visible from the SEG010 module and let trainers
# record the exact source-of-truth file in their artifacts.


def prompt_contract() -> dict[str, Any]:
    """Return the exact prompt/timestamp contract recorded in SEG010 outputs."""

    return {
        "design_version": DESIGN_VERSION,
        "source_path": str(PROMPT_SPEC_PATH),
        "source_sha256": PROMPT_SPEC_SHA256,
        "segment": {
            "evidence_instruction_id": "P2",
            "evidence_instruction": SEGMENT_EVIDENCE_INSTRUCTION,
            "timestamp_variant": SEGMENT_TIMESTAMP_VARIANT,
            "timestamp_frame_template": SEGMENT_TIMESTAMP_FRAME_TEMPLATE,
            "timestamp_context": SEGMENT_TIMESTAMP_CONTEXT,
        },
        "frame": {
            "evidence_instruction_id": "P2_FRAME",
            "evidence_instruction": FRAME_EVIDENCE_INSTRUCTION,
        },
        "selection": PROMPT_SPEC["selection"],
        "preflight": PROMPT_SPEC["preflight"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensors including names, dtype, shape, and raw bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scope_specs(artifact_root: Path) -> dict[str, dict[str, Any]]:
    return read_json(artifact_root / "adaptation_scopes.json")


def condition_specs(artifact_root: Path) -> list[dict[str, Any]]:
    return read_json(artifact_root / "conditions.json")


def condition_spec(artifact_root: Path, condition: str) -> dict[str, Any]:
    values = [item for item in condition_specs(artifact_root) if item["condition"] == condition]
    if len(values) != 1:
        raise ValueError(f"unknown SEG010 condition: {condition}")
    return values[0]


def scope_spec(artifact_root: Path, canonical_name: str) -> dict[str, Any]:
    values = [
        item
        for item in scope_specs(artifact_root).values()
        if item["canonical_name"] == canonical_name
    ]
    if len(values) != 1:
        raise ValueError(f"unknown SEG010 scope: {canonical_name}")
    return values[0]


def task_scope(artifact_root: Path, condition: str, task: str) -> dict[str, Any] | None:
    condition_value = condition_spec(artifact_root, condition)
    if task == "F":
        name = condition_value["FRAME_trainable_target"]
    elif task == "S":
        name = condition_value["canonical_scope"]
    else:
        raise ValueError(f"unknown SEG010 task: {task}")
    return None if name is None else scope_spec(artifact_root, name)


def expected_union_names(artifact_root: Path) -> set[str]:
    return set(scope_spec(artifact_root, "SEGMENT_LMV")["trainable_parameter_names"])


def union_parameter_names(model: torch.nn.Module) -> set[str]:
    return {
        name
        for name, _parameter in model.named_parameters()
        if ("lora_" in name and (".language_model." in name or ".visual.blocks." in name))
        or ".visual.merger." in name
        or ".visual.deepstack_merger_list." in name
    }


def vision_last_blocks_modules(
    model: torch.nn.Module, block_count: int = 4
) -> tuple[list[int], list[tuple[str, torch.nn.Linear]]]:
    blocks = model.model.visual.blocks
    if len(blocks) < block_count:
        raise ValueError(f"model has {len(blocks)} visual blocks, expected at least {block_count}")
    indices = list(range(len(blocks) - block_count, len(blocks)))
    modules: list[tuple[str, torch.nn.Linear]] = []
    for index in indices:
        for name, module in blocks[index].named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((f"model.visual.blocks.{index}.{name}", module))
    return indices, modules


def install_union(model: torch.nn.Module) -> dict[str, Any]:
    """Install the exact common L+V union and leave every base parameter frozen."""

    if any("lora_" in name for name, _parameter in model.named_parameters()):
        raise ValueError("SEG010 union installation requires a model without existing LoRA")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    language_modules = language_linear_modules(model)
    for full_name, base in language_modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(
            parent,
            child,
            LoRALinear(
                base,
                rank=LANGUAGE_RANK,
                alpha=LANGUAGE_ALPHA,
                dropout=LANGUAGE_DROPOUT,
            ),
        )

    vision_indices, vision_modules = vision_last_blocks_modules(model)
    for full_name, base in vision_modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(
            parent,
            child,
            LoRALinear(
                base,
                rank=VISION_RANK,
                alpha=VISION_ALPHA,
                dropout=VISION_DROPOUT,
            ),
        )

    # Merger parameters remain base-valued but are part of the union checkpoint
    # because FRAME-MV and SEGMENT-LMV may update them later.
    trainable_names = union_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_names)
    return {
        "language_linear_count": len(language_modules),
        "vision_linear_count": len(vision_modules),
        "vision_block_indices": vision_indices,
        "union_parameter_names": sorted(trainable_names),
        "union_parameter_count": int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if name in trainable_names)
        ),
    }


def validate_union(model: torch.nn.Module, artifact_root: Path) -> None:
    actual = union_parameter_names(model)
    expected = expected_union_names(artifact_root)
    if actual != expected:
        raise ValueError(
            "SEG010 union scope mismatch: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )


def select_task(model: torch.nn.Module, artifact_root: Path, condition: str, task: str) -> list[str]:
    """Set the active mask and clear all stale gradients before a task update."""

    expected = expected_union_names(artifact_root)
    actual = union_parameter_names(model)
    if actual != expected:
        raise ValueError("cannot select task on a model with a different union")
    selected_spec = task_scope(artifact_root, condition, task)
    if selected_spec is None:
        raise ValueError(f"condition {condition} has no task {task}")
    selected = set(selected_spec["trainable_parameter_names"])
    for name, parameter in model.named_parameters():
        parameter.grad = None
        parameter.requires_grad_(name in selected)
    return sorted(selected)


def make_optimizer(
    model: torch.nn.Module, artifact_root: Path, condition: str, task: str
) -> torch.optim.Optimizer | None:
    selected_spec = task_scope(artifact_root, condition, task)
    if selected_spec is None:
        return None
    parameters = dict(model.named_parameters())
    groups = []
    covered: set[str] = set()
    for group_name, group_spec in selected_spec["parameter_groups"].items():
        names = list(group_spec["names"])
        if not names:
            continue
        if any(name not in parameters for name in names):
            raise ValueError(f"optimizer group {group_name} has missing model parameter")
        groups.append(
            {
                "params": [parameters[name] for name in names],
                # The preflight conditions artifact is historical and contains
                # the earlier 5e-5 language value.  SEG010's final LR decision
                # is owned by this trainer contract, not by that raw artifact.
                "lr": float(OPTIMIZER_LEARNING_RATES[group_name]),
                "weight_decay": float(group_spec["weight_decay"]),
                "scope": group_name,
            }
        )
        covered.update(names)
    expected = set(selected_spec["trainable_parameter_names"])
    if covered != expected:
        raise ValueError("optimizer groups do not cover the selected task scope")
    return torch.optim.AdamW(groups)


def optimizer_group_names(
    artifact_root: Path, condition: str, task: str
) -> dict[str, list[str]] | None:
    selected_spec = task_scope(artifact_root, condition, task)
    if selected_spec is None:
        return None
    return {
        group_name: list(group_spec["names"])
        for group_name, group_spec in selected_spec["parameter_groups"].items()
    }


def optimizer_group_learning_rates(
    artifact_root: Path, condition: str, task: str
) -> dict[str, float] | None:
    selected_spec = task_scope(artifact_root, condition, task)
    if selected_spec is None:
        return None
    return {
        group_name: float(OPTIMIZER_LEARNING_RATES[group_name])
        for group_name in selected_spec["parameter_groups"]
    }


def task_sequence(artifact_root: Path, schedule_name: str) -> list[dict[str, Any]]:
    schedules = read_json(artifact_root / "task_schedules.json")
    if schedule_name not in schedules:
        raise ValueError(f"unknown SEG010 schedule: {schedule_name}")
    return list(schedules[schedule_name])


def phase2_s_only_schedule(artifact_root: Path, condition: str) -> list[dict[str, Any]]:
    """Append the fixed 128-update SEGMENT-only Phase 2 tail to a condition."""

    phase1 = task_sequence(artifact_root, condition_spec(artifact_root, condition)["schedule"])
    continuation = task_sequence(artifact_root, "CTRL")
    segment_updates_before = sum(item["task"] == "S" for item in phase1)
    result = list(phase1)
    for item in continuation:
        result.append(
            {
                "global_step": len(phase1) + int(item["global_step"]),
                "task": "S",
                "task_step": segment_updates_before + int(item["task_step"]),
                "qa_start": int(item["qa_start"]),
                "qa_end_exclusive": int(item["qa_end_exclusive"]),
            }
        )
    return result


def checkpoint_union_state(model: torch.nn.Module, artifact_root: Path) -> dict[str, torch.Tensor]:
    expected = expected_union_names(artifact_root)
    parameters = dict(model.named_parameters())
    if set(parameters) & expected != expected:
        raise ValueError("model is missing union checkpoint parameters")
    return {name: parameters[name].detach().cpu().contiguous() for name in sorted(expected)}


def optimizer_state_hash(optimizer: torch.optim.Optimizer | None) -> str:
    if optimizer is None:
        return "absent"
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for param_id in sorted(state["state"]):
        digest.update(str(param_id).encode())
        for key in sorted(state["state"][param_id]):
            value = state["state"][param_id][key]
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def set_train_modes(model: torch.nn.Module, task: str, selected_names: set[str]) -> None:
    """Keep frozen visual blocks deterministic while retaining LoRA dropout."""

    model.train(True)
    model.model.visual.eval()
    if task in {"F", "S"} and any(".visual.blocks." in name for name in selected_names):
        for index in range(len(model.model.visual.blocks) - 4, len(model.model.visual.blocks)):
            model.model.visual.blocks[index].train()


def parameter_group_norms(
    model: torch.nn.Module, group_names: dict[str, list[str]] | None
) -> dict[str, float]:
    if not group_names:
        return {}
    parameters = dict(model.named_parameters())
    result = {}
    for group, names in group_names.items():
        values = [parameters[name].grad.detach().float().square().sum() for name in names if parameters[name].grad is not None]
        result[group] = float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0
    return result


def parameter_group_snapshot(
    model: torch.nn.Module, group_names: dict[str, list[str]] | None
) -> dict[str, dict[str, torch.Tensor]]:
    if not group_names:
        return {}
    parameters = dict(model.named_parameters())
    return {
        group: {name: parameters[name].detach().cpu().clone() for name in names}
        for group, names in group_names.items()
    }


def parameter_group_update_norms(
    model: torch.nn.Module,
    group_names: dict[str, list[str]] | None,
    before: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    if not group_names:
        return {}
    parameters = dict(model.named_parameters())
    result = {}
    for group, names in group_names.items():
        values = [
            (parameters[name].detach().float().cpu() - before[group][name].float()).square().sum()
            for name in names
        ]
        result[group] = float(torch.sqrt(torch.stack(values).sum())) if values else 0.0
    return result
