"""Contracts and module surgery for the SEG005 Qwen temporal SFT pilot.

SEG005 deliberately reuses the EXP102 L/LM/LMV module policy and adds only
the registered LMV-all condition.  The temporal input contract is independent
of the legacy fixed-frame EXP102 micro-overfit contract: source-cache frames
are sampled at one frame per second from the request start.

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
from orena_procedure.exp102 import (
    CANDIDATES as EXP102_CANDIDATES,
)
from orena_procedure.exp102 import (
    configure_train_modes as configure_exp102_train_modes,
)
from orena_procedure.exp102 import (
    install_candidate as install_exp102_candidate,
)
from orena_procedure.exp102 import (
    merger_modules,
    vision_final_block_modules,
)

EXPERIMENT = "SEG005"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_PIXELS = 50176
MAX_NEW_TOKENS = 128
CACHE_FPS = 5.0
SAMPLE_FPS = 1.0
GRADIENT_ACCUMULATION = 8
MAX_UPDATES = 100
CHECKPOINT_STEPS = (0, 10, 25, 50, 100)
SEED = 5005
LANGUAGE_LR = 5e-5
VISION_LR = 1e-5
WEIGHT_DECAY = 0.01
VISION_RANK = 4
VISION_ALPHA = 8.0
VISION_DROPOUT = 0.05

CANDIDATES: dict[str, dict[str, Any]] = {
    "L": EXP102_CANDIDATES["L"],
    "LM": EXP102_CANDIDATES["LM"],
    "LMV-last": EXP102_CANDIDATES["LMV"],
    "LMV-all": {
        **EXP102_CANDIDATES["LMV"],
        "vision": {
            "scope": "all_blocks_all_linear_lora",
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


def nearest_cache_frame_index(timestamp_seconds: float, cache_fps: float = CACHE_FPS) -> int:
    """Map an absolute source timestamp to the nearest 5-fps cache frame."""

    if not math.isfinite(float(timestamp_seconds)) or timestamp_seconds < 0:
        raise ValueError(f"invalid source timestamp: {timestamp_seconds}")
    if cache_fps <= 0:
        raise ValueError("cache_fps must be positive")
    return math.floor(float(timestamp_seconds) * cache_fps + 0.5)


def one_fps_timestamps(
    start_seconds: float, end_seconds: float, sample_fps: float = SAMPLE_FPS
) -> list[float]:
    """Return start-anchored timestamps in the half-open request window.

    The half-open rule gives 29 samples for a 29-second request and avoids a
    duplicated endpoint.  The request bounds remain source-procedure time.
    """

    start = float(start_seconds)
    end = float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid request window: {start_seconds}, {end_seconds}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    count = max(1, math.ceil((end - start) * sample_fps - 1e-9))
    return [start + index / sample_fps for index in range(count)]


def one_fps_frame_indices(
    start_seconds: float,
    end_seconds: float,
    total_cache_frames: int,
    cache_fps: float = CACHE_FPS,
    sample_fps: float = SAMPLE_FPS,
) -> tuple[list[int], list[float]]:
    """Create the fixed-1-fps source-cache frame and timestamp manifest."""

    if total_cache_frames < 1:
        raise ValueError("total_cache_frames must be positive")
    timestamps = one_fps_timestamps(start_seconds, end_seconds, sample_fps)
    indices = [nearest_cache_frame_index(value, cache_fps) for value in timestamps]
    if any(index < 0 or index >= total_cache_frames for index in indices):
        raise ValueError("1-fps sample reaches outside the cached video")
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("1-fps cache indices are not ordered and unique")
    return indices, timestamps


def _parent_and_child(root: torch.nn.Module, relative_name: str) -> tuple[torch.nn.Module, str]:
    parts = relative_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _install_vision_block_lora(
    model: torch.nn.Module,
    block_index: int,
    rank: int = VISION_RANK,
    alpha: float = VISION_ALPHA,
    dropout: float = VISION_DROPOUT,
) -> list[str]:
    block = model.model.visual.blocks[block_index]
    modules = [
        (f"model.visual.blocks.{block_index}.{name}", module)
        for name, module in block.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    installed: list[str] = []
    for full_name, base in modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(parent, child, LoRALinear(base, rank, alpha, dropout))
        installed.append(full_name)
    return installed


def install_candidate(model: torch.nn.Module, candidate_name: str) -> dict[str, Any]:
    """Install one exact SEG005 trainable scope on an actual model tree."""

    if candidate_name not in CANDIDATES:
        raise ValueError(f"unknown SEG005 candidate: {candidate_name}")
    if candidate_name != "LMV-all":
        base_name = "LMV" if candidate_name == "LMV-last" else candidate_name
        installed = install_exp102_candidate(model, base_name)
    else:
        installed = install_exp102_candidate(model, "LMV")
        block_count = len(model.model.visual.blocks)
        extra: list[str] = []
        for index in range(block_count - 1):
            extra.extend(_install_vision_block_lora(model, index))
        installed = dict(installed)
        installed["vision_modules"] = [
            *installed["vision_modules"],
            *extra,
        ]
        installed["vision_linear_count"] = len(installed["vision_modules"])
        installed["candidate"] = candidate_name

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    vision_names = [name for name, _parameter in trainable if ".visual.blocks." in name]
    language_names = [name for name, _parameter in trainable if ".language_model." in name]
    merger_names = [
        name
        for name, _parameter in trainable
        if ".visual.merger." in name or ".visual.deepstack_merger_list." in name
    ]
    if candidate_name == "LMV-all":
        expected_blocks = len(model.model.visual.blocks)
        actual_blocks = {
            int(name.split(".visual.blocks.", 1)[1].split(".", 1)[0]) for name in vision_names
        }
        if actual_blocks != set(range(expected_blocks)):
            raise RuntimeError(f"LMV-all did not install every vision block: {actual_blocks}")
    installed.update(
        {
            "candidate": candidate_name,
            "trainable_parameter_names": [name for name, _parameter in trainable],
            "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
            "trainable_parameter_count_by_scope": {
                "language_lora": int(sum(parameter.numel() for name, parameter in trainable if name in language_names)),
                "merger_full": int(sum(parameter.numel() for name, parameter in trainable if name in merger_names)),
                "vision_lora": int(sum(parameter.numel() for name, parameter in trainable if name in vision_names)),
            },
            "vision_block_indices": sorted(
                {
                    int(name.split(".visual.blocks.", 1)[1].split(".", 1)[0])
                    for name in vision_names
                }
            ),
        }
    )
    return installed


def module_tree_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize the actual model paths used by all four conditions."""

    final_index, final_modules = vision_final_block_modules(model)
    vision_blocks = []
    for index, block in enumerate(model.model.visual.blocks):
        linears = [
            {
                "name": f"model.visual.blocks.{index}.{name}",
                "shape": list(module.weight.shape),
            }
            for name, module in block.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        vision_blocks.append({"index": index, "linear_modules": linears})
    return {
        "language_model_linear_count": len(
            [module for module in model.model.language_model.modules() if isinstance(module, torch.nn.Linear)]
        ),
        "merger_module_count": len(merger_modules(model)),
        "vision_block_count": len(model.model.visual.blocks),
        "final_block_index": final_index,
        "final_block_linear_count": len(final_modules),
        "vision_blocks": vision_blocks,
    }


def configure_train_modes(model: torch.nn.Module, candidate_name: str, training: bool) -> None:
    """Keep frozen vision blocks in eval mode while enabling LoRA dropout."""

    if candidate_name != "LMV-all":
        configure_exp102_train_modes(model, "LMV" if candidate_name == "LMV-last" else candidate_name, training)
        return
    model.train(training)
    model.model.visual.eval()
    if training:
        for block in model.model.visual.blocks:
            block.train()


def candidate_parameter_norms(model: torch.nn.Module) -> dict[str, float]:
    trainable = [parameter.detach().float().square().sum() for parameter in model.parameters() if parameter.requires_grad]
    lora = [
        parameter.detach().float().square().sum()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    return {
        "trainable_parameter_l2": float(torch.sqrt(torch.stack(trainable).sum()).cpu()) if trainable else 0.0,
        "lora_parameter_l2": float(torch.sqrt(torch.stack(lora).sum()).cpu()) if lora else 0.0,
    }
