"""Self-contained SEG010 LMV specialist installation and switching runtime.

This module intentionally contains only the small adapter surgery used by the
SEG010 trainer.  The submission image does not contain the development
``src/`` tree, so importing the training package here would make the runtime
non-reproducible.  Specialist banks are ordinary tensors rather than
parameters and are never included in the model state dict.
"""

from __future__ import annotations

import gc
import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file


EXPECTED_TENSOR_COUNT = 560
EXPECTED_PARAMETER_COUNT = 205_284_608
LANGUAGE_RANK = 16
LANGUAGE_ALPHA = 32.0
LANGUAGE_DROPOUT = 0.05
VISION_RANK = 16
VISION_ALPHA = 32.0
VISION_DROPOUT = 0.05


class SpecialistRuntimeError(RuntimeError):
    """Raised when a specialist cannot be safely installed or activated."""


class SpecialistActivationError(SpecialistRuntimeError):
    """A state copy failed; the caller must terminate without another forward."""


class LoRALinear(torch.nn.Module):
    """The exact dependency-free LoRA wrapper used by SEG010 training."""

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
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_B(self.lora_A(self.dropout(value))) * self.scaling


def _parent_and_child(root: torch.nn.Module, relative_name: str) -> tuple[torch.nn.Module, str]:
    parts = relative_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _module_path(prefix: str, relative_name: str) -> str:
    return f"{prefix}.{relative_name}" if relative_name else prefix


def language_linear_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    return [
        (_module_path("model.language_model", name), module)
        for name, module in model.model.language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def vision_last_block_modules(
    model: torch.nn.Module, block_count: int = 4
) -> tuple[list[int], list[tuple[str, torch.nn.Linear]]]:
    blocks = model.model.visual.blocks
    if len(blocks) < block_count:
        raise SpecialistRuntimeError(f"model has {len(blocks)} visual blocks; expected at least {block_count}")
    indices = list(range(len(blocks) - block_count, len(blocks)))
    modules: list[tuple[str, torch.nn.Linear]] = []
    for index in indices:
        modules.extend(
            (_module_path("model.visual", f"blocks.{index}.{name}"), module)
            for name, module in blocks[index].named_modules()
            if isinstance(module, torch.nn.Linear)
        )
    return indices, modules


def union_parameter_names(model: torch.nn.Module) -> set[str]:
    return {
        name
        for name, _parameter in model.named_parameters()
        if ("lora_" in name and (".language_model." in name or ".visual.blocks." in name))
        or ".visual.merger." in name
        or ".visual.deepstack_merger_list." in name
    }


def install_union(model: torch.nn.Module) -> dict[str, Any]:
    """Install the exact SEG010 L+V union on a fresh Qwen model."""

    was_training = model.training
    if any("lora_" in name for name, _parameter in model.named_parameters()):
        raise SpecialistRuntimeError("union installation requires a model without existing LoRA")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    language_modules = language_linear_modules(model)
    for full_name, base in language_modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(
            parent,
            child,
            LoRALinear(base, rank=LANGUAGE_RANK, alpha=LANGUAGE_ALPHA, dropout=LANGUAGE_DROPOUT),
        )

    vision_indices, vision_modules = vision_last_block_modules(model)
    for full_name, base in vision_modules:
        parent, child = _parent_and_child(model.model, full_name.removeprefix("model."))
        setattr(
            parent,
            child,
            LoRALinear(base, rank=VISION_RANK, alpha=VISION_ALPHA, dropout=VISION_DROPOUT),
        )

    names = union_parameter_names(model)
    if len(names) != EXPECTED_TENSOR_COUNT:
        raise SpecialistRuntimeError(f"unexpected union tensor count: {len(names)}")
    count = sum(parameter.numel() for name, parameter in model.named_parameters() if name in names)
    if count != EXPECTED_PARAMETER_COUNT:
        raise SpecialistRuntimeError(f"unexpected union parameter count: {count}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    # Modules inserted after ``model.eval()`` default to training mode.  Keep
    # the caller's mode for this generic installer; the inference runtime
    # below additionally enforces eval mode for production.
    model.train(was_training)
    return {
        "language_linear_count": len(language_modules),
        "vision_linear_count": len(vision_modules),
        "vision_block_indices": vision_indices,
        "union_parameter_names": sorted(names),
        "union_parameter_count": count,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_validate_state(
    path: Path, parameters: dict[str, torch.nn.Parameter], expected_names: set[str]
) -> dict[str, torch.Tensor]:
    state = load_file(str(path), device="cpu")
    if set(state) != expected_names:
        raise SpecialistRuntimeError(
            f"specialist state key mismatch for {path}: "
            f"missing={sorted(expected_names - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected_names)[:5]}"
        )
    for name, value in state.items():
        parameter = parameters[name]
        if tuple(value.shape) != tuple(parameter.shape):
            raise SpecialistRuntimeError(
                f"specialist state shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}"
            )
        if value.dtype != parameter.dtype:
            raise SpecialistRuntimeError(
                f"specialist state dtype mismatch for {name}: {value.dtype} != {parameter.dtype}"
            )
        if not value.is_floating_point() or not bool(value.isfinite().all()):
            raise SpecialistRuntimeError(f"specialist state contains non-finite/non-floating tensor: {name}")
    return {name: state[name].contiguous() for name in sorted(expected_names)}


class SpecialistRuntime:
    """One shared model with two ordinary tensor banks and one active state."""

    SELECTOR_ALIASES = {
        "GENERAL": "general",
        "GENERAL_FINAL": "general",
        "general": "general",
        "AGGREGATION": "agg",
        "AGG": "agg",
        "AGG_FINAL": "agg",
        "agg": "agg",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        bindings: dict[str, Path],
    ) -> None:
        if set(bindings) != {"general", "agg"}:
            raise SpecialistRuntimeError(f"both specialist bindings are required: {sorted(bindings)}")
        self.model = model
        self.device = device
        self._lock = threading.RLock()
        installation = install_union(model)
        model.eval()
        self._expected_names = set(installation["union_parameter_names"])
        self._parameters = {
            name: parameter for name, parameter in model.named_parameters() if name in self._expected_names
        }
        self._banks: dict[str, dict[str, torch.Tensor]] = {}
        self.source_paths = {name: Path(path).resolve() for name, path in bindings.items()}
        self.source_hashes = {name: sha256_file(path) for name, path in self.source_paths.items()}
        for name in ("general", "agg"):
            cpu_state = _load_and_validate_state(self.source_paths[name], self._parameters, self._expected_names)
            self._banks[name] = {
                key: value.to(device=device, dtype=self._parameters[key].dtype)
                for key, value in cpu_state.items()
            }
            del cpu_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.active_specialist: str | None = None
        self.installation = {
            **installation,
            "specialist_runtime": self,
            "specialist_names": ["general", "agg"],
            "specialist_bank_device": str(device),
            "specialist_source_hashes": dict(self.source_hashes),
            "active_specialist": None,
        }

    @property
    def banks(self) -> dict[str, dict[str, torch.Tensor]]:
        """Expose read-only-by-convention banks for diagnostics, not production mutation."""

        return self._banks

    @property
    def active_parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._parameters))

    def _normalize_selector(self, selector: str) -> str:
        try:
            return self.SELECTOR_ALIASES[str(selector)]
        except KeyError as exc:
            raise SpecialistRuntimeError(f"invalid specialist selector: {selector!r}") from exc

    def activate_specialist(self, selector: str) -> bool:
        """Copy a complete validated bank; activation failure is fatal to the caller."""

        name = self._normalize_selector(selector)
        with self._lock:
            if self.active_specialist == name:
                return False
            try:
                with torch.no_grad():
                    for key in self.active_parameter_names:
                        self._parameters[key].copy_(self._banks[name][key])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            except BaseException as exc:
                # A copy can have partially completed.  Never attempt another
                # forward or a fallback after this point; the caller must exit.
                self.active_specialist = None
                self.installation["active_specialist"] = None
                raise SpecialistActivationError(
                    f"specialist activation failed for {name}; process must terminate"
                ) from exc
            self.active_specialist = name
            self.installation["active_specialist"] = name
            return True

    @contextmanager
    def inference_guard(self) -> Iterator[None]:
        """Prevent a concurrent switch from overlapping a model forward."""

        with self._lock:
            yield

    def active_state_matches_bank(self, selector: str) -> bool:
        """Diagnostic exact comparison; not called by the production switch."""

        name = self._normalize_selector(selector)
        return all(torch.equal(self._parameters[key], self._banks[name][key]) for key in self.active_parameter_names)
