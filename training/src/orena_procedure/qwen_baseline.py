"""Qwen3-VL baseline utilities with reproducible temporal sampling."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PyNvVideoCodec as nvc
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

from orena_procedure.generation_contract import DEFAULT_MAX_NEW_TOKENS
from orena_procedure.multiple_choice import (
    extract_multiple_choice_spec,
    multiple_choice_prompt_instruction,
)


@dataclass(frozen=True)
class BaselineConfig:
    model_id: str
    model_revision: str
    device: str
    dtype: str
    num_frames: int
    max_pixels: int
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    sampling_strategy: str = "uniform"
    timeline_mode: str = "source_video"
    fo_class_guidance: bool = False
    retrieval_model_id: str = "google/siglip-base-patch16-224"
    retrieval_candidates: int = 32
    decoder_backend: str = "decord"
    decoder_fps: float | None = None


TIMESTAMP_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)\b")
TIME_LOCALIZATION_PATTERN = re.compile(
    r"\b(?:when|what time|at what time|time point)\b.*\b(?:hh:mm:ss|timestamp)\b", re.IGNORECASE
)
EARLY_EVENT_PATTERN = re.compile(r"\b(first|inserted|created)\b", re.IGNORECASE)
LATE_EVENT_PATTERN = re.compile(r"\b(last|retriev\w*)\b", re.IGNORECASE)


def uniform_indices(start_frame: int, end_frame: int, num_frames: int) -> list[int]:
    """Return ordered, unique indices spanning ``[start_frame, end_frame)``."""
    if end_frame <= start_frame:
        raise ValueError("Video window contains no frames.")
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    available = end_frame - start_frame
    count = min(num_frames, available)
    return np.linspace(start_frame, end_frame - 1, num=count, dtype=int).tolist()


def question_timestamps_seconds(question: str) -> list[int]:
    """Extract ordered, unique HH:MM:SS references from a question."""
    timestamps: list[int] = []
    for hour, minute, second in TIMESTAMP_PATTERN.findall(question):
        value = int(hour) * 3600 + int(minute) * 60 + int(second)
        if value not in timestamps:
            timestamps.append(value)
    return timestamps


def is_time_localization_question(question: str) -> bool:
    """Identify questions requesting an event timestamp rather than a duration."""
    return bool(TIME_LOCALIZATION_PATTERN.search(question))


def parse_timestamp_seconds(text: str) -> int | None:
    """Return the first valid HH:MM:SS value in a model response."""
    match = TIMESTAMP_PATTERN.search(text)
    if match is None:
        return None
    hour, minute, second = (int(value) for value in match.groups())
    return hour * 3600 + minute * 60 + second


def deterministic_random_local_center(
    qid: str, start_seconds: float, end_seconds: float, excluded_center_seconds: float
) -> float:
    """Return a reproducible local-control center away from the oracle event."""
    if end_seconds <= start_seconds:
        raise ValueError("Random-local sampling requires a non-empty request interval.")
    fraction = int.from_bytes(hashlib.sha256(qid.encode("utf-8")).digest()[:8], "big") / 2**64
    candidate = start_seconds + fraction * (end_seconds - start_seconds)
    if end_seconds - start_seconds > 120.0 and abs(candidate - excluded_center_seconds) < 60.0:
        candidate = start_seconds + ((fraction + 0.5) % 1.0) * (end_seconds - start_seconds)
    return float(candidate)


def directional_time_window_seconds(
    question: str, start_seconds: float, end_seconds: float
) -> tuple[float, float] | None:
    """Return a directional search window for an anchored insertion/retrieval event."""
    timestamps = question_timestamps_seconds(question)
    if len(timestamps) != 1:
        return None
    anchor = timestamps[0]
    if re.search(r"retriev\w*", question, re.IGNORECASE):
        lower, upper = anchor, end_seconds
    elif re.search(r"insert\w*", question, re.IGNORECASE):
        lower, upper = start_seconds, anchor
    else:
        return None
    lower = max(start_seconds, min(end_seconds, lower))
    upper = max(start_seconds, min(end_seconds, upper))
    if upper <= lower:
        return None
    return ((lower + upper) / 2, (upper - lower) / 2)


def timestamp_aware_indices(
    start_frame: int,
    end_frame: int,
    num_frames: int,
    timestamp_seconds: list[int],
    fps: float,
    source_start_seconds: float,
) -> list[int]:
    """Mix global anchors with frames across the question's explicit time range.

    The output always contains at most ``num_frames`` ordered, unique frame indices.
    When the question has no usable timestamp, it is exactly uniform sampling.
    """
    if not timestamp_seconds:
        return uniform_indices(start_frame, end_frame, num_frames)
    if end_frame <= start_frame:
        raise ValueError("Video window contains no frames.")
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")

    global_count = num_frames // 2
    local_count = num_frames - global_count
    global_indices = uniform_indices(start_frame, end_frame, global_count)
    relative_seconds = [timestamp - source_start_seconds for timestamp in timestamp_seconds]
    local_start = max(start_frame, min(end_frame - 1, int(min(relative_seconds) * fps)))
    local_end = max(local_start + 1, min(end_frame, int(max(relative_seconds) * fps) + 1))
    # A point timestamp needs a small temporal neighborhood rather than one repeated frame.
    if local_end - local_start <= 1:
        radius_frames = max(1, round(2 * fps))
        local_start = max(start_frame, local_start - radius_frames)
        local_end = min(end_frame, local_start + 2 * radius_frames + 1)
    local_indices = uniform_indices(local_start, local_end, local_count)
    indices = sorted(set(global_indices + local_indices))
    if len(indices) < min(num_frames, end_frame - start_frame):
        for index in uniform_indices(start_frame, end_frame, num_frames):
            if index not in indices:
                indices.append(index)
            if len(indices) == min(num_frames, end_frame - start_frame):
                break
    return sorted(indices)


def nested_timestamp_aware_indices(
    start_frame: int,
    end_frame: int,
    num_frames: int,
    timestamp_seconds: list[int],
    fps: float,
    source_start_seconds: float,
    anchor_frames: int = 64,
) -> list[int]:
    """Add temporal evidence while preserving the existing timestamp-aware anchors.

    This supports a controlled 64-versus-96 comparison: every 64-frame
    timestamp-aware anchor remains present, and only the extra frames differ.
    """
    if end_frame <= start_frame:
        raise ValueError("Video window contains no frames.")
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    target = min(num_frames, end_frame - start_frame)
    anchor_count = min(target, anchor_frames)
    anchors = timestamp_aware_indices(
        start_frame,
        end_frame,
        anchor_count,
        timestamp_seconds,
        fps,
        source_start_seconds,
    )
    if target == len(anchors):
        return anchors

    candidate_count = min(end_frame - start_frame, max(target * 2, anchor_frames * 2))
    candidates = timestamp_aware_indices(
        start_frame,
        end_frame,
        candidate_count,
        timestamp_seconds,
        fps,
        source_start_seconds,
    )
    extra_pool = [index for index in candidates if index not in set(anchors)]
    extra_count = target - len(anchors)
    extras = (
        [
            extra_pool[index]
            for index in np.linspace(0, len(extra_pool) - 1, num=extra_count, dtype=int).tolist()
        ]
        if extra_pool
        else []
    )
    indices = sorted(set(anchors + extras))
    if len(indices) < target:
        for index in uniform_indices(start_frame, end_frame, target):
            if index not in indices:
                indices.append(index)
            if len(indices) == target:
                break
    return sorted(indices)


def query_aware_route(question: str) -> str:
    """Choose a temporal rule from explicit question wording without a learned retriever."""
    if question_timestamps_seconds(question):
        return "explicit_timestamp"
    if LATE_EVENT_PATTERN.search(question):
        return "late_event"
    if EARLY_EVENT_PATTERN.search(question):
        return "early_event"
    return "uniform"


def query_aware_indices(
    start_frame: int,
    end_frame: int,
    num_frames: int,
    question: str,
    fps: float,
    source_start_seconds: float,
) -> list[int]:
    """Mix global anchors with a query-selected early or late temporal region."""
    route = query_aware_route(question)
    if route == "explicit_timestamp":
        return timestamp_aware_indices(
            start_frame,
            end_frame,
            num_frames,
            question_timestamps_seconds(question),
            fps,
            source_start_seconds,
        )
    if route == "uniform":
        return uniform_indices(start_frame, end_frame, num_frames)
    global_count = num_frames // 2
    local_count = num_frames - global_count
    available = end_frame - start_frame
    region_size = max(local_count, round(available * 0.35))
    if route == "early_event":
        local_start, local_end = start_frame, min(end_frame, start_frame + region_size)
    else:
        local_start, local_end = max(start_frame, end_frame - region_size), end_frame
    indices = sorted(
        set(
            uniform_indices(start_frame, end_frame, global_count)
            + uniform_indices(local_start, local_end, local_count)
        )
    )
    for index in uniform_indices(start_frame, end_frame, num_frames):
        if len(indices) >= min(num_frames, available):
            break
        if index not in indices:
            indices.append(index)
    return sorted(indices)


def concise_answer(text: str) -> str:
    """Apply only submission-safe whitespace and length normalization."""
    return " ".join(text.strip().split())[:300]


FO_CLASS_NAMES = (
    "Sponge",
    "Clip",
    "Specimen Bag",
    "Silicone Loop",
    "External Drain",
    "Needle",
    "Gallstone",
    "Specimen",
    "Mesh",
    "Absorbable Hemostatic Agent",
)


class ModuleTimer:
    """Collect CUDA time for visual, prefill, and decode module calls."""

    def __init__(self, model: Qwen3VLForConditionalGeneration) -> None:
        self._device = next(model.parameters()).device
        self._stacks: dict[str, list[tuple[torch.cuda.Event, str]]] = {
            "vision_encoder": [],
            "language_model": [],
        }
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []
        self._handles = [
            model.model.visual.register_forward_pre_hook(
                self._pre_hook("vision_encoder"), with_kwargs=True
            ),
            model.model.visual.register_forward_hook(self._post_hook("vision_encoder")),
            model.model.language_model.register_forward_pre_hook(
                self._pre_hook("language_model"), with_kwargs=True
            ),
            model.model.language_model.register_forward_hook(self._post_hook("language_model")),
        ]

    def _pre_hook(self, module_name: str):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            category = module_name
            if module_name == "language_model":
                tensor = kwargs.get("inputs_embeds")
                if tensor is None:
                    tensor = kwargs.get("input_ids")
                sequence_length = tensor.shape[1] if tensor is not None and tensor.ndim > 1 else 1
                category = "llm_prefill" if sequence_length > 1 else "generation"
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._stacks[module_name].append((start, category))

        return hook

    def _post_hook(self, module_name: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            start, category = self._stacks[module_name].pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events.append((start, end, category))

        return hook

    def reset(self) -> None:
        self._events.clear()
        for stack in self._stacks.values():
            stack.clear()

    def results_ms(self) -> dict[str, float]:
        torch.cuda.synchronize(self._device)
        result = {"vision_encoder": 0.0, "llm_prefill": 0.0, "generation": 0.0}
        for start, end, category in self._events:
            result[category] += start.elapsed_time(end)
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def procedure_prompt(
    question: str,
    procedure_type: str,
    timestamps: list[float],
    images: list[Image.Image],
    answer_format: str | None,
    fo_class_guidance: bool = True,
) -> list[dict[str, Any]]:
    """Build the shared EXP092/EXP094 user prompt."""

    if len(timestamps) != len(images):
        raise ValueError("prompt timestamp/image count mismatch")
    content: list[dict[str, Any]] = []
    for timestamp, image in zip(timestamps, images, strict=True):
        content.extend(
            [
                {"type": "text", "text": f"Frame timestamp: {timestamp:0.1f} seconds."},
                {"type": "image", "image": image},
            ]
        )
    format_instruction = ""
    option_instruction = multiple_choice_prompt_instruction(question)
    if answer_format == "multiple_choice" and option_instruction is None:
        # Do not silently fall back when the official format says that this is
        # multiple-choice but the question template cannot be parsed.
        extract_multiple_choice_spec(question)
    if option_instruction is not None:
        format_instruction = f" {option_instruction}"
    if fo_class_guidance and answer_format == "fo_class":
        format_instruction = (
            " For foreign-object class answers, use only comma-separated names from: "
            f"{', '.join(FO_CLASS_NAMES)}; use none only when no class applies."
        )
    content.append(
        {
            "type": "text",
            "text": (
                "You are assisting with laparoscopic surgery. Use only the sampled frames as evidence. "
                f"Procedure type: {procedure_type}. Answer the following question with only a concise answer, "
                "without explanation. Use yes/no for binary questions, a non-negative integer for counts, "
                "and HH:MM:SS for requested timestamps."
                f"{format_instruction}\n"
                f"Question: {question}"
            ),
        }
    )
    return [{"role": "user", "content": content}]


class QwenUniformBaseline:
    """Qwen3-VL inference over uniformly sampled frames from a request window."""

    def __init__(self, config: BaselineConfig) -> None:
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.config = config
        dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            dtype=dtype,
            device_map={"": str(self.device)},
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            max_pixels=config.max_pixels,
        )
        self.retrieval_model = None
        self.retrieval_processor = None
        if config.sampling_strategy == "semantic_retrieval":
            self.retrieval_processor = AutoProcessor.from_pretrained(config.retrieval_model_id)
            self.retrieval_model = (
                AutoModel.from_pretrained(config.retrieval_model_id, dtype=dtype)
                .to(self.device)
                .eval()
            )
        self.timer = ModuleTimer(self.model)

    def _select_frame_indices(
        self,
        total_frames: int,
        fps: float,
        start_seconds: float,
        end_seconds: float,
        question: str,
        sampling_override: str | None,
        local_center_seconds: float | None,
        local_radius_seconds: float | None,
        num_frames: int,
        reader: Any | None = None,
    ) -> tuple[list[int], list[float]]:
        if self.config.timeline_mode == "source_video":
            window_start = max(0, min(total_frames - 1, int(np.floor(start_seconds * fps))))
            window_end = max(window_start + 1, min(total_frames, int(np.ceil(end_seconds * fps))))
            source_start_seconds = 0.0
            timestamps_offset = 0.0
        elif self.config.timeline_mode == "trimmed_clip":
            window_start = 0
            window_end = total_frames
            source_start_seconds = start_seconds
            timestamps_offset = start_seconds
        else:
            raise ValueError(f"Unsupported timeline mode: {self.config.timeline_mode}")

        sampling_strategy = sampling_override or self.config.sampling_strategy
        if local_center_seconds is not None:
            if local_radius_seconds is None or local_radius_seconds <= 0:
                raise ValueError("A positive local_radius_seconds is required for local sampling.")
            center_frame = min(
                window_end - 1,
                max(window_start, round((local_center_seconds - timestamps_offset) * fps)),
            )
            radius_frames = max(1, round(local_radius_seconds * fps))
            local_start = max(window_start, center_frame - radius_frames)
            local_end = min(window_end, center_frame + radius_frames + 1)
            indices = uniform_indices(local_start, local_end, num_frames)
        elif sampling_strategy == "uniform":
            indices = uniform_indices(window_start, window_end, num_frames)
        elif sampling_strategy == "timestamp_aware":
            indices = timestamp_aware_indices(
                window_start,
                window_end,
                num_frames,
                question_timestamps_seconds(question),
                fps,
                source_start_seconds,
            )
        elif sampling_strategy == "nested_timestamp_aware":
            indices = nested_timestamp_aware_indices(
                window_start,
                window_end,
                num_frames,
                question_timestamps_seconds(question),
                fps,
                source_start_seconds,
            )
        elif sampling_strategy == "query_aware":
            indices = query_aware_indices(
                window_start,
                window_end,
                num_frames,
                question,
                fps,
                source_start_seconds,
            )
        elif sampling_strategy == "semantic_retrieval":
            if reader is None:
                raise ValueError("semantic_retrieval requires a decoder with reader.get_batch")
            indices = self._semantic_indices(reader, window_start, window_end, question, num_frames)
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")
        return indices, [index / fps + timestamps_offset for index in indices]

    def _semantic_indices(
        self, reader, start_frame: int, end_frame: int, question: str, num_frames: int
    ) -> list[int]:
        assert self.retrieval_model is not None and self.retrieval_processor is not None
        candidates = uniform_indices(start_frame, end_frame, self.config.retrieval_candidates)
        candidate_images = [
            Image.fromarray(frame) for frame in reader.get_batch(candidates).asnumpy()
        ]
        inputs = self.retrieval_processor(
            text=[question] * len(candidate_images),
            images=candidate_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.retrieval_model(**inputs).logits_per_image
        scores = logits.diag().float().cpu().tolist()
        anchors = uniform_indices(start_frame, end_frame, num_frames // 2)
        ranked = [index for _, index in sorted(zip(scores, candidates, strict=True), reverse=True)]
        selected = list(anchors)
        for index in ranked:
            if index not in selected:
                selected.append(index)
            if len(selected) >= num_frames:
                break
        return sorted(selected)

    def _frames(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        question: str,
        sampling_override: str | None = None,
        local_center_seconds: float | None = None,
        local_radius_seconds: float | None = None,
        num_frames_override: int | None = None,
        decoder_fps_override: float | None = None,
    ) -> tuple[list[Image.Image], list[float], dict[str, float]]:
        if self.config.decoder_backend == "pynvcodec":
            return self._frames_pynvcodec(
                video_path,
                start_seconds,
                end_seconds,
                question,
                sampling_override,
                local_center_seconds,
                local_radius_seconds,
                num_frames_override,
                decoder_fps_override,
            )
        if self.config.decoder_backend != "decord":
            raise ValueError(f"Unsupported decoder backend: {self.config.decoder_backend}")

        # Import after model initialization.  In the development image,
        # importing decord before CUDA initialization can make Qwen's CUDA
        # allocator fail to initialize.
        import decord

        timings: dict[str, float] = {}
        start = time.perf_counter()
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        timings["video_loading"] = time.perf_counter() - start
        fps = decoder_fps_override or self.config.decoder_fps or float(reader.get_avg_fps())
        total_frames = len(reader)
        num_frames = self.config.num_frames if num_frames_override is None else num_frames_override
        if num_frames < 1:
            raise ValueError("num_frames_override must be positive")

        start = time.perf_counter()
        indices, timestamps = self._select_frame_indices(
            total_frames,
            fps,
            start_seconds,
            end_seconds,
            question,
            sampling_override,
            local_center_seconds,
            local_radius_seconds,
            num_frames,
            reader,
        )
        timings["sampling"] = time.perf_counter() - start

        start = time.perf_counter()
        frames = reader.get_batch(indices).asnumpy()
        del reader
        timings["frame_decoding"] = time.perf_counter() - start

        start = time.perf_counter()
        images = [Image.fromarray(frame) for frame in frames]
        timings["image_conversion"] = time.perf_counter() - start
        return images, timestamps, timings

    def _frames_pynvcodec(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        question: str,
        sampling_override: str | None = None,
        local_center_seconds: float | None = None,
        local_radius_seconds: float | None = None,
        num_frames_override: int | None = None,
        decoder_fps_override: float | None = None,
    ) -> tuple[list[Image.Image], list[float], dict[str, float]]:
        """Decode normalized 5-fps clips with NVDEC and preserve PIL input semantics."""
        if self.device.type != "cuda":
            raise ValueError("PyNvVideoCodec requires a CUDA device")
        if self.config.timeline_mode not in {"source_video", "trimmed_clip"}:
            raise ValueError(f"Unsupported timeline mode: {self.config.timeline_mode}")
        if self.config.sampling_strategy == "semantic_retrieval":
            raise ValueError("semantic_retrieval is not supported with PyNvVideoCodec")

        timings: dict[str, float] = {}
        gpu_index = (
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        )
        start = time.perf_counter()
        decoder = nvc.SimpleDecoder(
            str(video_path),
            gpu_id=gpu_index,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
            bWaitForSessionWarmUp=True,
        )
        timings["video_loading"] = time.perf_counter() - start

        # Existing procedure experiments use a normalized 5-fps cache.  A
        # caller may override this for native-resolution frame diagnostics.
        fps = decoder_fps_override or self.config.decoder_fps or 5.0
        total_frames = len(decoder)
        num_frames = self.config.num_frames if num_frames_override is None else num_frames_override
        if num_frames < 1:
            del decoder
            raise ValueError("num_frames_override must be positive")

        start = time.perf_counter()
        indices, timestamps = self._select_frame_indices(
            total_frames,
            fps,
            start_seconds,
            end_seconds,
            question,
            sampling_override,
            local_center_seconds,
            local_radius_seconds,
            num_frames,
        )
        timings["sampling"] = time.perf_counter() - start

        start = time.perf_counter()
        surfaces = decoder.get_batch_frames_by_index(indices)
        # The first implementation intentionally transfers RGB to CPU and
        # creates PIL images so the processor/input contract remains unchanged.
        arrays = [torch.utils.dlpack.from_dlpack(surface).cpu().numpy() for surface in surfaces]
        torch.cuda.synchronize(self.device)
        timings["frame_decoding"] = time.perf_counter() - start
        del surfaces, decoder

        start = time.perf_counter()
        images = [Image.fromarray(frame) for frame in arrays]
        timings["image_conversion"] = time.perf_counter() - start
        return images, timestamps, timings

    def _prompt(
        self,
        question: str,
        procedure_type: str,
        timestamps: list[float],
        images: list[Image.Image],
        answer_format: str | None,
    ) -> list[dict[str, Any]]:
        return procedure_prompt(
            question,
            procedure_type,
            timestamps,
            images,
            answer_format,
            self.config.fo_class_guidance,
        )

    def answer(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        procedure_type: str,
        question: str,
        answer_format: str | None = None,
        sampling_override: str | None = None,
        local_center_seconds: float | None = None,
        local_radius_seconds: float | None = None,
        num_frames_override: int | None = None,
        decoder_fps_override: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        wall_start = time.perf_counter()
        images, timestamps, timings = self._frames(
            video_path,
            start_seconds,
            end_seconds,
            question,
            sampling_override,
            local_center_seconds,
            local_radius_seconds,
            num_frames_override,
            decoder_fps_override,
        )

        return self.answer_from_images(
            images,
            timestamps,
            procedure_type,
            question,
            answer_format,
            initial_timings=timings,
            wall_start=wall_start,
            sampling_strategy=sampling_override or self.config.sampling_strategy,
            local_center_seconds=local_center_seconds,
            local_radius_seconds=local_radius_seconds,
        )

    def answer_from_images(
        self,
        images: list[Image.Image],
        timestamps: list[float],
        procedure_type: str,
        question: str,
        answer_format: str | None = None,
        initial_timings: dict[str, float] | None = None,
        wall_start: float | None = None,
        sampling_strategy: str | None = None,
        local_center_seconds: float | None = None,
        local_radius_seconds: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Answer using preselected images while preserving the normal VLM path."""

        if not images or len(images) != len(timestamps):
            raise ValueError("answer_from_images requires matching non-empty images and timestamps")
        wall_start = time.perf_counter() if wall_start is None else wall_start
        timings = dict(initial_timings or {})

        start = time.perf_counter()
        messages = self._prompt(question, procedure_type, timestamps, images, answer_format)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        timings["preprocessing"] = time.perf_counter() - start

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.timer.reset()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False
            )
        timings.update({name: value / 1000.0 for name, value in self.timer.results_ms().items()})

        start = time.perf_counter()
        new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
        answer = concise_answer(
            self.processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        )
        timings["postprocessing"] = time.perf_counter() - start
        timings["total"] = time.perf_counter() - wall_start

        merge_size = self.model.config.vision_config.spatial_merge_size
        visual_tokens = int(inputs.image_grid_thw.prod(dim=1).sum().item() // (merge_size**2))
        patch_size = int(getattr(self.model.config.vision_config, "patch_size", 14))
        grid = inputs.image_grid_thw[0].tolist()
        processor_width = int(grid[2] * patch_size)
        processor_height = int(grid[1] * patch_size)
        effective_sampling = sampling_strategy or self.config.sampling_strategy
        metadata = {
            "timings_seconds": timings,
            "sampled_timestamps_seconds": timestamps,
            "question_timestamps_seconds": question_timestamps_seconds(question),
            "sampling_strategy": effective_sampling,
            "local_center_seconds": local_center_seconds,
            "local_radius_seconds": local_radius_seconds,
            "sampling_route": query_aware_route(question)
            if effective_sampling == "query_aware"
            else effective_sampling,
            "timeline_mode": self.config.timeline_mode,
            "sampled_frames": len(images),
            "source_frame_resolution_width_height": list(images[0].size),
            "image_grid_thw": inputs.image_grid_thw.cpu().tolist(),
            "processor_image_resolution_width_height": [processor_width, processor_height],
            "processor_image_pixels": processor_width * processor_height,
            "processor_patch_size": patch_size,
            "visual_tokens": visual_tokens,
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(new_tokens.shape[1]),
            "max_new_tokens": self.config.max_new_tokens,
            "generation_cap_reached": int(new_tokens.shape[1]) >= self.config.max_new_tokens,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }
        return answer, metadata

    def close(self) -> None:
        self.timer.close()
