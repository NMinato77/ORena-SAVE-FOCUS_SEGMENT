"""Offline Qwen3-VL SEGMENT submission using the official template interface.

The container receives one batch at ``/input`` and writes one response for
every request in ``request.json`` to ``/output/answer.json``. The video files
are already question-specific trimmed clips; their first decoded frame is
clip-relative time zero.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from focus import Request, Response, load_requests, save_items
from PIL import Image
from resources.candidate_policy import SEGMENT_EVIDENCE_INSTRUCTION
from resources.format_reducer import ReducedAnswer, format_instruction, reduce_answer
from resources.q1_router import Q1_ARTIFACT_PATH, OperationalRoute, RouterRuntime, load_q1_router
from resources.q2_siglip_router import H2Router, load_h2_router
from resources.routing import Routing, is_static_d_hires
from resources.routing import route_question as q0_route_question
from resources.specialist_state import require_final_specialist_bindings
from resources.specialist_runtime import SpecialistActivationError, SpecialistRuntime
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCES_PATH = Path(__file__).parent / "resources"
MODEL_PATH = RESOURCES_PATH / "qwen3-vl-8b"
VIDEO_DIR = INPUT_PATH / "plain"

MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
NORMAL_MAX_PIXELS = 50176
R1_MAX_PIXELS = 100352
STATIC_HIRES_MAX_PIXELS = 200704
STATIC_HIRES_MAX_FRAMES = 64
MAX_NEW_TOKENS = 128
SOURCE_FPS = 5.0
SHORT_TARGET_FPS = 2.0
MAX_LONG_FRAMES = 240
FINAL_CANDIDATE_CONFIG_PATH = RESOURCES_PATH / "final_candidate_config.json"


def log_environment(device: torch.device) -> None:
    log.info("--- Environment ---")
    log.info("torch=%s CUDA=%s", torch.__version__, torch.version.cuda)
    flags = torch._C._cuda_getArchFlags() or ""
    log.info("compiled CUDA architectures=%s", flags)
    if device.type != "cuda":
        log.warning("No CUDA device visible; CPU inference is a compatibility fallback only")
        return
    capability = torch.cuda.get_device_capability(device)
    free, total = torch.cuda.mem_get_info(device)
    log.info("GPU=%s capability=sm_%d%d", torch.cuda.get_device_name(device), *capability)
    log.info("VRAM=%.1f GiB free of %.1f GiB", free / 1024**3, total / 1024**3)


def clip_path_for(req: Request) -> Path:
    """Resolve the plain, already-trimmed clip for one request."""

    return VIDEO_DIR / f"{req.qID}.mp4"


def _source_frame_count(total_frames: int, fps: float, request_duration: float) -> int:
    """Count source-grid frames in the request's half-open clip interval."""

    if total_frames < 1 or fps <= 0 or request_duration <= 0:
        raise ValueError("invalid clip frame or duration metadata")
    count = max(1, math.ceil(request_duration * fps - 1e-9))
    return min(total_frames, count)


def nearest_source_indices(total_frames: int, fps: float, duration: float) -> list[int]:
    """Select unique nearest source frames for a 2-fps target grid.

    Target timestamps are clip-relative ``0, 0.5, ...`` in the half-open
    interval. The selected timestamps are later shifted by request.start_time
    for the model prompt and diagnostics.
    """

    source_count = _source_frame_count(total_frames, fps, duration)
    source_indices = list(range(source_count))
    target_count = max(1, math.ceil(duration * SHORT_TARGET_FPS - 1e-9))
    desired = [index / SHORT_TARGET_FPS for index in range(target_count)]
    selected: list[int] = []
    for target in desired:
        index = min(
            (candidate for candidate in source_indices if candidate not in selected),
            key=lambda candidate: (abs(candidate / fps - target), candidate),
        )
        if selected and index <= selected[-1]:
            raise AssertionError("nearest sampling must preserve frame order")
        selected.append(index)
    return selected


def uniform_indices(total_frames: int, budget: int) -> list[int]:
    """Return a unique, endpoint-preserving uniform subset of a clip."""

    source_count = int(total_frames)
    if source_count < 1 or budget < 1:
        raise ValueError("total_frames and budget must be positive")
    count = min(source_count, int(budget))
    if count == source_count:
        return list(range(source_count))
    indices = np.rint(np.linspace(0, source_count - 1, count)).astype(int).tolist()
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise AssertionError("uniform sampling produced duplicate or unordered indices")
    return indices


def endpoint_uniform_floor_indices(total_frames: int, budget: int) -> list[int]:
    """Return the source-of-truth floor-interpolated endpoint-preserving subset.

    The frozen SEG010 R0/R1 screens use integer floor interpolation rather than
    NumPy's rounding behavior.  The older ``uniform_indices`` helper remains
    for historical legacy paths; candidate R0/R1 paths use this helper.
    """

    source_count = int(total_frames)
    if source_count < 1 or budget < 1:
        raise ValueError("total_frames and budget must be positive")
    count = min(source_count, int(budget))
    if count == source_count:
        return list(range(source_count))
    if count == 1:
        return [0]
    indices = [index * (source_count - 1) // (count - 1) for index in range(count)]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise AssertionError("floor uniform sampling produced duplicate or unordered indices")
    return indices


def select_frame_indices(total_frames: int, fps: float, request_duration: float) -> tuple[list[int], str]:
    """Resolve the retained legacy adaptive policy from duration only."""

    source_count = _source_frame_count(total_frames, fps, request_duration)
    if request_duration <= 30.0:
        return nearest_source_indices(source_count, fps, request_duration), "short_F2"
    return uniform_indices(source_count, MAX_LONG_FRAMES), "long_M240"


def fixed_1fps_indices(total_frames: int, fps: float, duration: float) -> list[int]:
    """Select the source frames nearest to the fixed 1-fps clip-relative grid."""

    source_count = _source_frame_count(total_frames, fps, duration)
    target_count = max(1, math.ceil(duration - 1e-9))
    selected: list[int] = []
    for target_index in range(target_count):
        target = float(target_index)
        index = min(
            (candidate for candidate in range(source_count) if candidate not in selected),
            key=lambda candidate: (abs(candidate / fps - target), candidate),
        )
        if selected and index <= selected[-1]:
            raise AssertionError("fixed 1-fps sampling must preserve frame order")
        selected.append(index)
    return selected


def select_r0_frame_indices(
    total_frames: int, fps: float, request_duration: float
) -> tuple[list[int], str]:
    """Resolve the frozen R0 fixed-1-fps timeline with its 240-frame cap."""

    timeline = fixed_1fps_indices(total_frames, fps, request_duration)
    if len(timeline) <= MAX_LONG_FRAMES:
        return timeline, "R0_FIXED_1FPS"
    positions = endpoint_uniform_floor_indices(len(timeline), MAX_LONG_FRAMES)
    return [timeline[position] for position in positions], "R0_FIXED_1FPS_UNIFORM_240"


def select_r1_frame_indices(
    total_frames: int, fps: float, request_duration: float
) -> tuple[list[int], str]:
    """Resolve R1 as endpoint-preserving half of the frozen R0 timeline."""

    r0_timeline, _r0_policy = select_r0_frame_indices(total_frames, fps, request_duration)
    if len(r0_timeline) < 2:
        return r0_timeline, "R1_BALANCED_HALF_R0"
    target_count = max(2, math.ceil(len(r0_timeline) / 2))
    positions = endpoint_uniform_floor_indices(len(r0_timeline), target_count)
    return [r0_timeline[position] for position in positions], "R1_BALANCED_HALF_R0"


def initialize_router(
    *,
    q1_path: Path = Q1_ARTIFACT_PATH,
    q1_loader: Callable[[Path], Any] = load_q1_router,
    h2_loader: Callable[[Any, Any], H2Router] = load_h2_router,
    logger: Any = log,
) -> tuple[H2Router | None, RouterRuntime]:
    """Resolve the startup-only H2 -> Q1 -> Q0 fallback chain.

    H2 failure retries Q1 so a transient/partial H2 resource failure cannot
    leave an unvalidated route. Q0 is used only when the Q1 startup load also
    fails. No per-question confidence or disagreement fallback is performed.
    """

    try:
        q1 = q1_loader(Path(q1_path))
    except Exception as exc:
        logger.warning("Q1_INIT_FAILED_USING_Q0: %s", exc)
        return None, RouterRuntime(None, q0_route_question, load_error=str(exc))
    try:
        h2 = h2_loader(q0_route_question, q1)
    except Exception as exc:
        logger.warning("H2_INIT_FAILED_USING_Q1: %s", exc)
        try:
            q1_fallback = q1_loader(Path(q1_path))
        except Exception as q1_exc:
            logger.warning("Q1_INIT_FAILED_USING_Q0: %s", q1_exc)
            return None, RouterRuntime(None, q0_route_question, load_error=str(q1_exc))
        logger.info("Q1 router active after H2 startup failure")
        return None, RouterRuntime(q1_fallback, q0_route_question, load_error=str(exc))
    return h2, RouterRuntime(q1, q0_route_question)


def decode_clip(
    req: Request,
    route: Routing | None = None,
    operational_route: OperationalRoute | None = None,
) -> tuple[list[Image.Image], list[int], list[float], dict[str, Any]]:
    """Decode selected frames from the clip beginning with one batched read."""

    # Import only after torch has been imported and model/CUDA initialization
    # has happened. A top-level decord import can break CUDA initialization.
    import decord

    video_path = clip_path_for(req)
    if not video_path.is_file():
        raise FileNotFoundError(f"plain clip not found: {video_path}")
    reader_started = time.perf_counter()
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    reader_seconds = time.perf_counter() - reader_started
    fps = float(reader.get_avg_fps())
    total_frames = len(reader)
    request_duration = float(req.end_time) - float(req.start_time)
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"invalid video metadata for {video_path}: frames={total_frames}, fps={fps}")
    if abs(fps - SOURCE_FPS) > 0.05:
        log.warning("%s reports %.6f fps; expected the official 5-fps clip grid", req.qID, fps)
    if operational_route is not None and operational_route.router in {"Q1-R5", "H2"}:
        static_d_hires = False
        route = None
        if operational_route.resolution_policy == "R1_BALANCED":
            indices, policy = select_r1_frame_indices(total_frames, fps, request_duration)
        elif operational_route.resolution_policy == "R0":
            indices, policy = select_r0_frame_indices(total_frames, fps, request_duration)
        else:
            raise ValueError(f"unknown operational resolution policy: {operational_route.resolution_policy}")
        requested_frame_count = operational_route.max_frames
        max_pixels = operational_route.max_pixels
    else:
        route = route if route is not None else q0_route_question(str(req.question))
        static_d_hires = is_static_d_hires(route)
        if static_d_hires:
            indices = uniform_indices(total_frames, STATIC_HIRES_MAX_FRAMES)
            policy = "static_d_hires"
            requested_frame_count = STATIC_HIRES_MAX_FRAMES
        else:
            indices, policy = select_frame_indices(total_frames, fps, request_duration)
            requested_frame_count = (
                max(1, math.ceil(request_duration * SHORT_TARGET_FPS - 1e-9))
                if request_duration <= 30.0
                else MAX_LONG_FRAMES
            )
        max_pixels = STATIC_HIRES_MAX_PIXELS if static_d_hires else NORMAL_MAX_PIXELS
    decode_started = time.perf_counter()
    frames = reader.get_batch(indices).asnumpy()
    decode_seconds = time.perf_counter() - decode_started
    del reader
    images_started = time.perf_counter()
    images = [Image.fromarray(frame) for frame in frames]
    image_seconds = time.perf_counter() - images_started
    timestamps = [float(req.start_time) + index / fps for index in indices]
    actual_clip_duration = total_frames / fps
    diagnostics = {
        "policy": policy,
        "sampling_route": (
            operational_route.route_name
            if operational_route is not None and operational_route.router in {"Q1-R5", "H2"}
            else "static_d_hires" if static_d_hires else "adaptive_baseline"
        ),
        "route_medium": (
            operational_route.route_name
            if operational_route is not None and operational_route.router in {"Q1-R5", "H2"}
            else route.route_medium
        ),
        "route_fine": (
            operational_route.capability
            if operational_route is not None and operational_route.router in {"Q1-R5", "H2"}
            else route.route_fine
        ),
        "sampling_policy": (
            operational_route.sampling_policy
            if operational_route is not None and operational_route.router in {"Q1-R5", "H2"}
            else route.sampling_policy
        ),
        "model_selector": operational_route.model_selector if operational_route is not None else "Q0_LEGACY",
        "resolution_policy": operational_route.resolution_policy if operational_route is not None else "Q0_LEGACY",
        "requested_frame_count": requested_frame_count,
        "max_pixels": max_pixels,
        "request_duration_seconds": request_duration,
        "clip_duration_seconds": actual_clip_duration,
        "decoded_clip_duration_seconds": actual_clip_duration,
        "decoder_fps": fps,
        "source_frame_count": total_frames,
        "selected_clip_relative_frame_indices": indices,
        "selected_clip_relative_timestamps_seconds": [index / fps for index in indices],
        "selected_absolute_timestamps_seconds": timestamps,
        "frame_count": len(images),
        "video_loading_seconds": reader_seconds,
        "frame_decoding_seconds": decode_seconds,
        "image_conversion_seconds": image_seconds,
    }
    log.debug(
        "%s: policy=%s frames=%d/%d fps=%.3f clip=%.2fs abs_time=%.3f..%.3f",
        req.qID, policy, len(images), total_frames, fps, actual_clip_duration,
        timestamps[0], timestamps[-1],
    )
    return images, indices, timestamps, diagnostics


def prompt_messages(req: Request, images: list[Image.Image], timestamps: list[float]) -> list[dict[str, Any]]:
    """Build the frozen request-only P2 prompt exactly once per question."""

    if len(images) != len(timestamps):
        raise ValueError("image/timestamp count mismatch")
    content: list[dict[str, Any]] = []
    for timestamp, image in zip(timestamps, images, strict=True):
        content.extend([
            {"type": "text", "text": f"Frame timestamp (absolute source-procedure timeline): {timestamp:0.1f} seconds."},
            {"type": "image", "image": image},
        ])
    instruction = format_instruction(
        str(req.question), request_start_seconds=float(req.start_time), request_end_seconds=float(req.end_time)
    )
    content.append({
        "type": "text",
        "text": (
            "You are assisting with laparoscopic surgery. "
            f"{SEGMENT_EVIDENCE_INSTRUCTION} "
            f"Procedure type: {req.procedure_type}. The request window is from {float(req.start_time):.1f} "
            f"to {float(req.end_time):.1f} seconds on the original source-procedure timeline. "
            "Sampled-frame timestamps are absolute source-procedure timeline timestamps. "
            f"{instruction}\nQuestion: {req.question}"
        ),
    })
    return [{"role": "user", "content": content}]


def load_model(
    device: torch.device,
) -> tuple[Qwen3VLForConditionalGeneration, Any, Any, Any, dict[str, Any]]:
    """Load one frozen base and both validated LMV specialist banks."""

    bindings = require_final_specialist_bindings(FINAL_CANDIDATE_CONFIG_PATH)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"offline model snapshot is missing: {MODEL_PATH}")
    normal_processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH), max_pixels=NORMAL_MAX_PIXELS, local_files_only=True
    )
    r1_processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH), max_pixels=R1_MAX_PIXELS, local_files_only=True
    )
    static_hires_processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH), max_pixels=STATIC_HIRES_MAX_PIXELS, local_files_only=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        local_files_only=True,
    ).eval()
    specialist_runtime = SpecialistRuntime(model, device, bindings)
    model.config.use_cache = True
    installation = specialist_runtime.installation
    installation.update(
        {
            "base_model_revision": MODEL_REVISION,
            "base_model_path": str(MODEL_PATH),
            "max_pixels": {
                "normal": NORMAL_MAX_PIXELS,
                "r1": R1_MAX_PIXELS,
                "static_hires": STATIC_HIRES_MAX_PIXELS,
            },
        }
    )
    return model, normal_processor, r1_processor, static_hires_processor, installation


def generate_answer(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    device: torch.device,
    req: Request,
    images: list[Image.Image],
    timestamps: list[float],
) -> tuple[str, ReducedAnswer, int, int, float]:
    messages = prompt_messages(req, images, timestamps)
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)
    eos_token_id = getattr(model.generation_config, "eos_token_id", None) or processor.tokenizer.eos_token_id
    pad_token_id = getattr(model.generation_config, "pad_token_id", None) or processor.tokenizer.pad_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("generation EOS and PAD token IDs must be defined")
    eos_ids = [int(value) for value in eos_token_id] if isinstance(eos_token_id, (list, tuple)) else [int(eos_token_id)]
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            eos_token_id=eos_ids, pad_token_id=int(pad_token_id),
        )
    generation_seconds = time.perf_counter() - generation_started
    new_tokens = generated[:, encoded.input_ids.shape[1] :]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    reduced = reduce_answer(
        str(req.question), raw, request_start_seconds=float(req.start_time), request_end_seconds=float(req.end_time)
    )
    return raw, reduced, int(encoded.input_ids.shape[1]), int(new_tokens.shape[1]), generation_seconds


def _read_fo_definitions() -> str:
    path = INPUT_PATH / "FO_definitions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, str):
        raise TypeError("FO_definitions.json must contain a JSON-encoded text string")
    log.info("FO definitions loaded: %d chars, sha256=%s", len(value), hashlib.sha256(value.encode()).hexdigest())
    # The active request-only prompt is used without injecting the full
    # definitions text. The official input is still read and audited.
    return value


def run() -> int:
    started = time.perf_counter()
    log.info("=== ORena SAVE FOCUS — SEGMENT inference start ===")
    require_final_specialist_bindings(FINAL_CANDIDATE_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log_environment(device)
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    qids = [str(req.qID) for req in requests]
    if len(qids) != len(set(qids)):
        raise ValueError("request.json contains duplicate qIDs")
    log.info("Batch: %d question(s); plain clips: %d", len(requests), len(list(VIDEO_DIR.glob("*.mp4"))))
    _read_fo_definitions()

    h2_router, router_runtime = initialize_router()
    log.info("Question router loaded once: mode=%s", "H2" if h2_router is not None else router_runtime.mode)

    model_started = time.perf_counter()
    model, normal_processor, r1_processor, static_hires_processor, installation = load_model(device)
    specialist_runtime = installation["specialist_runtime"]
    log.info(
        "Model loaded once in %.2fs: revision=%s trainable_params=%s normal_max_pixels=%d r1_max_pixels=%d static_hires_max_pixels=%d",
        time.perf_counter() - model_started,
        MODEL_REVISION,
        installation.get("trainable_parameter_count", "unknown"),
        NORMAL_MAX_PIXELS,
        R1_MAX_PIXELS,
        STATIC_HIRES_MAX_PIXELS,
    )
    if device.type == "cuda":
        log.info("VRAM after setup: %.1f MiB allocated / %.1f MiB reserved", torch.cuda.memory_allocated(device) / 2**20, torch.cuda.memory_reserved(device) / 2**20)

    responses: list[Response] = []
    failed = 0
    batch_started = time.perf_counter()
    for number, req in enumerate(requests, start=1):
        question_started = time.perf_counter()
        log.info("[%d/%d] qID=%s window=[%.2f, %.2f]", number, len(requests), req.qID, req.start_time, req.end_time)
        decision: OperationalRoute | None = None
        q0_route: Routing | None = None
        static_d_hires = False
        processor = normal_processor
        diagnostics: dict[str, Any] = {}
        generation_seconds = 0.0
        try:
            decision = (
                h2_router.route_question(str(req.question))
                if h2_router is not None
                else router_runtime.route_question(str(req.question))
            )
            q0_route = decision.legacy_q0_route
            static_d_hires = q0_route is not None and is_static_d_hires(q0_route)
            if static_d_hires:
                processor = static_hires_processor
            elif decision.resolution_policy == "R1_BALANCED":
                processor = r1_processor
            else:
                processor = normal_processor
            images, _indices, timestamps, diagnostics = decode_clip(
                req, route=q0_route, operational_route=decision
            )
            assert decision is not None
            specialist_runtime.activate_specialist(decision.model_selector)
            with specialist_runtime.inference_guard():
                raw, reduced, prompt_tokens, generated_tokens, generation_seconds = generate_answer(
                    model, processor, device, req, images, timestamps
                )
            diagnostics["active_specialist"] = specialist_runtime.active_specialist
            diagnostics["generation_seconds"] = generation_seconds
            if not raw.strip():
                raise ValueError("model generated an empty answer")
            content = reduced.content
            if reduced.format_valid:
                log.info("[%d/%d] qID=%s raw=%r reduced=%r format=%s", number, len(requests), req.qID, raw, content, reduced.answer_format)
            else:
                log.warning("[%d/%d] qID=%s reducer invalid: raw=%r error=%s", number, len(requests), req.qID, raw, reduced.error)
            log.debug("qID=%s prompt_tokens=%d generated_tokens=%d diagnostics=%s", req.qID, prompt_tokens, generated_tokens, diagnostics)
        except SpecialistActivationError:
            log.exception(
                "[%d/%d] qID=%s specialist activation failed; terminating without fallback",
                number,
                len(requests),
                req.qID,
            )
            raise
        except Exception:
            failed += 1
            log.exception("[%d/%d] qID=%s failed; emitting empty answer", number, len(requests), req.qID)
            content = ""
        latency = time.perf_counter() - question_started
        responses.append(Response(qID=req.qID, content=content, latency=latency))
        log.info(
            "[%d/%d] qID=%s end-to-end %.3fs sampling_route=%s route_medium=%s route_fine=%s "
            "sampling_policy=%s frames=%d/%d max_pixels=%d clip_duration=%.3fs",
            number,
            len(requests),
            req.qID,
            latency,
            diagnostics.get("sampling_route", "route_failed"),
            diagnostics.get("route_medium", decision.route_name if decision is not None else "route_failed"),
            diagnostics.get("route_fine", decision.capability if decision is not None else "route_failed"),
            diagnostics.get("sampling_policy", decision.sampling_policy if decision is not None else "route_failed"),
            diagnostics.get("frame_count", 0),
            diagnostics.get("requested_frame_count", 0),
            diagnostics.get("max_pixels", STATIC_HIRES_MAX_PIXELS if static_d_hires else NORMAL_MAX_PIXELS),
            diagnostics.get("clip_duration_seconds", 0.0),
        )
        log.info("[%d/%d] qID=%s generation %.3fs", number, len(requests), req.qID, generation_seconds)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    batch_seconds = time.perf_counter() - batch_started
    log.info("Batch inference: %d answered, %d failed, %.3fs total, %.3fs/question", len(responses) - failed, failed, batch_seconds, batch_seconds / len(responses))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d responses to %s", len(responses), OUTPUT_PATH / "answer.json")
    log.info("=== inference done in %.3fs total ===", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
