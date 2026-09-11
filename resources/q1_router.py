"""Question-only Q1-R5 production router.

The serialized bundle is loaded once at startup.  This module contains no
reference-answer, dataset, video, or qID inputs and does not import the
training-side router implementation.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

Q1_ARTIFACT_NAME = "q1_tfidf_r5_router.joblib"
Q1_ARTIFACT_PATH = Path(__file__).with_name(Q1_ARTIFACT_NAME)
CAPABILITIES = (
    "aggregation",
    "complex_reasoning",
    "event_understanding",
    "object_recognition",
    "temporal_grounding",
)
Q0_FINE_TO_R5 = {
    "object_attributes": "object_recognition",
    "object_identity_presence": "object_recognition",
    "object_count": "aggregation",
    "event_object_identity": "object_recognition",
    "spatial_camera": "object_recognition",
    "spatial_situs": "object_recognition",
    "instance_matching": "object_recognition",
    "event_count": "aggregation",
    "occupancy_percentage": "aggregation",
    "duration": "temporal_grounding",
    "event_localization": "temporal_grounding",
    "temporal_ordering": "event_understanding",
    "event_occurrence": "event_understanding",
    "clinical_reasoning": "complex_reasoning",
    "general_other": "object_recognition",
}


@dataclass(frozen=True)
class OperationalRoute:
    """Frozen model and input policy selected for one question."""

    capability: str
    route_name: str
    model_selector: str
    resolution_policy: str
    sampling_policy: str
    max_frames: int
    max_pixels: int
    frame_policy: str
    router: str
    # Only populated on the explicit Q0 load-failure fallback.  Normal Q1
    # operation never carries a legacy route object.
    legacy_q0_route: Any | None = None


def capability_to_route(capability: str, *, router: str = "Q1-R5") -> OperationalRoute:
    capability = str(capability)
    if capability == "aggregation":
        return OperationalRoute(
            capability=capability,
            route_name="Route_A_AGG_R0",
            model_selector="AGGREGATION",
            resolution_policy="R0",
            sampling_policy="FIXED_1FPS",
            max_frames=240,
            max_pixels=50176,
            frame_policy="FIXED_1FPS_MAX_240",
            router=router,
        )
    if capability == "temporal_grounding":
        return OperationalRoute(
            capability=capability,
            route_name="Route_B_GENERAL_R0",
            model_selector="GENERAL",
            resolution_policy="R0",
            sampling_policy="FIXED_1FPS",
            max_frames=240,
            max_pixels=50176,
            frame_policy="FIXED_1FPS_MAX_240",
            router=router,
        )
    if capability in {"object_recognition", "event_understanding", "complex_reasoning"}:
        return OperationalRoute(
            capability=capability,
            route_name="Route_C_GENERAL_R1",
            model_selector="GENERAL",
            resolution_policy="R1_BALANCED",
            sampling_policy="FIXED_1FPS_THEN_HALF_R0_TIMELINE",
            max_frames=240,
            max_pixels=100352,
            frame_policy="HALF_R0_TIMELINE_ENDPOINT_PRESERVING",
            router=router,
        )
    raise ValueError(f"unknown Q1 capability: {capability!r}")


class Q1Router:
    """Loaded Q1-R5 model with serialized class-order validation."""

    def __init__(self, bundle: dict[str, Any], artifact_path: Path) -> None:
        if not isinstance(bundle, dict) or set(bundle) != {"vectorizer", "classifier"}:
            raise ValueError("Q1 artifact must contain exactly vectorizer and classifier")
        vectorizer = bundle["vectorizer"]
        classifier = bundle["classifier"]
        classes = tuple(str(value) for value in getattr(classifier, "classes_", ()))
        if set(classes) != set(CAPABILITIES) or len(classes) != len(CAPABILITIES):
            raise ValueError(f"unexpected serialized Q1 classes_: {classes}")
        self._vectorizer = vectorizer
        self._classifier = classifier
        self._classes = classes
        self.artifact_path = artifact_path
        # sklearn transform/predict is read-only here.  The lock makes the
        # singleton safe if the submission wrapper later calls it concurrently.
        self._predict_lock = threading.RLock()

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    def predict_capability(self, question: str) -> str:
        return self.predict_capabilities([str(question)])[0]

    def predict_capabilities(self, questions: list[str]) -> list[str]:
        with self._predict_lock:
            features = self._vectorizer.transform([str(question) for question in questions])
            predictions = [str(value) for value in self._classifier.predict(features)]
        unknown = sorted(set(predictions) - set(self._classes))
        if unknown:
            raise ValueError(f"Q1 predicted unknown classes: {unknown}")
        return predictions

    def route_question(self, question: str) -> OperationalRoute:
        return capability_to_route(self.predict_capability(str(question)))


def load_q1_router(path: Path = Q1_ARTIFACT_PATH) -> Q1Router:
    """Load and validate the Q1 bundle once; callers retain the instance."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Q1 router artifact is missing: {path}")
    return Q1Router(joblib.load(path), path)


def q0_to_operational_route(q0_route: Any) -> OperationalRoute:
    """Map the unchanged legacy Q0 route to a downstream operational route."""

    fine = str(q0_route.route_fine)
    capability = Q0_FINE_TO_R5.get(fine)
    if capability is None:
        raise ValueError(f"Q0 fallback produced an unknown fine route: {fine!r}")
    route = capability_to_route(capability, router="Q0_FALLBACK")
    return OperationalRoute(**{**route.__dict__, "legacy_q0_route": q0_route})


class RouterRuntime:
    """Startup-selected Q1 runtime with load-failure-only Q0 fallback."""

    def __init__(self, q1: Q1Router | None, q0_route_question: Callable[[str], Any] | None, load_error: str | None = None) -> None:
        if q1 is None and q0_route_question is None:
            raise ValueError("Q0 fallback callback is required when Q1 is unavailable")
        self.q1 = q1
        self._q0_route_question = q0_route_question
        self.load_error = load_error

    @property
    def mode(self) -> str:
        return "Q1-R5" if self.q1 is not None else "Q0_FALLBACK"

    def route_question(self, question: str) -> OperationalRoute:
        if self.q1 is not None:
            return self.q1.route_question(str(question))
        assert self._q0_route_question is not None
        return q0_to_operational_route(self._q0_route_question(str(question)))


def load_router_runtime(
    path: Path = Q1_ARTIFACT_PATH,
    *,
    q0_route_question: Callable[[str], Any] | None,
    logger: logging.Logger | None = None,
) -> RouterRuntime:
    """Load Q1 at startup and activate Q0 only when that load fails."""

    try:
        return RouterRuntime(load_q1_router(path), q0_route_question)
    except Exception as exc:
        message = f"Q1_INIT_FAILED_USING_Q0: {exc}"
        (logger or logging.getLogger(__name__)).warning(message)
        return RouterRuntime(None, q0_route_question, load_error=str(exc))
