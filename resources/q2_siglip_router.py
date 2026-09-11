"""Offline Q2 SigLIP text router for the H2 candidate.

The encoder is text-only, loaded once on CPU at startup, and receives only the
request question.  No reference metadata, video, answer, or answer format is
consulted.  Loading failures are explicit so the caller can retain its tested
fallback rather than silently activating a partial route.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from resources.candidate_policy import route_from_predictions
from resources.q1_router import (
    Q0_FINE_TO_R5,
    OperationalRoute,
    Q1Router,
    capability_to_route,
)

Q2_ENCODER_PATH = Path(__file__).with_name("q2_siglip_base_patch16_384")
Q2_CLASSIFIER_PATH = Path(__file__).with_name("q2_siglip_text_r5.joblib")
Q2_MANIFEST_PATH = Path(__file__).with_name("q2_siglip_manifest.json")
Q2_CLASSES = (
    "aggregation",
    "complex_reasoning",
    "event_understanding",
    "object_recognition",
    "temporal_grounding",
)
EXPECTED_Q2_CLASSIFIER_SHA256 = "a335c568991db95b7633dd9fb9c23eef2ee276ae51485592781056c1917c9158"


def _load_q2_classifier(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Q2 classifier is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_Q2_CLASSIFIER_SHA256:
        raise ValueError(f"unexpected Q2 classifier SHA256: {actual}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"vectorizer", "classifier"}:
        raise ValueError("Q2 artifact must contain exactly vectorizer and classifier")
    if bundle["vectorizer"] is not None:
        raise ValueError("Q2 SigLIP artifact must not contain a text vectorizer")
    classifier = bundle["classifier"]
    classes = tuple(str(value) for value in getattr(classifier, "classes_", ()))
    if classes != Q2_CLASSES:
        raise ValueError(f"unexpected Q2 serialized classes_: {classes}")
    return classifier


def _verify_encoder_manifest(path: Path, encoder_path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Q2 encoder manifest has no file hashes")
    for name, expected in files.items():
        asset = encoder_path / str(name)
        if not asset.is_file():
            raise FileNotFoundError(f"Q2 encoder asset is missing: {asset}")
        actual = hashlib.sha256(asset.read_bytes()).hexdigest()
        if actual != str(expected):
            raise ValueError(f"Q2 encoder asset SHA256 mismatch for {name}: {actual}")


class Q2SigLIPRouter:
    """One-time CPU SigLIP encoder plus frozen Q2 classifier."""

    def __init__(self, encoder: Any, processor: Any, classifier: Any) -> None:
        self._encoder = encoder
        self._processor = processor
        self._classifier = classifier
        self._predict_lock = threading.RLock()

    def predict_capability(self, question: str) -> str:
        import torch

        with self._predict_lock, torch.inference_mode():
            inputs = self._processor(
                text=[str(question)],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            output = self._encoder.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            embedding = pooled.float().cpu().numpy().astype(np.float32, copy=False)
            prediction = str(self._classifier.predict(embedding)[0])
        if prediction not in Q2_CLASSES:
            raise ValueError(f"Q2 predicted unknown class: {prediction!r}")
        return prediction


class H2Router:
    """Compose Q0, Q1, and Q2 predictions using H2 precedence."""

    def __init__(self, q0_route_question: Any, q1: Q1Router, q2: Q2SigLIPRouter) -> None:
        self._q0_route_question = q0_route_question
        self._q1 = q1
        self._q2 = q2

    def route_question(self, question: str) -> OperationalRoute:
        text = str(question)
        q0_route = self._q0_route_question(text)
        q0_capability = Q0_FINE_TO_R5.get(str(q0_route.route_fine))
        if q0_capability is None:
            raise ValueError(f"Q0 produced an unknown fine route: {q0_route.route_fine!r}")
        q1_capability = self._q1.predict_capability(text)
        q2_capability = self._q2.predict_capability(text)
        candidate = route_from_predictions(q0_capability, q1_capability, q2_capability)
        if candidate.route_id == "A":
            return capability_to_route("aggregation", router="H2")
        if candidate.route_id == "B":
            return capability_to_route("temporal_grounding", router="H2")
        # Route C has one operational policy for all non-temporal capabilities.
        # Preserve a useful capability label when Q2 is non-temporal, while
        # avoiding accidental conversion of a C decision into the B route.
        c_capability = q2_capability if q2_capability != "temporal_grounding" else "object_recognition"
        return capability_to_route(c_capability, router="H2")


def load_h2_router(
    q0_route_question: Any,
    q1: Q1Router,
    *,
    encoder_path: Path = Q2_ENCODER_PATH,
    classifier_path: Path = Q2_CLASSIFIER_PATH,
    manifest_path: Path = Q2_MANIFEST_PATH,
) -> H2Router:
    """Load and validate all H2 assets once, entirely offline."""

    encoder_path = Path(encoder_path)
    classifier_path = Path(classifier_path)
    manifest_path = Path(manifest_path)
    if not encoder_path.is_dir():
        raise FileNotFoundError(f"Q2 encoder directory is missing: {encoder_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Q2 encoder manifest is missing: {manifest_path}")
    from transformers import AutoModel, AutoProcessor

    _verify_encoder_manifest(manifest_path, encoder_path)
    classifier = _load_q2_classifier(classifier_path)
    encoder = AutoModel.from_pretrained(
        str(encoder_path), local_files_only=True
    ).cpu().eval()
    processor = AutoProcessor.from_pretrained(
        str(encoder_path), local_files_only=True
    )
    return H2Router(q0_route_question, q1, Q2SigLIPRouter(encoder, processor, classifier))
