from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.inference.aggregation import AGGREGATION_STRATEGY
from app.inference.config import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    SCORE_TYPE_DECISION_FUNCTION,
    InferenceConfig,
)
from app.inference.errors import ModelArtifactError, PredictionError
from app.inference.features import feature_columns
from app.inference.schemas import ModelMetadata


REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "feature_columns",
    "preprocessing",
    "mfcc",
    "selected_params",
    "positive_label",
    "seed",
}


class MfccSvmModel:
    def __init__(
        self,
        artifact: dict[str, Any],
        path: Path,
        config: InferenceConfig,
        sha256: str,
    ) -> None:
        self.artifact = artifact
        self.path = path
        self.config = config
        self.sha256 = sha256
        self.pipeline = artifact["pipeline"]
        self.classes = tuple(int(value) for value in np.asarray(self.pipeline.classes_).tolist())
        self.metadata = ModelMetadata(
            model_id=config.model_id,
            sha256=sha256,
            loaded_path=path,
            classes=self.classes,
            positive_label=config.positive_label,
            score_type=SCORE_TYPE_DECISION_FUNCTION,
            target_sample_rate=config.target_sample_rate,
            fragment_duration_seconds=config.fragment_duration_seconds,
            n_mfcc=config.n_mfcc,
            n_features=len(feature_columns(config)),
            decision_threshold=config.decision_threshold,
            aggregation_strategy=AGGREGATION_STRATEGY,
            score_is_calibrated_probability=False,
        )

    @classmethod
    def load(cls, path: Path, config: InferenceConfig) -> "MfccSvmModel":
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.exists():
            raise ModelArtifactError(f"Model artifact does not exist: {artifact_path}")
        if not artifact_path.is_file():
            raise ModelArtifactError(f"Model artifact path is not a file: {artifact_path}")

        sha256 = _sha256_file(artifact_path)
        try:
            artifact = joblib.load(artifact_path)
        except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
            raise ModelArtifactError(f"Could not load model artifact {artifact_path}: {exc}") from exc

        _validate_artifact(artifact, config)
        return cls(artifact=artifact, path=artifact_path, config=config, sha256=sha256)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise PredictionError(f"Expected a 2D feature matrix, found shape {matrix.shape}")
        if matrix.shape[0] == 0:
            raise PredictionError("Cannot predict zero fragments")
        expected_features = len(feature_columns(self.config))
        if matrix.shape[1] != expected_features:
            raise PredictionError(
                f"Expected {expected_features} features, found {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise PredictionError("Feature matrix contains NaN or infinite values")

        try:
            scores = self.pipeline.decision_function(matrix)
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Could not compute SVM decision scores: {exc}") from exc

        oriented = np.asarray(scores, dtype=np.float64).reshape(-1)
        if oriented.shape != (matrix.shape[0],):
            raise PredictionError(
                f"Expected one score per fragment, found shape {oriented.shape}"
            )
        oriented = self._orient_scores(oriented)
        if not np.isfinite(oriented).all():
            raise PredictionError("Decision scores contain NaN or infinite values")
        return oriented

    def _orient_scores(self, scores: np.ndarray) -> np.ndarray:
        classes = tuple(int(value) for value in np.asarray(self.pipeline.classes_).tolist())
        if len(classes) != 2:
            raise PredictionError(f"Expected binary classes, found {classes}")
        if self.config.positive_label == classes[1]:
            return scores
        if self.config.positive_label == classes[0]:
            return -scores
        raise PredictionError(
            f"Positive label {self.config.positive_label} is not present in classes {classes}"
        )


def predict_labels_from_scores(
    scores: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise PredictionError(f"Expected a 1D score vector, found shape {values.shape}")
    if not np.isfinite(values).all():
        raise PredictionError("Scores contain NaN or infinite values")
    return np.where(values >= threshold, AI_GENERATED_LABEL, HUMAN_LABEL).astype(int)


def _validate_artifact(artifact: Any, config: InferenceConfig) -> None:
    if not isinstance(artifact, dict):
        raise ModelArtifactError("Model artifact root object must be a dictionary")

    missing = REQUIRED_ARTIFACT_KEYS.difference(artifact)
    if missing:
        raise ModelArtifactError(f"Model artifact missing required keys: {sorted(missing)}")

    if artifact["positive_label"] != config.positive_label:
        raise ModelArtifactError(
            f"Expected positive_label {config.positive_label}, found {artifact['positive_label']}"
        )

    artifact_columns = tuple(artifact["feature_columns"])
    expected_columns = feature_columns(config)
    if len(artifact_columns) != len(expected_columns):
        raise ModelArtifactError(
            f"Expected {len(expected_columns)} feature columns, found {len(artifact_columns)}"
        )
    if artifact_columns != expected_columns:
        raise ModelArtifactError("Model artifact feature column order is incompatible")

    preprocessing = artifact["preprocessing"]
    if preprocessing.get("target_sample_rate") != config.target_sample_rate:
        raise ModelArtifactError(
            "Model artifact target_sample_rate is incompatible: "
            f"{preprocessing.get('target_sample_rate')}"
        )
    if float(preprocessing.get("duration_seconds")) != config.fragment_duration_seconds:
        raise ModelArtifactError(
            "Model artifact duration_seconds is incompatible: "
            f"{preprocessing.get('duration_seconds')}"
        )

    mfcc = artifact["mfcc"]
    if mfcc.get("n_mfcc") != config.n_mfcc:
        raise ModelArtifactError(f"Model artifact n_mfcc is incompatible: {mfcc.get('n_mfcc')}")

    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "classes_"):
        raise ModelArtifactError("Model pipeline is not fitted: missing classes_")
    classes = tuple(int(value) for value in np.asarray(pipeline.classes_).tolist())
    if set(classes) != {0, 1} or len(classes) != 2:
        raise ModelArtifactError(f"Expected classes 0 and 1, found {classes}")

    expected_feature_count = len(expected_columns)
    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is None and hasattr(pipeline, "named_steps"):
        final_step = list(pipeline.named_steps.values())[-1]
        n_features = getattr(final_step, "n_features_in_", None)
    if n_features != expected_feature_count:
        raise ModelArtifactError(
            f"Expected pipeline to require {expected_feature_count} features, found {n_features}"
        )

    if not hasattr(pipeline, "decision_function"):
        raise ModelArtifactError("Model pipeline is missing decision_function")
    if not hasattr(pipeline, "predict"):
        raise ModelArtifactError("Model pipeline is missing predict")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
