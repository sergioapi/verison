from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from app.inference.config import InferenceConfig
from app.inference.errors import ModelArtifactError
from app.inference.features import feature_columns
from app.inference.model import MfccSvmModel, predict_labels_from_scores


def test_load_valid_synthetic_artifact(artifact_factory, config: InferenceConfig) -> None:
    path = artifact_factory()

    model = MfccSvmModel.load(path, config)

    assert model.path == path.resolve()
    assert model.metadata.n_features == 40
    assert model.metadata.positive_label == 1
    assert model.metadata.sha256


def test_load_errors_if_required_key_is_missing(artifact_factory, rng, config, tmp_path: Path) -> None:
    path = artifact_factory()
    artifact = joblib.load(path)
    artifact.pop("pipeline")
    broken = tmp_path / "broken.joblib"
    joblib.dump(artifact, broken)

    with pytest.raises(ModelArtifactError, match="missing required keys"):
        MfccSvmModel.load(broken, config)


def test_load_errors_if_positive_label_is_not_one(artifact_factory, config) -> None:
    path = artifact_factory(positive_label=0)

    with pytest.raises(ModelArtifactError, match="positive_label"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_feature_column_count_is_not_forty(artifact_factory, config) -> None:
    path = artifact_factory(columns=feature_columns(config)[:-1])

    with pytest.raises(ModelArtifactError, match="feature columns"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_feature_column_order_is_wrong(artifact_factory, config) -> None:
    columns = list(feature_columns(config))
    columns[0], columns[1] = columns[1], columns[0]
    path = artifact_factory(columns=tuple(columns))

    with pytest.raises(ModelArtifactError, match="column order"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_frequency_is_different(artifact_factory, config) -> None:
    path = artifact_factory(
        preprocessing={"target_sample_rate": 8_000, "duration_seconds": 10.0}
    )

    with pytest.raises(ModelArtifactError, match="target_sample_rate"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_duration_is_different(artifact_factory, config) -> None:
    path = artifact_factory(
        preprocessing={"target_sample_rate": 16_000, "duration_seconds": 5.0}
    )

    with pytest.raises(ModelArtifactError, match="duration_seconds"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_n_mfcc_is_different(artifact_factory, config) -> None:
    path = artifact_factory(mfcc={"n_mfcc": 13})

    with pytest.raises(ModelArtifactError, match="n_mfcc"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_classes_are_not_zero_and_one(artifact_factory, config) -> None:
    path = artifact_factory(classes=(0, 2))

    with pytest.raises(ModelArtifactError, match="classes"):
        MfccSvmModel.load(path, config)


def test_load_errors_if_pipeline_feature_count_is_different(artifact_factory, config) -> None:
    path = artifact_factory(n_features=39)

    with pytest.raises(ModelArtifactError, match="40 features"):
        MfccSvmModel.load(path, config)


def test_predict_scores_runs_in_batch_and_returns_finite_scores(artifact_factory, config, rng) -> None:
    model = MfccSvmModel.load(artifact_factory(), config)
    features = rng.normal(size=(3, 40)).astype(np.float32)

    scores = model.predict_scores(features)

    assert scores.shape == (3,)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()


def test_score_orientation_with_classes_zero_one(artifact_factory, config, rng) -> None:
    model = MfccSvmModel.load(artifact_factory(), config)
    features = rng.normal(size=(2, 40)).astype(np.float32)
    raw = np.asarray(model.pipeline.decision_function(features), dtype=np.float64)

    assert np.allclose(model.predict_scores(features), raw)


class ReversedClassesPipeline:
    classes_ = np.array([1, 0])
    n_features_in_ = 40

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.array([2.0, -3.0], dtype=np.float64)[: features.shape[0]]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(features.shape[0], dtype=int)


def test_score_orientation_is_inverted_when_positive_label_is_classes_zero(config: InferenceConfig) -> None:
    artifact = {
        "pipeline": ReversedClassesPipeline(),
        "feature_columns": list(feature_columns(config)),
        "preprocessing": {
            "target_sample_rate": config.target_sample_rate,
            "duration_seconds": config.fragment_duration_seconds,
        },
        "mfcc": {"n_mfcc": config.n_mfcc},
        "selected_params": {"C": 10, "gamma": 0.01},
        "positive_label": 1,
        "seed": 42,
    }
    model = MfccSvmModel(artifact, Path("dummy.joblib"), config, sha256="abc")

    scores = model.predict_scores(np.zeros((2, 40), dtype=np.float32))

    assert np.allclose(scores, np.array([-2.0, 3.0]))


def test_predict_labels_from_scores_uses_threshold_zero() -> None:
    labels = predict_labels_from_scores(np.array([-0.1, 0.0, 0.2]), threshold=0.0)

    assert labels.tolist() == [0, 1, 1]


def test_load_reports_sha256(artifact_factory, config) -> None:
    model = MfccSvmModel.load(artifact_factory(), config)

    assert len(model.metadata.sha256) == 64
    assert all(char in "0123456789abcdef" for char in model.metadata.sha256)
