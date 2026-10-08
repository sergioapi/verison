from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app.inference.errors import ModelArtifactError
from app.main import create_app


@dataclass(frozen=True)
class FakeMetadata:
    model_id: str = "mfcc-svm-baseline"
    sha256: str = "abc123"
    loaded_path: Path = Path("C:/private/model.joblib")
    classes: tuple[int, ...] = (0, 1)
    positive_label: int = 1
    score_type: str = "fake_decision_score"
    target_sample_rate: int = 16_000
    fragment_duration_seconds: float = 10.0
    n_mfcc: int = 20
    n_features: int = 40
    decision_threshold: float = 0.0
    aggregation_strategy: str = "duration_weighted_mean_decision_score"
    score_is_calibrated_probability: bool = False


class FakeModel:
    metadata = FakeMetadata()


class FakeConfig:
    usage_warning = "El score no es una probabilidad calibrada."


class FakeService:
    model = FakeModel()
    config = FakeConfig()

    @property
    def metadata(self) -> FakeMetadata:
        return self.model.metadata

    @property
    def usage_warning(self) -> str:
        return self.config.usage_warning

    def predict_file(self, path):  # pragma: no cover - must not be called by /api/v1/model.
        raise AssertionError("/api/v1/model must not run inference")


def test_model_info_reports_loaded_model_metadata() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "mfcc-svm-baseline",
        "sha256": "abc123",
        "classes": [0, 1],
        "positive_label": 1,
        "score_type": "fake_decision_score",
        "score_is_calibrated_probability": False,
        "decision_threshold": 0.0,
        "target_sample_rate": 16_000,
        "fragment_duration_seconds": 10.0,
        "n_mfcc": 20,
        "n_features": 40,
        "aggregation_strategy": "duration_weighted_mean_decision_score",
        "usage_warning": "El score no es una probabilidad calibrada.",
    }


def test_model_info_does_not_expose_internal_paths() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.get("/api/v1/model")

    body = response.text
    assert response.status_code == 200
    assert "loaded_path" not in body
    assert "MODEL_PATH" not in body
    assert "C:/private/model.joblib" not in body


def test_model_info_returns_503_when_model_is_not_available() -> None:
    def service_factory() -> FakeService:
        raise ModelArtifactError("Could not load C:/private/model.joblib")

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = client.get("/api/v1/model")

    assert response.status_code == 503
    assert "Could not load" not in response.text
    assert "C:/private/model.joblib" not in response.text


def test_model_info_does_not_call_predict_file() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json()["score_type"] == "fake_decision_score"


def test_service_factory_is_called_once_for_health_and_model_info() -> None:
    calls = {"count": 0}

    def service_factory() -> FakeService:
        calls["count"] += 1
        return FakeService()

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        health_response = client.get("/health")
        model_response = client.get("/api/v1/model")

    assert health_response.status_code == 200
    assert model_response.status_code == 200
    assert calls["count"] == 1
