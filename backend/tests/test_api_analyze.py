from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import ApiSettings
from app.api.uploads import uploaded_audio_path, validate_upload_extension
from app.inference.errors import (
    AudioDecodingError,
    AudioValidationError,
    ModelArtifactError,
    PredictionError,
)
from app.inference.schemas import (
    FragmentPrediction,
    InferenceTimings,
    ModelMetadata,
    PredictionResult,
)
from app.main import create_app


class FakeConfig:
    usage_warning = "El score no es una probabilidad calibrada."


class FakeModel:
    metadata = ModelMetadata(
        model_id="mfcc-svm-baseline",
        sha256="abc123",
        loaded_path=Path("C:/private/model.joblib"),
        classes=(0, 1),
        positive_label=1,
        score_type="fake_decision_score",
        target_sample_rate=16_000,
        fragment_duration_seconds=10.0,
        n_mfcc=20,
        n_features=40,
        decision_threshold=0.0,
        aggregation_strategy="duration_weighted_mean_decision_score",
        score_is_calibrated_probability=False,
    )


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.config = FakeConfig()
        self.model = FakeModel()
        self.error = error
        self.calls = 0
        self.received_paths: list[Path] = []
        self.exists_during_predict = False
        self.can_open_during_predict = False
        self.parent_during_predict: Path | None = None

    def predict_file(self, path: str | Path) -> PredictionResult:
        self.calls += 1
        received_path = Path(path)
        self.received_paths.append(received_path)
        self.exists_during_predict = received_path.exists()
        self.parent_during_predict = received_path.parent
        with received_path.open("rb") as handle:
            self.can_open_during_predict = bool(handle.read(1))
        if self.error is not None:
            raise self.error
        return _prediction_result()


class UnexpectedErrorService(FakeService):
    def predict_file(self, path: str | Path) -> PredictionResult:
        super().predict_file(path)
        raise RuntimeError("boom C:/private/audio.wav")


class BrokenUploadFile:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.content_type = "audio/wav"
        self.file = BrokenReadFile()


class BrokenReadFile:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        self.calls += 1
        if self.calls == 1:
            return b"partial"
        raise OSError("read failed")

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def stub_audio_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.routes.get_audio_duration_seconds", lambda path: 1.0)


def test_analyze_valid_wav_returns_complete_schema() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    payload = response.json()
    assert response.status_code == 200
    assert payload["predicted_label"] == 1
    assert payload["predicted_class"] == "ai_generated"
    assert payload["ai_score"] == pytest.approx(0.42)
    assert "probability" not in payload
    assert payload["decision_threshold"] == 0.0
    assert payload["audio_duration_seconds"] == pytest.approx(1.5)
    assert payload["original_sample_rate"] == 44_100
    assert payload["n_fragments"] == 1
    assert payload["fragments"][0]["was_padded"] is True
    assert payload["timings"]["total_seconds"] == pytest.approx(0.7)
    assert payload["model"]["score_type"] == "fake_decision_score"
    assert payload["model"]["score_is_calibrated_probability"] is False
    assert payload["usage_warning"] == FakeConfig.usage_warning
    assert "usage_warning" not in payload["model"]
    assert "decision_threshold" not in payload["model"]


def test_analyze_allows_configured_cors_origin() -> None:
    service = FakeService()
    settings = ApiSettings(cors_allowed_origins=("http://localhost:5173",))
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(
            client,
            "sample.wav",
            b"audio",
            headers={"Origin": "http://localhost:5173"},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_analyze_does_not_allow_unconfigured_cors_origin() -> None:
    service = FakeService()
    settings = ApiSettings(cors_allowed_origins=("http://localhost:5173",))
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(
            client,
            "sample.wav",
            b"audio",
            headers={"Origin": "http://127.0.0.1:5173"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_analyze_missing_file_returns_422() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.post("/api/v1/analyze")

    assert response.status_code == 422


def test_analyze_rejects_extensionless_filename() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "audio", b"audio")

    assert response.status_code == 415
    assert response.json()["detail"] == {
        "code": "unsupported_file_type",
        "message": "Unsupported audio file type",
    }
    assert service.calls == 0


@pytest.mark.parametrize("filename", ["", None])
def test_upload_extension_validation_rejects_empty_filename(filename: str | None) -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_extension(filename, (".wav",))

    assert getattr(exc_info.value, "status_code") == 415
    assert exc_info.value.detail == {
        "code": "unsupported_file_type",
        "message": "Unsupported audio file type",
    }


def test_upload_extension_validation_recognizes_mp3() -> None:
    settings = ApiSettings()

    assert validate_upload_extension("sample.mp3", settings.allowed_audio_extensions) == ".mp3"


def test_analyze_rejects_disallowed_extension() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.flac", b"audio", content_type="audio/flac")

    assert response.status_code == 415
    assert service.calls == 0


def test_analyze_accepts_mp3_with_mpeg_mime_type() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.mp3", b"audio", content_type="audio/mpeg")

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_rejects_disallowed_mime_type_without_prediction() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio", content_type="text/plain")

    assert response.status_code == 415
    assert response.json()["detail"] == {
        "code": "unsupported_media_type",
        "message": "Unsupported audio media type",
    }
    assert service.calls == 0


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("sample.wav", "audio/mpeg"),
        ("sample.mp3", "audio/wav"),
    ],
)
def test_analyze_rejects_mime_type_for_different_supported_extension(
    filename: str,
    content_type: str,
) -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, filename, b"audio", content_type=content_type)

    assert response.status_code == 415
    assert response.json()["detail"] == {
        "code": "unsupported_media_type",
        "message": "Unsupported audio media type",
    }
    assert service.calls == 0


@pytest.mark.parametrize("content_type", [None, ""])
def test_analyze_allows_missing_mime_type_when_extension_is_valid(
    content_type: str | None,
) -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio", content_type=content_type)

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_accepts_extension_case_insensitively() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.WAV", b"audio")

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_rejects_empty_file_without_prediction_or_temp_file(tmp_path: Path) -> None:
    service = FakeService()
    settings = ApiSettings(temp_dir=tmp_path)
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"")

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "empty_file",
        "message": "Uploaded file is empty",
    }
    assert service.calls == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [b"1234", b"12345"])
def test_analyze_accepts_file_at_or_below_size_limit(payload: bytes) -> None:
    service = FakeService()
    settings = ApiSettings(max_upload_size_bytes=5)
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", payload)

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_rejects_file_above_size_limit_without_prediction_or_temp_file(
    tmp_path: Path,
) -> None:
    service = FakeService()
    settings = ApiSettings(max_upload_size_bytes=5, temp_dir=tmp_path)
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"123456")

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "code": "file_too_large",
        "message": "Uploaded file is too large",
    }
    assert service.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_analyze_accepts_audio_at_duration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService()
    settings = ApiSettings(max_audio_duration_seconds=5.0)
    application = create_app(service_factory=lambda: service, settings=settings)
    monkeypatch.setattr("app.api.routes.get_audio_duration_seconds", lambda path: 5.0)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_rejects_audio_above_duration_limit_without_prediction_or_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = FakeService()
    settings = ApiSettings(max_audio_duration_seconds=5.0, temp_dir=tmp_path)
    application = create_app(service_factory=lambda: service, settings=settings)
    monkeypatch.setattr("app.api.routes.get_audio_duration_seconds", lambda path: 5.1)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "code": "audio_too_long",
        "message": "Audio duration exceeds the allowed limit",
    }
    assert service.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_upload_copy_failure_leaves_no_temp_file(tmp_path: Path) -> None:
    upload = BrokenUploadFile("sample.wav")
    settings = ApiSettings(temp_dir=tmp_path)

    with pytest.raises(OSError, match="read failed"):
        with uploaded_audio_path(upload, settings):
            pass

    assert upload.file.closed is True
    assert list(tmp_path.iterdir()) == []


def test_analyze_temp_file_exists_and_is_closed_during_prediction() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 200
    assert service.exists_during_predict is True
    assert service.can_open_during_predict is True


def test_analyze_removes_temp_file_after_success() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 200
    assert service.received_paths
    assert not service.received_paths[0].exists()


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (AudioDecodingError("decode failed C:/private/audio.wav"), 422),
        (AudioValidationError("validation failed C:/private/audio.wav"), 422),
        (PredictionError("prediction failed C:/private/audio.wav"), 500),
    ],
)
def test_analyze_removes_temp_file_after_inference_errors(
    error: Exception,
    status_code: int,
) -> None:
    service = FakeService(error=error)
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == status_code
    assert service.received_paths
    assert not service.received_paths[0].exists()
    assert "C:/private/audio.wav" not in response.text


def test_analyze_removes_temp_file_after_unexpected_error() -> None:
    service = UnexpectedErrorService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 500
    assert service.received_paths
    assert not service.received_paths[0].exists()
    assert "C:/private/audio.wav" not in response.text


def test_analyze_model_unavailable_returns_503_without_prediction() -> None:
    calls = {"count": 0}

    def service_factory() -> FakeService:
        calls["count"] += 1
        raise ModelArtifactError("missing C:/private/model.joblib")

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 503
    assert calls["count"] == 1
    assert "C:/private/model.joblib" not in response.text


def test_analyze_response_does_not_expose_internal_paths() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    body = response.text
    assert response.status_code == 200
    assert "loaded_path" not in body
    assert "MODEL_PATH" not in body
    assert "C:/private/model.joblib" not in body


def test_analyze_invokes_predict_file_exactly_once() -> None:
    service = FakeService()
    application = create_app(service_factory=lambda: service)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 200
    assert service.calls == 1


def test_analyze_uses_configured_temp_dir_and_removes_file(tmp_path: Path) -> None:
    service = FakeService()
    settings = ApiSettings(temp_dir=tmp_path)
    application = create_app(service_factory=lambda: service, settings=settings)

    with TestClient(application) as client:
        response = _post_file(client, "sample.wav", b"audio")

    assert response.status_code == 200
    assert service.parent_during_predict == tmp_path
    assert service.received_paths
    assert not service.received_paths[0].exists()


def _post_file(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str | None = "audio/wav",
    headers: dict[str, str] | None = None,
):
    if content_type is None:
        return client.post(
            "/api/v1/analyze",
            files={"file": (filename, content)},
            headers=headers,
        )
    return client.post(
        "/api/v1/analyze",
        files={"file": (filename, content, content_type)},
        headers=headers,
    )


def _prediction_result() -> PredictionResult:
    metadata = FakeModel.metadata
    return PredictionResult(
        predicted_label=1,
        predicted_class="ai_generated",
        ai_score=0.42,
        decision_threshold=0.0,
        audio_duration_seconds=1.5,
        original_sample_rate=44_100,
        n_fragments=1,
        fragments=(
            FragmentPrediction(
                index=0,
                start_seconds=0.0,
                end_seconds=1.5,
                duration_seconds=1.5,
                ai_score=0.42,
                predicted_label=1,
                predicted_class="ai_generated",
                was_padded=True,
            ),
        ),
        timings=InferenceTimings(
            decode_seconds=0.1,
            segmentation_seconds=0.1,
            preprocessing_seconds=0.1,
            mfcc_seconds=0.1,
            prediction_seconds=0.1,
            aggregation_seconds=0.1,
            total_seconds=0.7,
        ),
        model=metadata,
        usage_warning=FakeConfig.usage_warning,
    )
