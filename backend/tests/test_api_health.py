from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from app.config import ApiSettings
from app.inference.config import InferenceConfig
from app.inference.errors import ModelArtifactError
from app.inference.schemas import StartupWarmupResult, WarmupResult
from app.main import create_app


class FakeService:
    config = InferenceConfig()

    def predict_file(self, path):  # pragma: no cover - must not be called by /health.
        raise AssertionError("/health must not run inference")


def test_health_reports_ok_when_service_loads() -> None:
    calls = {"count": 0}

    def service_factory() -> FakeService:
        calls["count"] += 1
        return FakeService()

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = client.get("/health")

    assert calls["count"] == 1
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True}


def test_health_reports_degraded_when_model_artifact_load_fails() -> None:
    def service_factory() -> FakeService:
        raise ModelArtifactError("Could not load C:/private/model.joblib")

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 503
    assert payload == {"status": "degraded", "model_ready": False}
    assert "Could not load" not in response.text
    assert "C:/private/model.joblib" not in response.text


def test_ready_reports_model_unavailable_without_internal_error_details() -> None:
    def service_factory() -> FakeService:
        raise ModelArtifactError("Could not load C:/private/model.joblib")

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")

    assert health_response.status_code == 503
    assert ready_response.status_code == 503
    assert ready_response.json() == {"status": "unavailable"}
    assert "Could not load" not in ready_response.text
    assert "C:/private/model.joblib" not in ready_response.text


def test_health_does_not_call_predict_file() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_ready_is_immediate_when_startup_warmups_are_disabled() -> None:
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=False),
    )

    with TestClient(application) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_startup_warmups_are_not_scheduled_when_disabled(monkeypatch) -> None:
    calls = {"count": 0}

    def warmup(*args) -> StartupWarmupResult:
        calls["count"] += 1
        raise AssertionError("Warm-ups must be disabled")

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=False),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200

    assert calls["count"] == 0


def test_startup_warmups_run_once_in_background_when_enabled(monkeypatch) -> None:
    calls = {"count": 0}
    completed = threading.Event()

    result = StartupWarmupResult(
        resampling=WarmupResult("resampling", True, 0.1),
        mfcc=WarmupResult("mfcc", True, 0.2),
    )

    def warmup(*args) -> StartupWarmupResult:
        calls["count"] += 1
        completed.set()
        return result

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert completed.wait(timeout=1)
        response = _wait_for_ready(client)

    assert calls["count"] == 1
    assert application.state.startup_warmup_result is result
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_is_pending_while_startup_warmups_run(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def warmup(*args) -> StartupWarmupResult:
        started.set()
        assert release.wait(timeout=1)
        return StartupWarmupResult(
            resampling=WarmupResult("resampling", True, 0.1),
            mfcc=WarmupResult("mfcc", True, 0.2),
        )

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert started.wait(timeout=1)
        health_response = client.get("/health")
        ready_response = client.get("/ready")
        release.set()

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    assert ready_response.json() == {"status": "pending"}


def test_startup_warmup_failure_does_not_degrade_health(monkeypatch) -> None:
    completed = threading.Event()
    result = StartupWarmupResult(
        resampling=WarmupResult("resampling", False, 0.1, "RuntimeError"),
        mfcc=WarmupResult("mfcc", False, 0.2, "RuntimeError"),
    )

    def warmup(*args) -> StartupWarmupResult:
        completed.set()
        return result

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert completed.wait(timeout=1)
        response = _wait_for_ready(client)

    assert application.state.startup_warmup_result is result
    assert response.status_code == 503
    assert response.json() == {"status": "failed"}


def test_ready_allows_configured_origin_and_get_only() -> None:
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(
            cors_allowed_origins=("https://verison-app.vercel.app",),
        ),
    )

    with TestClient(application) as client:
        response = client.get(
            "/ready",
            headers={"Origin": "https://verison-app.vercel.app"},
        )
        preflight = client.options(
            "/ready",
            headers={
                "Origin": "https://verison-app.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://verison-app.vercel.app"
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-methods"] == "GET, POST"


def _wait_for_ready(client: TestClient):
    for _ in range(20):
        response = client.get("/ready")
        if response.status_code != 503 or response.json() != {"status": "pending"}:
            return response
        time.sleep(0.01)
    return response
