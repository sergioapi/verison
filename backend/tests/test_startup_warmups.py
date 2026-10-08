from __future__ import annotations

from app.inference.config import InferenceConfig
from app.inference.schemas import WarmupResult
from app.inference.warmups import run_startup_warmups


def test_startup_warmups_run_resampling_before_mfcc_once(monkeypatch) -> None:
    calls: list[str] = []
    config = InferenceConfig()

    def warm_resampling():
        calls.append("resampling")
        return WarmupResult("resampling", succeeded=True, duration_seconds=0.1)

    def warm_mfcc(received_config):
        calls.append("mfcc")
        assert received_config is config
        return WarmupResult("mfcc", succeeded=True, duration_seconds=0.2)

    monkeypatch.setattr("app.inference.warmups.warm_up_resampling", warm_resampling)
    monkeypatch.setattr("app.inference.warmups.warm_up_mfcc", warm_mfcc)

    result = run_startup_warmups(config)

    assert calls == ["resampling", "mfcc"]
    assert result.succeeded is True
    assert result.resampling.name == "resampling"
    assert result.mfcc.name == "mfcc"


def test_startup_warmups_stops_after_failed_resampling(monkeypatch) -> None:
    calls: list[str] = []

    def warm_resampling():
        calls.append("resampling")
        return WarmupResult(
            "resampling",
            succeeded=False,
            duration_seconds=0.1,
            error_type="RuntimeError",
        )

    def warm_mfcc(config):
        calls.append("mfcc")
        return WarmupResult("mfcc", succeeded=True, duration_seconds=0.2)

    monkeypatch.setattr("app.inference.warmups.warm_up_resampling", warm_resampling)
    monkeypatch.setattr("app.inference.warmups.warm_up_mfcc", warm_mfcc)

    result = run_startup_warmups(InferenceConfig())

    assert calls == ["resampling"]
    assert result.succeeded is False
    assert result.resampling.error_type == "RuntimeError"
