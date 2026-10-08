from __future__ import annotations

import numpy as np
import pytest

from app.inference.config import InferenceConfig
from app.inference.errors import PredictionError
from app.inference.features import extract_mfcc_features, feature_columns, warm_up_mfcc


def test_extract_mfcc_features_shape_dtype_and_finiteness(config: InferenceConfig) -> None:
    signal = np.sin(
        2 * np.pi * 440 * np.arange(config.target_samples, dtype=np.float32) / config.target_sample_rate
    ).astype(np.float32)

    features = extract_mfcc_features(signal, config.target_sample_rate, config)

    assert features.shape == (40,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_columns_has_exact_order(config: InferenceConfig) -> None:
    columns = feature_columns(config)

    assert len(columns) == 40
    assert columns[:3] == ("mfcc_mean_00", "mfcc_mean_01", "mfcc_mean_02")
    assert columns[19] == "mfcc_mean_19"
    assert columns[20] == "mfcc_std_00"
    assert columns[-1] == "mfcc_std_19"


def test_extract_mfcc_rejects_wrong_shape(config: InferenceConfig) -> None:
    signal = np.zeros((config.target_samples, 2), dtype=np.float32)

    with pytest.raises(PredictionError, match="Expected preprocessed mono"):
        extract_mfcc_features(signal, config.target_sample_rate, config)


def test_extract_mfcc_rejects_non_finite_signal(config: InferenceConfig) -> None:
    signal = np.zeros(config.target_samples, dtype=np.float32)
    signal[0] = np.nan

    with pytest.raises(PredictionError, match="NaN or infinite"):
        extract_mfcc_features(signal, config.target_sample_rate, config)


def test_extract_mfcc_rejects_unexpected_sample_rate(config: InferenceConfig) -> None:
    signal = np.zeros(config.target_samples, dtype=np.float32)

    with pytest.raises(PredictionError, match="expects 16000 Hz"):
        extract_mfcc_features(signal, 8_000, config)


def test_mfcc_warmup_reuses_production_extractor_with_mono_float32_signal(
    monkeypatch,
    config: InferenceConfig,
) -> None:
    call: dict[str, object] = {}

    def extract(signal, sample_rate, received_config) -> np.ndarray:
        call["signal"] = signal
        call["sample_rate"] = sample_rate
        call["config"] = received_config
        return np.zeros(config.n_mfcc * 2, dtype=np.float32)

    monkeypatch.setattr("app.inference.features.extract_mfcc_features", extract)

    result = warm_up_mfcc(config)

    signal = call["signal"]
    assert result.succeeded is True
    assert result.name == "mfcc"
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert signal.dtype == np.float32
    assert signal.shape == (config.target_samples,)
    assert call["sample_rate"] == config.target_sample_rate == 16_000
    assert call["config"] is config


def test_mfcc_warmup_reports_failure_without_raising(monkeypatch, config) -> None:
    def fail_extract(*args, **kwargs) -> np.ndarray:
        raise RuntimeError("MFCC warm-up failed")

    monkeypatch.setattr("app.inference.features.extract_mfcc_features", fail_extract)

    result = warm_up_mfcc(config)

    assert result.succeeded is False
    assert result.error_type == "RuntimeError"


def test_mfcc_warmup_ignores_diagnostic_logging_failures(monkeypatch, config) -> None:
    def fail_logging(*args, **kwargs) -> None:
        raise RuntimeError("logging failed")

    monkeypatch.setattr(
        "app.inference.features.logger.info",
        fail_logging,
    )
    monkeypatch.setattr(
        "app.inference.features.extract_mfcc_features",
        lambda *args, **kwargs: np.zeros(config.n_mfcc * 2, dtype=np.float32),
    )

    result = warm_up_mfcc(config)

    assert result.succeeded is True
