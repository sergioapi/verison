from __future__ import annotations

import logging
import time

import librosa
import numpy as np

from app.inference.config import InferenceConfig
from app.inference.errors import PredictionError
from app.inference.schemas import WarmupResult


logger = logging.getLogger("uvicorn.error")


def extract_mfcc_features(
    signal: np.ndarray,
    sample_rate: int,
    config: InferenceConfig,
) -> np.ndarray:
    if sample_rate != config.target_sample_rate:
        raise PredictionError(
            f"MFCC extraction expects {config.target_sample_rate} Hz, found {sample_rate}"
        )
    audio = np.asarray(signal, dtype=np.float32)
    if audio.ndim != 1:
        raise PredictionError(f"Expected preprocessed mono audio, found shape {audio.shape}")
    if audio.size == 0:
        raise PredictionError("Preprocessed audio is empty")
    _validate_finite(audio, "preprocessed audio")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
    )
    features = np.concatenate(
        [
            mfcc.mean(axis=1, dtype=np.float64),
            mfcc.std(axis=1, dtype=np.float64),
        ]
    ).astype(np.float32)
    expected_shape = (config.n_mfcc * 2,)
    if features.shape != expected_shape:
        raise PredictionError(
            f"Expected {expected_shape[0]} MFCC features, found shape {features.shape}"
        )
    if features.dtype != np.float32:
        raise PredictionError(f"Expected float32 MFCC features, found {features.dtype}")
    _validate_finite(features, "MFCC features")
    return features


def warm_up_mfcc(config: InferenceConfig) -> WarmupResult:
    """Exercise the production MFCC extraction route once with synthetic audio."""
    _log_warmup("started")
    total_start = time.perf_counter()
    try:
        signal = np.zeros(config.target_samples, dtype=np.float32)
        extract_mfcc_features(signal, config.target_sample_rate, config)
    except Exception as exc:  # noqa: BLE001 - warm-up must not break startup.
        duration_seconds = _elapsed(total_start)
        _log_warmup(
            "failed",
            error_type=type(exc).__name__,
            duration_seconds=duration_seconds,
        )
        return WarmupResult(
            name="mfcc",
            succeeded=False,
            duration_seconds=duration_seconds,
            error_type=type(exc).__name__,
        )

    duration_seconds = _elapsed(total_start)
    _log_warmup("completed", duration_seconds=duration_seconds)
    return WarmupResult(
        name="mfcc",
        succeeded=True,
        duration_seconds=duration_seconds,
    )


def feature_columns(config: InferenceConfig | None = None) -> tuple[str, ...]:
    config = config or InferenceConfig()
    return tuple(f"mfcc_mean_{index:02d}" for index in range(config.n_mfcc)) + tuple(
        f"mfcc_std_{index:02d}" for index in range(config.n_mfcc)
    )


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise PredictionError(f"{name} contains NaN or infinite values")


def _log_warmup(
    status: str,
    *,
    error_type: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    try:
        fields = [f"mfcc_warmup status={status}"]
        if error_type is not None:
            fields.append(f"error_type={error_type}")
        if duration_seconds is not None:
            fields.append(f"total_seconds={duration_seconds:.4f}")
        if status == "failed":
            logger.warning(" ".join(fields))
        else:
            logger.info(" ".join(fields))
    except Exception:  # noqa: BLE001 - diagnostic logging must not break startup.
        return


def _elapsed(start: float) -> float:
    return max(0.0, time.perf_counter() - start)
