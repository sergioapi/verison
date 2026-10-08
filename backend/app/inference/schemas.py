from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioFragment:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    signal: np.ndarray
    sample_rate: int
    is_incomplete: bool


@dataclass(frozen=True)
class AudioFragmentMetadata:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    is_incomplete: bool


@dataclass(frozen=True)
class FragmentPrediction:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    ai_score: float
    predicted_label: int
    predicted_class: str
    was_padded: bool


@dataclass(frozen=True)
class InferenceTimings:
    decode_seconds: float
    segmentation_seconds: float
    preprocessing_seconds: float
    mfcc_seconds: float
    prediction_seconds: float
    aggregation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class WarmupResult:
    name: str
    succeeded: bool
    duration_seconds: float
    error_type: str | None = None


@dataclass(frozen=True)
class StartupWarmupResult:
    resampling: WarmupResult
    mfcc: WarmupResult

    @property
    def succeeded(self) -> bool:
        return self.resampling.succeeded and self.mfcc.succeeded


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    sha256: str
    loaded_path: Path
    classes: tuple[int, ...]
    positive_label: int
    score_type: str
    target_sample_rate: int
    fragment_duration_seconds: float
    n_mfcc: int
    n_features: int
    decision_threshold: float
    aggregation_strategy: str
    score_is_calibrated_probability: bool


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: int
    predicted_class: str
    ai_score: float
    decision_threshold: float
    audio_duration_seconds: float
    original_sample_rate: int
    n_fragments: int
    fragments: tuple[FragmentPrediction, ...]
    timings: InferenceTimings
    model: ModelMetadata
    usage_warning: str
