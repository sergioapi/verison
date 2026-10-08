from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool


class ReadinessResponse(BaseModel):
    status: Literal["unavailable", "pending", "ready", "failed"]


class ModelInfoResponse(BaseModel):
    model_id: str
    sha256: str
    classes: tuple[int, ...]
    positive_label: int
    score_type: str
    score_is_calibrated_probability: bool
    decision_threshold: float
    target_sample_rate: int
    fragment_duration_seconds: float
    n_mfcc: int
    n_features: int
    aggregation_strategy: str
    usage_warning: str


class AnalyzeModelResponse(BaseModel):
    model_id: str
    sha256: str
    classes: tuple[int, ...]
    positive_label: int
    score_type: str
    score_is_calibrated_probability: bool
    target_sample_rate: int
    fragment_duration_seconds: float
    n_mfcc: int
    n_features: int
    aggregation_strategy: str


class FragmentPredictionResponse(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    ai_score: float
    predicted_label: int
    predicted_class: str
    was_padded: bool


class InferenceTimingsResponse(BaseModel):
    decode_seconds: float
    segmentation_seconds: float
    preprocessing_seconds: float
    mfcc_seconds: float
    prediction_seconds: float
    aggregation_seconds: float
    total_seconds: float


class AnalyzeResponse(BaseModel):
    predicted_label: int
    predicted_class: str
    ai_score: float
    decision_threshold: float
    audio_duration_seconds: float
    original_sample_rate: int
    n_fragments: int
    fragments: tuple[FragmentPredictionResponse, ...]
    timings: InferenceTimingsResponse
    model: AnalyzeModelResponse
    usage_warning: str
