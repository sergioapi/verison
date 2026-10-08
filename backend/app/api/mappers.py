from __future__ import annotations

from app.api.schemas import (
    AnalyzeModelResponse,
    AnalyzeResponse,
    FragmentPredictionResponse,
    InferenceTimingsResponse,
    ModelInfoResponse,
)
from app.inference.schemas import ModelMetadata, PredictionResult


def model_info_response(metadata: ModelMetadata, usage_warning: str) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_id=metadata.model_id,
        sha256=metadata.sha256,
        classes=metadata.classes,
        positive_label=metadata.positive_label,
        score_type=metadata.score_type,
        score_is_calibrated_probability=metadata.score_is_calibrated_probability,
        decision_threshold=metadata.decision_threshold,
        target_sample_rate=metadata.target_sample_rate,
        fragment_duration_seconds=metadata.fragment_duration_seconds,
        n_mfcc=metadata.n_mfcc,
        n_features=metadata.n_features,
        aggregation_strategy=metadata.aggregation_strategy,
        usage_warning=usage_warning,
    )


def analyze_response(prediction: PredictionResult) -> AnalyzeResponse:
    metadata = prediction.model
    return AnalyzeResponse(
        predicted_label=prediction.predicted_label,
        predicted_class=prediction.predicted_class,
        ai_score=prediction.ai_score,
        decision_threshold=prediction.decision_threshold,
        audio_duration_seconds=prediction.audio_duration_seconds,
        original_sample_rate=prediction.original_sample_rate,
        n_fragments=prediction.n_fragments,
        fragments=tuple(
            FragmentPredictionResponse(
                index=fragment.index,
                start_seconds=fragment.start_seconds,
                end_seconds=fragment.end_seconds,
                duration_seconds=fragment.duration_seconds,
                ai_score=fragment.ai_score,
                predicted_label=fragment.predicted_label,
                predicted_class=fragment.predicted_class,
                was_padded=fragment.was_padded,
            )
            for fragment in prediction.fragments
        ),
        timings=InferenceTimingsResponse(
            decode_seconds=prediction.timings.decode_seconds,
            segmentation_seconds=prediction.timings.segmentation_seconds,
            preprocessing_seconds=prediction.timings.preprocessing_seconds,
            mfcc_seconds=prediction.timings.mfcc_seconds,
            prediction_seconds=prediction.timings.prediction_seconds,
            aggregation_seconds=prediction.timings.aggregation_seconds,
            total_seconds=prediction.timings.total_seconds,
        ),
        model=AnalyzeModelResponse(
            model_id=metadata.model_id,
            sha256=metadata.sha256,
            classes=metadata.classes,
            positive_label=metadata.positive_label,
            score_type=metadata.score_type,
            score_is_calibrated_probability=metadata.score_is_calibrated_probability,
            target_sample_rate=metadata.target_sample_rate,
            fragment_duration_seconds=metadata.fragment_duration_seconds,
            n_mfcc=metadata.n_mfcc,
            n_features=metadata.n_features,
            aggregation_strategy=metadata.aggregation_strategy,
        ),
        usage_warning=prediction.usage_warning,
    )
