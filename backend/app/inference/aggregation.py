from __future__ import annotations

import numpy as np

from app.inference.errors import PredictionError


AGGREGATION_STRATEGY = "duration_weighted_mean_decision_score"


def aggregate_duration_weighted_scores(
    scores: np.ndarray,
    durations_seconds: np.ndarray,
) -> float:
    score_values = np.asarray(scores, dtype=np.float64)
    duration_values = np.asarray(durations_seconds, dtype=np.float64)

    if score_values.ndim != 1:
        raise PredictionError(f"Expected a 1D score vector, found shape {score_values.shape}")
    if duration_values.ndim != 1:
        raise PredictionError(
            f"Expected a 1D duration vector, found shape {duration_values.shape}"
        )
    if score_values.size == 0:
        raise PredictionError("Cannot aggregate zero fragment scores")
    if score_values.shape != duration_values.shape:
        raise PredictionError(
            "Scores and durations must have the same length: "
            f"{score_values.size} scores, {duration_values.size} durations"
        )
    if not np.isfinite(score_values).all():
        raise PredictionError("Fragment scores contain NaN or infinite values")
    if not np.isfinite(duration_values).all():
        raise PredictionError("Fragment durations contain NaN or infinite values")
    if not np.all(duration_values > 0.0):
        raise PredictionError("Fragment durations must be greater than zero")

    total_duration = float(np.sum(duration_values, dtype=np.float64))
    if not np.isfinite(total_duration) or total_duration <= 0.0:
        raise PredictionError("Total fragment duration must be greater than zero")

    weighted_sum = float(np.sum(score_values * duration_values, dtype=np.float64))
    result = weighted_sum / total_duration
    if not np.isfinite(result):
        raise PredictionError("Aggregated decision score is not finite")
    return float(result)
