from __future__ import annotations

import numpy as np
import pytest

from app.inference.aggregation import aggregate_duration_weighted_scores
from app.inference.errors import PredictionError


def test_single_complete_fragment_returns_its_score() -> None:
    score = aggregate_duration_weighted_scores(
        np.array([1.25]),
        np.array([10.0]),
    )

    assert score == pytest.approx(1.25)


def test_single_incomplete_fragment_returns_its_score() -> None:
    score = aggregate_duration_weighted_scores(
        np.array([-0.75]),
        np.array([0.23]),
    )

    assert score == pytest.approx(-0.75)


def test_two_complete_fragments_equal_arithmetic_mean() -> None:
    scores = np.array([-1.0, 3.0])
    durations = np.array([10.0, 10.0])

    score = aggregate_duration_weighted_scores(scores, durations)

    assert score == pytest.approx(np.mean(scores))


def test_short_final_fragment_uses_real_duration_not_equal_weight() -> None:
    scores = np.array([-1.2088, 3.6799])
    durations = np.array([10.0, 0.23])

    score = aggregate_duration_weighted_scores(scores, durations)

    expected = (scores[0] * 10.0 + scores[1] * 0.23) / 10.23
    arithmetic = np.mean(scores)
    assert score == pytest.approx(expected)
    assert score != pytest.approx(arithmetic)


def test_three_complete_fragments_and_short_final_fragment() -> None:
    scores = np.array([0.1476, -0.0230, -0.3115, 3.3368])
    durations = np.array([10.0, 10.0, 10.0, 2.832])

    score = aggregate_duration_weighted_scores(scores, durations)

    expected = float(np.sum(scores * durations) / np.sum(durations))
    assert score == pytest.approx(expected)


def test_mismatched_lengths_raise_prediction_error() -> None:
    with pytest.raises(PredictionError, match="same length"):
        aggregate_duration_weighted_scores(np.array([1.0, 2.0]), np.array([10.0]))


def test_zero_fragments_raise_prediction_error() -> None:
    with pytest.raises(PredictionError, match="zero fragment"):
        aggregate_duration_weighted_scores(np.array([]), np.array([]))


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_non_finite_scores_raise_prediction_error(bad_score: float) -> None:
    with pytest.raises(PredictionError, match="scores contain NaN or infinite"):
        aggregate_duration_weighted_scores(np.array([bad_score]), np.array([10.0]))


@pytest.mark.parametrize("bad_duration", [0.0, -1.0])
def test_non_positive_durations_raise_prediction_error(bad_duration: float) -> None:
    with pytest.raises(PredictionError, match="greater than zero"):
        aggregate_duration_weighted_scores(np.array([1.0]), np.array([bad_duration]))


@pytest.mark.parametrize("bad_duration", [np.nan, np.inf, -np.inf])
def test_non_finite_durations_raise_prediction_error(bad_duration: float) -> None:
    with pytest.raises(PredictionError, match="durations contain NaN or infinite"):
        aggregate_duration_weighted_scores(np.array([1.0]), np.array([bad_duration]))
