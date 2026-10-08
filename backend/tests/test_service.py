from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.api.mappers import analyze_response
from app.inference.aggregation import AGGREGATION_STRATEGY, aggregate_duration_weighted_scores
from app.inference.audio import decode_audio_file, preprocess_fragment, segment_audio
from app.inference.config import class_name_for_label
from app.inference.errors import ModelArtifactError
from app.inference.features import extract_mfcc_features
from app.inference.model import predict_labels_from_scores
from app.inference.schemas import FragmentPrediction, InferenceTimings, PredictionResult
from app.inference.service import AudioInferenceService


def sine(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_constructor_accepts_explicit_path(artifact_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())

    assert service.model.metadata.model_id == "mfcc-svm-baseline"


def test_constructor_reads_model_path_environment(monkeypatch: pytest.MonkeyPatch, artifact_factory) -> None:
    path = artifact_factory()
    monkeypatch.setenv("MODEL_PATH", str(path))

    service = AudioInferenceService()

    assert service.model.path == path.resolve()


def test_explicit_path_has_priority_over_environment(
    monkeypatch: pytest.MonkeyPatch,
    artifact_factory,
) -> None:
    env_path = artifact_factory(positive_label=0)
    explicit_path = artifact_factory()
    monkeypatch.setenv("MODEL_PATH", str(env_path))

    service = AudioInferenceService(model_path=explicit_path)

    assert service.model.path == explicit_path.resolve()


def test_constructor_errors_if_model_path_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(ModelArtifactError, match="MODEL_PATH"):
        AudioInferenceService()


def test_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch, artifact_factory) -> None:
    calls = {"count": 0}
    original_load = AudioInferenceService.__init__.__globals__["MfccSvmModel"].load

    def counting_load(path, config):
        calls["count"] += 1
        return original_load(path, config)

    monkeypatch.setattr(
        "app.inference.service.MfccSvmModel.load",
        staticmethod(counting_load),
    )
    service = AudioInferenceService(model_path=artifact_factory())

    assert calls["count"] == 1
    assert service.model is not None


def test_predict_short_audio_returns_one_fragment(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(3.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 1
    assert result.fragments[0].was_padded is True
    assert result.audio_duration_seconds == pytest.approx(3.0, abs=1e-3)


def test_predict_exactly_ten_seconds_returns_one_fragment(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(10.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 1
    assert result.fragments[0].was_padded is False


def test_predict_multiple_fragments_returns_ordered_results(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 3
    starts = [fragment.start_seconds for fragment in result.fragments]
    assert starts == sorted(starts)
    assert result.fragments[-1].was_padded is True


def test_global_score_is_duration_weighted_mean_of_fragment_scores(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)

    scores = np.array([fragment.ai_score for fragment in result.fragments])
    durations = np.array([fragment.duration_seconds for fragment in result.fragments])
    expected = np.sum(scores * durations) / np.sum(durations)
    assert result.ai_score == pytest.approx(expected)


def test_global_label_and_class_follow_threshold(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(10.0), 16_000)

    result = service.predict_file(path)

    expected_label = 1 if result.ai_score >= 0.0 else 0
    assert result.predicted_label == expected_label
    expected_class = "ai_generated" if expected_label == 1 else "human"
    assert result.predicted_class == expected_class


def test_result_contains_metadata_warning_timings_and_original_audio_info(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(1.0, 8_000), 8_000)

    result = service.predict_file(path)

    assert result.model.model_id == "mfcc-svm-baseline"
    assert result.model.aggregation_strategy == AGGREGATION_STRATEGY
    assert result.model.score_is_calibrated_probability is False
    assert "no es una probabilidad calibrada" in result.usage_warning
    assert "no constituye un veredicto forense" in result.usage_warning
    assert result.original_sample_rate == 8_000
    assert result.audio_duration_seconds == pytest.approx(1.0, abs=1e-3)
    assert all(value >= 0.0 for value in result.timings.__dict__.values())


def test_fragment_results_are_chronological(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(35.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 4
    assert [fragment.index for fragment in result.fragments] == [0, 1, 2, 3]
    assert [fragment.start_seconds for fragment in result.fragments] == pytest.approx(
        [0.0, 10.0, 20.0, 30.0]
    )


def test_weighted_aggregation_does_not_change_individual_scores(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)
    fragment_scores = np.array([fragment.ai_score for fragment in result.fragments])
    labels_from_scores = np.where(fragment_scores >= result.decision_threshold, 1, 0)

    assert [fragment.predicted_label for fragment in result.fragments] == labels_from_scores.tolist()
    assert result.fragments[-1].duration_seconds == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("seconds", "sample_rate", "stereo"),
    [
        (3.0, 16_000, False),
        (10.0, 44_100, True),
        (21.0, 48_000, False),
        (21.0, 48_000, True),
    ],
)
def test_streaming_service_matches_full_read_baseline(
    artifact_factory,
    wav_factory,
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    sample_rate: int,
    stereo: bool,
) -> None:
    signal = sine(seconds, sample_rate)
    if stereo:
        signal = np.column_stack([signal, signal * 0.5])
    path = wav_factory(signal, sample_rate)
    service = AudioInferenceService(model_path=artifact_factory())
    expected, expected_features, expected_scores = _full_read_baseline(service, path)
    captured_features: dict[str, np.ndarray] = {}
    original_predict_scores = service.model.predict_scores

    def capture_predict_scores(features: np.ndarray) -> np.ndarray:
        captured_features["value"] = features.copy()
        return original_predict_scores(features)

    monkeypatch.setattr(service.model, "predict_scores", capture_predict_scores)
    actual = service.predict_file(path)

    np.testing.assert_allclose(captured_features["value"], expected_features, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        [fragment.ai_score for fragment in actual.fragments],
        expected_scores,
        rtol=1e-7,
        atol=1e-7,
    )
    assert actual.predicted_label == expected.predicted_label
    assert actual.predicted_class == expected.predicted_class
    assert actual.ai_score == pytest.approx(expected.ai_score, abs=1e-7)
    assert actual.audio_duration_seconds == pytest.approx(expected.audio_duration_seconds)
    assert actual.original_sample_rate == expected.original_sample_rate
    assert actual.n_fragments == expected.n_fragments
    for observed, reference in zip(actual.fragments, expected.fragments):
        assert observed.index == reference.index
        assert observed.start_seconds == pytest.approx(reference.start_seconds)
        assert observed.end_seconds == pytest.approx(reference.end_seconds)
        assert observed.duration_seconds == pytest.approx(reference.duration_seconds)
        assert observed.was_padded is reference.was_padded
        assert observed.predicted_label == reference.predicted_label
        assert observed.predicted_class == reference.predicted_class

    actual_payload = analyze_response(actual).model_dump()
    expected_payload = analyze_response(expected).model_dump()
    assert actual_payload.keys() == expected_payload.keys()
    assert actual_payload.pop("timings").keys() == expected_payload.pop("timings").keys()
    assert actual_payload["model"] == expected_payload["model"]
    assert actual_payload["usage_warning"] == expected_payload["usage_warning"]
    assert actual_payload["decision_threshold"] == expected_payload["decision_threshold"]
    assert [fragment.keys() for fragment in actual_payload["fragments"]] == [
        fragment.keys() for fragment in expected_payload["fragments"]
    ]


def test_streaming_service_uses_decoded_frames_when_metadata_overstates_length(
    artifact_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ShortDecoderSource:
        samplerate = 16_000
        frames = 400_000

        def __init__(self) -> None:
            self.blocks = iter(
                (
                    np.ones(160_000, dtype=np.float64),
                    np.ones(160_000, dtype=np.float64),
                    np.ones(48_000, dtype=np.float64),
                    np.array([], dtype=np.float64),
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self, *, frames: int, dtype: str, always_2d: bool) -> np.ndarray:
            assert frames == 160_000
            assert dtype == "float64"
            assert always_2d is False
            return next(self.blocks)

    path = tmp_path / "metadata-overstates.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "app.inference.audio.sf.SoundFile",
        lambda _: ShortDecoderSource(),
    )
    service = AudioInferenceService(model_path=artifact_factory())

    result = service.predict_file(path)

    assert result.n_fragments == 3
    assert result.audio_duration_seconds == pytest.approx(23.0)
    assert [fragment.duration_seconds for fragment in result.fragments] == pytest.approx(
        [10.0, 10.0, 3.0]
    )
    assert result.fragments[-1].was_padded is True


def _full_read_baseline(
    service: AudioInferenceService,
    path: Path,
) -> tuple[PredictionResult, np.ndarray, np.ndarray]:
    audio, sample_rate = decode_audio_file(path)
    fragments = segment_audio(audio, sample_rate, service.config.fragment_duration_seconds)
    feature_rows = tuple(
        extract_mfcc_features(
            preprocess_fragment(fragment.signal, fragment.sample_rate, service.config),
            service.config.target_sample_rate,
            service.config,
        )
        for fragment in fragments
    )
    features = np.vstack(feature_rows).astype(np.float32, copy=False)
    scores = service.model.predict_scores(features)
    labels = predict_labels_from_scores(scores, threshold=service.config.decision_threshold)
    durations = np.asarray([fragment.duration_seconds for fragment in fragments], dtype=np.float64)
    global_score = aggregate_duration_weighted_scores(scores, durations)
    global_label = int(
        predict_labels_from_scores(
            np.asarray([global_score], dtype=np.float64),
            threshold=service.config.decision_threshold,
        )[0]
    )
    predictions = tuple(
        FragmentPrediction(
            index=fragment.index,
            start_seconds=fragment.start_seconds,
            end_seconds=fragment.end_seconds,
            duration_seconds=fragment.duration_seconds,
            ai_score=float(score),
            predicted_label=int(label),
            predicted_class=class_name_for_label(int(label)),
            was_padded=fragment.is_incomplete,
        )
        for fragment, score, label in zip(fragments, scores, labels)
    )
    result = PredictionResult(
        predicted_label=global_label,
        predicted_class=class_name_for_label(global_label),
        ai_score=global_score,
        decision_threshold=service.config.decision_threshold,
        audio_duration_seconds=audio.shape[0] / sample_rate,
        original_sample_rate=sample_rate,
        n_fragments=len(predictions),
        fragments=predictions,
        timings=InferenceTimings(
            decode_seconds=0.0,
            segmentation_seconds=0.0,
            preprocessing_seconds=0.0,
            mfcc_seconds=0.0,
            prediction_seconds=0.0,
            aggregation_seconds=0.0,
            total_seconds=0.0,
        ),
        model=service.metadata,
        usage_warning=service.usage_warning,
    )
    return result, features, scores
