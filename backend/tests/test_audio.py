from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.inference.audio import (
    decode_audio_file,
    get_audio_duration_seconds,
    open_audio_fragments,
    preprocess_fragment,
    segment_audio,
)
from app.inference.errors import AudioDecodingError, AudioValidationError


def sine(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_decode_rejects_missing_path() -> None:
    with pytest.raises(AudioDecodingError, match="does not exist"):
        decode_audio_file(Path("missing.wav"))


def test_decode_rejects_empty_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr("app.inference.audio.sf.read", lambda *args, **kwargs: (np.array([]), 16_000))

    with pytest.raises(AudioValidationError, match="empty"):
        decode_audio_file(path)


def test_decode_accepts_mono_audio(wav_factory) -> None:
    path = wav_factory(sine(0.2), 16_000)

    audio, sample_rate = decode_audio_file(path)

    assert sample_rate == 16_000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_decode_accepts_stereo_audio(wav_factory) -> None:
    mono = sine(0.2)
    stereo = np.column_stack([mono, mono * 0.5])
    path = wav_factory(stereo, 16_000)

    audio, sample_rate = decode_audio_file(path)

    assert sample_rate == 16_000
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_audio_duration_uses_file_metadata(wav_factory) -> None:
    path = wav_factory(sine(0.25), 16_000)

    assert get_audio_duration_seconds(path) == pytest.approx(0.25)


def test_audio_duration_rejects_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"placeholder")
    fake_info = type("FakeInfo", (), {"samplerate": 16_000, "frames": 0})()
    monkeypatch.setattr("app.inference.audio.sf.info", lambda *args, **kwargs: fake_info)

    with pytest.raises(AudioValidationError, match="empty"):
        get_audio_duration_seconds(path)


def test_decode_rejects_non_finite_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "app.inference.audio.sf.read",
        lambda *args, **kwargs: (np.array([0.0, np.nan], dtype=np.float32), 16_000),
    )

    with pytest.raises(AudioValidationError, match="NaN or infinite"):
        decode_audio_file(path)


def test_streaming_decoder_rejects_initial_empty_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class EmptyDecoderSource:
        samplerate = 16_000
        frames = 160_000

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self, *, frames: int, dtype: str, always_2d: bool) -> np.ndarray:
            return np.array([], dtype=np.float64)

    path = tmp_path / "empty-decoder.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "app.inference.audio.sf.SoundFile",
        lambda _: EmptyDecoderSource(),
    )

    with open_audio_fragments(path, 10.0) as (_, fragments):
        with pytest.raises(AudioValidationError, match="Audio signal is empty"):
            tuple(fragments)


@pytest.mark.parametrize(
    ("seconds", "sample_rate", "stereo"),
    [
        (3.0, 16_000, False),
        (10.0, 16_000, True),
        (21.0, 44_100, False),
        (21.0, 44_100, True),
        (21.0, 48_000, False),
        (21.0, 48_000, True),
    ],
)
def test_sequential_soundfile_blocks_match_full_decode_and_segmentation(
    wav_factory,
    seconds: float,
    sample_rate: int,
    stereo: bool,
) -> None:
    signal = sine(seconds, sample_rate)
    if stereo:
        signal = np.column_stack([signal, signal * 0.5])
    path = wav_factory(signal, sample_rate)

    full_audio, full_sample_rate = decode_audio_file(path)
    expected = segment_audio(full_audio, full_sample_rate, 10.0)
    with open_audio_fragments(path, 10.0) as (audio_info, fragment_iterator):
        actual = tuple(fragment_iterator)

    assert audio_info.sample_rate == full_sample_rate
    assert audio_info.duration_seconds == pytest.approx(seconds, abs=1e-3)
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        assert observed.index == reference.index
        assert observed.start_seconds == pytest.approx(reference.start_seconds)
        assert observed.end_seconds == pytest.approx(reference.end_seconds)
        assert observed.duration_seconds == pytest.approx(reference.duration_seconds)
        assert observed.is_incomplete is reference.is_incomplete
        np.testing.assert_array_equal(observed.signal, reference.signal)


def test_segment_audio_short_audio_produces_one_incomplete_fragment() -> None:
    fragments = segment_audio(sine(3.0), 16_000, 10.0)

    assert len(fragments) == 1
    assert fragments[0].is_incomplete is True
    assert fragments[0].duration_seconds == pytest.approx(3.0)


def test_segment_audio_exactly_ten_seconds_produces_one_complete_fragment() -> None:
    fragments = segment_audio(sine(10.0), 16_000, 10.0)

    assert len(fragments) == 1
    assert fragments[0].is_incomplete is False
    assert fragments[0].end_seconds == pytest.approx(10.0)


def test_segment_audio_more_than_ten_seconds() -> None:
    fragments = segment_audio(sine(12.0), 16_000, 10.0)

    assert len(fragments) == 2
    assert fragments[0].start_seconds == pytest.approx(0.0)
    assert fragments[1].start_seconds == pytest.approx(10.0)
    assert fragments[1].duration_seconds == pytest.approx(2.0)


def test_segment_audio_thirty_five_seconds_produces_four_fragments() -> None:
    fragments = segment_audio(sine(35.0), 16_000, 10.0)

    assert len(fragments) == 4
    assert [f.start_seconds for f in fragments] == pytest.approx([0.0, 10.0, 20.0, 30.0])
    assert [f.end_seconds for f in fragments] == pytest.approx([10.0, 20.0, 30.0, 35.0])


def test_segment_audio_multiple_of_ten_has_no_empty_fragment() -> None:
    fragments = segment_audio(sine(30.0), 16_000, 10.0)

    assert len(fragments) == 3
    assert all(fragment.duration_seconds > 0 for fragment in fragments)
    assert fragments[-1].is_incomplete is False


def test_segment_audio_marks_last_fragment_incomplete() -> None:
    fragments = segment_audio(sine(21.0), 16_000, 10.0)

    assert fragments[-1].is_incomplete is True
    assert fragments[-1].duration_seconds == pytest.approx(1.0)


def test_segment_audio_preserves_stereo_channels() -> None:
    mono = sine(12.0)
    stereo = np.column_stack([mono, mono])

    fragments = segment_audio(stereo, 16_000, 10.0)

    assert fragments[0].signal.ndim == 2
    assert fragments[0].signal.shape[1] == 2


def test_preprocess_converts_stereo_to_mono(config) -> None:
    left = np.ones(16_000, dtype=np.float32)
    right = np.zeros(16_000, dtype=np.float32)
    stereo = np.column_stack([left, right])

    processed = preprocess_fragment(stereo, 16_000, config)

    assert processed.shape == (160_000,)
    assert np.allclose(processed[:16_000], 0.5)


def test_preprocess_pads_to_target_samples(config) -> None:
    processed = preprocess_fragment(sine(1.0), 16_000, config)

    assert processed.shape == (160_000,)
    assert np.count_nonzero(processed[16_000:]) == 0


def test_preprocess_resamples_from_different_sample_rate(config) -> None:
    processed = preprocess_fragment(sine(1.0, sample_rate=8_000), 8_000, config)

    assert processed.shape == (160_000,)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()


def test_preprocess_output_is_float32_finite_and_exact_length(config) -> None:
    processed = preprocess_fragment(sine(10.0), 16_000, config)

    assert processed.dtype == np.float32
    assert processed.shape == (160_000,)
    assert np.isfinite(processed).all()


def test_preprocess_rejects_non_finite_values(config) -> None:
    signal = sine(1.0)
    signal[10] = np.inf

    with pytest.raises(AudioValidationError, match="NaN or infinite"):
        preprocess_fragment(signal, 16_000, config)


def test_preprocess_uses_maximum_energy_window_when_too_long(config) -> None:
    signal = np.zeros(25, dtype=np.float32)
    signal[5:15] = 0.2
    signal[15:25] = 1.0
    short_config = type(config)(
        target_sample_rate=10,
        fragment_duration_seconds=1.0,
        n_mfcc=config.n_mfcc,
        positive_label=config.positive_label,
        decision_threshold=config.decision_threshold,
    )

    processed = preprocess_fragment(signal, 10, short_config)

    assert processed.shape == (10,)
    assert np.allclose(processed, 1.0)
