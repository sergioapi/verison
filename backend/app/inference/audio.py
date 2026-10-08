from __future__ import annotations

import math
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import librosa
import numpy as np
import soundfile as sf

from app.inference.config import InferenceConfig
from app.inference.errors import AudioDecodingError, AudioValidationError
from app.inference.schemas import AudioFragment, WarmupResult


logger = logging.getLogger("uvicorn.error")
_RESAMPLE_WARMUP_SOURCE_SAMPLE_RATE = 48_000
_RESAMPLE_WARMUP_TARGET_SAMPLE_RATE = 16_000
_RESAMPLE_WARMUP_SAMPLES = _RESAMPLE_WARMUP_SOURCE_SAMPLE_RATE


@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    frames: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def get_audio_duration_seconds(path: Path) -> float:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")

    try:
        info = sf.info(audio_path)
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not inspect audio file {audio_path}: {exc}") from exc

    audio_info = _validate_audio_info(int(info.samplerate), int(info.frames))
    duration_seconds = audio_info.duration_seconds
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise AudioValidationError(f"Invalid audio duration: {duration_seconds!r}")
    return float(duration_seconds)


@contextmanager
def open_audio_fragments(
    path: Path,
    fragment_duration_seconds: float,
) -> Iterator[tuple[AudioFileInfo, Iterator[AudioFragment]]]:
    """Open an audio file once and expose consecutive decoded fragments."""
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")
    if fragment_duration_seconds <= 0:
        raise AudioValidationError(
            f"Invalid fragment duration: {fragment_duration_seconds}"
        )

    try:
        with sf.SoundFile(audio_path) as source:
            audio_info = _validate_audio_info(int(source.samplerate), int(source.frames))
            fragment_samples = int(round(audio_info.sample_rate * fragment_duration_seconds))
            if fragment_samples <= 0:
                raise AudioValidationError(f"Invalid fragment length: {fragment_samples}")
            yield audio_info, _read_audio_fragments(source, audio_info, fragment_samples)
    except (AudioDecodingError, AudioValidationError):
        raise
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not decode audio file {audio_path}: {exc}") from exc


def decode_audio_file(path: Path) -> tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not decode audio file {audio_path}: {exc}") from exc

    decoded = validate_decoded_audio(audio, int(sample_rate))
    return decoded


def validate_decoded_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim not in (1, 2):
        raise AudioValidationError(f"Expected mono or stereo audio, found shape {signal.shape}")
    if signal.shape[0] == 0:
        raise AudioValidationError("Audio signal is empty")
    _validate_finite(signal, "decoded audio")
    return signal, sample_rate


def _read_audio_fragments(
    source: sf.SoundFile,
    audio_info: AudioFileInfo,
    fragment_samples: int,
) -> Iterator[AudioFragment]:
    index = 0
    decoded_frames = 0
    while True:
        try:
            audio = source.read(
                frames=fragment_samples,
                dtype="float64",
                always_2d=False,
            )
        except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
            raise AudioDecodingError(f"Could not decode audio fragment: {exc}") from exc

        if np.asarray(audio).shape[0] == 0:
            if decoded_frames == 0:
                raise AudioValidationError("Audio signal is empty")
            break
        signal, sample_rate = validate_decoded_audio(audio, audio_info.sample_rate)
        frames_read = signal.shape[0]
        start = decoded_frames
        end = decoded_frames + frames_read
        yield AudioFragment(
            index=index,
            start_seconds=start / sample_rate,
            end_seconds=end / sample_rate,
            duration_seconds=frames_read / sample_rate,
            signal=signal,
            sample_rate=sample_rate,
            is_incomplete=frames_read < fragment_samples,
        )
        index += 1
        decoded_frames = end


def _validate_audio_info(sample_rate: int, frames: int) -> AudioFileInfo:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")
    if frames <= 0:
        raise AudioValidationError("Audio signal is empty")
    duration_seconds = frames / sample_rate
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise AudioValidationError(f"Invalid audio duration: {duration_seconds!r}")
    return AudioFileInfo(sample_rate=sample_rate, frames=frames)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    fragment_duration_seconds: float,
) -> tuple[AudioFragment, ...]:
    signal, sample_rate = validate_decoded_audio(audio, sample_rate)
    if fragment_duration_seconds <= 0:
        raise AudioValidationError(
            f"Invalid fragment duration: {fragment_duration_seconds}"
        )

    fragment_samples = int(round(sample_rate * fragment_duration_seconds))
    if fragment_samples <= 0:
        raise AudioValidationError(f"Invalid fragment length: {fragment_samples}")

    fragments: list[AudioFragment] = []
    total_samples = signal.shape[0]
    for index, start in enumerate(range(0, total_samples, fragment_samples)):
        end = min(start + fragment_samples, total_samples)
        if end <= start:
            continue
        fragment_signal = signal[start:end].copy()
        duration = (end - start) / sample_rate
        fragments.append(
            AudioFragment(
                index=index,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                duration_seconds=duration,
                signal=fragment_signal,
                sample_rate=sample_rate,
                is_incomplete=(end - start) < fragment_samples,
            )
        )

    if not fragments:
        raise AudioValidationError("Audio segmentation produced no fragments")
    return tuple(fragments)


def preprocess_fragment(
    audio: np.ndarray,
    sample_rate: int,
    config: InferenceConfig,
) -> np.ndarray:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1, dtype=np.float32)
    elif signal.ndim != 1:
        raise AudioValidationError(f"Expected mono or stereo fragment, found shape {signal.shape}")

    if signal.size == 0:
        raise AudioValidationError("Audio fragment is empty")
    _validate_finite(signal, "audio fragment")

    source_window_samples = int(round(sample_rate * config.fragment_duration_seconds))
    signal = _select_or_pad_window(signal, source_window_samples)

    if sample_rate != config.target_sample_rate:
        resample_fn = librosa.resample
        signal = resample_fn(
            signal,
            orig_sr=sample_rate,
            target_sr=config.target_sample_rate,
        ).astype(np.float32, copy=False)

    signal = _fix_length(signal, config.target_samples)
    _validate_finite(signal, "preprocessed audio")
    return signal.astype(np.float32, copy=False)


def warm_up_resampling() -> WarmupResult:
    """Exercise the production resampling route once without reading audio."""
    total_start = time.perf_counter()
    try:
        _log_resample_warmup(logging.INFO, "resample_warmup status=started")
        signal = np.zeros(_RESAMPLE_WARMUP_SAMPLES, dtype=np.float32)
        resample_fn = librosa.resample
        resample_fn(
            signal,
            orig_sr=_RESAMPLE_WARMUP_SOURCE_SAMPLE_RATE,
            target_sr=_RESAMPLE_WARMUP_TARGET_SAMPLE_RATE,
        )
    except Exception as exc:  # noqa: BLE001 - a warm-up must not break startup.
        _log_resample_warmup(
            logging.WARNING,
            "resample_warmup status=failed error_type=%s",
            type(exc).__name__,
        )
        return WarmupResult(
            name="resampling",
            succeeded=False,
            duration_seconds=max(0.0, time.perf_counter() - total_start),
            error_type=type(exc).__name__,
        )
    else:
        duration_seconds = max(0.0, time.perf_counter() - total_start)
        _log_resample_warmup(
            logging.INFO,
            "resample_warmup status=completed total_seconds=%.4f",
            duration_seconds,
        )
        return WarmupResult(
            name="resampling",
            succeeded=True,
            duration_seconds=duration_seconds,
        )
def _log_resample_warmup(level: int, message: str, *args: object) -> None:
    try:
        logger.log(level, message, *args)
    except Exception:  # noqa: BLE001 - diagnostic logging must not break startup.
        return


def _select_or_pad_window(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 0:
        raise AudioValidationError(f"Invalid window length: {window_samples}")
    if signal.size <= window_samples:
        return _fix_length(signal, window_samples)

    squared = np.square(signal, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    energies = cumulative[window_samples:] - cumulative[:-window_samples]
    best_start = int(np.argmax(energies))
    return signal[best_start : best_start + window_samples].astype(np.float32, copy=False)


def _fix_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        raise AudioValidationError(f"Invalid target length: {target_samples}")
    if signal.size == target_samples:
        return signal.astype(np.float32, copy=False)
    if signal.size > target_samples:
        return signal[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: signal.size] = signal.astype(np.float32, copy=False)
    return padded


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise AudioValidationError(f"{name} contains NaN or infinite values")