from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    DEFAULT_ALLOWED_AUDIO_EXTENSIONS,
    DEFAULT_ALLOWED_AUDIO_MIME_TYPES,
    DEFAULT_CORS_ALLOWED_ORIGINS,
    DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    DEFAULT_RESAMPLE_WARMUP_ENABLED,
    ApiSettings,
)


def test_api_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_api_env(monkeypatch)

    settings = ApiSettings.from_env()

    assert DEFAULT_MAX_UPLOAD_SIZE_BYTES == 64 * 1024 * 1024
    assert settings.max_upload_size_bytes == 64 * 1024 * 1024
    assert settings.max_audio_duration_seconds == DEFAULT_MAX_AUDIO_DURATION_SECONDS
    assert settings.allowed_audio_extensions == DEFAULT_ALLOWED_AUDIO_EXTENSIONS
    assert settings.allowed_audio_mime_types == DEFAULT_ALLOWED_AUDIO_MIME_TYPES
    assert settings.cors_allowed_origins == DEFAULT_CORS_ALLOWED_ORIGINS
    assert settings.resample_warmup_enabled is DEFAULT_RESAMPLE_WARMUP_ENABLED
    assert settings.temp_dir is None


def test_api_settings_reads_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "12345")
    monkeypatch.setenv("MAX_AUDIO_DURATION_SECONDS", "12.5")
    monkeypatch.setenv("ALLOWED_AUDIO_EXTENSIONS", "wav, .mp3")
    monkeypatch.setenv("ALLOWED_AUDIO_MIME_TYPES", "audio/wav, audio/mpeg")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        " http://localhost:5173, ,https://example.com ",
    )
    monkeypatch.setenv("TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("RESAMPLE_WARMUP_ENABLED", "true")

    settings = ApiSettings.from_env()

    assert settings.max_upload_size_bytes == 12345
    assert settings.max_audio_duration_seconds == pytest.approx(12.5)
    assert settings.allowed_audio_extensions == (".wav", ".mp3")
    assert settings.allowed_audio_mime_types == ("audio/wav", "audio/mpeg")
    assert settings.cors_allowed_origins == (
        "http://localhost:5173",
        "https://example.com",
    )
    assert settings.temp_dir == tmp_path
    assert settings.resample_warmup_enabled is True


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MAX_UPLOAD_SIZE_BYTES", "abc", "integer"),
        ("MAX_UPLOAD_SIZE_BYTES", "0", "greater than zero"),
        ("MAX_AUDIO_DURATION_SECONDS", "abc", "number"),
        ("MAX_AUDIO_DURATION_SECONDS", "-1", "greater than zero"),
        ("MAX_AUDIO_DURATION_SECONDS", "nan", "greater than zero"),
        ("MAX_AUDIO_DURATION_SECONDS", "inf", "greater than zero"),
        ("RESAMPLE_WARMUP_ENABLED", "maybe", "boolean"),
    ],
)
def test_api_settings_rejects_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    _clear_api_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        ApiSettings.from_env()


def test_api_settings_rejects_empty_extension_after_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_api_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_AUDIO_EXTENSIONS", ".")

    with pytest.raises(ValueError, match="invalid extension"):
        ApiSettings.from_env()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_upload_size_bytes": 0}, "greater than zero"),
        ({"max_audio_duration_seconds": float("nan")}, "finite"),
        ({"max_audio_duration_seconds": float("inf")}, "finite"),
        ({"max_audio_duration_seconds": float("-inf")}, "finite"),
        ({"allowed_audio_extensions": ()}, "at least one"),
        ({"allowed_audio_extensions": (".",)}, "invalid extension"),
        ({"allowed_audio_extensions": ("audio/wav",)}, "invalid extension"),
        ({"allowed_audio_extensions": (".flac",)}, "unsupported extensions"),
        ({"allowed_audio_mime_types": ()}, "at least one"),
        ({"allowed_audio_mime_types": ("audio/flac",)}, "unsupported values"),
        ({"cors_allowed_origins": ("*",)}, "wildcard"),
        ({"cors_allowed_origins": ("localhost:5173",)}, "absolute HTTP or HTTPS"),
        (
            {"cors_allowed_origins": ("http://localhost:5173/path",)},
            "absolute HTTP or HTTPS",
        ),
        (
            {"allowed_audio_extensions": (".wav",), "allowed_audio_mime_types": ("audio/mpeg",)},
            "no compatible format",
        ),
    ],
)
def test_api_settings_direct_construction_validates_invariants(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ApiSettings(**kwargs)


def test_api_settings_direct_construction_normalizes_values() -> None:
    settings = ApiSettings(
        allowed_audio_extensions=("WAV", " .MP3 "),
        allowed_audio_mime_types=(" Audio/WAV ", " Audio/MPEG "),
    )

    assert settings.allowed_audio_extensions == (".wav", ".mp3")
    assert settings.allowed_audio_mime_types == ("audio/wav", "audio/mpeg")


def test_api_settings_rejects_missing_temp_dir() -> None:
    with pytest.raises(ValueError, match="temp_dir must exist"):
        ApiSettings(temp_dir=Path("missing-temp-dir"))


def _clear_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MAX_UPLOAD_SIZE_BYTES",
        "MAX_AUDIO_DURATION_SECONDS",
        "ALLOWED_AUDIO_EXTENSIONS",
        "ALLOWED_AUDIO_MIME_TYPES",
        "CORS_ALLOWED_ORIGINS",
        "RESAMPLE_WARMUP_ENABLED",
        "TEMP_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
