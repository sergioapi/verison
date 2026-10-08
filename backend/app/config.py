from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_MAX_UPLOAD_SIZE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 300.0
SUPPORTED_AUDIO_FORMATS = {
    ".wav": ("audio/wav", "audio/wave", "audio/x-wav"),
    ".mp3": ("audio/mpeg",),
}
DEFAULT_ALLOWED_AUDIO_EXTENSIONS = tuple(SUPPORTED_AUDIO_FORMATS)
DEFAULT_ALLOWED_AUDIO_MIME_TYPES = tuple(
    mime_type
    for mime_types in SUPPORTED_AUDIO_FORMATS.values()
    for mime_type in mime_types
)
DEFAULT_CORS_ALLOWED_ORIGINS: tuple[str, ...] = ()
DEFAULT_RESAMPLE_WARMUP_ENABLED = False


@dataclass(frozen=True)
class ApiSettings:
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    max_audio_duration_seconds: float = DEFAULT_MAX_AUDIO_DURATION_SECONDS
    allowed_audio_extensions: tuple[str, ...] = DEFAULT_ALLOWED_AUDIO_EXTENSIONS
    allowed_audio_mime_types: tuple[str, ...] = DEFAULT_ALLOWED_AUDIO_MIME_TYPES
    cors_allowed_origins: tuple[str, ...] = DEFAULT_CORS_ALLOWED_ORIGINS
    resample_warmup_enabled: bool = DEFAULT_RESAMPLE_WARMUP_ENABLED
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_upload_size_bytes, int):
            raise ValueError("max_upload_size_bytes must be an integer")
        if self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be greater than zero")
        if not _is_positive_finite(self.max_audio_duration_seconds):
            raise ValueError("max_audio_duration_seconds must be a finite value greater than zero")
        if not isinstance(self.resample_warmup_enabled, bool):
            raise ValueError("resample_warmup_enabled must be a boolean")

        object.__setattr__(
            self,
            "allowed_audio_extensions",
            _validate_supported_extensions(
                _normalize_extensions(
                    self.allowed_audio_extensions,
                    "allowed_audio_extensions",
                )
            ),
        )
        object.__setattr__(
            self,
            "allowed_audio_mime_types",
            _validate_supported_mime_types(
                _normalize_values(self.allowed_audio_mime_types, "allowed_audio_mime_types")
            ),
        )
        _validate_compatible_audio_formats(
            self.allowed_audio_extensions,
            self.allowed_audio_mime_types,
        )
        object.__setattr__(
            self,
            "cors_allowed_origins",
            _validate_cors_allowed_origins(
                _normalize_optional_values(self.cors_allowed_origins)
            ),
        )
        object.__setattr__(self, "temp_dir", _validate_temp_dir(self.temp_dir))

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            max_upload_size_bytes=_read_int(
                "MAX_UPLOAD_SIZE_BYTES",
                DEFAULT_MAX_UPLOAD_SIZE_BYTES,
            ),
            max_audio_duration_seconds=_read_float(
                "MAX_AUDIO_DURATION_SECONDS",
                DEFAULT_MAX_AUDIO_DURATION_SECONDS,
            ),
            allowed_audio_extensions=_read_extensions(
                "ALLOWED_AUDIO_EXTENSIONS",
                DEFAULT_ALLOWED_AUDIO_EXTENSIONS,
            ),
            allowed_audio_mime_types=_read_values(
                "ALLOWED_AUDIO_MIME_TYPES",
                DEFAULT_ALLOWED_AUDIO_MIME_TYPES,
            ),
            cors_allowed_origins=_read_optional_values(
                "CORS_ALLOWED_ORIGINS",
                DEFAULT_CORS_ALLOWED_ORIGINS,
            ),
            resample_warmup_enabled=_read_bool(
                "RESAMPLE_WARMUP_ENABLED",
                DEFAULT_RESAMPLE_WARMUP_ENABLED,
            ),
            temp_dir=_read_optional_path("TEMP_DIR"),
        )


def _read_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not _is_positive_finite(value):
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _read_extensions(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = _read_values(name, default)
    return _normalize_extensions(values, name)


def _read_values(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    return tuple(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )


def _read_optional_values(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )


def _read_optional_path(name: str) -> Path | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return Path(raw_value).expanduser()


def _is_positive_finite(value: float) -> bool:
    numeric_value = float(value)
    return numeric_value > 0.0 and math.isfinite(numeric_value)


def _normalize_extensions(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in values
        if value.strip()
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    if any(value == "." or "/" in value or "\\" in value for value in normalized):
        raise ValueError(f"{name} contains an invalid extension")
    return normalized


def _normalize_values(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(value.strip().lower() for value in values if value.strip())
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    return normalized


def _normalize_optional_values(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())


def _validate_supported_extensions(values: tuple[str, ...]) -> tuple[str, ...]:
    unsupported = tuple(value for value in values if value not in SUPPORTED_AUDIO_FORMATS)
    if unsupported:
        raise ValueError(
            f"allowed_audio_extensions contains unsupported extensions: {unsupported}"
        )
    return values


def _validate_supported_mime_types(values: tuple[str, ...]) -> tuple[str, ...]:
    supported = set(DEFAULT_ALLOWED_AUDIO_MIME_TYPES)
    unsupported = tuple(value for value in values if value not in supported)
    if unsupported:
        raise ValueError(
            f"allowed_audio_mime_types contains unsupported values: {unsupported}"
        )
    return values


def _validate_compatible_audio_formats(
    extensions: tuple[str, ...],
    mime_types: tuple[str, ...],
) -> None:
    allowed_mime_types = set(mime_types)
    compatible_extensions = tuple(
        extension
        for extension in extensions
        if allowed_mime_types.intersection(SUPPORTED_AUDIO_FORMATS[extension])
    )
    if not compatible_extensions:
        raise ValueError("allowed audio extensions and MIME types have no compatible format")


def _validate_cors_allowed_origins(values: tuple[str, ...]) -> tuple[str, ...]:
    for origin in values:
        if origin == "*":
            raise ValueError("cors_allowed_origins must not contain wildcard origins")

        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "cors_allowed_origins must contain absolute HTTP or HTTPS origins"
            )
    return values


def _validate_temp_dir(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError("temp_dir must exist")
    if not path.is_dir():
        raise ValueError("temp_dir must be a directory")
    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
    except OSError as exc:
        raise ValueError("temp_dir must be writable") from exc
    return path
