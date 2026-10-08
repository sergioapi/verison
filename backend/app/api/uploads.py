from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException, UploadFile, status

from app.config import SUPPORTED_AUDIO_FORMATS, ApiSettings


logger = logging.getLogger(__name__)
UPLOAD_COPY_CHUNK_SIZE = 1024 * 1024


@contextmanager
def uploaded_audio_path(
    file: UploadFile,
    settings: ApiSettings,
) -> Iterator[Path]:
    temp_path: Path | None = None
    try:
        suffix = validate_upload_extension(
            file.filename,
            settings.allowed_audio_extensions,
        )
        validate_upload_mime_type(
            file.content_type,
            suffix,
            settings.allowed_audio_mime_types,
        )
        temp_path = _copy_upload_to_temp(file, suffix, settings)
        yield temp_path
    finally:
        if temp_path is not None:
            _remove_temp_upload(temp_path)
        file.file.close()


def validate_upload_extension(
    filename: str | None,
    allowed_extensions: tuple[str, ...],
) -> str:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_file_type",
                "message": "Unsupported audio file type",
            },
        )

    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_file_type",
                "message": "Unsupported audio file type",
            },
        )
    return suffix


def validate_upload_mime_type(
    content_type: str | None,
    extension: str,
    allowed_mime_types: tuple[str, ...],
) -> None:
    if content_type is None or content_type.strip() == "":
        return

    mime_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    extension_mime_types = SUPPORTED_AUDIO_FORMATS[extension]
    if mime_type not in extension_mime_types or mime_type not in allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_media_type",
                "message": "Unsupported audio media type",
            },
        )


def _copy_upload_to_temp(
    file: UploadFile,
    suffix: str,
    settings: ApiSettings,
) -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix, dir=settings.temp_dir)
    temp_path = Path(path)
    bytes_written = 0

    try:
        with os.fdopen(fd, "wb") as target:
            while True:
                chunk = file.file.read(UPLOAD_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail={
                            "code": "file_too_large",
                            "message": "Uploaded file is too large",
                        },
                    )
                target.write(chunk)

        if bytes_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "empty_file", "message": "Uploaded file is empty"},
            )
        return temp_path
    except Exception:
        _remove_temp_upload(temp_path)
        raise


def _remove_temp_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove temporary upload")
