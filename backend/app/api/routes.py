from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status

from app.api.mappers import analyze_response, model_info_response
from app.api.schemas import (
    AnalyzeResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadinessResponse,
)
from app.api.uploads import uploaded_audio_path
from app.config import ApiSettings
from app.inference.audio import get_audio_duration_seconds
from app.inference.errors import AudioDecodingError, AudioValidationError, PredictionError
from app.inference.interfaces import InferenceService
from app.readiness import StartupReadiness


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    if model_ready:
        return HealthResponse(status="ok", model_ready=True)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="degraded", model_ready=False)


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request, response: Response) -> ReadinessResponse:
    if not bool(getattr(request.app.state, "model_ready", False)):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable")

    readiness = getattr(request.app.state, "startup_readiness", StartupReadiness.FAILED)
    if readiness is StartupReadiness.READY:
        return ReadinessResponse(status=readiness.value)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status=readiness.value)


@router.get("/api/v1/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = getattr(request.app.state, "inference_service", None)
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    if not model_ready or service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "message": "Model is not available"},
        )

    return model_info_response(service.metadata, service.usage_warning)


@router.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: Request, file: UploadFile = File(...)) -> AnalyzeResponse:
    try:
        service = _ready_service(request)
        settings = _api_settings(request)
        with uploaded_audio_path(file, settings) as path:
            _ensure_audio_duration_allowed(path, settings)
            prediction = service.predict_file(path)
        return analyze_response(prediction)
    except HTTPException:
        raise
    except (AudioDecodingError, AudioValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_audio",
                "message": "Audio file could not be processed",
            },
        ) from exc
    except PredictionError as exc:
        logger.error("Audio prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "prediction_failed", "message": "Prediction failed"},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected API error while analyzing audio")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "internal_error", "message": "Internal server error"},
        ) from exc


def _ready_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference_service", None)
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    if not model_ready or service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "message": "Model is not available"},
        )
    return service


def _api_settings(request: Request) -> ApiSettings:
    return getattr(request.app.state, "settings")


def _ensure_audio_duration_allowed(path: Path, settings: ApiSettings) -> None:
    duration_seconds = get_audio_duration_seconds(path)
    if duration_seconds > settings.max_audio_duration_seconds:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "audio_too_long",
                "message": "Audio duration exceeds the allowed limit",
            },
        )
