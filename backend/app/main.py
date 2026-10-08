from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import ApiSettings
from app.inference.config import InferenceConfig
from app.inference.errors import ModelArtifactError
from app.inference.interfaces import InferenceService
from app.inference.service import AudioInferenceService
from app.inference.warmups import run_startup_warmups
from app.readiness import StartupReadiness, readiness_from_warmup_result


logger = logging.getLogger(__name__)
ServiceFactory = Callable[[], InferenceService]


def _default_service_factory() -> AudioInferenceService:
    return AudioInferenceService()


def create_app(
    service_factory: ServiceFactory | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings.from_env()
    factory = service_factory or _default_service_factory

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        warmup_task: asyncio.Task[None] | None = None
        application.state.inference_service = None
        application.state.model_ready = False
        application.state.startup_warmup_result = None
        application.state.startup_readiness = None
        try:
            service = factory()
        except ModelArtifactError as exc:
            logger.error(
                "Inference service could not be loaded; API starts degraded: %s",
                exc,
            )
        else:
            application.state.inference_service = service
            application.state.model_ready = True
            if api_settings.resample_warmup_enabled:
                application.state.startup_readiness = StartupReadiness.PENDING
                warmup_task = asyncio.create_task(
                    _run_startup_warmups(
                        application,
                        service.config,
                    )
                )
            else:
                application.state.startup_readiness = StartupReadiness.READY

        try:
            yield
        finally:
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task

    application = FastAPI(
        title="AI Music Detection API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if api_settings.cors_allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=[],
        )
    application.state.settings = api_settings
    application.include_router(router)
    return application


app = create_app()


async def _run_startup_warmups(
    application: FastAPI,
    config: InferenceConfig,
) -> None:
    try:
        result = await asyncio.to_thread(
            run_startup_warmups,
            config,
        )
    except Exception:
        logger.exception("Startup warm-ups failed unexpectedly")
        application.state.startup_readiness = StartupReadiness.FAILED
        return

    application.state.startup_warmup_result = result
    application.state.startup_readiness = readiness_from_warmup_result(result)
