from __future__ import annotations

from enum import StrEnum

from app.inference.schemas import StartupWarmupResult


class StartupReadiness(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


def readiness_from_warmup_result(result: StartupWarmupResult) -> StartupReadiness:
    return StartupReadiness.READY if result.succeeded else StartupReadiness.FAILED
