from __future__ import annotations

from app.inference.audio import warm_up_resampling
from app.inference.config import InferenceConfig
from app.inference.features import warm_up_mfcc
from app.inference.schemas import StartupWarmupResult, WarmupResult


def run_startup_warmups(config: InferenceConfig) -> StartupWarmupResult:
    """Run startup warm-ups in production order and retain their outcomes."""
    resampling = warm_up_resampling()
    if not resampling.succeeded:
        return StartupWarmupResult(resampling=resampling, mfcc=WarmupResult(
            name="mfcc",
            succeeded=False,
            duration_seconds=0.0,
            error_type="SkippedAfterResamplingFailure",
        ))
    mfcc = warm_up_mfcc(config)
    return StartupWarmupResult(resampling=resampling, mfcc=mfcc)
