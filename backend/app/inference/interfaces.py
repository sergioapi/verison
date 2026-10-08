from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.inference.config import InferenceConfig
from app.inference.schemas import ModelMetadata, PredictionResult


class InferenceService(Protocol):
    @property
    def config(self) -> InferenceConfig:
        ...

    @property
    def metadata(self) -> ModelMetadata:
        ...

    @property
    def usage_warning(self) -> str:
        ...

    def predict_file(self, path: str | Path) -> PredictionResult:
        ...
