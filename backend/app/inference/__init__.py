"""Autonomous MFCC + SVM inference package."""

from app.inference.config import InferenceConfig
from app.inference.service import AudioInferenceService

__all__ = ["AudioInferenceService", "InferenceConfig"]
