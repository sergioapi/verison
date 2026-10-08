from __future__ import annotations


class InferenceError(RuntimeError):
    """Base error for local audio inference failures."""


class ModelArtifactError(InferenceError):
    """Raised when the model artifact is missing or incompatible."""


class AudioDecodingError(InferenceError):
    """Raised when an audio file cannot be decoded."""


class AudioValidationError(InferenceError):
    """Raised when decoded audio violates inference invariants."""


class PredictionError(InferenceError):
    """Raised when prediction or score aggregation fails."""
