from __future__ import annotations

from dataclasses import dataclass

from app.inference.errors import PredictionError


HUMAN_LABEL = 0
AI_GENERATED_LABEL = 1
CLASS_NAMES = ("human", "ai_generated")
SCORE_TYPE_DECISION_FUNCTION = "decision_function"


def class_name_for_label(label: int) -> str:
    if label == HUMAN_LABEL:
        return CLASS_NAMES[HUMAN_LABEL]
    if label == AI_GENERATED_LABEL:
        return CLASS_NAMES[AI_GENERATED_LABEL]
    raise PredictionError(f"Unknown class label: {label}")


@dataclass(frozen=True)
class InferenceConfig:
    target_sample_rate: int = 16_000
    fragment_duration_seconds: float = 10.0
    n_mfcc: int = 20
    positive_label: int = AI_GENERATED_LABEL
    decision_threshold: float = 0.0
    model_id: str = "mfcc-svm-baseline"
    usage_warning: str = (
        "La salida es una estimación; el score no es una probabilidad calibrada "
        "y el resultado no constituye un veredicto forense."
    )

    @property
    def target_samples(self) -> int:
        return int(round(self.target_sample_rate * self.fragment_duration_seconds))
