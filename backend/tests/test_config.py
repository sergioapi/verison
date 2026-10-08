from __future__ import annotations

import pytest

from app.inference.config import class_name_for_label
from app.inference.errors import PredictionError


def test_class_name_for_human_label() -> None:
    assert class_name_for_label(0) == "human"


def test_class_name_for_ai_generated_label() -> None:
    assert class_name_for_label(1) == "ai_generated"


def test_class_name_rejects_unknown_label() -> None:
    with pytest.raises(PredictionError, match="Unknown class label"):
        class_name_for_label(2)
