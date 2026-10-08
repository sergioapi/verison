from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.inference.config import InferenceConfig  # noqa: E402
from app.inference.features import feature_columns  # noqa: E402


@pytest.fixture
def config() -> InferenceConfig:
    return InferenceConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def wav_factory(tmp_path: Path):
    def _write_wav(
        signal: np.ndarray,
        sample_rate: int = 16_000,
        name: str = "audio.wav",
    ) -> Path:
        path = tmp_path / name
        sf.write(path, np.asarray(signal, dtype=np.float32), sample_rate)
        return path

    return _write_wav


def build_pipeline(
    rng: np.random.Generator,
    n_features: int = 40,
    classes: tuple[int, int] = (0, 1),
) -> Pipeline:
    X0 = rng.normal(loc=-1.0, scale=0.2, size=(16, n_features))
    X1 = rng.normal(loc=1.0, scale=0.2, size=(16, n_features))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([classes[0]] * 16 + [classes[1]] * 16)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma=0.01, probability=False)),
        ]
    ).fit(X, y)


def build_artifact(
    rng: np.random.Generator,
    config: InferenceConfig | None = None,
    *,
    n_features: int = 40,
    columns: tuple[str, ...] | None = None,
    preprocessing: dict[str, float | int] | None = None,
    mfcc: dict[str, int] | None = None,
    positive_label: int = 1,
    classes: tuple[int, int] = (0, 1),
) -> dict[str, object]:
    config = config or InferenceConfig()
    return {
        "pipeline": build_pipeline(rng, n_features=n_features, classes=classes),
        "feature_columns": list(columns if columns is not None else feature_columns(config)),
        "preprocessing": preprocessing
        if preprocessing is not None
        else {
            "target_sample_rate": config.target_sample_rate,
            "duration_seconds": config.fragment_duration_seconds,
        },
        "mfcc": mfcc if mfcc is not None else {"n_mfcc": config.n_mfcc},
        "selected_params": {"C": 10, "gamma": 0.01},
        "positive_label": positive_label,
        "seed": 42,
    }


@pytest.fixture
def artifact_factory(tmp_path: Path, rng: np.random.Generator, config: InferenceConfig):
    counter = {"value": 0}

    def _write_artifact(**overrides) -> Path:
        counter["value"] += 1
        artifact = build_artifact(rng, config, **overrides)
        path = tmp_path / f"model-{counter['value']}.joblib"
        joblib.dump(artifact, path)
        return path

    return _write_artifact
