from __future__ import annotations

import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42


def normalize_proba(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


class MulticlassPlattCalibrator:
    """Compatibility class matching the calibrator saved by the training notebook."""

    def __init__(self, seed: int = SEED):
        self.model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)

    def fit(self, raw_proba, y):
        self.model.fit(np.log(normalize_proba(raw_proba)), y)
        return self

    def predict_proba(self, raw_proba):
        return normalize_proba(self.model.predict_proba(np.log(normalize_proba(raw_proba))))


def install_pickle_compat() -> None:
    # The notebook created MulticlassPlattCalibrator while executing as __main__.
    # Expose the same class there before joblib.load resolves the saved object.
    setattr(sys.modules["__main__"], "MulticlassPlattCalibrator", MulticlassPlattCalibrator)


def align_proba(model: Any, x) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.arange(p.shape[1])), dtype=int)
    aligned = np.zeros((len(x), 3), dtype=float)
    aligned[:, classes] = p
    return normalize_proba(aligned)


class CalibratedClassifierBundle:
    """Adapter for the {'model', 'calibrator'} classifier artifacts saved by Phase 2."""

    def __init__(self, payload: dict[str, Any]):
        self.model = payload["model"]
        self.calibrator = payload.get("calibrator")
        self.classes_ = np.array([0, 1, 2], dtype=int)

    @property
    def named_steps(self):
        return getattr(self.model, "named_steps", {})

    def predict_proba(self, x) -> np.ndarray:
        raw = align_proba(self.model, x)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(raw)


def patch_loaded_app(app_module) -> None:
    if isinstance(app_module.PRE_OUTCOME_MODEL, dict):
        app_module.PRE_OUTCOME_MODEL = CalibratedClassifierBundle(app_module.PRE_OUTCOME_MODEL)
    if isinstance(app_module.LIVE_OUTCOME_MODEL, dict):
        app_module.LIVE_OUTCOME_MODEL = CalibratedClassifierBundle(app_module.LIVE_OUTCOME_MODEL)
