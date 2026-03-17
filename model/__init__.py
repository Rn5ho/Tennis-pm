"""Tennis prediction model package."""

import numpy as np


class CalibratedModel:
    """Wraps a classifier with isotonic regression calibration."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        cal = np.clip(cal, 0.001, 0.999)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
