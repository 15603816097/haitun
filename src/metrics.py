from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition-style Macro F1.

    Computes F1 independently for each label and skips labels that contain
    no positive samples in y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"expected 2D arrays, got ndim={y_true.ndim}")

    valid = y_true.sum(axis=0) > 0
    if not np.any(valid):
        return 0.0

    scores = [
        f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        for j in np.flatnonzero(valid)
    ]
    return float(np.mean(scores))


def apply_thresholds(scores: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    scores = np.asarray(scores)
    thresholds = np.asarray(thresholds)
    return (scores >= thresholds).astype(np.int8)
