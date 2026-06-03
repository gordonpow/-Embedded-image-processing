"""
Acceptance metrics — compare estimator output against ground truth.

  * angular_error_stats — for yaw/pitch/roll vs known angles (wrap-aware so that
    179 deg and -179 deg are 2 deg apart, not 358).
  * classification_accuracy — for indoor/outdoor vs labels, with a confusion
    tally keyed by (predicted, truth).
"""
import math

import numpy as np


def _wrap180(deg):
    """Wrap an angle (or array) to (-180, 180]."""
    d = np.asarray(deg, dtype=np.float64)
    return (d + 180.0) % 360.0 - 180.0


def angular_error_stats(pred, gt) -> dict:
    """Return {'mae', 'rmse', 'max'} of |pred - gt| in degrees (wrap-aware)."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    if pred.size == 0 or pred.size != gt.size:
        return {"mae": 0.0, "rmse": 0.0, "max": 0.0}
    err = np.abs(_wrap180(pred - gt))
    return {
        "mae": float(err.mean()),
        "rmse": float(math.sqrt((err ** 2).mean())),
        "max": float(err.max()),
    }


def classification_accuracy(pred_labels, gt_labels) -> tuple[float, dict]:
    """
    Accuracy + confusion tally for categorical predictions.

    Returns (accuracy, confusion) where confusion[(pred, truth)] = count.
    """
    pred_labels = list(pred_labels)
    gt_labels = list(gt_labels)
    n = min(len(pred_labels), len(gt_labels))
    if n == 0:
        return 0.0, {}
    correct = 0
    confusion: dict = {}
    for p, g in zip(pred_labels[:n], gt_labels[:n]):
        confusion[(p, g)] = confusion.get((p, g), 0) + 1
        if p == g:
            correct += 1
    return correct / n, confusion
