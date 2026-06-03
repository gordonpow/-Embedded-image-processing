"""
RED test for cumulative-rotation accuracy.

Generates a short controlled 3D rotation+translation sequence with EXACT ground
truth and asserts the estimator's accumulated yaw/pitch/roll track it.

This fails with the over-accumulation bug (R_global multiplied by the
reference->current rotation every frame even when the reference persists) and
passes once accumulation is keyframe-relative.
"""
import math

import cv2
import numpy as np

from estimator import PoseEstimator, _rot_to_ypr
from orient import ypr_to_R
from metrics import angular_error_stats

W, H = 640, 480
K = np.array([[W, 0, W / 2.0], [0, W, H / 2.0], [0, 0, 1.0]], dtype=np.float64)
DIST = np.zeros(5)

_rng = np.random.default_rng(11)
_PTS = np.column_stack([
    _rng.uniform(-4, 4, 450),
    _rng.uniform(-3, 3, 450),
    _rng.uniform(5, 14, 450),
]).astype(np.float64)
_RAD = _rng.integers(3, 7, 450).tolist()


def _render(yaw, pitch, roll, tvec):
    R = ypr_to_R(yaw, pitch, roll)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(_PTS, rvec, tvec, K, DIST)
    proj = proj.reshape(-1, 2)
    in_front = (R @ _PTS.T + tvec.reshape(3, 1))[2] > 0.1
    img = np.full((H, W), 25, dtype=np.uint8)
    for j, (x, y) in enumerate(proj):
        if in_front[j] and -20 <= x < W + 20 and -20 <= y < H + 20:
            cv2.circle(img, (int(x), int(y)), _RAD[j], 220, -1, cv2.LINE_AA)
    return img


def test_accumulated_rotation_tracks_ground_truth():
    est = PoseEstimator()
    est.set_K(K)
    preds, gts = [], []
    n = 48
    for i in range(n):
        t = i / 30.0
        yaw = 18.0 * math.sin(2 * math.pi * t / 2.4)
        pitch = 10.0 * math.sin(2 * math.pi * t / 3.0)
        roll = 12.0 * math.sin(2 * math.pi * t / 3.6)
        tvec = np.array([0.5 * math.sin(t), 0.25 * math.sin(1.3 * t), 0.4 * math.sin(0.8 * t)])
        gray = _render(yaw, pitch, roll, tvec)
        y, p, r, *_ = est.process(gray)
        preds.append((y, p, r))
        gts.append((yaw, pitch, roll))

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    worst = max(angular_error_stats(preds[:, i], gts[:, i])["mae"] for i in range(3))
    # Honest tolerance for ORB + 5-point + EMA smoothing on a clean synthetic clip.
    assert worst < 6.0, f"worst-axis MAE {worst:.1f} deg too high (accumulation bug?)"
