"""
Relative sparse depth via two-view triangulation.

Reuses what ``PoseEstimator.process`` already computed (K, R, unit-t, inlier
matches), so the extra cost is just one ``cv2.triangulatePoints`` call on a
few hundred points — cheap enough for the Raspberry Pi 4B.

Because monocular translation is only known up to scale (recoverPose returns a
*unit* t), the recovered depth is **relative**: distances are expressed in
"camera-baseline units", not metres.  We therefore report a median relative
depth plus a coarse NEAR / MID / FAR level.
"""
import cv2
import math

import numpy as np

_MIN_POINTS = 8           # need a reasonable set for a stable median
# Coarse thresholds in baseline units (unit-t triangulation).
_NEAR_MAX = 5.0
_FAR_MIN = 15.0


def relative_depth(K, R, t, pts1_in, pts2_in) -> tuple[float, str]:
    """
    Median relative scene depth + coarse level from two views.

    Returns
    -------
    (median_depth, level)
        median_depth : float (NaN when no depth could be computed)
        level        : "NEAR" | "MID" | "FAR" | "N/A"
    """
    if t is None:
        return float("nan"), "N/A"

    p1 = np.asarray(pts1_in, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2_in, dtype=np.float64).reshape(-1, 2)
    if p1.shape[0] < _MIN_POINTS or p1.shape != p2.shape:
        return float("nan"), "N/A"

    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)   # (4, N) homogeneous
    w = pts4d[3]
    valid = np.abs(w) > 1e-9
    if not np.any(valid):
        return float("nan"), "N/A"
    z = pts4d[2, valid] / w[valid]
    z = z[z > 0]                                          # keep points in front
    if z.size == 0:
        return float("nan"), "N/A"

    median_depth = float(np.median(z))
    if median_depth < _NEAR_MAX:
        level = "NEAR"
    elif median_depth > _FAR_MIN:
        level = "FAR"
    else:
        level = "MID"
    return median_depth, level
