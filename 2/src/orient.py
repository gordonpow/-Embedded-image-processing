"""
Single-image camera tilt from the horizon — the only orientation a lone still
image can reveal.

A video gets XYZ from *relative* rotation accumulated across frames; a single
photo has no second frame, so instead we read the camera's tilt against the
ground:

  * roll  — slant of the dominant near-horizontal line (the horizon / strong
            structural edges).  A horizon whose right end sits lower -> +roll.
  * pitch — from the horizon's height in the frame: above centre -> camera
            looking down (+pitch); below centre -> looking up (-pitch).
  * yaw   — heading is unobservable from one image -> always 0.

The estimated (yaw=0, pitch, roll) builds a rotation matrix that drives the
same XYZ indicator used for video, so a tilted photo shows a tilted gizmo.
Everything here is plain OpenCV (Canny + HoughLinesP).
"""
import math

import cv2
import numpy as np

_MAX_HORIZON_ANGLE = 40.0    # lines within ±this of horizontal count as horizon


def ypr_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build R = Ry(yaw)·Rx(pitch)·Rz(roll) — matches estimator._rot_to_ypr."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    return Ry @ Rx @ Rz


def estimate_orientation_from_image(image: np.ndarray) -> tuple[float, float]:
    """
    Estimate (roll_deg, pitch_deg) of the camera from a single image.

    Returns (0.0, 0.0) when no reliable horizon can be found.
    """
    if image is None:
        raise ValueError("image is None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=80,
        minLineLength=int(w * 0.25), maxLineGap=15,
    )
    if lines is None:
        return 0.0, 0.0

    angles, weights, mid_ys = [], [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Fold into (-90, 90]
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        if abs(ang) <= _MAX_HORIZON_ANGLE:
            length = math.hypot(x2 - x1, y2 - y1)
            angles.append(ang)
            weights.append(length)
            mid_ys.append((y1 + y2) / 2.0)

    if not weights:
        return 0.0, 0.0

    weights = np.asarray(weights, dtype=np.float64)
    roll = float(np.average(angles, weights=weights))

    # Pitch from horizon height: above centre -> positive. fy ≈ image width.
    horizon_y = float(np.average(mid_ys, weights=weights))
    fy = float(w)
    pitch = math.degrees(math.atan2((h / 2.0) - horizon_y, fy))
    return roll, pitch
