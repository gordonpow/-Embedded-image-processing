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


def _fold_angle(deg: float) -> float:
    """Fold a line angle into (-90, 90]."""
    if deg > 90:
        return deg - 180
    if deg <= -90:
        return deg + 180
    return deg


def detect_horizon(image: np.ndarray) -> dict:
    """
    Detect the horizon from near-horizontal Hough segments.

    Returns a dict with:
        roll       : weighted tilt angle (deg)
        pitch      : from horizon height (deg)
        horizon_y  : aggregate horizon row (px)
        center     : (cx, cy)
        segments   : list of (x1, y1, x2, y2) horizontal segments actually used
    """
    if image is None:
        raise ValueError("image is None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=80,
        minLineLength=int(w * 0.25), maxLineGap=15,
    )
    empty = {"roll": 0.0, "pitch": 0.0, "horizon_y": h / 2.0,
             "center": center, "segments": []}
    if lines is None:
        return empty

    angles, weights, mid_ys, segs = [], [], [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = _fold_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if abs(ang) <= _MAX_HORIZON_ANGLE:
            angles.append(ang)
            weights.append(math.hypot(x2 - x1, y2 - y1))
            mid_ys.append((y1 + y2) / 2.0)
            segs.append((int(x1), int(y1), int(x2), int(y2)))

    if not weights:
        return empty

    weights = np.asarray(weights, dtype=np.float64)
    roll = float(np.average(angles, weights=weights))
    horizon_y = float(np.average(mid_ys, weights=weights))
    pitch = math.degrees(math.atan2((h / 2.0) - horizon_y, float(w)))
    return {"roll": roll, "pitch": pitch, "horizon_y": horizon_y,
            "center": center, "segments": segs}


def estimate_orientation_from_image(image: np.ndarray) -> tuple[float, float]:
    """Convenience wrapper returning just (roll_deg, pitch_deg)."""
    det = detect_horizon(image)
    return det["roll"], det["pitch"]


def detect_vertical_lines(image: np.ndarray, max_lines: int = 12) -> list:
    """Detect near-vertical Hough segments (structural / man-made edges)."""
    if image is None:
        raise ValueError("image is None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=80,
        minLineLength=int(h * 0.25), maxLineGap=15,
    )
    if lines is None:
        return []
    out = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = _fold_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if abs(ang) >= 90 - _MAX_HORIZON_ANGLE:   # near vertical
            out.append((int(x1), int(y1), int(x2), int(y2)))
    # longest first
    out.sort(key=lambda s: (s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2, reverse=True)
    return out[:max_lines]


def vanishing_point(segments: list):
    """
    Least-squares intersection of >=2 line segments, or None.

    Each segment defines a line a·x + b·y + c = 0 with normal (a,b)=(dy,-dx);
    stacking them and solving A·p = -c gives the common (vanishing) point.
    """
    if segments is None or len(segments) < 2:
        return None
    A, rhs = [], []
    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy)
        if n < 1e-6:
            continue
        a, b = dy / n, -dx / n
        c = -(a * x1 + b * y1)
        A.append([a, b])
        rhs.append(-c)
    if len(A) < 2:
        return None
    p, *_ = np.linalg.lstsq(np.asarray(A, float), np.asarray(rhs, float), rcond=None)
    return float(p[0]), float(p[1])
