"""RED tests for src/orient.py — single-image tilt (roll/pitch) from horizon.

Conventions (image coords x-right, y-down):
  roll  = tilt angle of the dominant horizon line; a horizon whose right end is
          lower (y grows to the right) -> positive roll.
  pitch = derived from horizon height; horizon ABOVE image centre -> positive.
  yaw   = unobservable from one image -> always 0.
"""
import math

import cv2
import numpy as np

from orient import estimate_orientation_from_image, ypr_to_R


def _tilted_horizon(angle_deg, y_center=None, w=640, h=480):
    """White image with one strong black line through (w/2, y_center) at angle."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cx = w / 2.0
    cy = h / 2.0 if y_center is None else float(y_center)
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    L = w
    p1 = (int(cx - L * dx), int(cy - L * dy))
    p2 = (int(cx + L * dx), int(cy + L * dy))
    cv2.line(img, p1, p2, (0, 0, 0), 5, cv2.LINE_AA)
    return img


def test_level_horizon_gives_near_zero_roll():
    roll, pitch = estimate_orientation_from_image(_tilted_horizon(0.0))
    assert abs(roll) < 3.0


def test_positive_tilt_recovered():
    roll, _ = estimate_orientation_from_image(_tilted_horizon(15.0))
    assert abs(roll - 15.0) < 4.0


def test_negative_tilt_recovered():
    roll, _ = estimate_orientation_from_image(_tilted_horizon(-20.0))
    assert abs(roll - (-20.0)) < 4.0


def test_horizon_above_centre_gives_positive_pitch():
    _, pitch_high = estimate_orientation_from_image(_tilted_horizon(0.0, y_center=120))
    _, pitch_low = estimate_orientation_from_image(_tilted_horizon(0.0, y_center=360))
    assert pitch_high > pitch_low          # above-centre horizon -> larger pitch
    assert pitch_high > 0 > pitch_low


def test_featureless_image_returns_zero():
    blank = np.full((480, 640, 3), 127, dtype=np.uint8)
    roll, pitch = estimate_orientation_from_image(blank)
    assert roll == 0.0 and pitch == 0.0


def test_ypr_to_R_is_valid_rotation():
    R = ypr_to_R(0.0, 10.0, -15.0)
    assert R.shape == (3, 3)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_ypr_to_R_roundtrips_with_estimator_decomposition():
    from estimator import _rot_to_ypr
    yaw, pitch, roll = 0.0, 12.0, -8.0
    R = ypr_to_R(yaw, pitch, roll)
    y2, p2, r2 = _rot_to_ypr(R)
    assert abs(p2 - pitch) < 1e-6
    assert abs(r2 - roll) < 1e-6
