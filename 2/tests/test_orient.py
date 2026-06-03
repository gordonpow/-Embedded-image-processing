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

from orient import (
    estimate_orientation_from_image,
    ypr_to_R,
    detect_horizon,
    detect_vertical_lines,
    vanishing_point,
    image_pose,
)


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


# --------------------- detection helpers (for drawing evidence) ---------------------

def test_detect_horizon_exposes_segments():
    det = detect_horizon(_tilted_horizon(15.0))
    assert "segments" in det and "roll" in det and "horizon_y" in det
    assert len(det["segments"]) >= 1
    assert abs(det["roll"] - 15.0) < 4.0
    # each segment is (x1, y1, x2, y2)
    assert all(len(s) == 4 for s in det["segments"])


def _vertical_lines_image(xs=(200, 440), w=640, h=480):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for x in xs:
        cv2.line(img, (x, 20), (x, h - 20), (0, 0, 0), 4, cv2.LINE_AA)
    return img


def test_detect_vertical_lines_finds_segments():
    segs = detect_vertical_lines(_vertical_lines_image())
    assert len(segs) >= 1
    # each detected segment should be closer to vertical than horizontal
    for x1, y1, x2, y2 in segs:
        assert abs(y2 - y1) > abs(x2 - x1)


def test_vanishing_point_of_two_converging_lines():
    # both lines pass through (320, 100)
    seg1 = (320, 100, 200, 400)
    seg2 = (320, 100, 440, 400)
    vp = vanishing_point([seg1, seg2])
    assert vp is not None
    assert abs(vp[0] - 320) < 5.0
    assert abs(vp[1] - 100) < 5.0


def test_vanishing_point_needs_two_lines():
    assert vanishing_point([(0, 0, 10, 10)]) is None
    assert vanishing_point([]) is None


# --------------------- single-image yaw via vanishing point ---------------------

W, H = 640, 480


def _det(segments, roll=0.0, pitch=0.0):
    return {"roll": roll, "pitch": pitch, "horizon_y": H / 2.0,
            "center": (W / 2.0, H / 2.0), "segments": segments}


def test_image_pose_recovers_yaw_from_converging_horizontal_lines():
    # Near-horizontal lines all passing through a right-of-centre point (520,240)
    # => a real horizontal vanishing point => observable yaw > 0.
    segs = [(40, 180, 520, 240), (40, 240, 520, 240),
            (40, 300, 520, 240), (100, 140, 520, 240)]
    pose = image_pose(_det(segs), W, H)
    assert pose["yaw_valid"] is True
    assert pose["yaw"] > 5.0                  # VP to the right -> positive yaw


def test_image_pose_parallel_lines_have_no_observable_yaw():
    # Truly parallel horizontal lines (no convergence) -> yaw not observable.
    segs = [(40, 200, 600, 200), (40, 260, 600, 260), (40, 320, 600, 320)]
    pose = image_pose(_det(segs), W, H)
    assert pose["yaw_valid"] is False
    assert pose["yaw"] == 0.0


def test_image_pose_passes_through_roll_and_pitch():
    pose = image_pose(_det([], roll=4.0, pitch=-3.0), W, H)
    assert abs(pose["roll"] - 4.0) < 1e-9
    assert abs(pose["pitch"] - (-3.0)) < 1e-9
    assert pose["yaw_valid"] is False          # no segments -> yaw N/A
