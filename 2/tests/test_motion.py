"""RED tests for src/motion.py — camera motion (t) + optical-flow direction.

Direction conventions locked here:
  camera frame  : X-right, Y-down, Z-forward (OpenCV).
      +Z -> FWD   -Z -> BACK
      +X -> RIGHT -X -> LEFT
      +Y -> DOWN  -Y -> UP   (Y points down)
  optical flow  : a scene that moves RIGHT in the image means the camera
      panned LEFT, so median +dx -> "PAN-L", -dx -> "PAN-R";
      median +dy (scene moves down) -> camera tilted up -> "TILT-U".
"""
import numpy as np
import pytest

from motion import camera_motion, flow_direction


# --------------------------- camera_motion ---------------------------

@pytest.mark.parametrize("vec,expected", [
    ([0, 0, 1], "FWD"),
    ([0, 0, -1], "BACK"),
    ([1, 0, 0], "RIGHT"),
    ([-1, 0, 0], "LEFT"),
    ([0, 1, 0], "DOWN"),
    ([0, -1, 0], "UP"),
])
def test_camera_motion_dominant_axis(vec, expected):
    t = np.array(vec, dtype=np.float64).reshape(3, 1)
    assert camera_motion(t) == expected


def test_camera_motion_none_is_still():
    assert camera_motion(None) == "STILL"


def test_camera_motion_mixed_picks_largest_component():
    # forward-dominant with small sideways drift -> FWD
    t = np.array([0.2, -0.1, 0.95], dtype=np.float64).reshape(3, 1)
    assert camera_motion(t) == "FWD"


# --------------------------- flow_direction ---------------------------

def _ring(center=(100.0, 100.0), radius=40.0, n=8):
    cx, cy = center
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1).astype(np.float32)


def test_flow_scene_moves_right_is_pan_left():
    p1 = _ring()
    p2 = p1 + np.array([10.0, 0.0], dtype=np.float32)
    label, zoom = flow_direction(p1, p2)
    assert label == "PAN-L"
    assert zoom is False


def test_flow_scene_moves_left_is_pan_right():
    p1 = _ring()
    p2 = p1 + np.array([-10.0, 0.0], dtype=np.float32)
    label, _ = flow_direction(p1, p2)
    assert label == "PAN-R"


def test_flow_scene_moves_down_is_tilt_up():
    p1 = _ring()
    p2 = p1 + np.array([0.0, 10.0], dtype=np.float32)
    label, _ = flow_direction(p1, p2)
    assert label == "TILT-U"


def test_flow_scene_moves_up_is_tilt_down():
    p1 = _ring()
    p2 = p1 + np.array([0.0, -10.0], dtype=np.float32)
    label, _ = flow_direction(p1, p2)
    assert label == "TILT-D"


def test_flow_expanding_points_is_zoom_in():
    p1 = _ring()
    center = p1.mean(axis=0)
    p2 = (center + 1.2 * (p1 - center)).astype(np.float32)  # radial expansion
    _, zoom = flow_direction(p1, p2)
    assert zoom is True


def test_flow_empty_points_is_na():
    empty = np.empty((0, 2), dtype=np.float32)
    label, zoom = flow_direction(empty, empty)
    assert label == "N/A"
    assert zoom is False
