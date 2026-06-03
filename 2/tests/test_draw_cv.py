"""RED tests for src/draw_cv.py — Chinese localisation + CV-evidence drawing."""
import numpy as np

from draw_cv import tr_value, draw_flow_arrows, draw_horizon


def test_localisation_maps_known_values():
    assert tr_value("indoor") == "室內"
    assert tr_value("outdoor") == "戶外"
    assert tr_value("FWD") == "前進"
    assert tr_value("PAN-L") == "左平移"
    assert tr_value("NEAR") == "近"
    assert tr_value("N/A") == "無"


def test_localisation_passthrough_unknown():
    assert tr_value("12.3") == "12.3"


def test_draw_flow_arrows_modifies_frame_and_caps():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    p1 = rng.uniform(50, 590, (200, 2)).astype(np.float32)
    p2 = p1 + np.array([8.0, 3.0], dtype=np.float32)
    before = frame.sum()
    drawn = draw_flow_arrows(frame, p1, p2, max_arrows=60)
    assert 0 < drawn <= 60                 # capped
    assert frame.sum() > before            # actually drew something


def test_draw_flow_arrows_empty_is_safe():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 2), dtype=np.float32)
    drawn = draw_flow_arrows(frame, empty, empty)
    assert drawn == 0
    assert frame.sum() == 0


def test_draw_horizon_marks_frame_when_segments_present():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = {"roll": 5.0, "pitch": 0.0, "horizon_y": 240.0,
           "center": (320.0, 240.0), "segments": [(50, 230, 590, 250)]}
    draw_horizon(frame, det)
    assert frame.sum() > 0


def test_draw_horizon_empty_segments_is_safe():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = {"roll": 0.0, "pitch": 0.0, "horizon_y": 240.0,
           "center": (320.0, 240.0), "segments": []}
    draw_horizon(frame, det)   # must not raise
