"""RED tests for src/scene.py — indoor/outdoor classifier (classical CV + DNN)."""
import numpy as np
import pytest

from scene import classify_indoor_outdoor, softmax, aggregate_io


def _blue_sky_over_green(w=160, h=120):
    """Top half bright blue sky, bottom half green vegetation — clearly outdoor."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = (235, 180, 90)   # BGR: bright blue-ish sky
    img[h // 2 :] = (40, 160, 40)    # BGR: green foliage
    return img


def _dim_gray_room(w=160, h=120):
    """Uniform dim gray — no sky, no vegetation — clearly indoor."""
    return np.full((h, w, 3), 70, dtype=np.uint8)


def test_blue_sky_over_green_is_outdoor():
    label, conf = classify_indoor_outdoor(_blue_sky_over_green())
    assert label == "outdoor"
    assert 0.0 <= conf <= 1.0
    # Confidence should be meaningful for an unambiguous outdoor scene.
    assert conf > 0.5


def test_dim_gray_room_is_indoor():
    label, conf = classify_indoor_outdoor(_dim_gray_room())
    assert label == "indoor"
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5


def test_returns_label_and_float_confidence():
    label, conf = classify_indoor_outdoor(_dim_gray_room())
    assert label in ("indoor", "outdoor")
    assert isinstance(conf, float)


def test_none_input_raises():
    with pytest.raises((ValueError, TypeError)):
        classify_indoor_outdoor(None)


def test_tiny_input_does_not_crash():
    tiny = np.full((4, 4, 3), 120, dtype=np.uint8)
    label, conf = classify_indoor_outdoor(tiny)
    assert label in ("indoor", "outdoor")
    assert 0.0 <= conf <= 1.0


# --------------------- DNN path: pure aggregation logic ---------------------

def test_softmax_sums_to_one():
    p = softmax(np.array([2.0, 1.0, 0.1, -3.0]))
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert np.all(p >= 0)


def test_aggregate_io_indoor_dominant():
    # io flags: 1=indoor, 2=outdoor (Places365 convention)
    io = np.array([1, 1, 2, 2])
    probs = np.array([0.4, 0.4, 0.1, 0.1])
    label, conf = aggregate_io(probs, io)
    assert label == "indoor"
    assert abs(conf - 0.8) < 1e-6


def test_aggregate_io_outdoor_dominant():
    io = np.array([1, 1, 2, 2])
    probs = np.array([0.1, 0.1, 0.5, 0.3])
    label, conf = aggregate_io(probs, io)
    assert label == "outdoor"
    assert abs(conf - 0.8) < 1e-6
    assert 0.0 <= conf <= 1.0


def test_aggregate_io_would_catch_flipped_mapping():
    # If indoor/outdoor were swapped, this all-indoor vote would read 'outdoor'.
    io = np.array([1, 1, 1])
    probs = np.array([0.5, 0.3, 0.2])
    label, _ = aggregate_io(probs, io)
    assert label == "indoor"
