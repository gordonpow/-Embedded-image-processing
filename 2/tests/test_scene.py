"""RED tests for src/scene.py — indoor/outdoor classifier (classical CV)."""
import numpy as np
import pytest

from scene import classify_indoor_outdoor


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
