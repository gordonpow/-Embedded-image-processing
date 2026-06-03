"""RED tests for src/depth.py — relative sparse depth via triangulation.

Scale is ambiguous (unit baseline), so we assert *ordering* and coarse levels,
never absolute distances.
"""
import math

import numpy as np

from depth import relative_depth

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
R = np.eye(3, dtype=np.float64)
T_UNIT = np.array([1.0, 0.0, 0.0], dtype=np.float64).reshape(3, 1)  # unit baseline


def _project(pts3d, t):
    """Project Nx3 world points into a camera at (R=I, translation t)."""
    cam = pts3d + t.reshape(1, 3)
    uv = (K @ cam.T).T
    return (uv[:, :2] / uv[:, 2:3]).astype(np.float32)


def _cloud(zmin, zmax, n=12, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.6, 0.6, n)
    y = rng.uniform(-0.4, 0.4, n)
    z = rng.uniform(zmin, zmax, n)
    return np.stack([x, y, z], axis=1)


def _views(pts3d):
    p1 = _project(pts3d, np.zeros(3))          # cam 1 at origin
    p2 = _project(pts3d, T_UNIT.reshape(3))    # cam 2 shifted by unit baseline
    return p1, p2


def test_near_cloud_has_smaller_depth_than_far_cloud():
    near = _cloud(1.5, 3.0, seed=1)
    far = _cloud(18.0, 25.0, seed=2)
    d_near, lvl_near = relative_depth(K, R, T_UNIT, *_views(near))
    d_far, lvl_far = relative_depth(K, R, T_UNIT, *_views(far))

    assert math.isfinite(d_near) and d_near > 0
    assert math.isfinite(d_far) and d_far > 0
    assert d_near < d_far                       # core: triangulation ordering


def test_depth_levels_are_coarsely_correct():
    near = _cloud(1.5, 3.0, seed=3)
    far = _cloud(18.0, 25.0, seed=4)
    _, lvl_near = relative_depth(K, R, T_UNIT, *_views(near))
    _, lvl_far = relative_depth(K, R, T_UNIT, *_views(far))
    assert lvl_near == "NEAR"
    assert lvl_far == "FAR"
    assert lvl_near != lvl_far


def test_too_few_points_returns_nan_and_na_level():
    p = np.empty((0, 2), dtype=np.float32)
    d, lvl = relative_depth(K, R, T_UNIT, p, p)
    assert math.isnan(d)
    assert lvl == "N/A"


def test_none_translation_returns_na():
    p1, p2 = _views(_cloud(2.0, 4.0, seed=5))
    d, lvl = relative_depth(K, R, None, p1, p2)
    assert math.isnan(d)
    assert lvl == "N/A"
