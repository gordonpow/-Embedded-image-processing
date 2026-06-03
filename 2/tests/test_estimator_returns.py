"""RED tests for estimator.process() extended return: + t, pts1_in, pts2_in."""
import cv2
import numpy as np

from estimator import PoseEstimator


def _textured_frame(seed=7, w=640, h=480):
    """Dense, ORB-friendly texture so matching reliably succeeds."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for _ in range(450):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 14))
        c = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.circle(img, (cx, cy), r, c, -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _approx_K(w=640, h=480):
    return np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _rotated(gray, deg=2.0):
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h))


def test_process_returns_nine_tuple():
    est = PoseEstimator()
    est.set_K(_approx_K())
    out = est.process(_textured_frame())
    assert len(out) == 9


def test_reference_frame_returns_none_pose_and_empty_points():
    est = PoseEstimator()
    est.set_K(_approx_K())
    yaw, pitch, roll, inliers, npts, R_rel, t, pts1, pts2 = est.process(_textured_frame())
    assert inliers == -1            # first frame becomes the reference
    assert R_rel is None
    assert t is None
    assert np.asarray(pts1).size == 0
    assert np.asarray(pts2).size == 0


def test_tracked_frame_returns_unit_t_and_matched_inlier_points():
    est = PoseEstimator()
    est.set_K(_approx_K())
    g1 = _textured_frame()
    est.process(g1)                       # sets reference
    out = est.process(_rotated(g1, 2.0))  # tracked against reference
    inliers, R_rel, t, pts1, pts2 = out[3], out[5], out[6], out[7], out[8]

    assert inliers > 0                    # a pose was actually recovered
    assert t is not None
    # R_rel is a valid rotation: orthonormal with det +1
    R_rel = np.asarray(R_rel, dtype=np.float64)
    assert R_rel.shape == (3, 3)
    assert abs(np.linalg.det(R_rel) - 1.0) < 1e-6
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    assert t.shape == (3,)
    assert abs(np.linalg.norm(t) - 1.0) < 1e-6   # recoverPose t is unit length

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    assert pts1.shape == pts2.shape
    assert pts1.shape[0] == inliers       # returned points are the inlier subset
    assert pts1.shape[1] == 2
