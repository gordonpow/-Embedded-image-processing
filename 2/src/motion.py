"""
Dynamic motion direction — two complementary cues, both essentially free
because the inputs are already produced by ``PoseEstimator.process``.

  * camera_motion(t)        — direction the camera translated, from recoverPose's
                              unit translation vector (camera frame X-right,
                              Y-down, Z-forward).
  * flow_direction(p1, p2)  — apparent picture motion from matched inlier points:
                              median displacement -> pan/tilt label, radial
                              divergence -> zoom-in flag.

Direction conventions (kept identical to tests/test_motion.py):
  +Z->FWD -Z->BACK  +X->RIGHT -X->LEFT  +Y->DOWN -Y->UP
  scene moves right (+dx) -> camera panned left  -> "PAN-L"
  scene moves down  (+dy) -> camera tilted up    -> "TILT-U"
"""
import numpy as np

_STILL_EPS = 1e-6          # treat near-zero translation / flow as no motion
_FLOW_MIN_PX = 1.0         # median displacement below this -> no pan/tilt
_ZOOM_MIN = 0.0            # mean radial expansion above this -> zoom in

_CAM_LABELS = {
    0: ("RIGHT", "LEFT"),   # X axis: +X right, -X left
    1: ("DOWN", "UP"),      # Y axis: +Y down,  -Y up
    2: ("FWD", "BACK"),     # Z axis: +Z forward, -Z back
}


def camera_motion(t) -> str:
    """Dominant translation axis as FWD/BACK/LEFT/RIGHT/UP/DOWN, or STILL."""
    if t is None:
        return "STILL"
    v = np.asarray(t, dtype=np.float64).reshape(-1)
    if v.shape[0] != 3:
        return "STILL"
    n = np.linalg.norm(v)
    if n < _STILL_EPS:
        return "STILL"
    v = v / n
    axis = int(np.argmax(np.abs(v)))
    pos, neg = _CAM_LABELS[axis]
    return pos if v[axis] >= 0 else neg


def flow_direction(pts1, pts2) -> tuple[str, bool]:
    """
    Apparent picture motion from matched points.

    Returns
    -------
    (label, zoom_in)
        label   : "PAN-L" | "PAN-R" | "TILT-U" | "TILT-D" | "STILL" | "N/A"
        zoom_in : True when points diverge radially from their centroid.
    """
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if p1.shape[0] == 0 or p1.shape != p2.shape:
        return "N/A", False

    disp = p2 - p1
    med = np.median(disp, axis=0)
    dx, dy = float(med[0]), float(med[1])

    # --- zoom: mean radial component of displacement w.r.t. point centroid ---
    center = p1.mean(axis=0)
    radial_dir = p1 - center
    norms = np.linalg.norm(radial_dir, axis=1, keepdims=True)
    norms[norms < _STILL_EPS] = 1.0
    radial_unit = radial_dir / norms
    radial_proj = float(np.mean(np.sum(disp * radial_unit, axis=1)))
    zoom_in = radial_proj > _ZOOM_MIN and abs(radial_proj) > _FLOW_MIN_PX

    # --- pan / tilt: dominant median displacement axis ---
    if max(abs(dx), abs(dy)) < _FLOW_MIN_PX:
        return "STILL", zoom_in
    if abs(dx) >= abs(dy):
        label = "PAN-L" if dx >= 0 else "PAN-R"
    else:
        label = "TILT-U" if dy >= 0 else "TILT-D"
    return label, zoom_in
