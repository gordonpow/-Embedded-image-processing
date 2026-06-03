"""
Generate a synthetic test video with a real 3D point cloud + parallax.

Unlike gen_synthetic.py (a pure-rotation homography warp, which is *degenerate*
for a 5-point essential-matrix estimator), this renders a fixed 3D scene viewed
by a camera that both ROTATES (the signal we validate) and TRANSLATES (to give
the parallax the essential matrix needs) — exactly the regime a real hand-held
clip lives in.

Ground truth (synthetic3d_pose_gt.csv) is the camera's world→camera rotation as
yaw/pitch/roll, i.e. precisely what PoseEstimator accumulates into R_global, so
the two are directly comparable.
"""
import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from orient import ypr_to_R

W, H = 640, 480
FPS = 30
DURATION_S = 12
N_POINTS = 500

_HERE = os.path.dirname(__file__)
out_path = os.path.join(_HERE, "synthetic3d_pose_test.mp4")
gt_path = os.path.join(_HERE, "synthetic3d_pose_gt.csv")

K = np.array([[W, 0, W / 2.0], [0, W, H / 2.0], [0, 0, 1.0]], dtype=np.float64)
DIST = np.zeros(5)

# Fixed 3D scene: a cloud of coloured points spread in front of the camera.
rng = np.random.default_rng(7)
pts3d = np.column_stack([
    rng.uniform(-4.0, 4.0, N_POINTS),     # X
    rng.uniform(-3.0, 3.0, N_POINTS),     # Y
    rng.uniform(4.0, 14.0, N_POINTS),     # Z (depth) — varied depth => parallax
]).astype(np.float64)
colors = rng.integers(90, 255, (N_POINTS, 3)).tolist()
radii = rng.integers(3, 8, N_POINTS).tolist()


def _profile(t):
    """Smooth yaw/pitch/roll (deg) + small translation (units) over time."""
    yaw = 20.0 * math.sin(2 * math.pi * t / 6.0)
    pitch = 12.0 * math.sin(2 * math.pi * t / 8.0)
    roll = 15.0 * math.sin(2 * math.pi * t / 10.0)
    # Gentle sway gives parallax without dominating the rotation signal.
    tx = 0.6 * math.sin(2 * math.pi * t / 5.0)
    ty = 0.3 * math.sin(2 * math.pi * t / 7.0)
    tz = 0.4 * math.sin(2 * math.pi * t / 9.0)
    return yaw, pitch, roll, np.array([tx, ty, tz], dtype=np.float64)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
    gt_file = open(gt_path, "w", newline="", encoding="utf-8")
    gt = csv.writer(gt_file)
    gt.writerow(["frame_idx", "yaw_deg", "pitch_deg", "roll_deg"])

    total = DURATION_S * FPS
    for i in range(total):
        t = i / FPS
        yaw, pitch, roll, tvec = _profile(t)
        R_wc = ypr_to_R(yaw, pitch, roll)            # world -> camera rotation
        rvec, _ = cv2.Rodrigues(R_wc)

        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, DIST)
        proj = proj.reshape(-1, 2)

        frame = np.full((H, W, 3), 25, dtype=np.uint8)
        in_front = (R_wc @ pts3d.T + tvec.reshape(3, 1))[2] > 0.1
        for j, (x, y) in enumerate(proj):
            if not in_front[j]:
                continue
            xi, yi = int(round(x)), int(round(y))
            if -20 <= xi < W + 20 and -20 <= yi < H + 20:
                cv2.circle(frame, (xi, yi), radii[j], colors[j], -1, cv2.LINE_AA)

        writer.write(frame)
        gt.writerow([i, f"{yaw:.4f}", f"{pitch:.4f}", f"{roll:.4f}"])

    writer.release()
    gt_file.close()
    print(f"[gen3d] video -> {out_path}  ({total} frames)")
    print(f"[gen3d] ground truth -> {gt_path}")


if __name__ == "__main__":
    main()
