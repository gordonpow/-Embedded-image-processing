#!/usr/bin/env python3
"""
Single-image camera-pose acceptance on the York Urban Database (YUD) — the
classic vanishing-point benchmark.

YUD ships, per image, the orthogonal Manhattan vanishing directions as a 3x3
rotation `vp_orthogonal` (world axes expressed in the camera frame) plus the
camera intrinsics. From the vertical world axis we derive the TRUE horizon line
(vanishing line of the ground plane, l = K^-T · n_vertical) and hence the
ground-truth camera roll/pitch. We compare against our `orient` estimate
(horizon + vanishing point), and draw GT vs OURS horizons for a montage.

    python benchmarks/validate_yud.py --base test_inputs/yud/db/YorkUrbanDB \
        --out docs/yud_demo.png
"""
import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from orient import detect_horizon


def _line_y(line, x):
    a, b, c = line
    return -(a * x + c) / b


def gt_horizon(R, Kit, cx, cy, fx):
    """GT roll/pitch + horizon line from the orthogonal Manhattan rotation."""
    cols = [R[:, i] for i in range(3)]
    n = max(cols, key=lambda c: abs(c[1]))      # vertical world axis = max |y|
    a, b, c = Kit @ n                            # horizon line a·x+b·y+c=0
    roll = math.degrees(math.atan2(-a, b))
    # A line's tilt is only defined mod 180; fold into (-90, 90] so the sign of
    # the vertical axis can't fake a ~180 deg error.
    if roll > 90:
        roll -= 180
    elif roll <= -90:
        roll += 180
    pitch = math.degrees(math.atan2(cy - _line_y((a, b, c), cx), fx))
    return roll, pitch, (a, b, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="test_inputs/yud/db/YorkUrbanDB")
    ap.add_argument("--out", default="docs/yud_demo.png")
    ap.add_argument("--panels", type=int, default=8)
    args = ap.parse_args()

    cam = sio.loadmat(os.path.join(args.base, "cameraParameters.mat"))
    fx = float(cam["focal"][0, 0]) / float(cam["pixelSize"][0, 0])
    cx, cy = map(float, cam["pp"][0])
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
    Kit = np.linalg.inv(K).T

    rows = []
    for fo in sorted(glob.glob(os.path.join(args.base, "P*"))):
        name = os.path.basename(fo)
        mp = os.path.join(fo, name + "GroundTruthVP_Orthogonal_CamParams.mat")
        jp = os.path.join(fo, name + ".jpg")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            continue
        R = sio.loadmat(mp)["vp_orthogonal"]
        g_roll, g_pitch, gl = gt_horizon(R, Kit, cx, cy, fx)
        det = detect_horizon(cv2.imread(jp))
        rows.append((name, jp, g_roll, g_pitch, det["roll"], det["pitch"], gl, det))

    G = np.array([r[2] for r in rows]); O = np.array([r[4] for r in rows])
    sign = -1 if np.abs(-O - G).mean() < np.abs(O - G).mean() else 1
    roll_err = np.abs(sign * O - G)
    pitch_err = np.abs(np.array([r[5] for r in rows]) - np.array([r[3] for r in rows]))

    print(f"YUD single-image pose: {len(rows)} images")
    print(f"roll  MAE {roll_err.mean():5.1f}  median {np.median(roll_err):4.1f}  (sign {sign:+d})")
    print(f"pitch MAE {pitch_err.mean():5.1f}  median {np.median(pitch_err):4.1f}")
    print(f"roll within 5 deg: {(roll_err < 5).mean()*100:.0f}%")

    # Montage: GT horizon (green) vs OUR horizon (blue) on a spread of images.
    idx = np.linspace(0, len(rows) - 1, args.panels).astype(int)
    panels = []
    for i in idx:
        name, jp, g_roll, g_pitch, o_roll, o_pitch, gl, det = rows[i]
        im = cv2.imread(jp); w = im.shape[1]
        cv2.line(im, (0, int(_line_y(gl, 0))), (w, int(_line_y(gl, w))), (80, 255, 80), 2, cv2.LINE_AA)
        cxp, hy = det["center"][0], det["horizon_y"]; s = math.tan(math.radians(det["roll"]))
        cv2.line(im, (0, int(hy - cxp * s)), (w, int(hy + (w - cxp) * s)), (255, 130, 60), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(im, f"GT roll {g_roll:+.0f}", (6, 18), 0, 0.55, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(im, f"OUR roll {sign*o_roll:+.0f}", (6, 38), 0, 0.55, (255, 150, 80), 2, cv2.LINE_AA)
        panels.append(cv2.resize(im, (320, 240)))
    cols = 4
    grid = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    montage = np.vstack(grid)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
