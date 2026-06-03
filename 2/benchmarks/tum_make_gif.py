#!/usr/bin/env python3
"""
Build an animated GIF comparing our estimate against the TUM ground truth,
frame by frame (GT in green, OURS in blue, running geodesic error).

    python benchmarks/tum_make_gif.py --seq test_inputs/tum/rgbd_dataset_freiburg1_xyz \
        --out docs/tum_xyz.gif --step 3 --max-frames 360
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator, _rot_to_ypr
from orient import ypr_to_R
from validate_tum import quat_to_R, _read_pairs, geodesic_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", default="docs/tum_xyz.gif")
    ap.add_argument("--step", type=int, default=3, help="keep every Nth frame in the gif")
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--size", type=int, default=400, help="gif frame width")
    ap.add_argument("--ms", type=int, default=90, help="frame duration (ms)")
    ap.add_argument("--colors", type=int, default=96, help="GIF palette size")
    args = ap.parse_args()

    rgb = _read_pairs(os.path.join(args.seq, "rgb.txt"))
    gt = _read_pairs(os.path.join(args.seq, "groundtruth.txt"))
    gts = np.array([float(r[0]) for r in gt])
    gR = [quat_to_R(*map(float, r[4:8])) for r in gt]

    est = PoseEstimator()
    K = None
    g0 = None
    gif = []
    n = min(args.max_frames, len(rgb))
    for i in range(n):
        ts = float(rgb[i][0])
        img = cv2.imread(os.path.join(args.seq, rgb[i][1]))
        if K is None:
            h, w = img.shape[:2]
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]])
            est.set_K(K)
        y, p, r, *_ = est.process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        j = int(np.argmin(np.abs(gts - ts)))
        if g0 is None:
            g0 = gR[j]
        if i % args.step:
            continue
        gy = _rot_to_ypr(g0.T @ gR[j])
        pr = ypr_to_R(y, p, r).T
        py = _rot_to_ypr(pr)
        err = geodesic_deg(pr, g0.T @ gR[j])

        cv2.rectangle(img, (0, 0), (img.shape[1], 74), (0, 0, 0), -1)
        cv2.putText(img, f"TUM {os.path.basename(args.seq).split('_')[-1]}  frame {i}  err {err:3.0f} deg",
                    (8, 22), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"TUM TRUTH Y{gy[0]:+5.0f} P{gy[1]:+5.0f} R{gy[2]:+5.0f}", (8, 46),
                    0, 0.55, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(img, f"OUR EST.  Y{py[0]:+5.0f} P{py[1]:+5.0f} R{py[2]:+5.0f}", (8, 68),
                    0, 0.55, (80, 130, 255), 2, cv2.LINE_AA)

        hgt = int(img.shape[0] * args.size / img.shape[1])
        small = cv2.resize(img, (args.size, hgt))
        # Quantise to a small adaptive palette to keep the GIF README-friendly.
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        gif.append(pil.convert("P", palette=Image.ADAPTIVE, colors=args.colors))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gif[0].save(args.out, save_all=True, append_images=gif[1:],
                duration=args.ms, loop=0, optimize=True)
    mb = os.path.getsize(args.out) / 1e6
    print(f"saved -> {args.out}  ({len(gif)} frames, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
