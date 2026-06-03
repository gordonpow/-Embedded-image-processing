#!/usr/bin/env python3
"""
Side-by-side comparison on real TUM RGB-D test frames:
for chosen frames, draw the actual image + the dataset's GROUND-TRUTH angles
(green) and OUR estimate (blue), so "ours vs theirs" is visible at a glance.

    python benchmarks/tum_frame_compare.py --seq test_inputs/tum/rgbd_dataset_freiburg1_desk
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator, _rot_to_ypr
from orient import ypr_to_R
from validate_tum import quat_to_R, _read_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames", default="40,120,200,400")
    ap.add_argument("--out", default="runs/tum_compare_frames.png")
    args = ap.parse_args()

    picks = [int(x) for x in args.frames.split(",")]
    n = max(picks) + 1
    rgb = _read_pairs(os.path.join(args.seq, "rgb.txt"))
    gt = _read_pairs(os.path.join(args.seq, "groundtruth.txt"))
    gts = np.array([float(r[0]) for r in gt])
    gR = [quat_to_R(*map(float, r[4:8])) for r in gt]

    est = PoseEstimator()
    K = None
    P, G, imgs = [], [], []
    for i in range(n):
        ts = float(rgb[i][0])
        img = cv2.imread(os.path.join(args.seq, rgb[i][1]))
        if K is None:
            h, w = img.shape[:2]
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]])
            est.set_K(K)
        y, p, r, *_ = est.process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        j = int(np.argmin(np.abs(gts - ts)))
        P.append(ypr_to_R(y, p, r).T)        # our pose (recoverPose-inverse convention)
        G.append(gR[j])
        imgs.append(img)

    g0 = G[0]
    panels = []
    for fi in picks:
        gy = _rot_to_ypr(g0.T @ G[fi])       # GT relative to frame 0
        py = _rot_to_ypr(P[fi])
        im = imgs[fi].copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 74), (0, 0, 0), -1)
        cv2.putText(im, f"frame {fi}", (8, 22), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"GT   Y{gy[0]:+5.0f} P{gy[1]:+5.0f} R{gy[2]:+5.0f}",
                    (8, 46), 0, 0.6, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(im, f"OURS Y{py[0]:+5.0f} P{py[1]:+5.0f} R{py[2]:+5.0f}",
                    (8, 68), 0, 0.6, (80, 130, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(im, (480, 360)))

    half = (len(panels) + 1) // 2
    top = np.hstack(panels[:half])
    bot = np.hstack(panels[half:])
    if bot.shape[1] < top.shape[1]:
        bot = np.hstack([bot, np.zeros((bot.shape[0], top.shape[1] - bot.shape[1], 3), np.uint8)])
    out = np.vstack([top, bot])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
