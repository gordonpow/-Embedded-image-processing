#!/usr/bin/env python3
"""
Pose acceptance test — run the estimator on a video with KNOWN per-frame
yaw/pitch/roll and report the angular error against that ground truth.

Default target is the synthetic clip (exact ground truth from gen_synthetic.py);
the same script accepts any video + GT CSV (frame_idx,yaw_deg,pitch_deg,roll_deg),
e.g. a TUM RGB-D sequence converted to Euler angles.

Usage
-----
    python benchmarks/validate_pose.py
    python benchmarks/validate_pose.py --video clip.mp4 --gt gt.csv --threshold 8
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator
from metrics import angular_error_stats

_ROOT = os.path.dirname(os.path.dirname(__file__))


def _approx_K(w, h):
    return np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _run_estimator(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    est = PoseEstimator()
    est.set_K(_approx_K(w, h))
    ypr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yaw, pitch, roll, *_ = est.process(gray)
        ypr.append((yaw, pitch, roll))
    cap.release()
    return np.asarray(ypr, dtype=np.float64)


def _load_gt(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["yaw_deg"]), float(r["pitch_deg"]), float(r["roll_deg"])))
    return np.asarray(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Validate yaw/pitch/roll vs ground truth")
    ap.add_argument("--video", default=os.path.join(_ROOT, "test_inputs", "synthetic_pose_test.mp4"))
    ap.add_argument("--gt", default=os.path.join(_ROOT, "test_inputs", "synthetic_pose_gt.csv"))
    # Monocular VO without loop closure drifts ~linearly with sequence length;
    # 15 deg is a defensible per-axis MAE bound for a ~12 s clip.
    ap.add_argument("--threshold", type=float, default=15.0, help="PASS if every-axis MAE < this (deg)")
    args = ap.parse_args()

    pred = _run_estimator(args.video)
    gt = _load_gt(args.gt)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    print(f"\n=== Pose acceptance ===  frames={n}  video={os.path.basename(args.video)}")
    print(f"{'axis':6s} {'MAE':>8s} {'RMSE':>8s} {'MAX':>8s}")
    worst_mae = 0.0
    for i, axis in enumerate(("yaw", "pitch", "roll")):
        s = angular_error_stats(pred[:, i], gt[:, i])
        worst_mae = max(worst_mae, s["mae"])
        print(f"{axis:6s} {s['mae']:8.2f} {s['rmse']:8.2f} {s['max']:8.2f}")

    verdict = "PASS" if worst_mae < args.threshold else "FAIL"
    print(f"\nworst-axis MAE = {worst_mae:.2f} deg   threshold = {args.threshold:.1f} deg   -> {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
