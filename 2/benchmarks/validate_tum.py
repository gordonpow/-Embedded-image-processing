#!/usr/bin/env python3
"""
Pose acceptance on the TUM RGB-D benchmark — a real-world standard answer.

TUM provides motion-capture ground-truth camera poses (groundtruth.txt:
  timestamp tx ty tz qx qy qz qw). We run our estimator on the RGB frames and
compare the accumulated rotation against the GT rotation (both expressed
relative to the first frame).

TUM uses the optical-frame convention (x-right, y-down, z-forward) — the same
as OpenCV — so no axis remap is needed. recoverPose's rotation may be the
inverse of the GT frame-change, so we auto-pick whichever convention (R or Rᵀ)
fits better and report it (a geodesic-angle comparison, convention-robust).

Usage
-----
    python benchmarks/validate_tum.py --seq test_inputs/tum/rgbd_dataset_freiburg1_desk
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator, _rot_to_ypr
from orient import ypr_to_R
from metrics import angular_error_stats


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="extracted TUM sequence folder")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--assoc", type=float, default=0.02, help="rgb<->gt time match (s)")
    ap.add_argument("--threshold", type=float, default=10.0, help="PASS if geodesic MAE < this")
    ap.add_argument("--plot", default=None, help="save a pred-vs-GT comparison PNG")
    args = ap.parse_args()

    rgb = _read_pairs(os.path.join(args.seq, "rgb.txt"))           # ts, filename
    gt = _read_pairs(os.path.join(args.seq, "groundtruth.txt"))    # ts tx ty tz qx qy qz qw
    gt_ts = np.array([float(r[0]) for r in gt])
    gt_R = [quat_to_R(*map(float, r[4:8])) for r in gt]

    est = PoseEstimator()
    K = None
    pred_R, paired_gt_R = [], []
    n = min(args.max_frames, len(rgb))
    for i in range(n):
        ts = float(rgb[i][0])
        img = cv2.imread(os.path.join(args.seq, rgb[i][1]))
        if img is None:
            continue
        if K is None:
            h, w = img.shape[:2]
            K = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
            est.set_K(K)
        yaw, pitch, roll, *_ = est.process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # nearest GT pose in time
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) > args.assoc:
            continue
        pred_R.append(ypr_to_R(yaw, pitch, roll))
        paired_gt_R.append(gt_R[j])

    if len(pred_R) < 5:
        sys.exit("[error] too few rgb<->gt associations")

    # Make both rotations relative to the first paired frame.
    gt0 = paired_gt_R[0]
    gt_rel = [gt0.T @ R for R in paired_gt_R]

    # Auto-pick convention: R_pred vs R_pred^T against GT (geodesic).
    errA = [geodesic_deg(p, g) for p, g in zip(pred_R, gt_rel)]
    errB = [geodesic_deg(p.T, g) for p, g in zip(pred_R, gt_rel)]
    use_T = np.mean(errB) < np.mean(errA)
    geo = errB if use_T else errA
    pred_use = [p.T if use_T else p for p in pred_R]

    # Per-axis (after convention choice).
    py = np.array([_rot_to_ypr(R) for R in pred_use])
    gy = np.array([_rot_to_ypr(R) for R in gt_rel])

    print(f"\n=== TUM acceptance ===  seq={os.path.basename(args.seq)}  paired_frames={len(geo)}")
    print(f"convention: {'R^T (recoverPose inverse)' if use_T else 'R'}")
    print(f"{'axis':6s} {'MAE':>8s} {'RMSE':>8s}")
    for i, axis in enumerate(("yaw", "pitch", "roll")):
        s = angular_error_stats(py[:, i], gy[:, i])
        print(f"{axis:6s} {s['mae']:8.2f} {s['rmse']:8.2f}")
    geo = np.asarray(geo)
    print(f"\ngeodesic rotation error:  MAE={geo.mean():.2f}  RMSE={math.sqrt((geo**2).mean()):.2f}  MAX={geo.max():.2f} deg")
    verdict = "PASS" if geo.mean() < args.threshold else "FAIL"
    print(f"threshold {args.threshold:.1f} deg -> {verdict}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for i, axis in enumerate(("yaw", "pitch", "roll")):
            axs[i].plot(gy[:, i], "g-", lw=2, label="TUM ground truth")
            axs[i].plot(py[:, i], "r--", lw=1.3, label="our estimate")
            axs[i].set_ylabel(f"{axis} (deg)")
            axs[i].grid(alpha=0.3)
            axs[i].legend(loc="upper left", fontsize=8)
        axs[0].set_title(f"TUM {os.path.basename(args.seq)} — estimate vs ground truth "
                         f"(geodesic MAE {geo.mean():.1f}°)")
        axs[2].set_xlabel("paired frame index")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
