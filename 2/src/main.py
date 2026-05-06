#!/usr/bin/env python3
"""
Camera Pose Estimator — yaw / pitch / roll via ORB + 5-point Essential Matrix.

Target hardware: Raspberry Pi 4B (CPU-only).
Input: pre-recorded video file or image sequence.

Usage
-----
    python src/main.py test_inputs/indoor_office.mp4
    python src/main.py test_inputs/outdoor.mp4 --imgsz 480 --no-show
    python src/main.py video.mp4 --calib my_camera.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monocular relative pose estimator (yaw/pitch/roll) for Raspberry Pi 4B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input / Output ---
    p.add_argument("source", help="video file or image path")
    p.add_argument(
        "--output", "-o", default=None,
        help="output annotated mp4 (default: runs/<stem>_pose.mp4)",
    )

    # --- Resolution ---
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="target width in pixels (height scaled proportionally; 0 = keep original)",
    )
    p.add_argument(
        "--pi-sim", action="store_true",
        help="force 640×480 to simulate Pi 4B headless performance",
    )

    # --- Estimator tuning ---
    p.add_argument("--nfeatures",        type=int,   default=500,  help="ORB max features per frame")
    p.add_argument("--ratio-thresh",     type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--ransac-thresh",    type=float, default=1.0,  help="RANSAC inlier threshold (px)")
    p.add_argument("--keyframe-ratio",   type=float, default=0.5,  help="spawn keyframe if inlier ratio < X")
    p.add_argument("--keyframe-inliers", type=int,   default=60,   help="spawn keyframe if inlier count < X")

    # --- Calibration ---
    p.add_argument(
        "--calib", default=None,
        help="camera_matrix.yaml produced by calibrate.py",
    )

    # --- Display / write flags ---
    p.add_argument("--no-show",  action="store_true", help="headless mode — no display window")
    p.add_argument("--no-video", action="store_true", help="skip writing annotated output mp4")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"[error] source not found: {args.source}")

    stats = run(args)

    print(
        f"\n[result] avg_fps={stats['avg_fps']:.1f}  "
        f"resolution={stats['w']}x{stats['h']}  "
        f"total_frames={stats['total_frames']}  "
        f"csv={stats['csv_path']}"
    )
    if stats.get("mp4_path"):
        print(f"[result] mp4={stats['mp4_path']}")
    print(
        f"\n[hint]   python plot_poses.py {stats['csv_path']}"
    )
