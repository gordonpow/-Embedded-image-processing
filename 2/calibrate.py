#!/usr/bin/env python3
"""
Camera calibration using chessboard — produces camera_matrix.yaml for main.py.

Usage
-----
    # From images in a folder
    python calibrate.py --images calib_images/ --output camera_matrix.yaml

    # From a calibration video (extract frames every N frames)
    python calibrate.py --video calib_video.mp4 --interval 30 --output camera_matrix.yaml

Chessboard default: 9×6 inner corners (i.e. a 10×7 squares board).
Print one at: https://calib.io/pages/camera-calibration-pattern-generator
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OpenCV chessboard camera calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images",   default=None, help="folder with .jpg/.png calibration images")
    ap.add_argument("--video",    default=None, help="calibration video file")
    ap.add_argument("--interval", type=int, default=30,  help="frame interval when using --video")
    ap.add_argument("--pattern",  default="9x6",         help="chessboard inner corners WxH")
    ap.add_argument("--output",   default="camera_matrix.yaml", help="output YAML file")
    ap.add_argument("--show",     action="store_true",   help="show detected corners during calibration")
    args = ap.parse_args()

    cols, rows = map(int, args.pattern.split("x"))
    print(f"[calib] pattern={cols}×{rows} inner corners")

    # World-space corner coordinates (unit = one square side)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    img_size = None

    frames = _gather_frames(args)
    if not frames:
        sys.exit("[error] No images found. Supply --images or --video.")

    found_total = 0
    for i, img in enumerate(frames):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            continue

        cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_pts.append(objp)
        img_pts.append(corners)
        found_total += 1
        print(f"  frame {i:4d}: corners found  [{found_total} total]")

        if args.show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
            cv2.imshow("corners", vis)
            cv2.waitKey(200)

    if args.show:
        cv2.destroyAllWindows()

    if found_total < 10:
        sys.exit(f"[error] Only {found_total} usable frames (need ≥ 10). "
                 "Take more images / reduce --interval.")

    print(f"\n[calib] calibrating with {found_total} frames …")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}  "
          f"cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}")
    print(f"  dist coeffs: {dist.ravel().round(5)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs",   dist)
    fs.write("rms",           float(rms))
    fs.release()
    print(f"\n[calib] saved → {args.output}")
    print(f"[hint]  python src/main.py video.mp4 --calib {args.output}")


def _gather_frames(args) -> list[np.ndarray]:
    if args.images:
        paths = sorted(
            glob.glob(os.path.join(args.images, "*.jpg")) +
            glob.glob(os.path.join(args.images, "*.JPG")) +
            glob.glob(os.path.join(args.images, "*.png")) +
            glob.glob(os.path.join(args.images, "*.PNG"))
        )
        return [f for p in paths if (f := cv2.imread(p)) is not None]

    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % args.interval == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    return []


if __name__ == "__main__":
    main()
