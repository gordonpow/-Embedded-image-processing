import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator
from visualize import draw_pose_overlay, draw_orientation_indicator

_EMA_ALPHA = 0.1   # same smoothing factor as folder-1 fire detector


def run(args) -> dict:
    """
    Main processing loop.

    Returns a stats dict with avg_fps, w, h, total_frames, csv_path, mp4_path.
    """
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 <= src_fps <= 120.0):
        src_fps = 30.0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve target resolution
    if getattr(args, "pi_sim", False):
        tgt_w, tgt_h = 640, 480
    elif args.imgsz > 0:
        tgt_w = args.imgsz
        tgt_h = max(1, int(src_h * args.imgsz / src_w))
    else:
        tgt_w, tgt_h = src_w, src_h

    # Camera intrinsics — approximate when no calibration file given
    K = _build_K(args, tgt_w, tgt_h)

    print(
        f"[init] source={args.source}  "
        f"resolution={tgt_w}x{tgt_h}  "
        f"src_fps={src_fps:.1f}  "
        f"intrinsics={'calib' if args.calib else f'approx fx=fy={tgt_w}'}  "
        f"nfeatures={args.nfeatures}"
    )

    # Output paths
    stem = os.path.splitext(os.path.basename(args.source))[0]
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    out_mp4 = getattr(args, "output", None) or os.path.join(runs_dir, f"{stem}_pose.mp4")
    out_csv = os.path.splitext(out_mp4)[0] + ".csv"

    writer = None
    if not getattr(args, "no_video", False):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, src_fps, (tgt_w, tgt_h))

    estimator = PoseEstimator(
        nfeatures=args.nfeatures,
        ratio_thresh=args.ratio_thresh,
        ransac_thresh=args.ransac_thresh,
        keyframe_min_ratio=args.keyframe_ratio,
        keyframe_min_inliers=args.keyframe_inliers,
    )
    estimator.set_K(K)

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame_idx", "timestamp_s", "yaw_deg", "pitch_deg", "roll_deg",
         "fps", "inliers", "nfeatures"]
    )

    ema_fps = 0.0
    frame_idx = 0
    fps_sum = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (tgt_w, tgt_h) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, (tgt_w, tgt_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        yaw, pitch, roll, inliers, npts = estimator.process(gray)
        dt = time.perf_counter() - t0

        inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
        ema_fps = (_EMA_ALPHA * inst_fps + (1 - _EMA_ALPHA) * ema_fps
                   if frame_idx > 0 else inst_fps)
        fps_sum += inst_fps

        timestamp = frame_idx / src_fps
        csv_writer.writerow([
            frame_idx, f"{timestamp:.3f}",
            f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}",
            f"{ema_fps:.1f}", inliers, npts,
        ])

        draw_pose_overlay(frame, yaw, pitch, roll, ema_fps, inliers, npts, tgt_w, tgt_h)
        draw_orientation_indicator(frame, estimator.R_global)

        if writer:
            writer.write(frame)

        if not getattr(args, "no_show", False):
            cv2.imshow("Camera Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    csv_file.close()

    avg_fps = fps_sum / frame_idx if frame_idx > 0 else 0.0
    return {
        "avg_fps": avg_fps,
        "w": tgt_w,
        "h": tgt_h,
        "total_frames": frame_idx,
        "csv_path": out_csv,
        "mp4_path": out_mp4 if not getattr(args, "no_video", False) else None,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_K(args, w: int, h: int) -> np.ndarray:
    calib = getattr(args, "calib", None)
    if calib:
        fs = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        fs.release()
        return K.astype(np.float64)
    # Approximate: fx = fy = W, principal point at image center
    return np.array(
        [[w, 0, w / 2.0],
         [0, w, h / 2.0],
         [0, 0, 1.0]],
        dtype=np.float64,
    )
