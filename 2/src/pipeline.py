import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from estimator import PoseEstimator
from visualize import draw_pose_overlay, draw_orientation_indicator
from scene import classify_indoor_outdoor, SceneClassifier
from motion import camera_motion, flow_direction
from depth import relative_depth
from orient import (
    ypr_to_R, detect_horizon, detect_vertical_lines, vanishing_point, image_pose,
)
from draw_cv import (
    draw_flow_arrows, draw_horizon, draw_feature_gradient, draw_vertical_vp,
)

_EMA_ALPHA = 0.1        # FPS smoothing — same factor as folder-1 fire detector
_SCENE_INTERVAL = 10    # recompute indoor/outdoor every N frames (Pi-friendly)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_DEFAULT_MODEL = os.path.join(_MODELS_DIR, "places365_resnet18.onnx")
_DEFAULT_IO = os.path.join(_MODELS_DIR, "io_places365.txt")

CSV_HEADER = [
    "frame_idx", "timestamp_s", "yaw_deg", "pitch_deg", "roll_deg",
    "fps", "inliers", "nfeatures",
    "scene", "scene_conf", "cam_motion", "flow_motion", "zoom_in",
    "rel_depth", "depth_level",
]


def run(args) -> dict:
    """
    Main entry point.

    Routes to one of two paths depending on the source:
      * single image  -> indoor/outdoor only (pose/motion/depth need >=2 frames)
      * video / webcam -> full per-frame pose + scene + motion + depth
    """
    cap_arg, kind = _resolve_source(args.source)

    if kind == "image":
        return _run_image(args, cap_arg)
    return _run_stream(args, cap_arg)


# ------------------------------------------------------------------
# Stream (video file or webcam) path
# ------------------------------------------------------------------

def _load_scene_classifier(args):
    """Load the Places365 DNN backend, or None (-> heuristic fallback)."""
    model = getattr(args, "scene_model", None) or _DEFAULT_MODEL
    io = getattr(args, "scene_io", None) or _DEFAULT_IO
    return SceneClassifier.try_load(model, io)


def _run_stream(args, cap_arg) -> dict:
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 <= src_fps <= 120.0):
        src_fps = 30.0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    tgt_w, tgt_h = _target_res(args, src_w, src_h)
    K = _build_K(args, tgt_w, tgt_h)

    print(
        f"[init] source={args.source}  resolution={tgt_w}x{tgt_h}  "
        f"src_fps={src_fps:.1f}  "
        f"intrinsics={'calib' if args.calib else f'approx fx=fy={tgt_w}'}  "
        f"nfeatures={args.nfeatures}"
    )

    out_mp4, out_csv = _output_paths(args)
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

    classifier = _load_scene_classifier(args)

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    ema_fps = 0.0
    frame_idx = 0
    fps_sum = 0.0
    scene_label, scene_conf = "indoor", 0.5   # cached between scene updates

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (tgt_w, tgt_h) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, (tgt_w, tgt_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        yaw, pitch, roll, inliers, npts, R_rel, t_vec, pts1_in, pts2_in = estimator.process(gray)
        dt = time.perf_counter() - t0

        inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
        ema_fps = (_EMA_ALPHA * inst_fps + (1 - _EMA_ALPHA) * ema_fps
                   if frame_idx > 0 else inst_fps)
        fps_sum += inst_fps

        # --- Scene (indoor/outdoor): recompute every N frames, reuse otherwise ---
        if frame_idx % _SCENE_INTERVAL == 0:
            scene_label, scene_conf = classify_indoor_outdoor(frame, classifier)

        # --- Motion direction: camera translation + apparent picture flow ---
        cam_dir = camera_motion(t_vec)
        flow_dir, zoom_in = flow_direction(pts1_in, pts2_in)

        # --- Relative sparse depth from this view pair ---
        rel_d, depth_level = relative_depth(K, R_rel, t_vec, pts1_in, pts2_in)
        rel_d_str = "N/A" if (rel_d != rel_d) else f"{rel_d:.2f}"   # nan check

        # Real optical-flow arrows from the matched inliers (drawn under the HUD).
        draw_flow_arrows(frame, pts1_in, pts2_in)

        timestamp = frame_idx / src_fps
        csv_writer.writerow([
            frame_idx, f"{timestamp:.3f}",
            f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}",
            f"{ema_fps:.1f}", inliers, npts,
            scene_label, f"{scene_conf:.2f}", cam_dir, flow_dir, int(zoom_in),
            rel_d_str, depth_level,
        ])

        draw_pose_overlay(
            frame, yaw, pitch, roll, ema_fps, inliers, npts, tgt_w, tgt_h,
            scene=scene_label, scene_conf=scene_conf,
            cam_motion=cam_dir, flow=flow_dir, zoom_in=zoom_in,
            depth_level=depth_level,
        )
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
        "avg_fps": avg_fps, "w": tgt_w, "h": tgt_h,
        "total_frames": frame_idx, "csv_path": out_csv,
        "mp4_path": out_mp4 if not getattr(args, "no_video", False) else None,
    }


# ------------------------------------------------------------------
# Single image path — only indoor/outdoor is meaningful
# ------------------------------------------------------------------

def _run_image(args, path) -> dict:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    src_h, src_w = img.shape[:2]
    tgt_w, tgt_h = _target_res(args, src_w, src_h)
    if (tgt_w, tgt_h) != (src_w, src_h):
        img = cv2.resize(img, (tgt_w, tgt_h))

    print(
        f"[init] source={args.source}  resolution={tgt_w}x{tgt_h}  "
        f"mode=single-image (yaw needs >=2 frames -> N/A; roll/pitch from horizon)"
    )

    classifier = _load_scene_classifier(args)
    scene_label, scene_conf = classify_indoor_outdoor(img, classifier)

    # Single-image orientation: roll/pitch from the horizon; yaw from a
    # horizontal vanishing point when the scene is structured (else N/A).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = detect_horizon(img)
    pose = image_pose(det, tgt_w, tgt_h)
    roll, pitch, yaw = pose["roll"], pose["pitch"], pose["yaw"]
    yaw_na = not pose["yaw_valid"]
    R_img = ypr_to_R(yaw, pitch, roll)
    vsegs = detect_vertical_lines(img)
    vp = vanishing_point(vsegs)

    out_mp4, out_csv = _output_paths(args)
    out_png = os.path.splitext(out_csv)[0] + ".png"

    with open(out_csv, "w", newline="", encoding="utf-8") as csv_file:
        w = csv.writer(csv_file)
        w.writerow(CSV_HEADER)
        w.writerow([
            0, "0.000", ("N/A" if yaw_na else f"{yaw:.2f}"), f"{pitch:.2f}", f"{roll:.2f}",
            "N/A", "N/A", "N/A",
            scene_label, f"{scene_conf:.2f}", "N/A", "N/A", "N/A",
            "N/A", "N/A",
        ])

    # CV evidence drawn on the image (under the HUD): ORB features + gradient
    # field, detected horizon lines, vertical lines + vanishing point.
    draw_feature_gradient(img, gray)
    draw_vertical_vp(img, vsegs, vp)
    draw_horizon(img, det)

    # Slim HUD (no FPS / stats) + XYZ gizmo from the estimated orientation.
    draw_pose_overlay(
        img, yaw, pitch, roll, 0.0, 0, 0, tgt_w, tgt_h,
        scene=scene_label, scene_conf=scene_conf,
        yaw_na=yaw_na, show_fps=False, show_stats=False,
    )
    draw_orientation_indicator(img, R_img)
    cv2.imwrite(out_png, img)
    yaw_txt = "N/A" if yaw_na else f"{yaw:+.1f}"
    print(
        f"[result] scene={scene_label} ({scene_conf:.2f})  "
        f"yaw={yaw_txt} roll={roll:+.1f} pitch={pitch:+.1f}  png={out_png}  csv={out_csv}"
    )

    return {
        "avg_fps": 0.0, "w": tgt_w, "h": tgt_h,
        "total_frames": 1, "csv_path": out_csv, "mp4_path": None,
        "png_path": out_png, "scene": scene_label,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_source(source):
    """Return (capture_arg, kind). kind is 'image' or 'stream'."""
    s = str(source)
    if s.isdigit():
        return int(s), "stream"          # webcam index
    ext = os.path.splitext(s)[1].lower()
    if ext in _IMAGE_EXTS:
        return s, "image"
    return s, "stream"                    # video file


def _target_res(args, src_w, src_h):
    if getattr(args, "pi_sim", False):
        return 640, 480
    if args.imgsz > 0:
        return args.imgsz, max(1, int(src_h * args.imgsz / src_w))
    return src_w, src_h


def _output_paths(args):
    stem = os.path.splitext(os.path.basename(str(args.source)))[0] or "cam"
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_mp4 = getattr(args, "output", None) or os.path.join(runs_dir, f"{stem}_pose.mp4")
    out_csv = os.path.splitext(out_mp4)[0] + ".csv"
    return out_mp4, out_csv


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
