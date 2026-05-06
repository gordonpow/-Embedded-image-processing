import time
import cv2
import numpy as np

from visualize import draw_bboxes, draw_fps_overlay


_EMA_ALPHA = 0.1   # smoothing factor for FPS display


def run_detection_pipeline(
    video_path: str,
    detector,
    output_path: str = None,
    show: bool = True,
    frame_skip: int = 1,   # process every N-th frame (1 = all frames)
):
    """
    Main frame loop.

    Args:
        video_path  : path to input video file
        detector    : YoloFireSmokeDetector instance
        output_path : if given, writes annotated video here
        show        : open imshow window
        frame_skip  : infer only on every N-th frame, repeat bbox in between
    Returns:
        dict with benchmark stats: avg_fps, total_frames, inferred_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (orig_w, orig_h))

    ema_fps   = 0.0
    fps_times = []
    frame_idx = 0
    last_detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            if frame_idx % frame_skip == 0:
                last_detections = detector.detect(frame)

            draw_bboxes(frame, last_detections)

            elapsed = time.perf_counter() - t0
            fps_times.append(elapsed)

            # EMA FPS (based on inference + draw time)
            inst_fps = 1.0 / elapsed if elapsed > 0 else 0
            if ema_fps == 0.0:
                ema_fps = inst_fps
            else:
                ema_fps = _EMA_ALPHA * inst_fps + (1 - _EMA_ALPHA) * ema_fps

            draw_fps_overlay(frame, ema_fps, (orig_w, orig_h))

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Fire & Smoke Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    inferred = max(1, len([i for i in range(frame_idx) if i % frame_skip == 0]))
    avg_fps  = frame_idx / sum(fps_times) if fps_times else 0.0

    return {
        'avg_fps':        round(avg_fps, 2),
        'total_frames':   frame_idx,
        'inferred_frames': inferred,
        'resolution':     f"{orig_w}x{orig_h}",
    }
