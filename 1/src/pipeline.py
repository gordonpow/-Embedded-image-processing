import time
import cv2
import numpy as np

from visualize import draw_bboxes, draw_contours, draw_fps_overlay


_EMA_ALPHA = 0.1   # smoothing factor for FPS display
_PI_SIZE   = (640, 480)  # Pi 4B camera default at 480p


def run_detection_pipeline(
    video_path: str,
    detector,
    output_path: str = None,
    show: bool = True,
    frame_skip: int = 1,   # process every N-th frame (1 = all frames)
    pi_sim: bool = False,  # resize frames to 640x480 to simulate Pi 4B
    contour: bool = False, # draw HSV contours instead of bounding boxes
    speed: int = 1,        # read every N-th input frame (shrinks output by 1/N)
    start_sec: float = 0.0, # seek to this timestamp before processing
):
    """
    Main frame loop.

    Args:
        video_path  : path to input video file
        detector    : YoloFireSmokeDetector instance
        output_path : if given, writes annotated video here
        show        : open imshow window
        frame_skip  : infer only on every N-th frame, repeat bbox in between
        pi_sim      : resize every frame to 640x480 (Pi 4B 480p simulation)
        contour     : use HSV-segmented contours instead of plain rectangles
        speed       : stride for input reading — only every N-th frame is decoded
        start_sec   : seek to this position (seconds) before the main loop
    Returns:
        dict with benchmark stats: avg_fps, total_frames, inferred_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    if start_sec > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Guard against corrupted FPS metadata (avoids multi-million-frame output)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1.0 <= src_fps <= 120.0 else 30.0

    out_w, out_h = (_PI_SIZE if pi_sim else (orig_w, orig_h))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (out_w, out_h))

    ema_fps   = 0.0
    fps_times = []
    frame_idx = 0
    last_detections = []

    try:
        while True:
            # --speed N: skip N-1 frames without decoding, then decode the kept one
            for _ in range(speed - 1):
                if not cap.grab():
                    break

            ret, frame = cap.read()
            if not ret:
                break

            if pi_sim:
                frame = cv2.resize(frame, _PI_SIZE, interpolation=cv2.INTER_LINEAR)

            t0 = time.perf_counter()

            if frame_idx % frame_skip == 0:
                last_detections = detector.detect(frame)

            if contour:
                draw_contours(frame, last_detections)
            else:
                draw_bboxes(frame, last_detections)

            elapsed = time.perf_counter() - t0
            fps_times.append(elapsed)

            # EMA FPS (based on inference + draw time)
            inst_fps = 1.0 / elapsed if elapsed > 0 else 0
            if ema_fps == 0.0:
                ema_fps = inst_fps
            else:
                ema_fps = _EMA_ALPHA * inst_fps + (1 - _EMA_ALPHA) * ema_fps

            draw_fps_overlay(frame, ema_fps, (out_w, out_h))

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
