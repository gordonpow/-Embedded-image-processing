"""
Fire & Smoke Detection — Entry Point

Usage (development machine / Pi):
    python src/main.py --video test_videos/fire.mp4 \
                       --model models/fire_smoke_yolov8n.onnx \
                       --imgsz 320 --conf 0.35 --output output.mp4
"""
import argparse
import os
import sys

# Allow running from repo root or from src/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from detector import YoloFireSmokeDetector
from pipeline  import run_detection_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Fire & Smoke Detector (YOLOv8n ONNX)")
    p.add_argument('--video',  required=True,                    help='Input video path')
    p.add_argument('--model',  default='models/fire_smoke_yolov8n.onnx', help='ONNX model path')
    p.add_argument('--imgsz',  type=int,   default=320,          help='Inference resolution (square)')
    p.add_argument('--conf',   type=float, default=0.35,         help='Confidence threshold')
    p.add_argument('--iou',    type=float, default=0.45,         help='NMS IoU threshold')
    p.add_argument('--output', default=None,                     help='Output video path (optional)')
    p.add_argument('--skip',    type=int,   default=1,            help='Run inference every N frames')
    p.add_argument('--no-show', action='store_true',             help='Disable imshow (headless)')
    p.add_argument('--pi-sim',    action='store_true',           help='Resize frames to 640x480 (Pi 4B simulation)')
    p.add_argument('--contour',   action='store_true',           help='Draw HSV contours instead of bounding boxes')
    p.add_argument('--speed',     type=int,   default=1,        help='Read every N-th frame (shrinks output duration by N)')
    p.add_argument('--start-sec', type=float, default=0.0,      help='Seek to this timestamp (seconds) before processing')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[init] model  : {args.model}")
    print(f"[init] imgsz  : {args.imgsz}")
    print(f"[init] conf   : {args.conf}  iou: {args.iou}")
    print(f"[init] pi_sim : {args.pi_sim}  contour: {args.contour}")
    print(f"[init] speed  : {args.speed}x  start_sec: {args.start_sec}s")

    detector = YoloFireSmokeDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    stats = run_detection_pipeline(
        video_path=args.video,
        detector=detector,
        output_path=args.output,
        show=not args.no_show,
        frame_skip=args.skip,
        pi_sim=args.pi_sim,
        contour=args.contour,
        speed=args.speed,
        start_sec=args.start_sec,
    )

    print(f"\n[result] avg_fps={stats['avg_fps']}  "
          f"frames={stats['total_frames']}  "
          f"inferred={stats['inferred_frames']}  "
          f"resolution={stats['resolution']}")


if __name__ == '__main__':
    main()
