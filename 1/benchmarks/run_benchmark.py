"""
Benchmark script — measure FPS at multiple resolutions.

Usage:
    python benchmarks/run_benchmark.py \
        --videos test_videos/ \
        --model  models/fire_smoke_yolov8n.onnx \
        --resolutions 320 416 640 \
        --frames 300
"""
import argparse
import os
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
from detector import YoloFireSmokeDetector


def benchmark_video(video_path: str, detector, num_frames: int = 300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    latencies = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    cap.release()
    if not latencies:
        return None

    latencies = sorted(latencies)
    return {
        'avg_ms':  round(sum(latencies) / len(latencies), 1),
        'p50_ms':  round(latencies[len(latencies) // 2], 1),
        'p99_ms':  round(latencies[int(len(latencies) * 0.99)], 1),
        'avg_fps': round(1000 / (sum(latencies) / len(latencies)), 2),
        'frames':  len(latencies),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos',      required=True,   help='Video folder or single file')
    p.add_argument('--model',       default='models/fire_smoke_yolov8n.onnx')
    p.add_argument('--resolutions', nargs='+', type=int, default=[320, 416, 640])
    p.add_argument('--conf',        type=float, default=0.35)
    p.add_argument('--frames',      type=int,   default=300)
    p.add_argument('--out',         default='benchmarks/results.csv')
    args = p.parse_args()

    videos_path = Path(args.videos)
    if videos_path.is_dir():
        videos = list(videos_path.glob('*.mp4')) + list(videos_path.glob('*.avi'))
    else:
        videos = [videos_path]

    if not videos:
        print("No videos found.")
        return

    rows = []
    header = ['video', 'imgsz', 'avg_fps', 'avg_ms', 'p50_ms', 'p99_ms', 'frames']

    for imgsz in args.resolutions:
        print(f"\n=== imgsz={imgsz} ===")
        detector = YoloFireSmokeDetector(args.model, imgsz=imgsz, conf_thres=args.conf)

        for vp in videos:
            stats = benchmark_video(str(vp), detector, args.frames)
            if stats is None:
                print(f"  SKIP {vp.name} (cannot open)")
                continue
            print(f"  {vp.name:30s}  FPS={stats['avg_fps']:6.2f}  avg={stats['avg_ms']}ms  p99={stats['p99_ms']}ms")
            rows.append([vp.name, imgsz, stats['avg_fps'], stats['avg_ms'], stats['p50_ms'], stats['p99_ms'], stats['frames']])

    # Write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\nResults saved to {args.out}")

    # Print markdown table
    print(f"\n{'Video':<30} {'imgsz':>6} {'FPS':>7} {'avg ms':>8} {'p99 ms':>8}")
    print('-' * 65)
    for r in rows:
        print(f"{r[0]:<30} {r[1]:>6} {r[2]:>7.2f} {r[3]:>8.1f} {r[5]:>8.1f}")


if __name__ == '__main__':
    main()
