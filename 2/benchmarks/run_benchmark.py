#!/usr/bin/env python3
"""
Multi-resolution × multi-scene benchmark for the camera pose estimator.

Sweeps:
    resolutions : 320 / 480 / 720 / 1080 (target width in px)
    scenes      : indoor / outdoor / dynamic  (detected from filename)

Usage
-----
    python benchmarks/run_benchmark.py --videos test_inputs/

Name your test videos to include the scene label, e.g.:
    indoor_office.mp4
    outdoor_street.mp4
    dynamic_car.mp4

Results are written to benchmarks/results.md as a markdown table.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time


RESOLUTIONS  = [320, 480, 720, 1080]
SCENE_LABELS = ["indoor", "outdoor", "dynamic"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark pose estimator across resolutions and scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--videos",  default="test_inputs/",        help="folder with test videos")
    ap.add_argument("--results", default="benchmarks/results.md", help="output markdown file")
    ap.add_argument("--timeout", type=int, default=300,          help="per-run timeout (s)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_py   = os.path.join(repo_root, "src", "main.py")

    videos = sorted(
        glob.glob(os.path.join(args.videos, "*.mp4")) +
        glob.glob(os.path.join(args.videos, "*.MP4")) +
        glob.glob(os.path.join(args.videos, "*.avi")) +
        glob.glob(os.path.join(args.videos, "*.mov"))
    )

    if not videos:
        sys.exit(f"[benchmark] No videos found in {args.videos}\n"
                 "Name files like indoor_*.mp4 / outdoor_*.mp4 / dynamic_*.mp4")

    print(f"[benchmark] {len(videos)} video(s) × {len(RESOLUTIONS)} resolution(s) "
          f"= {len(videos) * len(RESOLUTIONS)} runs\n")

    records = []
    for vid in videos:
        scene = _detect_scene(vid)
        for imgsz in RESOLUTIONS:
            label = f"[{scene:>8}] {os.path.basename(vid):30s} @ {imgsz:4d}px"
            print(f"  {label}  … ", end="", flush=True)

            rec = _run_once(main_py, vid, imgsz, args.timeout)
            records.append({
                "video": os.path.basename(vid),
                "scene": scene,
                "imgsz": imgsz,
                **rec,
            })

            fps_str = f"{rec['avg_fps']:.1f}" if rec["avg_fps"] > 0 else "ERR"
            print(f"fps={fps_str}  frames={rec.get('total_frames', '?')}  "
                  f"res={rec.get('resolution', str(imgsz)+'px')}")

    _write_markdown(args.results, records)
    print(f"\n[benchmark] results → {os.path.abspath(args.results)}")


def _run_once(main_py: str, vid: str, imgsz: int, timeout: int) -> dict:
    cmd = [
        sys.executable, main_py, vid,
        "--imgsz",    str(imgsz),
        "--no-show",
        "--no-video",
    ]
    try:
        raw = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=timeout
        ).decode(errors="replace")
    except subprocess.CalledProcessError as exc:
        return {"avg_fps": 0.0, "total_frames": 0,
                "resolution": f"{imgsz}px", "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"avg_fps": 0.0, "total_frames": 0,
                "resolution": f"{imgsz}px", "error": "timeout"}

    # Parse [result] line produced by main.py
    m = re.search(
        r"avg_fps=([\d.]+).*?resolution=(\w+).*?total_frames=(\d+)",
        raw, re.DOTALL,
    )
    if m:
        return {
            "avg_fps":      float(m.group(1)),
            "resolution":   m.group(2),
            "total_frames": int(m.group(3)),
        }
    return {"avg_fps": 0.0, "total_frames": 0,
            "resolution": f"{imgsz}px", "raw_tail": raw[-300:]}


def _detect_scene(path: str) -> str:
    name = os.path.basename(path).lower()
    for label in SCENE_LABELS:
        if label in name:
            return label
    return "unknown"


def _write_markdown(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Benchmark Results\n",
        f"_Generated {now}_\n",
        "## Summary\n",
        "| Video | Scene | Resolution | FPS | Frames |",
        "|-------|-------|-----------|-----|--------|",
    ]
    for r in records:
        fps = r.get("avg_fps", 0)
        fps_str = f"{fps:.1f}" if fps > 0 else "—"
        lines.append(
            f"| `{r['video']}` | {r['scene']} | {r.get('resolution', str(r['imgsz'])+'px')} "
            f"| {fps_str} | {r.get('total_frames', 0)} |"
        )

    lines += [
        "\n## Notes",
        "- `--nfeatures 500`, `--ratio-thresh 0.75`, `--ransac-thresh 1.0` (defaults)",
        "- `--no-video` used (encoding cost excluded from FPS)",
        "- Intrinsics: approximate `fx = fy = W`",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
