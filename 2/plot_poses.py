#!/usr/bin/env python3
"""
Offline pose curve plotter — reads pose CSV, saves 3-panel PNG.

Usage
-----
    python plot_poses.py runs/indoor_office_pose.csv
    python plot_poses.py runs/outdoor_pose.csv -o results/outdoor_curves.png
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


_BG_OUTER = "#1a1a2e"
_BG_PANEL = "#16213e"
_SPINE    = "#444466"

_ANGLE_COLORS = {
    "yaw":   "#ff6b6b",
    "pitch": "#4ecdc4",
    "roll":  "#ffe66d",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot yaw/pitch/roll curves from pose CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("csv", help="pose CSV produced by main.py")
    ap.add_argument(
        "-o", "--output", default=None,
        help="output PNG path (default: <csv_stem>_curves.png)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[error] file not found: {args.csv}")

    rows = _read_csv(args.csv)
    if len(rows) < 2:
        sys.exit(f"[error] not enough data rows in {args.csv}")

    t     = np.array([r["timestamp_s"] for r in rows])
    yaw   = np.array([r["yaw_deg"]     for r in rows])
    pitch = np.array([r["pitch_deg"]   for r in rows])
    roll  = np.array([r["roll_deg"]    for r in rows])
    fps_vals = np.array([r["fps"] for r in rows if r["fps"] > 0])

    out_png = args.output or os.path.splitext(args.csv)[0] + "_curves.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             facecolor=_BG_OUTER)
    fig.suptitle(
        f"Camera Pose Estimation — {os.path.basename(args.csv)}\n"
        f"frames: {len(rows)}   avg FPS: {fps_vals.mean():.1f}   "
        f"duration: {t[-1]:.1f} s",
        color="white", fontsize=12, fontweight="bold",
    )

    _panel(axes[0], t, yaw,   "Yaw (°)",   _ANGLE_COLORS["yaw"])
    _panel(axes[1], t, pitch, "Pitch (°)", _ANGLE_COLORS["pitch"])
    _panel(axes[2], t, roll,  "Roll (°)",  _ANGLE_COLORS["roll"])
    axes[2].set_xlabel("Time (s)", color="white", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=_BG_OUTER)
    plt.close()
    print(f"[plot] saved → {out_png}")


def _panel(ax, t, vals, label: str, color: str) -> None:
    ax.set_facecolor(_BG_PANEL)
    ax.plot(t, vals, color=color, linewidth=1.1, alpha=0.9)
    ax.fill_between(t, vals, alpha=0.12, color=color)
    ax.axhline(0, color=_SPINE, linewidth=0.6, linestyle="--")
    ax.set_ylabel(label, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
    # Range annotation
    rng = float(vals.max() - vals.min())
    ax.annotate(
        f"range {rng:.1f}°",
        xy=(0.01, 0.88), xycoords="axes fraction",
        color=color, fontsize=8, alpha=0.8,
    )


def _read_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except ValueError:
                pass
    return rows


if __name__ == "__main__":
    main()
