"""RED tests for pipeline.run: new CSV columns + single-image branch."""
import csv
import os
from argparse import Namespace

import cv2
import numpy as np
import pytest

from pipeline import run

NEW_COLUMNS = {
    "scene", "scene_conf", "cam_motion", "flow_motion",
    "zoom_in", "rel_depth", "depth_level",
}


def _args(source, **over):
    base = dict(
        source=source, output=None, imgsz=0, pi_sim=False,
        nfeatures=500, ratio_thresh=0.75, ransac_thresh=1.0,
        keyframe_ratio=0.5, keyframe_inliers=60, calib=None,
        no_show=True, no_video=True,
    )
    base.update(over)
    return Namespace(**base)


def _textured(seed, w=640, h=480):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for _ in range(400):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 14))
        c = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.circle(img, (cx, cy), r, c, -1)
    return img


def _make_video(path, n=8):
    base = _textured(7)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    for i in range(n):
        M = cv2.getRotationMatrix2D((320, 240), 1.2 * i, 1.0)
        wr.write(cv2.warpAffine(base, M, (640, 480)))
    wr.release()


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_video_csv_has_new_columns_and_consistent_width(tmp_path):
    vid = str(tmp_path / "clip.mp4")
    _make_video(vid)
    stats = run(_args(vid))
    header, data = _read_csv(stats["csv_path"])

    assert NEW_COLUMNS.issubset(set(header))
    assert len(data) > 0
    for row in data:
        assert len(row) == len(header)        # every row matches the header width


def test_video_rows_carry_scene_label(tmp_path):
    vid = str(tmp_path / "clip2.mp4")
    _make_video(vid)
    stats = run(_args(vid))
    header, data = _read_csv(stats["csv_path"])
    scene_idx = header.index("scene")
    # Every frame must carry a non-empty indoor/outdoor label.
    for row in data:
        assert row[scene_idx] in ("indoor", "outdoor")


def test_single_image_yaw_na_pitch_roll_numeric_and_scene_filled(tmp_path):
    img_path = str(tmp_path / "photo.jpg")
    cv2.imwrite(img_path, _textured(3))
    stats = run(_args(img_path))
    header, data = _read_csv(stats["csv_path"])

    assert len(data) == 1
    row = dict(zip(header, data[0]))
    # Yaw is unobservable from one image; roll/pitch come from horizon estimate.
    assert row["yaw_deg"] == "N/A"
    float(row["pitch_deg"])               # parses as a number (not "N/A")
    float(row["roll_deg"])
    assert row["scene"] in ("indoor", "outdoor")
    # Pose-pipeline-only fields stay N/A for a still image.
    assert row["depth_level"] == "N/A"
    assert row["cam_motion"] == "N/A"
    # A PNG with the XYZ gizmo is produced.
    import os
    assert os.path.exists(stats["png_path"])
