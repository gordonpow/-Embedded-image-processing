#!/usr/bin/env python3
"""
Indoor/outdoor acceptance on the Places365 validation set — an image-based
standard answer (parallels the video-based TUM RGB-D pose test).

Places365 ships `places365_val.txt` (val_image -> category index) and
`IO_places365.txt` (category -> 1 indoor / 2 outdoor). We look up each image's
ground-truth indoor/outdoor label, run our DNN classifier, and report accuracy
plus a montage comparing THEIR answer (GT) vs OURS.

    python benchmarks/validate_places365.py --dir test_inputs/places365_val \
        --out docs/places365_demo.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from scene import SceneClassifier

_ROOT = os.path.dirname(os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="test_inputs/places365_val")
    ap.add_argument("--model", default=os.path.join(_ROOT, "models", "places365_resnet18.onnx"))
    ap.add_argument("--io", default=os.path.join(_ROOT, "models", "io_places365.txt"))
    ap.add_argument("--out", default="docs/places365_demo.png")
    ap.add_argument("--per-class", type=int, default=6, help="indoor/outdoor each in montage")
    args = ap.parse_args()

    io = [int(x) for x in open(args.io, encoding="utf-8") if x.strip()]
    labels = {}
    with open(os.path.join(args.dir, "places365_val.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                labels[os.path.basename(p[0])] = int(p[1])

    clf = SceneClassifier(args.model, args.io)
    items = []
    for fn in sorted(os.listdir(args.dir)):
        if not fn.lower().endswith(".jpg") or fn not in labels:
            continue
        gt = "indoor" if io[labels[fn]] == 1 else "outdoor"
        img = cv2.imread(os.path.join(args.dir, fn))
        if img is None:
            continue
        pred, conf = clf.predict(img)
        items.append((fn, img, gt, pred, conf))

    if not items:
        sys.exit("[error] no labelled images matched")
    acc = sum(g == p for _, _, g, p, _ in items) / len(items)
    print(f"Places365 val: accuracy {acc * 100:.1f}%  ({sum(g==p for _,_,g,p,_ in items)}/{len(items)})")

    # Balanced montage: per-class indoor + outdoor.
    ind = [x for x in items if x[2] == "indoor"][: args.per_class]
    out = [x for x in items if x[2] == "outdoor"][: args.per_class]
    panels = []
    for fn, img, gt, pred, conf in ind + out:
        im = cv2.resize(img, (320, 240))
        cv2.rectangle(im, (0, 0), (320, 48), (0, 0, 0), -1)
        cv2.putText(im, f"GT:  {gt}", (6, 19), 0, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
        col = (80, 255, 80) if gt == pred else (80, 80, 255)
        cv2.putText(im, f"OUR: {pred} {conf:.2f}", (6, 40), 0, 0.5, col, 1, cv2.LINE_AA)
        panels.append(im)

    cols = 4
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((240, 320, 3), np.uint8))
        rows.append(np.hstack(row))
    montage = np.vstack(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"saved -> {args.out}  {montage.shape}")


if __name__ == "__main__":
    main()
