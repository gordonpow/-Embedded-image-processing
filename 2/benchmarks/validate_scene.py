#!/usr/bin/env python3
"""
Indoor/outdoor acceptance test — run the scene classifier on a labelled image
folder and report accuracy + confusion against the ground-truth labels.

The labels CSV has two columns: ``filename,label`` (label in {indoor, outdoor}).
Point ``--images`` at the folder holding those files.  The same script scales
straight to a Places365 validation subset — just supply its labels CSV.

Usage
-----
    python benchmarks/validate_scene.py                       # demo: 6 images
    python benchmarks/validate_scene.py --images path --labels labels.csv
"""
import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))
from scene import classify_indoor_outdoor, SceneClassifier
from metrics import classification_accuracy

_ROOT = os.path.dirname(os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser(description="Validate indoor/outdoor vs labels")
    ap.add_argument("--images", default=os.path.join(_ROOT, "test_inputs"))
    ap.add_argument("--labels", default=os.path.join(os.path.dirname(__file__), "labels_demo.csv"))
    ap.add_argument("--model", default=os.path.join(_ROOT, "models", "places365_resnet18.onnx"))
    ap.add_argument("--io", default=os.path.join(_ROOT, "models", "io_places365.txt"))
    ap.add_argument("--threshold", type=float, default=0.8, help="PASS if accuracy >= this")
    args = ap.parse_args()

    classifier = SceneClassifier.try_load(args.model, args.io)
    backend = "DNN(Places365)" if classifier is not None else "heuristic"

    preds, gts = [], []
    print(f"\n=== Scene acceptance ===  backend={backend}")
    with open(args.labels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(args.images, row["filename"])
            img = cv2.imread(path)
            if img is None:
                print(f"  [skip] missing {path}")
                continue
            label, conf = classify_indoor_outdoor(img, classifier)
            hit = "OK " if label == row["label"] else "XX "
            print(f"  {hit} {row['filename']:14s} pred={label:8s} ({conf:.2f})  gt={row['label']}")
            preds.append(label)
            gts.append(row["label"])

    acc, conf = classification_accuracy(preds, gts)
    verdict = "PASS" if acc >= args.threshold else "FAIL"
    print(f"\naccuracy = {acc * 100:.1f}%  ({sum(p == g for p, g in zip(preds, gts))}/{len(preds)})"
          f"   threshold = {args.threshold * 100:.0f}%   -> {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
