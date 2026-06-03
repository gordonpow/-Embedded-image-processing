#!/usr/bin/env python3
"""
One-time helper: download the official Places365 ResNet18 weights and export
them to ONNX so the runtime can load the model through OpenCV's ``cv2.dnn``
module (no PyTorch needed at inference time, Raspberry-Pi friendly).

Why ResNet18?
  It is the smallest model the official CSAILVision/places365 release ships
  with reliable, pre-trained weights — the right size/accuracy trade-off for a
  Pi 4B that only runs scene classification once every N frames.

Outputs (written next to this script, git-ignored):
  models/places365_resnet18.onnx   — the network, ONNX opset 11
  models/io_places365.txt          — 365 lines, 1=indoor / 2=outdoor

Run once on a desktop:
    python models/export_places365_onnx.py
"""
import os
import urllib.request

import torch
import torchvision.models as tvm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PTH_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
_IO_URL = "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt"
_PTH_PATH = os.path.join(_HERE, "resnet18_places365.pth.tar")
_ONNX_PATH = os.path.join(_HERE, "places365_resnet18.onnx")
_IO_PATH = os.path.join(_HERE, "io_places365.txt")


def _download(url, path):
    if os.path.exists(path):
        print(f"[skip] already present: {path}")
        return
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[ok  ] -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def _build_io_file():
    """Parse IO_places365.txt ('/a/airfield 2') -> 365 lines of just the 1/2 flag."""
    tmp = os.path.join(_HERE, "_IO_raw.txt")
    _download(_IO_URL, tmp)
    flags = []
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                flags.append(parts[-1].strip())
    with open(_IO_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(flags) + "\n")
    os.remove(tmp)
    print(f"[ok  ] -> {_IO_PATH}  ({len(flags)} categories)")


def main():
    _download(_PTH_URL, _PTH_PATH)
    _build_io_file()

    # Places365 ResNet18: 365-way classifier head.
    model = tvm.resnet18(num_classes=365)
    checkpoint = torch.load(_PTH_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    # Strip the DataParallel 'module.' prefix.
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, _ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        opset_version=11, do_constant_folding=True,
    )
    print(f"[ok  ] -> {_ONNX_PATH}  ({os.path.getsize(_ONNX_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
