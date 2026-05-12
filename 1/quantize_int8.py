"""
在開發機上執行：將 FP32 ONNX 重新量化為 INT8（靜態 QDQ 格式）。

背景：
  onnxruntime quantize_dynamic + QInt8 會產生 ConvInteger 算子，
  ARM64 CPU 後端（Pi 4B 常見的 1.14–1.16 版）沒有 ConvInteger kernel，
  導致 InferenceSession 載入時拋出 NOT_IMPLEMENTED 錯誤。
  quantize_static + QDQ 格式產生的是 QLinearConv，ARM 後端原生支援。

用法：
  # 1. 先從測試影片抽取校正圖（一次性，約 30 張即可）
  python quantize_int8.py --extract-calib \
      --calib-video test_videos/fire.mp4 \
      --calib-dir   models/calib_images \
      --n-frames    30

  # 2. 執行靜態量化
  python quantize_int8.py \
      --fp32-onnx models/fire_smoke_yolov8n_320.onnx \
      --output    models/fire_smoke_yolov8n_320_int8.onnx \
      --calib-dir models/calib_images \
      --imgsz     320
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class _FireCalibReader:
    """
    Feeds letterboxed frames to onnxruntime quantization calibration.
    Implements the CalibrationDataReader interface.
    """

    def __init__(self, calib_dir: str, imgsz: int, input_name: str):
        self._input_name = input_name
        self._imgsz      = imgsz
        self._paths      = sorted(Path(calib_dir).glob('*.jpg')) + \
                           sorted(Path(calib_dir).glob('*.png'))
        if not self._paths:
            raise FileNotFoundError(
                f"calib_dir '{calib_dir}' にJPG/PNGが見つかりません。"
                "--extract-calib で先に抽出してください。"
            )
        self._idx = 0

    def _letterbox(self, img):
        h, w   = img.shape[:2]
        scale  = self._imgsz / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw     = (self._imgsz - nw) / 2
        dh     = (self._imgsz - nh) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

    def get_next(self):
        if self._idx >= len(self._paths):
            return None
        img = cv2.imread(str(self._paths[self._idx]))
        self._idx += 1
        if img is None:
            return self.get_next()
        lb  = self._letterbox(img)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), 0)   # NCHW
        return {self._input_name: tensor}

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# Calibration frame extraction
# ---------------------------------------------------------------------------

def extract_calib_frames(video: str, out_dir: str, n: int):
    os.makedirs(out_dir, exist_ok=True)
    cap   = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"動画が開けません: {video}")
    step  = max(1, total // n)
    saved = 0
    for i, pos in enumerate(range(0, total, step)):
        if saved >= n:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(os.path.join(out_dir, f'{saved:03d}.jpg'), frame)
        saved += 1
    cap.release()
    print(f"[calib] {saved} frames saved → {out_dir}/")


# ---------------------------------------------------------------------------
# Static QDQ quantization
# ---------------------------------------------------------------------------

def quantize(fp32_path: str, out_path: str, calib_dir: str, imgsz: int):
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError:
        raise SystemExit("pip install onnxruntime onnxruntime-tools を実行してください")

    # Derive input name from the FP32 model (same logic as detector.py:34)
    tmp_sess  = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    input_name = tmp_sess.get_inputs()[0].name
    del tmp_sess

    reader = _FireCalibReader(calib_dir, imgsz, input_name)

    print(f"[quant] input  : {fp32_path}")
    print(f"[quant] output : {out_path}")
    print(f"[quant] calib  : {calib_dir}  ({len(reader._paths)} images)")
    print(f"[quant] format : QDQ  (produces QLinearConv — ARM64 compatible)")

    quantize_static(
        model_input=fp32_path,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=['Conv', 'MatMul'],
    )
    print(f"[done] INT8 QDQ model saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Quantize YOLOv8 FP32 ONNX → INT8 QDQ (ARM-compatible)')
    p.add_argument('--fp32-onnx',     default='models/fire_smoke_yolov8n_320.onnx',
                   help='Path to FP32 ONNX model')
    p.add_argument('--output',        default='models/fire_smoke_yolov8n_320_int8.onnx',
                   help='Output path for INT8 model')
    p.add_argument('--calib-dir',     default='models/calib_images',
                   help='Directory of calibration images (JPG/PNG)')
    p.add_argument('--imgsz',         type=int, default=320,
                   help='Inference image size (must match --fp32-onnx export size)')
    # Frame extraction sub-command
    p.add_argument('--extract-calib', action='store_true',
                   help='Extract calibration frames from --calib-video first, then exit')
    p.add_argument('--calib-video',   default='test_videos/fire.mp4',
                   help='Source video for calibration frame extraction')
    p.add_argument('--n-frames',      type=int, default=30,
                   help='Number of frames to extract for calibration')
    args = p.parse_args()

    if args.extract_calib:
        extract_calib_frames(args.calib_video, args.calib_dir, args.n_frames)
        return

    quantize(args.fp32_onnx, args.output, args.calib_dir, args.imgsz)


if __name__ == '__main__':
    main()
