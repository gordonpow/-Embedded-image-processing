"""
在開發機（RTX 3060 Ti）上執行，將 YOLOv8n .pt 匯出為 ONNX。

步驟：
  1. pip install ultralytics
  2. 下載預訓練權重（見下方連結說明）
  3. python export_onnx.py --weights fire_smoke_yolov8n.pt --imgsz 320

權重來源（擇一）：
  主: https://github.com/luminous0219/fire-and-smoke-detection-yolov8 (AGPL-3.0)
  備: https://github.com/RichardoMrMu/yolov5-fire-smoke-detect       (GPL-3.0)
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to .pt weights file')
    p.add_argument('--imgsz',   type=int, default=320, help='Export image size')
    p.add_argument('--out',     default='models/', help='Output directory')
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("請先執行: pip install ultralytics")

    model = YOLO(args.weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] imgsz={args.imgsz}  →  {out_dir}/")
    model.export(
        format='onnx',
        imgsz=args.imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )

    # ultralytics 預設輸出在 weights 同目錄，搬到 models/
    src = Path(args.weights).with_suffix('.onnx')
    dst = out_dir / f"fire_smoke_yolov8n_{args.imgsz}.onnx"
    if src.exists():
        src.rename(dst)
        print(f"[done] saved → {dst}")
    else:
        print(f"[warn] ONNX not found at {src}, check ultralytics output directory")


if __name__ == '__main__':
    main()
