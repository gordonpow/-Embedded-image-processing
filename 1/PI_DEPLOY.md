# Raspberry Pi 4B 部署指南

## 硬體需求
- Raspberry Pi 4B（4GB / 8GB 皆可）
- microSD 16GB 以上（建議 Class 10）
- Raspberry Pi OS 64-bit (Bullseye 或 Bookworm)

## 安裝步驟

### 1. 系統套件
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev libopenjp2-7 libtiff5
```

### 2. Python 套件

使用專為 Pi 準備的 `requirements-pi.txt`（以 `opencv-python-headless` 取代 `opencv-python`，移除 Qt GUI 依賴，Pi headless 環境必要）：

```bash
pip install --upgrade pip
pip install -r requirements-pi.txt
```

> `requirements.txt`（開發機用）含完整 GUI 支援；Pi headless 請固定使用 `requirements-pi.txt`。
> 若 Pi 有桌面環境且需 `imshow`，可改執行 `pip install -r requirements.txt`。

### 3. 傳檔到 Pi

從開發機複製：
```bash
# 從 Windows / Linux 開發機（使用 scp 或 rsync）
scp -r src/ models/fire_smoke_yolov8n_320.onnx requirements.txt pi@<PI_IP>:~/fire_detect/
```

或用 USB / SD 卡轉。**至少需要這些檔案**：
```
fire_detect/
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── pipeline.py
│   └── visualize.py
├── models/
│   └── fire_smoke_yolov8n_320.onnx
└── benchmarks/
    └── run_benchmark.py    # 選填，跑效能測試用
```

## 執行

### 即時顯示（需桌面）
```bash
cd ~/fire_detect
python3 src/main.py --video sample.mp4 --imgsz 320 --conf 0.35
```

### 純輸出（無頭）
```bash
python3 src/main.py --video sample.mp4 --imgsz 320 \
                    --output output.mp4 --no-show
```

### Benchmark 多解析度
```bash
python3 benchmarks/run_benchmark.py \
    --videos test_videos/ \
    --resolutions 320 416 \
    --frames 200
```

## 預期效能（FP32 ONNX，開發機 ÷10 推算）

| Imgsz | 畫法 | 預估 FPS (Pi 4B 4GB) | 備註 |
|-------|------|---------------------|------|
| 320   | contour | **~11 FPS** ✓     | 推薦，滿足 ≥10 FPS 目標 |
| 320   | bbox    | ~12 FPS             | 稍快，視覺較差 |
| 416   | contour | ~7 FPS ✗           | 低於目標，不建議 |
| 640   | —       | <2 FPS（不建議）    | — |

> 實機數字待 Pi 量測後更新。預估來源：開發機 480p 測試結果 ÷ 10 換算。

## 優化選項（依需要啟用）

### Option 1：HSV 輪廓畫法（推薦開啟）
```bash
python3 src/main.py --video sample.mp4 --imgsz 320 --contour
# 僅多 4% 開銷，火/煙輪廓更精準
```

### Option 2：INT8 量化（在開發機做）
```python
# 在開發機執行
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input='models/fire_smoke_yolov8n_320.onnx',
    model_output='models/fire_smoke_yolov8n_320_int8.onnx',
    weight_type=QuantType.QInt8,
)
```
再把量化後 ONNX 放到 Pi 上跑，預期 FPS +30~50%（Pi NEON 加速，需實機驗證）。

### Option 3：Frame skip
```bash
python3 src/main.py --video sample.mp4 --imgsz 320 --skip 2   # 每 2 幀推理一次
```

### Option 4：CPU 親和性與優先順序
```bash
sudo nice -n -10 taskset -c 0-3 python3 src/main.py ...
```

## 故障排除

| 症狀 | 解法 |
|------|------|
| `onnxruntime.capi._pybind_state.NoSuchFile` | 確認 ONNX 路徑 |
| GUI 視窗無法顯示 | 改用 `--no-show --output result.mp4` |
| FPS < 2 | 啟用 INT8 量化 + frame skip |
| 推理時 CPU 溫度 > 80°C | 加裝散熱片 / 風扇，或啟用 frame skip |
| 偵測誤報多 | 提高 `--conf 0.5` |
