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

ONNX Runtime 在 Pi 4B 推薦使用官方 ARM64 wheel：

```bash
pip install --upgrade pip
pip install onnxruntime numpy
pip install opencv-python-headless    # headless 版較輕、無 GUI 依賴
```

> 若需要顯示視窗（imshow），改裝 `opencv-python` 並確認有桌面環境。

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

## 預期效能（FP32 ONNX）

| Imgsz | 預期 FPS (Pi 4B 4GB) |
|-------|----------------------|
| 320   | 4–8 FPS              |
| 416   | 2–4 FPS              |
| 640   | <2 FPS（不建議）     |

## 優化選項（依需要啟用）

### Option 1：INT8 量化（在開發機做）
```python
# 在開發機執行
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input='models/fire_smoke_yolov8n_320.onnx',
    model_output='models/fire_smoke_yolov8n_320_int8.onnx',
    weight_type=QuantType.QInt8,
)
```
再把量化後 ONNX 放到 Pi 上跑，預期 FPS +30~50%。

### Option 2：Frame skip
```bash
python3 src/main.py --video sample.mp4 --imgsz 320 --skip 2   # 每 2 幀推理一次
```

### Option 3：CPU 親和性與優先順序
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
