# 影片火煙偵測系統 (Video Fire and Smoke Detection)

## 專案簡介
本專案旨在開發一套基於影像處理的系統，專門用於分析影片內容以進行火災與煙霧的早期預警。透過這套系統，我們能夠在火災發生的初期階段，透過攝影機畫面迅速捕捉到火苗與煙霧的蹤跡。

## 需求定義
本系統主要達成以下兩項核心需求：
1. **火與煙偵測**：系統能夠持續分析影片畫面，並自動辨識畫面中是否出現「火 (Fire)」或「煙霧 (Smoke)」。
2. **範圍標示 (Bounding Box)**：當系統偵測到火或煙霧時，會在影片畫面上即時繪製邊界框，標示出火或煙霧發生的具體位置與範圍。


## 系統架構

```
影片輸入 → Letterbox 前處理 → YOLOv8n ONNX 推理 → NMS 後處理 → Bounding Box 輸出
```

- **模型**：YOLOv8n（火煙專用預訓練，mAP@0.5 ≈ 81.2%）
- **推理引擎**：ONNX Runtime CPU（適配 Raspberry Pi 4B）
- **類別**：`fire`（火焰）、`smoke`（濃煙）

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

### 偵測影片（顯示視窗）
```bash
python src/main.py --video test_videos/fire.mp4 \
                   --model models/fire_smoke_yolov8n_320.onnx \
                   --imgsz 320 --conf 0.35
```

### 無頭輸出（Pi / SSH 環境）
```bash
python src/main.py --video test_videos/fire.mp4 \
                   --model models/fire_smoke_yolov8n_320.onnx \
                   --output output.mp4 --no-show
```

### 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--video` | 必填 | 輸入影片路徑 |
| `--model` | models/fire_smoke_yolov8n_320.onnx | ONNX 模型路徑 |
| `--imgsz` | 320 | 推理解析度 |
| `--conf` | 0.35 | 信心閾值 |
| `--skip` | 1 | 每 N 幀推理一次（省效能） |
| `--output` | 無 | 輸出影片路徑 |
| `--no-show` | False | 不開視窗（無頭模式） |

### 效能測試
```bash
python benchmarks/run_benchmark.py \
    --videos test_videos/ \
    --resolutions 320 416 640
```

## 模型取得

預訓練權重來自 [luminous0219/fire-and-smoke-detection-yolov8](https://github.com/luminous0219/fire-and-smoke-detection-yolov8)（AGPL-3.0）。

已匯出 ONNX 格式：
- `models/fire_smoke_yolov8n_320.onnx`（推薦 Pi 使用）
- `models/fire_smoke_yolov8n_416.onnx`

## Raspberry Pi 4B 部署

詳見 [PI_DEPLOY.md](./PI_DEPLOY.md)

## 目前進度

- [x] 預訓練模型取得 (YOLOv8n, mAP 81.2%)
- [x] ONNX 匯出 (320 / 416)
- [x] 偵測 pipeline（前處理 / 推理 / NMS / 視覺化）
- [x] FPS 量測與 benchmark 腳本
- [x] 端對端測試通過（開發機 114 FPS）
- [ ] Pi 4B 實機 benchmark
- [ ] INT8 量化（Phase C 優化）
- [x] 真實火煙影片測試（test1: 火焰 24.4%, 煙霧 69.8%，最高信心 0.91）
- [ ] Pi 4B 實機 benchmark
- [ ] INT8 量化實機驗證

## 偵測效果展示

### 真實火煙偵測結果（test1.mp4）

![detection result](./gif/result_detect.gif)

### 遮罩參考
![](./gif/mass.gif)
