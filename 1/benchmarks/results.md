# Benchmark Results

## 環境

| 項目 | 開發機 | 部署目標 |
|------|--------|----------|
| 平台 | RTX 3060Ti / x86_64 | Raspberry Pi 4B (ARM Cortex-A72) |
| RAM | — | 4GB / 8GB |
| 推理引擎 | ONNX Runtime CPU | ONNX Runtime CPU (ARM wheel) |
| 模型 | YOLOv8n fire-smoke | 同左 |

## 模型資訊

| 檔案 | 大小 | 輸入 | 輸出 |
|------|------|------|------|
| `models/fire_smoke_yolov8n.pt` | 5.97 MB | 動態 | — |
| `models/fire_smoke_yolov8n_320.onnx` | 11.58 MB | [1,3,320,320] | [1,6,2100] |
| `models/fire_smoke_yolov8n_416.onnx` | 11.61 MB | [1,3,416,416] | [1,6,3549] |

類別：`{0: 'fire', 1: 'smoke'}`

## 開發機驗證

### 真實火煙影片（開發機 FP32, imgsz=320, conf=0.30）

| 影片 | 解析度 | 總幀數 | FPS | Fire 偵測率 | Smoke 偵測率 | Fire max conf | Smoke max conf |
|------|--------|--------|-----|-----------|------------|--------------|---------------|
| test1.mp4 | 640x480 | 3,161 | 124.6 | 24.4% | 69.8% | 0.863 | 0.916 |
| test2.mp4 | 854x480 | 26,913 | ~124 | 26.8%* | 27.3%* | 0.873 | 0.839 |

*test2 為每 30 幀抽樣統計

### 合成測試（640x480, 90 frames）

| 模型 | FPS | ms/frame | 備註 |
|------|-----|---------|------|
| FP32-320 | 132.2 | 7.6 | 開發機基準 |
| INT8-320 | 8.5  | 117.4 | x86 INT8 overhead（Pi ARM 需實機測） |

## Pi 4B 量測（待補）

執行 `python benchmarks/run_benchmark.py --videos test_videos/ --resolutions 320 416` 後填入：

| Video | Imgsz | Avg FPS | Avg ms | p99 ms |
|-------|-------|---------|--------|--------|
| TBD   | 320   | —       | —      | —      |
| TBD   | 416   | —       | —      | —      |

## 不同光影測試（待補）

| 場景 | Imgsz | FPS | 視覺檢查 |
|------|-------|-----|----------|
| 室外白天 | 320 | — | — |
| 室內昏暗 | 320 | — | — |
| 夜間強對比 | 320 | — | — |
