# 影片火煙偵測系統 (Video Fire & Smoke Detection)

> 嵌入式影像處理作業 — 目標平台：Raspberry Pi 4B

---

## 目錄

1. [專案簡介](#一專案簡介)
2. [功能分解圖 BREAKDOWN](#二功能分解圖-breakdown)
3. [系統架構圖](#三系統架構圖)
4. [處理流程圖](#四處理流程圖)
5. [實驗設計](#五實驗設計)
6. [安裝與使用](#六安裝與使用)
7. [案例效果展示](#七案例效果展示)
8. [效能 Benchmark](#八效能-benchmark)
9. [目前進度](#九目前進度)

---

## 一、專案簡介

本專案旨在開發一套可於 **Raspberry Pi 4B** 上即時運行的影像火煙偵測系統。透過 **YOLOv8n** 預訓練模型與 **ONNX Runtime CPU** 推理引擎，系統能在受限的嵌入式硬體上，對影片逐幀偵測火焰（Fire）與濃煙（Smoke），並以 **Bounding Box** 標示偵測區域與信心值。

### 核心需求

| 需求 | 說明 |
|------|------|
| 火焰偵測 | 偵測畫面中出現的明火 |
| 濃煙偵測 | 偵測濃密煙霧區域 |
| 位置標示 | 即時繪製 Bounding Box + 類別標籤 |
| 效能量測 | 顯示 FPS 與解析度資訊 |
| 嵌入式部署 | 可在 Pi 4B（無 GPU）上穩定運行 |

### 技術選型理由

- **不從頭訓練**：使用既有火煙預訓練權重（mAP@0.5 ≈ 81.2%），避免建置大量 dataset
- **ONNX Runtime**：比完整 PyTorch/TF 輕量，ARM 上安裝簡便
- **YOLOv8n**：nano 版本體積小（~12 MB ONNX），兼顧準確率與速度

---

## 二、功能分解圖 BREAKDOWN

```mermaid
mindmap
  root((火煙偵測系統))
    輸入模組
      影片檔讀取 VideoCapture
      逐幀解碼
      解析度與 FPS 取得
    前處理模組
      Letterbox 等比縮放
      灰邊 Padding 至正方形
      BGR→RGB 色彩轉換
      像素值正規化 /255
    推理模組
      ONNX Runtime Session 載入
      NCHW Tensor 封裝
      CPU 多核心推理 4 threads
    後處理模組
      原始輸出 transpose
      信心值過濾 conf threshold
      座標反算 imgsz→原圖
      NMS 去重複框
    視覺化模組
      Bounding Box 繪製
      類別標籤與信心值標注
      EMA FPS 計算與顯示
    輸出模組
      imshow 即時預覽
      VideoWriter 標注影片輸出
      Benchmark CSV 匯出
    優化模組
      INT8 動態量化
      Frame Skip 跳幀
      Motion Gating
```

---

## 三、系統架構圖

```mermaid
flowchart TB
    subgraph DEV["開發機（RTX 3060 Ti）"]
        A[原始 .pt 權重\nluminous0219/fire-yolov8] --> B[ultralytics export\nformat=onnx, simplify=True]
        B --> C[fire_smoke_yolov8n_320.onnx\n11.58 MB FP32]
        C --> D[onnxruntime.quantization\nquantize_dynamic INT8]
        D --> E[fire_smoke_yolov8n_320_int8.onnx\n3.08 MB]
    end

    subgraph PI["Raspberry Pi 4B（部署目標）"]
        F[影片輸入\n.mp4 / Pi Camera]
        G[detector.py\nYoloFireSmokeDetector]
        H[pipeline.py\n主幀迴圈]
        I[visualize.py\n視覺化渲染]
        J[輸出\nimshow / mp4]
    end

    C -.->|scp 傳檔| G
    E -.->|INT8 優化版| G

    F --> H
    H --> G
    G --> I
    I --> J
```

### 模組職責

| 模組 | 檔案 | 職責 |
|------|------|------|
| 偵測器 | `src/detector.py` | 封裝 ONNX 推理、letterbox 前處理、NMS 後處理 |
| 主迴圈 | `src/pipeline.py` | VideoCapture 幀迴圈、EMA FPS 計算、VideoWriter |
| 視覺化 | `src/visualize.py` | Bounding Box 繪製、FPS overlay |
| 入口 | `src/main.py` | CLI argparse、整合各模組 |
| 效能測試 | `benchmarks/run_benchmark.py` | 多解析度 latency/FPS 量測 |

---

## 四、處理流程圖

```mermaid
flowchart LR
    A([影片開始]) --> B[VideoCapture.read\n取得原始 BGR Frame]
    B --> C{有幀？}
    C -- 否 --> Z([結束])
    C -- 是 --> D[Letterbox 縮放\n等比 + 灰邊 padding]

    D --> E[BGR→RGB\n/255 正規化]
    E --> F[轉 NCHW Tensor\n1×3×320×320]
    F --> G[ONNX Runtime\nCPU 推理]
    G --> H[輸出 1×6×2100\ncx,cy,w,h,p_fire,p_smoke]

    H --> I[transpose → 2100×6\n提取 box + scores]
    I --> J{max_score\n≥ conf_thres?}
    J -- 否 --> M
    J -- 是 --> K[反算原圖座標\n去 padding, ÷ scale]
    K --> L[Per-class NMS\nIoU threshold]
    L --> M[draw_bboxes\n繪製框 + 標籤]

    M --> N[draw_fps_overlay\nEMA FPS 顯示]
    N --> O[VideoWriter.write\n輸出幀]
    O --> B
```

### 關鍵資料契約

```
ONNX 輸出格式（imgsz=320, nc=2）：
  shape: [1, 6, 2100]
  axis-1: [cx, cy, w, h, p_fire, p_smoke]   ← pixel space (0~320)
  axis-2: 2100 個候選框 (40×40 + 20×20 + 10×10 anchors)
```

---

## 五、實驗設計

### 5.1 研究問題

1. YOLOv8n 火煙預訓練模型在真實火場影片上的偵測率為何？
2. 不同解析度（320 / 416 / 640）對 FPS 與準確率的影響？
3. INT8 量化在 ARM 平台上的速度收益？

### 5.2 實驗變數

| 類型 | 變數 |
|------|------|
| 自變數 | 推理解析度（imgsz）、量化精度（FP32 / INT8）、frame skip 數 |
| 應變數 | 平均 FPS、推理延遲 ms、偵測率（%）、信心值分布 |
| 控制變數 | 模型權重、conf threshold（0.35）、IoU threshold（0.45） |

### 5.3 測試資料集

| 影片 | 解析度 | 時長 | 場景特性 |
|------|--------|------|----------|
| test1.mp4 | 640×480 | 105 秒 | 火焰＋濃煙並存，室外場景 |
| test2.mp4 | 854×480 | 899 秒 | 長時間監控，多光影條件 |

### 5.4 評估指標

- **偵測率**：有偵測到 fire/smoke 的幀數佔比
- **最高信心值**：量測模型對目標的最大確信度
- **平均 FPS**：`total_frames / total_inference_time`
- **p99 latency**：99th percentile 單幀推理時間（ms）

---

## 六、安裝與使用

### 環境需求

```
Python >= 3.8
onnxruntime >= 1.16.0
opencv-python >= 4.8.0
numpy >= 1.24.0
```

### 安裝

```bash
pip install -r requirements.txt
```

### 偵測影片（顯示視窗）

```bash
python src/main.py \
    --video  test_videos/fire.mp4 \
    --model  models/fire_smoke_yolov8n_320.onnx \
    --imgsz  320 \
    --conf   0.35
```

### 無頭輸出（Pi / SSH 環境）

```bash
python src/main.py \
    --video  test_videos/fire.mp4 \
    --model  models/fire_smoke_yolov8n_320.onnx \
    --output output.mp4 \
    --no-show
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--video` | 必填 | 輸入影片路徑 |
| `--model` | models/fire_smoke_yolov8n_320.onnx | ONNX 模型路徑 |
| `--imgsz` | 320 | 推理解析度（正方形邊長） |
| `--conf`  | 0.35 | 信心閾值，越高越嚴格 |
| `--iou`   | 0.45 | NMS IoU 閾值 |
| `--skip`  | 1 | 每 N 幀推理一次（省效能） |
| `--output` | 無 | 輸出標注影片路徑 |
| `--no-show` | False | 不顯示視窗（headless） |

### 效能 Benchmark

```bash
python benchmarks/run_benchmark.py \
    --videos test_videos/ \
    --resolutions 320 416 640 \
    --frames 300
```

---

## 七、案例效果展示

### 真實火場影片偵測結果

![偵測效果 GIF](./gif/result_detect.gif)

> 橘紅色框 = **fire（火焰）**　灰色框 = **smoke（濃煙）**　左上角顯示即時 FPS

### 案例量化結果

#### Case 1：test1.mp4（640×480，105 秒，室外火場）

| 指標 | 數值 |
|------|------|
| 推理解析度 | 320×320 |
| 平均 FPS（開發機） | **124.6** |
| Fire 偵測率 | **24.4%**（770 / 3161 幀） |
| Smoke 偵測率 | **69.8%**（2205 / 3161 幀） |
| Fire 最高信心值 | 0.863 |
| Smoke 最高信心值 | **0.916** |

#### Case 2：test2.mp4（854×480，899 秒，長時間多光影）

| 指標 | 數值 |
|------|------|
| 推理解析度 | 320×320 |
| 總幀數 | 26,913 幀 |
| Fire 偵測率（抽樣） | **26.8%** |
| Smoke 偵測率（抽樣） | **27.3%** |
| Fire 最高信心值 | 0.873 |
| Smoke 最高信心值 | 0.839 |

### 信心值分析

- 兩段影片的最高信心值均 > 0.80，顯示模型對目標具高確信度
- Smoke 偵測率在 test1（69.8%）高於 test2（27.3%），反映不同場景的煙霧密度差異

---

## 八、效能 Benchmark

### 開發機結果（RTX 3060 Ti，CPU-only 推理）

| 模型版本 | Imgsz | FPS | ms/frame |
|----------|-------|-----|---------|
| FP32 ONNX | 320 | 124.6 | 8.0 |
| INT8 ONNX | 320 | 8.5* | 117.4 |

*INT8 在 x86 上因 dequantization overhead 反而較慢；ARM Pi 4B 需實機驗證

### Pi 4B 預估（待實機量測）

| 模型版本 | Imgsz | 預估 FPS | 備註 |
|----------|-------|---------|------|
| FP32 | 320 | 4–8 | ONNX Runtime ARM |
| INT8 | 320 | 6–12 | ARM NEON INT8 加速 |
| FP32 + skip=2 | 320 | 8–16 | 每 2 幀推理 |

### 模型大小比較

| 格式 | 大小 | 縮減比 |
|------|------|--------|
| .pt（PyTorch） | 5.97 MB | — |
| ONNX FP32 | 11.58 MB | — |
| ONNX INT8 | **3.08 MB** | **73%** ↓ |

---

## 九、目前進度

- [x] 需求定義
- [x] 預訓練模型取得（YOLOv8n，mAP@0.5 ≈ 81.2%）
- [x] ONNX 匯出（320 / 416）
- [x] INT8 量化（FP32 11.58MB → 3.08MB）
- [x] 偵測 pipeline 實作（前處理 / 推理 / NMS / 視覺化 / FPS）
- [x] 端對端測試通過（開發機 124 FPS）
- [x] 真實火煙影片驗證（2 段，偵測率 24–70%，最高信心 0.916）
- [ ] Raspberry Pi 4B 實機 benchmark
- [ ] INT8 量化 Pi 實機加速驗證
- [ ] 不同光影條件壓力測試

---

## 授權

模型權重：[luminous0219/fire-and-smoke-detection-yolov8](https://github.com/luminous0219/fire-and-smoke-detection-yolov8)（AGPL-3.0）

程式碼：本 repo 依作業需求自行開發
