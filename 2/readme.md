# Camera Pose Estimator — Yaw / Pitch / Roll + Scene / Motion / Depth

> 從影片、即時攝影機或單張圖片提取相機姿態與場景資訊，專為 **Raspberry Pi 4B** 設計。  
> 純 CPU、無 ML 模型、僅依賴 `opencv-python`。

---

## What / Why / How

### What（這個專案做什麼）
輸入一段影片、即時 webcam 或單張圖片，輸出：
- **yaw / pitch / roll**（相機相對旋轉姿態，度）
- **室內 / 戶外**（indoor / outdoor）
- **動態方向**（相機運動方向 + 畫面光流方向 + zoom）
- **相對深度**（NEAR / MID / FAR，尺度相對非絕對）
- **FPS@解析度**

### Why（為什麼這樣設計）
- 目標硬體是 **Raspberry Pi 4B（CPU-only）**，所以姿態/動態/深度全用**純 OpenCV 幾何法**，才能即時跑。
- 動態方向與深度幾乎**零額外成本**：姿態管線 `recoverPose` 早已算出 `R / t / 內點匹配`，過去被丟棄，現在直接重用做運動方向與三角化深度。
- 室內/戶外改用 **`cv2.dnn` + Places365 ResNet18**（仍是 OpenCV，推論時不需 PyTorch）。古典啟發式在夜景/雪地/明亮室內會失準（實測 3/6），DNN 把同一組照片拉到 **6/6**。詳見下方〔室內/戶外〕專章。
- 單張圖片的 XYZ 用**水平線傾斜估計**（Canny + HoughLinesP）算出 roll/pitch。

### How（怎麼運作）
1. ORB 特徵 + BFMatcher + Lowe Ratio 找對應點
2. `findEssentialMat`(RANSAC) → `recoverPose` 取相對 `R, t` 與內點
3. 累積 `R_global` → YXZ Euler 分解出 yaw/pitch/roll
4. `t` → 相機運動方向；內點位移 → 光流方向 / zoom
5. `triangulatePoints(K[I|0], K[R|t])` → 相對稀疏深度 → 分級
6. 室內/戶外：`cv2.dnn` 跑 Places365（每 N 幀一次）→ 365 類機率 → 依官方 IO 表彙總成 indoor/outdoor
7. 每幀疊加 overlay + 寫 CSV

> ⚠️ **單張圖片限制**：yaw 需 ≥2 幀（無相對運動 → `N/A`）；roll/pitch 改由水平線估計；動態方向/深度仍需影片（`N/A`）。

---

## 功能特色

| 項目 | 說明 |
|------|------|
| **輸入** | 影片檔 (mp4/avi/mov)、即時攝影機（`0`/`1`…）、單張圖片 (jpg/png…) |
| **輸出** | yaw/pitch/roll (°)、室內/戶外、動態方向、相對深度、FPS@解析度、標注 mp4/PNG、CSV、曲線圖 |
| **演算法** | ORB + 5-point Essential Matrix (RANSAC) + 三角化深度 + 古典 CV 場景啟發式 |
| **內參** | 近似法（fx=fy=W）或 `camera_matrix.yaml`（校正） |
| **硬體目標** | Raspberry Pi 4B，15–25 FPS @ 480p |
| **場景** | 室內 / 戶外 / 動態，任意場景皆可 |

---

## 系統流程圖

```mermaid
flowchart LR
    VIDEO[影片檔] --> RESIZE[Resize\n目標解析度]
    RESIZE --> GRAY[BGR→Gray]
    GRAY --> ORB[ORB 特徵\n偵測+描述子]
    ORB --> MATCH[BFMatcher\nLowe Ratio]
    MATCH --> ESSENTIAL[findEssentialMat\nRANSAC]
    ESSENTIAL --> RECOVER[recoverPose\nR, unit-t]
    RECOVER --> ACCUM[R_global × R\n累積旋轉]
    ACCUM --> EULER[YXZ Euler\nyaw/pitch/roll]
    EULER --> VIZ[標注疊加\n方向指示器]
    VIZ --> OUT_MP4[標注 mp4]
    EULER --> OUT_CSV[pose CSV]
    OUT_CSV --> PLOT[plot_poses.py\n→ PNG 曲線]
```

詳細設計見 [`docs/architecture.md`](docs/architecture.md)；**驗收方法與標準答案**見 [`docs/validation.md`](docs/validation.md)。

---

## 快速開始

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 1.5（選用，建議）產生 Places365 模型 → 室內/戶外從 3/6 提升到 6/6
#     沒有模型時會自動退回古典啟發式，仍可執行
python models/export_places365_onnx.py

# 2. 執行姿態估計（影片 / 即時攝影機 / 單張圖片）
python src/main.py test_inputs/indoor_office.mp4
python src/main.py 0                 # 即時 webcam（索引 0）
python src/main.py photo.jpg         # 單張圖片 → 只輸出室內/戶外

# 3. 繪製曲線圖
python plot_poses.py runs/indoor_office_pose.csv

# 4. 多解析度 Benchmark
python benchmarks/run_benchmark.py --videos test_inputs/
```

---

## 輸出格式

### 標注影片 (mp4)

- 左上角：YAW / PIT / ROL 數值 + FPS@解析度 + 特徵點數
- 右上角：XYZ 方向指示器（正交投影）

### CSV (`runs/<stem>_pose.csv`)

```
frame_idx,timestamp_s,yaw_deg,pitch_deg,roll_deg,fps,inliers,nfeatures,scene,scene_conf,cam_motion,flow_motion,zoom_in,rel_depth,depth_level
0,0.000,0.00,0.00,0.00,11.2,-1,500,indoor,0.89,STILL,N/A,0,N/A,N/A
1,0.033,0.69,0.08,-0.17,14.6,67,500,indoor,0.89,FWD,PAN-L,0,42.48,FAR
```

| 欄位 | 說明 |
|------|------|
| `scene` / `scene_conf` | indoor/outdoor 與信心值 [0.5,1.0] |
| `cam_motion` | 相機運動方向 FWD/BACK/LEFT/RIGHT/UP/DOWN/STILL（由 `t`） |
| `flow_motion` | 畫面光流方向 PAN-L/PAN-R/TILT-U/TILT-D/STILL/N/A |
| `zoom_in` | 1 = 特徵向外發散（推近） |
| `rel_depth` / `depth_level` | 相對深度中位數（baseline 單位，**非絕對**）+ NEAR/MID/FAR |

> 單張圖片輸入時：`yaw/pitch/roll/cam_motion/flow_motion/depth_level` 皆為 `N/A`，僅 `scene` 有值，並輸出 `runs/<stem>_pose.png`。

### PNG 曲線圖

```bash
python plot_poses.py runs/<stem>_pose.csv
```

三欄圖：yaw / pitch / roll 對時間（秒）。

---

## CLI 參數

```
usage: python src/main.py <source> [options]

positional:
  source               影片或圖片檔路徑

輸出:
  --output -o PATH     輸出 mp4 路徑（預設: runs/<stem>_pose.mp4）
  --no-show            不顯示視窗（無頭模式）
  --no-video           不輸出 mp4（僅 CSV）

解析度:
  --imgsz INT          目標寬度 px（預設: 640，0=保持原始）
  --pi-sim             強制 640×480（Pi 4B 模擬）

演算法調整:
  --nfeatures INT      ORB 最大特徵數（預設: 500）
  --ratio-thresh FLOAT Lowe Ratio 閾值（預設: 0.75）
  --ransac-thresh FLOAT RANSAC 閾值 px（預設: 1.0）
  --keyframe-ratio FLOAT inlier ratio < X 時替換關鍵幀（預設: 0.5）
  --keyframe-inliers INT inlier count < X 時替換關鍵幀（預設: 60）

校正:
  --calib PATH         camera_matrix.yaml（由 calibrate.py 產生）

室內/戶外分類:
  --scene-model PATH   Places365 ONNX（預設 models/places365_resnet18.onnx）
  --scene-io PATH      indoor/outdoor 對照表（預設 models/io_places365.txt）
                       兩者皆不存在時 → 自動退回古典 HSV 啟發式
```

---

## 相機校正（選用）

若需更高精度（±1–2° 以下），先執行棋盤格校正：

```bash
# 拍攝 9×6 棋盤格（10×7 方格）≥ 15 張
python calibrate.py --images calib_images/ --output my_camera.yaml
python src/main.py video.mp4 --calib my_camera.yaml
```

不校正時使用 `fx = fy = W` 近似，對旋轉趨勢誤差約 5–10°。

---

## Benchmark

```bash
# 命名規則: indoor_*.mp4 / outdoor_*.mp4 / dynamic_*.mp4
python benchmarks/run_benchmark.py --videos test_inputs/
# 結果寫入 benchmarks/results.md
```

掃描 320 / 480 / 720 / 1080 px × 室內 / 戶外 / 動態場景。

---

## 演算法簡介

| 步驟 | 方法 | 說明 |
|------|------|------|
| 特徵偵測 | ORB (nFeatures=500, nlevels=4) | Pi 4B CPU-friendly，二進制描述子 |
| 特徵匹配 | BFMatcher Hamming + Lowe Ratio (0.75) | 濾除模糊對應 |
| 姿態求解 | findEssentialMat (RANSAC, 1px) | 5-point 演算法 |
| 旋轉分解 | recoverPose → R | 單位平移向量（尺度模糊不累積） |
| 累積 | R_global = R_global × R | 相對第 0 幀的累積旋轉 |
| Euler | YXZ 分解 (Ry·Rx·Rz) | yaw(Y) / pitch(X) / roll(Z) |
| 關鍵幀 | inlier ratio < 50% → 替換參考幀 | 限制漂移 |

---

## 室內/戶外分類：為什麼用 `cv2.dnn` + Places365（課堂報告重點）

### Q1. 為什麼古典 CV 啟發式不夠？
室內/戶外是**語意層級**的辨識，但亮度、藍天、綠色這些**低階特徵只跟「白天」場景相關**。實測你提供的 6 張照片：

| 圖 | 實際 | 古典啟發式 | DNN(Places365) |
|----|------|-----------|----------------|
| test1 明亮客廳 | 室內 | ❌ outdoor 0.57 | ✅ indoor 0.94 |
| test2 廚房 | 室內 | ✅ indoor | ✅ indoor 0.99 |
| test3 海邊建築 | 戶外 | ✅ outdoor | ✅ outdoor 1.00 |
| test4 草地藍天 | 戶外 | ✅ outdoor | ✅ outdoor 0.98 |
| test5 夜景房子 | 戶外 | ❌ indoor 0.89 | ✅ outdoor 0.99 |
| test6 黃昏老街 | 戶外 | ❌ indoor 0.59 | ✅ outdoor 0.96 |
| **準確率** | | **3/6** | **6/6** |

失敗模式很一致：**明亮室內被當戶外、夜景/黃昏/雪地被當室內**——因為它們缺乏「藍天」這個正向線索。這在古典 CV 下**本質難解**。

### Q2. 用 ML 不就違反「用 OpenCV」了嗎？→ 沒有
**OpenCV 內建 `cv2.dnn` 模組**，可直接在 CPU 載入並執行 ONNX 模型——**推論時完全不需要 PyTorch/TensorFlow**。

```python
net = cv2.dnn.readNetFromONNX("models/places365_resnet18.onnx")
net.setInput(blob); logits = net.forward()        # 純 OpenCV 推論
```

所以「用 OpenCV」與「跑 CNN」並不衝突。對**嵌入式**作業而言，展示「在 Pi 4B 上用 cv2.dnn 跑 CNN 推論」反而比手刻啟發式更貼近嵌入式視覺的實務。

### Q3. 為什麼選 Places365 ResNet18？
- **Places365**：MIT 專為**場景辨識**訓練（365 類場景），官方附 `IO_places365.txt` 把每類標記為 indoor(1)/outdoor(2)。我們跑一次推論 → softmax → 依 IO 表把 365 類機率**彙總**成室內 vs 戶外（見 `scene.aggregate_io`）。
- **ResNet18**：官方釋出權重中**最小**的一個（ONNX 約 45MB），是 Pi 4B 上**體積/準確率的最佳平衡**。

### Q4. Pi 4B 跑得動嗎？
跑得動，關鍵是**不需要每幀跑**：
- 室內/戶外**變化很慢**，每 ~30 幀（約 1 秒）推論一次即可（`_SCENE_INTERVAL`）。
- ResNet18 224² 在 Pi 4B CPU 約 **0.6–0.9s/次**（桌機實測 ~12ms）；因每秒才一次，**姿態主迴圈仍維持 15–25 FPS** 不受影響。

### Q5. 沒有模型檔會怎樣？→ 優雅退化
`SceneClassifier.try_load()` 找不到模型時回傳 `None`，自動退回**古典啟發式**（零依賴）。所以模型在就準、模型不在也能跑，**嵌入式部署有保險**。

```
scene.py:  有模型 → cv2.dnn 推論（準）   /   無模型 → HSV 啟發式（fallback）
```

### 模型如何產生（一次性，桌機執行）
```bash
python models/export_places365_onnx.py
# 下載官方 resnet18_places365 權重 + IO 表，用 torch 轉成
#   models/places365_resnet18.onnx  (cv2.dnn 載入用，runtime 不需 torch)
#   models/io_places365.txt         (365 行，1=室內 2=戶外)
```

---

## 單張圖片的 XYZ：水平線傾斜估計

影片的 XYZ 來自**跨幀累積的相對旋轉**；單張靜態圖片沒有第二幀，所以改判讀相機相對地面的**傾斜**：

| 角 | 從哪來 | 可觀測？ |
|----|--------|---------|
| **roll** | 主導水平線的傾斜角（Canny + HoughLinesP） | ✅ |
| **pitch** | 水平線在畫面中的高度（高於中心→俯角為正） | ✅ |
| **yaw** | 單圖無航向基準 | ❌ 固定 0 |

得到 `(0, pitch, roll)` 後用 `ypr_to_R` 建旋轉矩陣，餵給與影片**同一個** XYZ 指示器繪製——照片越傾斜，右上角的 XYZ 軸跟著轉。實作見 `src/orient.py`。

---

## Pi 4B 效能參考

| `--imgsz` | 解析度 | 預估 FPS |
|-----------|--------|---------|
| 320 | 320×240 | 25–35 |
| 480 | 640×480 | 15–25 |
| 720 | 1280×720 | 8–12 |
| 1080 | 1920×1080 | 4–6 |
