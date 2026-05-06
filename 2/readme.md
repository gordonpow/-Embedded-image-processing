# Camera Pose Estimator — Yaw / Pitch / Roll

> 從預錄影片提取相機相對旋轉姿態（yaw、pitch、roll），專為 **Raspberry Pi 4B** 設計。  
> 純 CPU、無 ML 模型、僅依賴 `opencv-python`。

---

## 功能特色

| 項目 | 說明 |
|------|------|
| **輸入** | 影片檔 (mp4/avi/mov) 或圖片 |
| **輸出** | yaw/pitch/roll (°)、標注 mp4、CSV、matplotlib PNG 曲線圖 |
| **演算法** | ORB + 5-point Essential Matrix (RANSAC) |
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

詳細設計見 [`docs/architecture.md`](docs/architecture.md)。

---

## 快速開始

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 2. 執行姿態估計
python src/main.py test_inputs/indoor_office.mp4

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
frame_idx,timestamp_s,yaw_deg,pitch_deg,roll_deg,fps,inliers,nfeatures
0,0.000,0.00,0.00,0.00,-1,0,500
1,0.033,-0.45,0.21,0.08,22.3,148,500
```

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

## Pi 4B 效能參考

| `--imgsz` | 解析度 | 預估 FPS |
|-----------|--------|---------|
| 320 | 320×240 | 25–35 |
| 480 | 640×480 | 15–25 |
| 720 | 1280×720 | 8–12 |
| 1080 | 1920×1080 | 4–6 |
