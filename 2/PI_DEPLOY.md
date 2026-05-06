# Raspberry Pi 4B 部署指南

## 需要傳輸的檔案

```
2/
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── estimator.py
│   └── visualize.py
├── calibrate.py       （選用）
├── plot_poses.py
├── requirements.txt
```

## Pi 4B 環境安裝

```bash
# 系統套件
sudo apt update && sudo apt install -y python3-pip

# Python 相依
pip3 install numpy>=1.24.0 matplotlib>=3.7.0

# OpenCV — 優先使用 headless 版（無 GUI 相依，較輕量）
pip3 install opencv-python-headless>=4.8.0

# 或安裝完整版（需 X11）
# pip3 install opencv-python>=4.8.0
```

## 傳輸檔案到 Pi

```bash
# 從開發機（Windows/Mac/Linux）
scp -r 2/src 2/plot_poses.py 2/calibrate.py 2/requirements.txt \
    pi@raspberrypi.local:~/pose_estimator/
```

## 執行（無頭模式）

```bash
# 基本執行（480p，僅輸出 CSV）
python3 src/main.py video.mp4 --imgsz 480 --no-show

# 輸出標注影片
python3 src/main.py video.mp4 --imgsz 480 --no-show --output result.mp4

# 高速模式（320p，最大 FPS）
python3 src/main.py video.mp4 --imgsz 320 --no-show --no-video

# 校正內參後使用
python3 src/main.py video.mp4 --calib camera_matrix.yaml --imgsz 480 --no-show
```

## 建議參數組合

| 使用情境 | 指令 |
|---------|------|
| 即時預覽（15–25 FPS）| `--imgsz 480 --nfeatures 500` |
| 最高精度（離線） | `--imgsz 720 --nfeatures 800 --ratio-thresh 0.70 --no-show` |
| 快速預覽 | `--imgsz 320 --nfeatures 300` |
| Benchmark | `--imgsz 480 --no-show --no-video` |

## 效能參考（Pi 4B，1.8 GHz，4 核）

| `--imgsz` | `--nfeatures` | 預估 FPS |
|-----------|--------------|---------|
| 320 | 300 | 30–40 |
| 480 | 500 | 15–25 |
| 720 | 500 | 8–12 |
| 1080 | 500 | 4–6 |

> `--no-video` 排除 VideoWriter 編碼成本（可提升 2–4 FPS）。

## 離線繪圖（在 Pi 上或傳回桌機）

```bash
# Pi 上執行（需安裝 matplotlib）
python3 plot_poses.py runs/video_pose.csv

# 或傳回桌機再繪圖
scp pi@raspberrypi.local:~/pose_estimator/runs/*.csv ./
python plot_poses.py runs/video_pose.csv
```
