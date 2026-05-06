# 測試結果總表

測試環境：開發機 CPU-only，影片均為 480p。Pi 4B 預估值以 **÷10** 換算（README 既有 baseline 推導）。

| Step | 模型 | imgsz | conf | 畫法 | speed | avg_fps（開發機） | Pi 4B 預估 | 備註 |
|------|------|-------|------|------|-------|-----------------|-----------|------|
| 2.1 | FP32-320 | 320 | 0.35 | bbox    | 4x | 118.87 | ~12 FPS | Baseline，無 contour |
| 2.2 | FP32-320 | 320 | 0.35 | contour | 4x | 113.56 | ~11 FPS | contour 開銷僅 4% |
| 2.3a | FP32-320 | 320 | 0.25 | contour | 4x | 108.61 | ~11 FPS | 低 conf，偵測較多 |
| 2.3b | FP32-320 | 320 | 0.35 | contour | 4x | 109.50 | ~11 FPS | 平衡點 |
| 2.3c | FP32-320 | 320 | 0.45 | contour | 4x | 111.10 | ~11 FPS | 高 conf，偵測較少 |
| 2.4 | FP32-416 | 416 | 0.35 | contour | 4x |  69.73 |  ~7 FPS | 解析度提升但速度邊緣 |
| 2.5 | INT8-320 | 320 | 0.35 | contour | 4x |   7.99 | TBD(Pi NEON) | x86 有 dequant 損耗，Pi 上才準 |
| 2.6 | FP32-320 | 320 | 0.35 | contour | 5x | 113.83 | ~11 FPS | test2 從 450s 跑，2687 幀穩定 |

---

## 優化結論

### 最佳組合（開發機 / Pi 4B 部署）

```
--model models/fire_smoke_yolov8n_320.onnx
--imgsz 320
--conf  0.35
--contour
--skip  1        # Pi 若實測 < 8 FPS 改 --skip 2
```

### 各項分析

| 維度 | 結論 |
|------|------|
| **畫法** | contour 僅多 4%，視覺效果明顯改善，無條件開啟 |
| **imgsz** | 320 最佳；416 在 Pi 預估僅 7 FPS，低於 10 FPS 目標 |
| **conf** | 0.35 不影響速度，維持 README 已知最佳偵測率 |
| **speed** | 測試用 4x/5x（縮短審查時間）；正式部署 = 1（Pi Camera 即時） |
| **INT8** | 需在 Pi NEON 實測，x86 結果（8 FPS）不具參考性 |

### Pi 4B 速度預測

- FP32-320 + contour：**~11 FPS** → 滿足 ≥ 10 FPS 目標 ✓
- 若 Pi 實測低於 8 FPS：加 `--skip 2`（推論率降一半，顯示仍 15 FPS）

---

## 輸出影片清單

| 檔案 | 說明 |
|------|------|
| `test1_baseline_4x.mp4` | bbox，可與 contour 版並排比較 |
| `test1_contour_4x.mp4` | **主要提交版** contour 畫法 |
| `test1_conf0.25_4x.mp4` | 低 conf，偵測較積極 |
| `test1_conf0.35_4x.mp4` | 平衡點 |
| `test1_conf0.45_4x.mp4` | 高 conf，誤報最少 |
| `test1_416_4x.mp4` | 416 解析度參考 |
| `test1_int8_4x.mp4` | INT8 參考（x86 不準） |
| `test2_mid_5x.mp4` | test2 後半段 450s→end，5x 速 |
