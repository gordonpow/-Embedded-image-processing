# 驗收方法書（Validation / Acceptance）

> 教授要求：**每個測試輸入都要有標準答案（ground truth），才能比對是否正確。**
> 本文說明：每項能力用**哪個資料集當標準答案**、**用什麼指標量誤差**、**通過門檻**、**怎麼跑**，以及**公開資料集**清單。

---

## 0. 驗收哲學：為什麼不能只看「畫面好看」

本系統多數輸出是**相對量**（單目尺度模糊、單圖無航向），所以驗收必須挑**對應的標準答案**與**合適的指標**：

| 輸出 | 性質 | 標準答案來源 | 指標 |
|------|------|-------------|------|
| yaw/pitch/roll | 相對第 0 幀的旋轉 | 合成影片（**精確 GT**）/ TUM RGB-D（動捕） | 角度 MAE / RMSE (°) |
| 室內/戶外 | 類別 | 標註檔 / Places365 驗證集 | 準確率 % + 混淆矩陣 |
| 深度 | **相對**（尺度模糊） | NYU V2 / KITTI / DIODE | 尺度對齊後 AbsRel / 排序相關 |
| 動態方向/光流 | 方向 | MPI Sintel / KITTI flow | EPE (px) / 方向一致率 |

> 重點：**相對量不能直接和絕對 GT 比**——深度要先做尺度對齊，姿態要用相同的旋轉慣例。

---

## 1. 姿態（yaw/pitch/roll）✅ 已可自動驗收

### 標準答案
`test_inputs/gen_synthetic3d.py`：以**已知**的逐幀 yaw/pitch/roll 渲染一個 3D 點雲場景，相機**同時旋轉＋平移**（提供視差），輸出：
- `synthetic3d_pose_test.mp4`
- `synthetic3d_pose_gt.csv`（精確標準答案，逐幀 yaw/pitch/roll）

> **為什麼不用純旋轉的舊合成片？** 純旋轉（單應變換）對 5-point Essential Matrix 是**退化情況**（無視差→本質矩陣病態），用它當標準答案會「冤枉」演算法。真實手持影片是旋轉＋平移，所以 3D 版才是公平的標準答案。

### 指標與門檻
逐幀角度誤差（環繞修正）的 **MAE / RMSE / MAX**；通過條件：**每軸 MAE < 15°**（12 秒單目影片、無回環校正的合理上限）。

### 怎麼跑
```bash
python test_inputs/gen_synthetic3d.py          # 產生影片 + 標準答案
python benchmarks/validate_pose.py \
    --video test_inputs/synthetic3d_pose_test.mp4 \
    --gt    test_inputs/synthetic3d_pose_gt.csv
```

### 實測結果
| 軸 | MAE | RMSE | MAX |
|----|-----|------|-----|
| yaw | 4.98° | 5.70° | 11.6° |
| pitch | 9.01° | 11.7° | 24.1° |
| roll | 13.29° | 16.7° | 31.8° |

→ 最差軸 MAE 13.3° < 15° **PASS**。誤差隨影片長度近似線性成長（單目 VO 漂移，需回環偵測才能消除）。

### 🐛 驗收抓到的真實 Bug（報告亮點）
這套標準答案**第一次跑就抓到一個既有 bug**：`R_global = R_global @ R` 每幀都乘，但 `R` 是「相對參考幀」的旋轉、參考幀又會持續多幀不變 → **重複累乘、過度累積**（合成片上 MAE 飆到 60–100°）。
- **根因**：累積邏輯把「相對參考幀」誤當「相對前一幀」。
- **修正**：改為 keyframe 相對累積 `R_global = R_base @ R`，並在換參考幀時凍結 `R_base`（`src/estimator.py::_set_reference`）。
- **效果**：MAE 100° → 13°。
- 回歸測試：`tests/test_pose_accuracy.py`（先寫失敗測試 → 修 → 通過）。

### 真實資料集實測：TUM RGB-D（已執行）

**TUM RGB-D**（[cvg.cit.tum.de](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)）是相機姿態的**業界標準 benchmark**——動作捕捉系統提供精確 GT 軌跡。我們已實際下載 `freiburg1_desk`（613 幀手持桌面拍攝）並比對。

```bash
# 下載並解壓 rgbd_dataset_freiburg1_desk 到 test_inputs/tum/ 後：
python benchmarks/validate_tum.py \
    --seq test_inputs/tum/rgbd_dataset_freiburg1_desk --plot docs/tum_desk_compare.png
```
驗收器自動：時間戳對齊 rgb↔GT、四元數→旋轉矩陣、相對第 0 幀比對、自動選慣例（幾何角度，慣例無關）。

**結果（估計 vs 真值）：**

![TUM 估計 vs 真值](tum_desk_compare.png)

| 區段 | 表現 |
|------|------|
| 前 ~200 幀（約 7 秒） | **緊貼真值**，yaw/pitch/roll 三軸都跟得很準（誤差數度） |
| 200 幀之後 | 開始**漂移**，到序列末端偏離 50–110° |
| 整段 | 每軸 MAE：yaw 30° / pitch 21° / roll 46°；幾何旋轉誤差 MAE 55.8° |

**誠實結論（報告重點）：**
- **短期準、長期漂移**是**單目視覺里程計的本質特性**——沒有回環偵測（loop closure）與光束法平差（bundle adjustment），誤差會隨時間累積。
- TUM `fr1` 序列是公認**最難**的（快速運動 + 動態模糊），連 ORB-SLAM 都需要完整後端才能處理。
- 我們驗證過：把 keyframe 更新頻率從 3～9999 幀掃了一輪，**任何設定都無法消除長期漂移**（34–56°）——這確認是**架構層級的界線**，不是參數問題。
- **適用界線**：本系統適合**短～中時長、中等速度**的片段（前 200 幀 MAE 約數度）；長時間快速序列需完整 SLAM 後端，超出 Pi 4B 即時預算。

> 其他可選真實資料集：EuRoC MAV、ICL-NUIM、KITTI odometry。

---

## 2. 室內/戶外 ✅ 已可自動驗收

### 標準答案
`benchmarks/labels_demo.csv`：人工標註你提供的 6 張照片（`filename,label`）。可無痛擴充成 **Places365 驗證集**（每類已有 indoor/outdoor 標籤）。

### 指標與門檻
**準確率 %** + 混淆矩陣；通過條件 **≥ 80%**。

### 怎麼跑
```bash
python benchmarks/validate_scene.py            # 6 張示範
python benchmarks/validate_scene.py --images <資料夾> --labels <labels.csv>
```

### 實測結果
DNN(Places365) 後端：**6/6 = 100% PASS**（古典啟發式僅 3/6，見 [readme](../readme.md) 對照表）。

### 真實資料集
**Places365**（[CSAILVision/places365](https://github.com/CSAILVision/places365)）含 `IO_places365.txt` 室內/戶外對照；**SUN397** 亦可。把驗證集影像 + 標籤做成 `labels.csv` 即可大規模驗收。

---

## 3. 深度（相對）📋 方法已定，需下載資料集

### 為什麼要特別處理
本系統輸出**相對深度**（單目尺度模糊，`t` 為單位向量），**不能**直接和公尺級 GT 比。標準做法：
1. **尺度對齊**：用 GT 與預測的**中位數比值**對齊尺度後算 **AbsRel**（`mean(|d_pred − d_gt| / d_gt)`）。
2. 或**排序相關**（Spearman）：只看「誰近誰遠」的次序是否正確（對尺度免疫）。

### 標準答案資料集
| 資料集 | 場景 | GT 來源 |
|--------|------|---------|
| **NYU Depth V2** | 室內 | Kinect 深度 |
| **KITTI** | 室外 | LiDAR |
| **DIODE** | 室內+室外 | 雷射掃描 |

### 怎麼驗收
在我們三角化出深度的**特徵點位置**取 GT 深度 → 尺度對齊 → 算 AbsRel / Spearman。門檻建議：Spearman ρ ≥ 0.6（次序大致正確）。
> 註：此項尚未附自動腳本（資料集需數十 GB 下載）；方法與指標如上，可依需要補 `validate_depth.py`。

---

## 4. 動態方向 / 光流 📋 方法已定，需下載資料集

### 標準答案資料集
- **MPI Sintel**：稠密光流 GT（[評測常用 EPE](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)）。
- **KITTI flow 2015**：200 訓練場景，動態 GT；判定標準「EPE < 3px 或 < 5%」。

### 指標
- 稠密光流：**EPE（End-Point Error, px）**。
- 我們的「動態方向」是**主方向**，可退化成**方向一致率**：把 GT 光流的主方向（中位向量）與我們的 `flow_motion` 標籤比對，算一致百分比。
> 註：此項同樣為方法定義（GT 需下載），可補 `validate_flow.py`。

---

## 5. 一鍵驗收（已實作部分）

```bash
# 姿態（合成 3D，精確 GT）
python test_inputs/gen_synthetic3d.py
python benchmarks/validate_pose.py --video test_inputs/synthetic3d_pose_test.mp4 \
                                   --gt test_inputs/synthetic3d_pose_gt.csv

# 室內/戶外（標註檔）
python benchmarks/validate_scene.py
```

| 能力 | 標準答案 | 指標 | 門檻 | 現況 |
|------|---------|------|------|------|
| 姿態 | 合成 3D GT | 每軸 MAE | < 15° | ✅ 13.3° PASS |
| 室內/戶外 | 標註檔 / Places365 | 準確率 | ≥ 80% | ✅ 100% PASS |
| 深度 | NYU/KITTI/DIODE | AbsRel / Spearman | ρ ≥ 0.6 | 📋 方法已定 |
| 光流 | Sintel / KITTI flow | EPE / 方向一致率 | EPE < 3px | 📋 方法已定 |

---

## 參考資料（標準答案資料集）
- TUM RGB-D SLAM Dataset：https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- Places365（含 IO 標籤）：https://github.com/CSAILVision/places365
- NYU Depth V2 / KITTI / DIODE：單目深度常用 GT 資料集
- MPI Sintel / KITTI flow 2015：https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php
