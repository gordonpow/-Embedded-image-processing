# 相機姿態與場景感知 — Yaw/Pitch/Roll · 室內外 · 動態 · 深度

> 輸入**影片 / 即時攝影機 / 單張圖片**，輸出相機姿態（yaw/pitch/roll）、室內或戶外、
> 動態方向、相對深度與 FPS@解析度。專為 **Raspberry Pi 4B（CPU-only）** 設計，
> 全程以 **OpenCV** 完成（含 `cv2.dnn` 場景分類，推論時不需 PyTorch）。

## 目錄
1. [需求](#1-需求)
2. [BreakDown](#2-breakdown)
3. [架構圖](#3-breakdown)
4. [系統總覽（What / Why / How）](#2-系統總覽whatwhyhow)
5. [快速開始](#3-快速開始)
6. [設計](#4-設計)
7. [方法說明](#5-方法說明)
8. [驗收（標準答案）](#6-驗收標準答案)
9. [CLI 參數](#7-cli-參數)
10. [Pi 4B 效能與參數調整](#8-pi-4b-效能與參數調整)

---

## 1. 需求

### 功能需求
| 項目     | 說明                                                                   |
| -------- | ---------------------------------------------------------------------- |
| 輸入     | 影片 (mp4/avi/mov)、即時攝影機（`0`/`1`…）、單張圖片 (jpg/png…)        |
| 姿態輸出 | yaw / pitch / roll（度）                                               |
| 場景輸出 | 室內 / 戶外（indoor / outdoor）                                        |
| 動態方向 | 相機運動方向（t）+ 畫面光流方向 + zoom                                 |
| 深度     | 相對深度分級 NEAR / MID / FAR（尺度相對，非絕對）                      |
| 效能     | FPS@解析度                                                             |
| 視覺化   | 影片畫光流箭頭、單圖畫水平線/特徵梯度場/消失點 + XYZ 指示器 + 中文 HUD |
| 無頭模式 | `--no-show` 於無螢幕 Pi 上執行                                         |

### 規格需求
| 項目     | 規格                                                               |
| -------- | ------------------------------------------------------------------ |
| 目標平台 | Raspberry Pi 4B（ARM Cortex-A72，CPU-only）                        |
| 解析度   | 預設 640×480（可調）                                               |
| 目標幀率 | 15–25 FPS @ 480p（姿態主迴圈）                                     |
| 依賴     | `opencv-python`、`numpy`、`matplotlib`、`Pillow`（中文 HUD，可選） |
| 語言     | Python 3.10+                                                       |

---

## 2. BreakDown

![Breakdown](./img/breakdown.png)


## 3. 架構圖

![Flowchart](./img/流程圖.png)


## 4. 系統總覽（What / Why / How）



#### 1. PRE (前處理: Resize → Gray)
*   **他是什麼**：在處理畫面之前，先幫照片「減肥」並「去掉顏色」的準備步驟。
*   **為什麼在這個專案要這樣做**：
    *   **照片瘦身 (Resize)**：我們用的樹莓派 4B 腦力（運算能力）有限。如果直接處理高清大圖，畫面會卡成投影片。把影像縮小到 $640 \times 480$，就像是把大張海報縮小成明信片，既看得清主要特徵，樹莓派跑起來又順暢，能維持每秒 15–25 張畫面的流暢度。
    *   **去掉顏色 (Gray)**：對電腦來說，辨識地標只需要看「明暗變化」，不需要知道是紅是綠。彩色照片的資料量是黑白照片的三倍，變成黑白（灰階）可以省下三分之二的傳輸與運算量，讓運算速度飛快。
*   **我們如何達成這個功能**：在 `pipeline.py` 中，呼叫 `cv2.resize` 與 `cv2.cvtColor`。
*   **函數參數意義與設定理由**：
    *   **`cv2.resize(frame, (tgt_w, tgt_h))`**：
        *   `frame`：輸入影像。
        *   `dsize`：目標解析度。專案設定值為 `(tgt_w, tgt_h)`，預設為 `(640, 480)`。這是在樹莓派 4B 算力限制下，特徵保留與速度的最佳平衡點。
    *   **`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`**：
        *   `frame`：BGR 彩色影像。
        *   `code`：色彩空間轉換常數。專案採用 `cv2.COLOR_BGR2GRAY`，將彩色轉為單通道灰階以利特徵點偵測。

#### 2. ORB (ORB + BFMatcher Lowe Ratio)
*   **他是什麼**：影像的「地標連連看」遊戲。它會在前後兩張畫面中，找出最特別的「地標」（像角落、亮點、邊緣），然後把前後兩張照片中代表同一個地方的地標連線配對。
*   **為什麼在這個專案要這樣做**：
    *   當相機移動時，畫面中的地標也會移動。我們必須找出前後畫面裡「同一個東西」在哪裡，才能推算相機是怎麼動的。
    *   **ORB 演算法**：是個公認速度極快、又不用版權費的「地標偵測器」，特別適合在樹莓派上跑。
    *   **BFMatcher**：用最快的方式（漢明距離）來比對這些地標的身分證（描述子），找出最像的配對。
    *   **Lowe's Ratio Test (嚴格篩選)**：為了避免連錯線，我們會做一個防呆機制。如果一個地標跟「最像的第一名」和「第二像的第二名」差不多像，代表很容易搞混，我們就放棄不用，只留下非常有把握的正確連線。
*   **我們如何達成這個功能**：在 `estimator.py` 中初始化，並呼叫 `self._matcher.knnMatch` 進行匹配篩選。
*   **函數參數意義與設定理由**：
    *   **`cv2.ORB_create(nfeatures=500, nlevels=4)`**：
        *   `nfeatures`：最大特徵點數。專案設定值為 `500`（可調整）。此值能維持本質矩陣求解的穩定性並控制 CPU 負載。
        *   `nlevels`：影像金字塔層數。專案設定為 `4`（預設為 8）。這減少了多尺度特徵計算，提速約 **30%**。
    *   **`cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)`**：
        *   `normType`：距離類型。設定為 `cv2.NORM_HAMMING`，因 ORB 是二進位描述子，使用漢明距離匹配速度最快。
        *   `crossCheck`：雙向交叉匹配。設定為 `False`，以保留 Top 2 匹配來進行 Lowe's Ratio 比對。
    *   **Lowe's Ratio Test 閥值**（專案代碼中的 `self._ratio_thresh = 0.75`）：
        *   最佳匹配距離必須小於次佳匹配的 `0.75` 倍，用以篩選出高置信度的物理對應點。

#### 3. EM (findEssentialMat & recoverPose → R,t,內點)
*   **他是什麼**：計算這一瞬間相機動了多少：相對旋轉矩陣 $R$、移動方向向量 $t$
*   **為什麼在這個專案要這樣做**：
    *   這是單鏡頭定位的靈魂。我們不需要安裝沉重又耗電的 AI 模型，只要用單純的幾何數學，就能從地標的移動軌跡，逆推回相機自身的動作。
*   **我們如何達成這個功能**：在 `estimator.py` 中呼叫 `cv2.findEssentialMat` 與 `cv2.recoverPose`。
*   **函數參數意義與設定理由**：
    *   **`cv2.findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold)`**：
        *   `points1, points2`：匹配好的點對坐標。
        *   `cameraMatrix`：相機內參矩陣（預設以 `fx=fy=W` 的近似陣代替）。
        *   `method`：演算法。設定為 `cv2.RANSAC`，可強健地剔除錯誤匹配。
        *   `prob`：信賴度。設定為 `0.999`，以最大機率找到最優單應/本質矩陣。
        *   `threshold`：RANSAC 最大重投影誤差。設定為 `1.0` 像素，此值可高精度篩除有微小漂移的雜點。
    *   **`cv2.recoverPose(E, points1, points2, cameraMatrix, mask)`**：
        *   `E`：本質矩陣。
        *   `mask`：RANSAC 內點遮罩，此函式會透過手性約束（點必須在相機前方）篩除無效點並解出旋轉矩陣 $R$ 與平移向量 $t$。

#### 4. POSE (keyframe 相對累積 & R_global → yaw/pitch/roll)
*   **他是什麼**：計算相對於最開始起點的全球三軸角度：Yaw / Pitch / Roll
*   **為什麼在這個專案要這樣做**：
    *   **關鍵影格累積**：如果我們每一步都跟前一張照片比，每一次的小誤差加起來，走沒幾步路地圖就全歪了（漂移）。所以我們只在必要時換「參考照片（Keyframe）」。在大部分時間裡，都拿目前畫面跟這張參考照片比，這樣誤差就不會一直疊加。
    *   **角度分解**：將旋轉矩陣 $R$ 轉為 Euler 角，能直觀呈現相機的偏航（Yaw）、俯仰（Pitch）、側傾（Roll）。
*   **我們如何達成這個功能**：在 `estimator.py` 中，計算 $R_{global} = R_{base} \times R$，並在 `_rot_to_ypr` 函式中完成分解。
*   **函數參數意義與設定理由**：
    *   **關鍵影格觸發門檻** (`self._kf_min_ratio = 0.5`, `self._kf_min_inliers = 60`)：
        *   當特徵匹配的內點比例低於 `50%` 或絕對內點數少於 `60` 個時，會觸發更換 Keyframe（更新 `R_base`）。這可最大程度減少參考影格的切換頻率，從而壓低誤差漂移。
    *   **歐拉角解算順序**（YXZ 順序）：
        *   專案依據相機坐標系定義（$Z$ 向前、$Y$ 向下、$X$ 向右）實作 $R = Ry(yaw) \times Rx(pitch) \times Rz(roll)$，以取得正確的三軸旋轉。

#### 5. MOT (t → 運動方向 & 內點位移 → 光流/zoom)
*   **他是什麼**：分析相機目前的移動趨勢（如前後左右上下），以及畫面中的地標是在朝我們靠近（拉近）還是遠離（拉遠）。
*   **為什麼在這個專案要這樣做**：
    *   這是一個「完全不花多餘力氣」的聰明設計！因為前面在算相機姿態時，已經把相機移動方向 $t$ 和地標位置都算好了，我們直接拿這些現成的資料，用簡單的幾何規則去判斷運動狀態，不需要在樹莓派上多跑一個沉重又慢的光流計算模型。
*   **我們如何達成這個功能**：在 `motion.py` 中實現 `camera_motion` 和 `flow_direction` 函式。
*   **函數參數意義與設定理由**：
    *   `_STILL_EPS = 1e-6`：平移量最小閾值。設定為 `1e-6` 可有效忽略浮點運算的微小噪訊，判定相機為靜止（STILL）。
    *   `_FLOW_MIN_PX = 1.0`（單位：像素）：點對移動的中位數若小於 1 像素，則判定畫面為 STILL，避免感測器噪點引發誤判。
    *   `_ZOOM_MIN = 0.0`：當特徵點徑向發散投影均值大於 `0` 且大於 `1.0` 像素時，即判定畫面正在推近（Zoom-in）。

#### 6. DEP (triangulatePoints → 相對深度)
*   **他是什麼**：利用「雙眼看物體」的原理（三角測量），計算地標距離相機有多遠，並分成「近、中、遠」三個等級。
*   **為什麼在這個專案要這樣做**：
    *   樹莓派沒辦法跑沉重的 AI 深度估算模型。我們利用前面算出來的相機位置、角度和地標座標，像人類兩隻眼睛看東西一樣進行「三角定位」，用極低的計算量就能估算出每個點的相對深度。
*   **我們如何達成這個功能**：在 `depth.py` 中，呼叫 `cv2.triangulatePoints`。
*   **函數參數意義與設定理由**：
    *   **`cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)`**：
        *   `projMatr1` (3x4)：第一相機投影矩陣，設為 $K \times [I | 0]$。
        *   `projMatr2` (3x4)：第二相機投影矩陣，設為 $K \times [R | t]$。
        *   `projPoints1, projPoints2`：匹配好的點對坐標（需轉置為 `2xN`）。
    *   **深度門檻** (`_NEAR_MAX = 5.0`, `_FAR_MIN = 15.0`)：
        *   單目估計的平移向量 $t$ 為單位向量（長度為 1），因此三角化結果代表的並非絕對公尺，而是「相對於相機基線的相對倍數」。中位數小於 5 倍基線時判定為 `NEAR`，大於 15 倍時判定為 `FAR`，其餘為 `MID`。此區間能有效區分室內外近中遠景。

#### 7. SCN (cv2.dnn Places365 → 室內/戶外)
*   **他是什麼**：用一個輕量型的 AI 大腦，幫我們一眼看出現在是在「室內」還是「室外」。
*   **為什麼要這樣做**：
    *   如果是傳統的方法（例如看天空的顏色、亮度等），只要遇到光線變化、晚上或是很亮的室內，就會猜錯（測試時六張照片只猜對三張）。
    *   所以我們引入了 Places365 專門認路辨識場景的輕量神經網路，把準確率直接拉高到 100%（六張全對）。
    *   為了不讓 AI 推理把樹莓派拖慢（跑一次 AI 要花將近一秒），我們設計了「每隔幾張照片才認一次路」的機制，既能維持高準度，畫面又不會卡。
*   **我們如何達成這個功能**：在 `scene.py` 中使用 `cv2.dnn.readNetFromONNX` 載入模型，並呼叫推理的 `setInput` 與 `forward` 方法。
*   **函數參數意義與設定理由**：
    *   **`cv2.dnn.readNetFromONNX(model_path)`**：
        *   `model_path`：ONNX 模型路徑。專案採用 `places365_resnet18.onnx`。ResNet18 結構簡單且推理速度快，符合嵌入式設備之限制。
    *   **DNN 影像預處理** (`_DNN_SIZE = 224`)：
        *   將輸入影格 `cv2.resize` 到 `(224, 224)`，並除以 255.0 做歸一化，減去 ImageNet 的 Mean 再除以 Std。這是確保模型輸入特徵分佈正確的標準做法。
    *   **推論間隔** (`_SCENE_INTERVAL = 10`，於 `pipeline.py`)：
        *   每隔 `10` 幀才重新推論一次場景，其餘幀使用快取結果。在 20 FPS 下約 0.5 秒執行一次，可將 DNN 造成的 CPU 佔用率降到最低。

#### 8. VIZ (疊圖 + 中文 HUD)
1. **他是什麼**：將姿態角度、FPS、運動方向、場景、相對深度，以及特徵點、光流箭頭、3D XYZ 旋轉指示軸實時繪製疊加在影像上的視覺化模組。
2. **為什麼在這個專案要這樣做**：
   * 提供直覺化的視覺回饋，讓使用者能直觀觀察系統的運作狀況（如特徵點分佈、光流是否符合移動方向、3D 姿態軸是否正確旋轉），便於調試與成果展示。
   * 支援繁體中文 HUD，提升介面的美觀度與專業感。
3. **我們如何達成這個功能**：在 `visualize.py` 與 `draw_cv.py` 中，使用 `cv2.circle`、`cv2.arrowedLine` 等函式繪圖，並利用 Pillow 繪製反鋸齒繁體中文（若無字型則自動退回 OpenCV 內建 ASCII 英文）。
4. **函數參數意義與設定理由**：
   * **`cv2.arrowedLine(frame, pt1, pt2, color=(255, 255, 0), thickness=1, line_type=cv2.LINE_AA, tipLength=0.3)`**（繪製光流箭頭）：
     * `pt1, pt2`：配對點的前後影格座標。
     * `color`：青藍色 `(255, 255, 0)`。
     * `line_type`：`cv2.LINE_AA`（反鋸齒線），可讓細小的箭頭邊緣極其平滑，提供高品質的視覺外觀。
   * **`cv2.circle(frame, (cx, cy), scale + 6, color=(25, 25, 25), thickness=-1, lineType=cv2.LINE_AA)`**（繪製右上角姿態底盤）：
     * `cx, cy`：圓盤中心點，專案定位於 `(w - 80, 80)`。
     * `scale + 6`：圓盤半徑（ scale=55，即半徑為 61 像素），實心填充（`thickness=-1`），作為 3D 姿態軸的背景板，防止影片亮色背景干擾。

#### 9. OUT (標注 mp4/PNG + CSV)
1. **他是什麼**：將寫有 HUD 與標註的影像壓縮存為 MP4 影片（或單張 PNG 圖片），並將各項感測器參數寫入 CSV 的資料輸出模組。
2. **為什麼在這個專案要這樣做**：
   * 作為專案的交付成果與後續性能驗證（Benchmark）的依據。CSV 檔案方便比對 Ground Truth，而標注影片則供使用者離線展示與結果驗收。
3. **我們如何達成這個功能**：在 `pipeline.py` 與 `L162-163` 中，呼叫 `cv2.VideoWriter` 寫入視訊，並以 Python 內建的 `csv` 模組寫入資料。
4. **函數參數意義與設定理由**：
   * **`cv2.VideoWriter(filename, fourcc, fps, frameSize)`**：
     * `filename`：輸出路徑。預設存至 `runs/<stem>_pose.mp4`。
     * `fourcc`：四字元編碼。設定為 `cv2.VideoWriter_fourcc(*"mp4v")`，這能輸出成高相容性的 MP4 格式。
     * `fps`：每秒影格數。設定為原始影片的幀率，防止播放速度失真。
     * `frameSize`：影像解析度大小。設為目標解析度 `(tgt_w, tgt_h)`，確保視訊編碼寫入正常。

---

## 3. 快速開始

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 2.（建議）產生 Places365 模型 → 室內/戶外 3/6 升到 6/6
#    沒有模型也能跑（自動退回古典 HSV 啟發式）
python models/export_places365_onnx.py

# 3. 執行（影片 / 即時攝影機 / 單張圖片）
python src/main.py test_inputs/synthetic3d_pose_test.mp4 --no-show
python src/main.py 0                      # 即時 webcam
python src/main.py test_inputs/test4.jpg  # 單張圖片

# 4. 驗收（與標準答案比對）
python benchmarks/validate_pose.py        # 姿態 vs 合成 GT
python benchmarks/validate_scene.py       # 室內/戶外 vs 標註

# 5. 繪製姿態曲線圖
python plot_poses.py runs/<stem>_pose.csv
```

---

## 4. 設計

### 模組職責
| 模組               | 職責                                                                           |
| ------------------ | ------------------------------------------------------------------------------ |
| `src/main.py`      | CLI 入口、來源解析（檔案/webcam 整數/圖片）                                    |
| `src/pipeline.py`  | 來源路由、幀迴圈、CSV、串接各模組                                              |
| `src/estimator.py` | ORB + Essential Matrix + **keyframe 相對累積** → yaw/pitch/roll、回傳 R/t/內點 |
| `src/scene.py`     | 室內/戶外：`cv2.dnn`+Places365（古典 HSV fallback）                            |
| `src/motion.py`    | `t` → 運動方向；內點位移 → 光流方向 / zoom                                     |
| `src/depth.py`     | `triangulatePoints` → 相對深度 + NEAR/MID/FAR                                  |
| `src/orient.py`    | 單圖 roll/pitch + 水平線/垂直線/消失點 + 單圖 yaw（VP）                        |
| `src/draw_cv.py`   | 中文 HUD（PIL，無字型退回英文）+ CV 證據繪製                                   |
| `src/visualize.py` | HUD + XYZ 方向指示器                                                           |
| `benchmarks/`      | `metrics` + `validate_pose` / `validate_scene`（驗收）                         |

完整資料流 / 公式推導見 [`docs/architecture.md`](docs/architecture.md)。

### 輸出格式
**標注影片/圖片**：左上角中文 HUD（偏航/俯仰/側傾/場景/運動/深度）、右上角 XYZ 指示器；影片畫光流箭頭，單圖畫水平線+特徵梯度場+消失點。

**CSV (`runs/<stem>_pose.csv`)** — 15 欄：
```
frame_idx,timestamp_s,yaw_deg,pitch_deg,roll_deg,fps,inliers,nfeatures,
scene,scene_conf,cam_motion,flow_motion,zoom_in,rel_depth,depth_level
```
| 欄位                        | 說明                                            |
| --------------------------- | ----------------------------------------------- |
| `scene` / `scene_conf`      | indoor/outdoor + 信心值                         |
| `cam_motion`                | FWD/BACK/LEFT/RIGHT/UP/DOWN/STILL（由 `t`）     |
| `flow_motion` / `zoom_in`   | PAN-L/PAN-R/TILT-U/TILT-D… / 1=推近             |
| `rel_depth` / `depth_level` | 相對深度（baseline 單位，非絕對）+ NEAR/MID/FAR |

> 單圖：`cam_motion/flow_motion/depth_level` 為 `N/A`；`yaw` 結構化場景才有值，否則 `N/A`；另輸出 `runs/<stem>_pose.png`。

---

## 5. 方法說明

| 技術                       | WHAT                              | WHY                                                          | HOW                                                                |
| -------------------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| **ORB + Essential Matrix** | 從兩幀特徵對應求相對旋轉          | Pi CPU 友善、任意紋理場景皆可                                | ORB→BFMatcher(Lowe)→`findEssentialMat`(RANSAC)→`recoverPose`→R,t   |
| **keyframe 相對累積**      | 把每對旋轉累積成相對第 0 幀的姿態 | 參考幀會持續多幀，直接每幀累乘會**過度累積**（已修正的 bug） | `R_global = R_base @ R`，換參考幀時凍結 `R_base`                   |
| **室內/戶外（cv2.dnn）**   | Places365 ResNet18 場景分類       | 古典啟發式對夜景/雪地/亮室內失準（3/6→DNN 6/6）              | `cv2.dnn.readNetFromONNX`→softmax→依官方 IO 表彙總；無模型退回 HSV |
| **單圖 yaw（消失點）**     | 結構化場景由水平消失點取 yaw      | 單圖無運動拿不到 yaw，但走廊/建築的平行線會收斂              | `vanishing_point` 求交點，x 偏移→`atan2(Δx,f)`；近平行→N/A         |
| **動態方向**               | 相機運動 + 畫面光流               | 重用既有 `t` 與內點，零額外成本                              | `t` 主導軸→方向；內點中位位移→pan/tilt；徑向發散→zoom              |
| **相對深度**               | 特徵點稀疏三角化                  | 重用 R/t/內點，Pi 可承受                                     | `triangulatePoints(K[I\|0],K[R\|t])`→正 Z 中位數→分級（尺度相對）  |

> **為什麼 `cv2.dnn` 不違反「用 OpenCV」**：cv2.dnn 是 OpenCV 內建模組，CPU 載入 ONNX 推論**不需** PyTorch；室內/戶外變化慢，每 ~30 幀跑一次（Pi ~0.6–0.9s/次），姿態主迴圈不受影響。深入見 [`docs/architecture.md`](docs/architecture.md)。

---

## 6. 驗收（標準答案）

每項能力都有標準答案與通過門檻，完整方法書見 [`docs/validation.md`](docs/validation.md)。

| 能力      | 標準答案                                     | 指標                       | 門檻    | 現況          |
| --------- | -------------------------------------------- | -------------------------- | ------- | ------------- |
| 姿態      | 合成 3D（精確逐幀 GT，`gen_synthetic3d.py`） | 每軸 MAE                   | < 15°   | ✅ 13.3°       |
| 室內/戶外 | 標註檔 / Places365 驗證集                    | 準確率                     | ≥ 80%   | ✅ 100%（6/6） |
| 深度      | NYU/KITTI/DIODE                              | 尺度對齊 AbsRel / Spearman | ρ ≥ 0.6 | 📋 方法已定    |
| 光流      | MPI Sintel / KITTI flow                      | EPE / 方向一致率           | < 3px   | 📋 方法已定    |

```bash
python benchmarks/validate_pose.py    # 姿態 → 每軸 MAE/RMSE + PASS/FAIL
python benchmarks/validate_scene.py   # 室內/戶外 → 準確率 + PASS/FAIL
```

### 單張圖片實測（6 張測試照片，室內/戶外 6/6）

同一支管線處理**單張照片**，疊上偵測到的 CV 證據與場景判斷：

![6 張測試照片標注輸出](docs/scene_demo.png)

*圖說：6 張測試照片的標注輸出。左上 HUD 顯示 `Yaw/Pitch/Roll`（單圖由水平線＋消失點估計，無法觀測時標 N/A）與 `場景：室內/戶外（信心值）`；畫面疊上偵測到的**水平線**（黃）、**特徵點＋梯度方向場**（綠）、**垂直線/消失點**（洋紅），右上為 **XYZ 姿態指示器**。分類結果 test1/2＝室內、test3–6＝戶外，與標準答案 **6/6 相符**（對照古典啟發式僅 3/6）。*

### 圖片標準答案對照：York Urban DB（單圖**相機姿態**）

圖片的核心任務和影片一樣是**相機姿態**——用 OpenCV 找**消失點 / 水平線**解出 yaw/pitch/roll 並畫箭頭；室內/戶外只是順帶輸出的**一個欄位**。對應的圖片型標準答案是 [York Urban Database](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)（102 張曼哈頓場景，附**正交消失點 GT** 與相機內參）。由 GT 消失點推出**真實水平線**，再比對我們估的水平線 / roll：

![YUD 單圖姿態：GT vs 我們](docs/yud_demo.png)

*圖說：8 張 York Urban 影像。🟢 綠線＝由 **GT 消失點**推得的**真實水平線**、🔵 藍線＝**我們**用 Hough 水平線/消失點估的水平線，文字為各自 roll。多數場景兩線貼合。*

| 指標（我們 vs YUD GT，102 張） | 結果                                      |
| ------------------------------ | ----------------------------------------- |
| **roll**（水平線傾斜）         | MAE **5.0°**、median 4.2°、57% 落在 5° 內 |
| **pitch**（水平線高度）        | MAE 6.6°、median 5.8°                     |

```bash
python benchmarks/validate_yud.py --base test_inputs/yud/db/YorkUrbanDB --out docs/yud_demo.png
```

> **室內/戶外**只是圖片輸出的**其中一個欄位**，非主任務；該欄位另在 Places365 驗證集達 **95%（57/60）**（`benchmarks/validate_places365.py`）。

> 此驗收框架**第一次跑就抓到一個既有 bug**：姿態旋轉過度累積（合成 GT 上 MAE 飆到 100°），修正為 keyframe 相對累積後降到 13°。詳見 [validation.md](docs/validation.md)。

### 真實世界對比：TUM RGB-D（我們的輸出 vs 動捕真值）

在 [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) 公開測試序列上，用**同一批測試影格**跑我們的系統，和他們的**動作捕捉真值**逐幀對比。

> **圖例**：🟢 綠字 `TUM TRUTH` = TUM 動捕**真值**（他們的標準答案）　🟠 橘字 `OUR EST.` = **我們**系統的估計。

**① 中速序列 `freiburg1_xyz` — 全程貼合真值：**

![TUM freiburg1_xyz 對比動畫](docs/tum_xyz.gif)

*圖說：逐幀播放 `freiburg1_xyz`。每格綠字 `TUM TRUTH` 是該幀相機的**真實姿態**、藍字 `OUR EST.` 是**我們的估計**，`err` 為當下旋轉誤差。三軸數值幾乎同步變化——中速序列全程貼合。*

**② 最難序列 `freiburg1_desk`（快速＋動態模糊）— 前段準、後段漂移：**

![TUM freiburg1_desk 對比動畫](docs/tum_desk.gif)

*圖說：逐幀播放 `freiburg1_desk`（公認最難）。前段藍≈綠，約 200 幀後藍字逐漸偏離綠字、`err` 攀升，呈現單目視覺里程計的**長期漂移**。*

| 序列             | 速度/難度                  | yaw   | pitch | roll     | 幾何旋轉 MAE                 |
| ---------------- | -------------------------- | ----- | ----- | -------- | ---------------------------- |
| `freiburg1_xyz`  | 中速（translation 為主）   | 8.0°  | 7.1°  | **2.0°** | **11.9°** ✅ 全程貼合         |
| `freiburg1_desk` | 快速＋動態模糊（公認最難） | 29.7° | 21.5° | 46.0°    | 55.8°（前 7 秒準、後段漂移） |

**三軸對照曲線**（綠實線＝TUM 真值、紅虛線＝我們的估計）：

`freiburg1_xyz`（中速，全程貼合）
![xyz 對照曲線](docs/tum_xyz_compare.png)

*圖說：橫軸＝配對幀序、縱軸＝角度（度），由上而下為 yaw／pitch／roll 三軸。綠實線（TUM 真值）與紅虛線（我們）幾乎重疊，尤其 roll 僅 2° 誤差——代表全程都跟得住真值。*

`freiburg1_desk`（最難，前段準、後段漂移）
![desk 對照曲線](docs/tum_desk_compare.png)

*圖說：同樣三軸對幀序。前 ~200 幀紅綠貼合，之後紅線（我們）發散偏離綠線（真值），對應幾何旋轉誤差由數度增至約 110°——這就是快速＋動態模糊序列上的漂移。*

**`freiburg1_desk` 逐幀並排**（真實測試影像 + 🟢TUM 真值 vs 🟠我們的估計）：
![desk 逐幀並排](docs/tum_compare_frames.png)

*圖說：四個代表幀（frame 40／150／200／400）的單格快照。黑底上排綠字＝TUM 真值、下排藍字＝我們估計的 Y/P/R，可逐格核對數字差距；可見前段接近、frame 400 已明顯偏離。*

> 另有 `freiburg1_xyz` 逐幀並排：[`docs/tum_xyz_frames.png`](docs/tum_xyz_frames.png)

```bash
# 下載並解壓 TUM 序列到 test_inputs/tum/ 後：
python benchmarks/validate_tum.py    --seq test_inputs/tum/rgbd_dataset_freiburg1_xyz --plot docs/tum_xyz_compare.png
python benchmarks/tum_make_gif.py    --seq test_inputs/tum/rgbd_dataset_freiburg1_xyz --out docs/tum_xyz.gif
python benchmarks/tum_frame_compare.py --seq test_inputs/tum/rgbd_dataset_freiburg1_xyz --out docs/tum_xyz_frames.png
```

**結論**：中等速度的真實序列我們**全程貼合真值**（roll 僅 2° 誤差）；只有在最難的快速＋模糊序列才會長期漂移——這是單目 VO（無 SLAM 後端）的本質界線。

### Raspberry Pi 4B 實測成果（GIF 對照）

以下為在 Raspberry Pi 4B 平台上執行的實測成果。表格左側為原始輸入的 GIF 動畫，右側為帶有相機姿態估計與場景感知（包含 `_pose` 後綴）的輸出結果。

|                      原始輸入檔案                       |                      姿態與場景感知輸出成果                       |
| :-----------------------------------------------------: | :---------------------------------------------------------------: |
| ![synthetic3d_pose_test](gif/synthetic3d_pose_test.gif) | ![synthetic3d_pose_test_pose](gif/synthetic3d_pose_test_pose.gif) |
|   ![synthetic_pose_test](gif/synthetic_pose_test.gif)   |   ![synthetic_pose_test_pose](gif/synthetic_pose_test_pose.gif)   |
|                ![test_1](gif/test_1.gif)                |                ![test_1_pose](gif/test_1_pose.gif)                |
|                ![test_2](gif/test_2.gif)                |                ![test_2_pose](gif/test_2_pose.gif)                |

---

## 7. CLI 參數
```
python src/main.py <source> [options]

輸出:   --output/-o PATH   --no-show   --no-video
解析度: --imgsz INT（預設 640，0=原始）   --pi-sim（強制 640×480）
演算法: --nfeatures 500   --ratio-thresh 0.75   --ransac-thresh 1.0
        --keyframe-ratio 0.5   --keyframe-inliers 60
校正:   --calib camera_matrix.yaml（由 calibrate.py 產生）
場景:   --scene-model models/places365_resnet18.onnx
        --scene-io    models/io_places365.txt（皆缺→退回 HSV 啟發式）
```

相機校正（選用，精度需 ±1–2° 時）：
```bash
python calibrate.py --images calib_images/ --output my_camera.yaml
python src/main.py video.mp4 --calib my_camera.yaml
```
不校正時以 `fx=fy=W` 近似，旋轉趨勢誤差約 5–10°。

---

## 8. Pi 4B 效能與參數調整

| `--imgsz` | 解析度    | 預估 FPS |
| --------- | --------- | -------- |
| 320       | 320×240   | 25–35    |
| 480       | 640×480   | 15–25    |
| 720       | 1280×720  | 8–12     |
| 1080      | 1920×1080 | 4–6      |

> 以 `--nfeatures 500 --no-video` 為基準；室內/戶外每 ~30 幀跑一次，對 FPS 影響可忽略。

| 問題                    | 調整方式                                                     |
| ----------------------- | ------------------------------------------------------------ |
| 特徵/內點太少、姿態跳動 | 提高 `--nfeatures`、放寬 `--ratio-thresh`                    |
| 姿態漂移過大            | 調低 `--keyframe-ratio`/`--keyframe-inliers`（更常換參考幀） |
| Pi 太慢                 | 降 `--imgsz`（如 320）、`--no-video`                         |
| 室內/戶外不準           | 確認已產生 Places365 模型（否則退回較弱的啟發式）            |
| 單圖 yaw 常 N/A         | 正常——僅結構化（建築/走廊/街道）場景才可觀測                 |
