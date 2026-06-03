"""
On-frame visualisation of the OpenCV evidence + a Traditional-Chinese HUD.

Why a separate module:
  * Chinese text can't be drawn with cv2.putText (Hershey fonts are ASCII only),
    so we render the HUD with Pillow + a CJK font, falling back to English via
    cv2.putText when no CJK font is found (e.g. a bare Raspberry Pi).
  * The画面 should *show* what the CV extracted — detected horizon lines, ORB
    features + gradient arrows, vertical lines / vanishing point, and (for video)
    the real optical-flow arrows — not just a summary gizmo in the corner.
"""
import math
import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ---- Traditional-Chinese localisation -------------------------------------

_LABEL_ZH = {
    "YAW": "偏航", "PIT": "俯仰", "ROL": "側傾", "FPS": "幀率",
    "PTS": "特徵", "INL": "內點", "SCN": "場景", "MOV": "運動", "DEP": "深度",
}
_VALUE_ZH = {
    "indoor": "室內", "outdoor": "戶外",
    "FWD": "前進", "BACK": "後退", "LEFT": "左移", "RIGHT": "右移",
    "UP": "上移", "DOWN": "下移", "STILL": "靜止",
    "PAN-L": "左平移", "PAN-R": "右平移", "TILT-U": "上仰", "TILT-D": "下俯",
    "NEAR": "近", "MID": "中", "FAR": "遠",
    "N/A": "無", "ZOOM": "推近",
}


def tr_label(code: str) -> str:
    return _LABEL_ZH.get(code, code)


def tr_value(code: str) -> str:
    return _VALUE_ZH.get(code, code)


# ---- CJK font (lazy) -------------------------------------------------------

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/mingliu.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_font_cache: dict = {}


def _cjk_font(size: int):
    if not _HAS_PIL:
        return None
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _font_cache[size] = font
    return font


def render_text_lines(frame, lines, x=10, y0=24, dy=30, size=22) -> None:
    """
    Draw HUD ``lines`` = list of ``(zh_text, en_text, (B,G,R))``.

    Uses the CJK font (Chinese) when available, else falls back to ASCII English
    via cv2.putText so the tool still runs on a font-less Raspberry Pi.
    """
    font = _cjk_font(size)
    if font is None:
        for i, (_zh, en, color) in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(frame, en, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, en, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)
        return

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for i, (zh, _en, color) in enumerate(lines):
        y = y0 + i * dy
        draw.text((x, y), zh, font=font, fill=(color[2], color[1], color[0]),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---- CV-evidence overlays --------------------------------------------------

def draw_horizon(frame, det, show_segments: int = 3) -> None:
    """
    Draw the estimated horizon precisely.

    Only the single aggregate horizon line (the actual "answer") is drawn boldly;
    a few of the longest contributing segments are shown faintly so the result
    stays readable instead of flooding the frame with every Hough line.
    """
    segs = det.get("segments", [])
    if not segs:
        return
    longest = sorted(segs, key=lambda s: (s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2,
                     reverse=True)[:show_segments]
    for x1, y1, x2, y2 in longest:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    # Aggregate horizon: a line through (cx, horizon_y) at the estimated roll.
    cx, _cy = det["center"]
    hy = det["horizon_y"]
    w = frame.shape[1]
    slope = math.tan(math.radians(det["roll"]))
    x_left, x_right = 0, w - 1
    y_left = int(hy + (x_left - cx) * slope)
    y_right = int(hy + (x_right - cx) * slope)
    cv2.line(frame, (x_left, y_left), (x_right, y_right), (0, 200, 255), 2, cv2.LINE_AA)


def draw_feature_gradient(frame, gray, step: int = 72, max_pts: int = 60) -> None:
    """
    ORB keypoints (dots) + a *sparse* Sobel gradient-direction arrow field.

    A coarse grid keeps it a readable vector field rather than a thicket of lines.
    """
    orb = cv2.ORB_create(nfeatures=max_pts)
    kps = orb.detect(gray, None)
    for kp in kps[:max_pts]:
        cv2.circle(frame, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 165, 255), -1, cv2.LINE_AA)

    # Smooth first so the sampled gradient reflects local structure, not noise.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]
    half = step // 2
    for y in range(half, h, step):
        for x in range(half, w, step):
            vx, vy = float(gx[y, x]), float(gy[y, x])
            mag = math.hypot(vx, vy)
            if mag < 60:                      # only strong, confident edges
                continue
            ux, uy = vx / mag, vy / mag
            ex, ey = int(x + ux * half * 0.7), int(y + uy * half * 0.7)
            cv2.arrowedLine(frame, (x, y), (ex, ey), (80, 255, 80), 1,
                            cv2.LINE_AA, tipLength=0.35)


def draw_vertical_vp(frame, segments, vp) -> None:
    """Vertical structural lines (magenta) + short arrows toward the vanishing point."""
    for x1, y1, x2, y2 in segments:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
    if vp is None:
        return
    vx, vy = int(vp[0]), int(vp[1])
    h, w = frame.shape[:2]
    if 0 <= vx < w and 0 <= vy < h:
        cv2.circle(frame, (vx, vy), 5, (255, 0, 255), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in segments[:6]:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        dx, dy = vp[0] - mx, vp[1] - my
        n = math.hypot(dx, dy)
        if n < 1e-3:
            continue
        ex, ey = int(mx + dx / n * 30), int(my + dy / n * 30)
        cv2.arrowedLine(frame, (mx, my), (ex, ey), (255, 120, 255), 1,
                        cv2.LINE_AA, tipLength=0.4)


def draw_flow_arrows(frame, pts1, pts2, max_arrows: int = 60) -> int:
    """
    Draw real optical-flow arrows from matched inliers (pts1 -> pts2).

    Returns the number of arrows actually drawn (subsampled to ``max_arrows``).
    """
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    n = min(len(p1), len(p2))
    if n == 0:
        return 0
    idx = np.arange(n)
    if n > max_arrows:
        idx = np.linspace(0, n - 1, max_arrows).astype(int)
    for i in idx:
        a = (int(p1[i, 0]), int(p1[i, 1]))
        b = (int(p2[i, 0]), int(p2[i, 1]))
        cv2.arrowedLine(frame, a, b, (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(frame, a, 2, (0, 165, 255), -1, cv2.LINE_AA)
    return int(len(idx))
