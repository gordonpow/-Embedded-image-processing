"""
Generate a synthetic test video with known yaw/pitch/roll profile.
Renders a textured 3D scene with rich features for ORB matching.
"""
import cv2
import numpy as np
import math
import os

W, H = 640, 480
FPS  = 30
DURATION_S = 20   # 10s yaw sweep, 5s pitch, 5s roll

out_path = os.path.join(os.path.dirname(__file__), "synthetic_pose_test.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

# --- Build a rich textured background (wall of dots + grid) ---
BG = np.zeros((H * 4, W * 4, 3), dtype=np.uint8) + 40
rng = np.random.default_rng(42)
# Grid lines
for x in range(0, BG.shape[1], 40):
    cv2.line(BG, (x, 0), (x, BG.shape[0]), (70, 70, 70), 1)
for y in range(0, BG.shape[0], 40):
    cv2.line(BG, (0, y), (BG.shape[1], y), (70, 70, 70), 1)
# Random coloured dots
for _ in range(600):
    cx = rng.integers(0, BG.shape[1])
    cy = rng.integers(0, BG.shape[0])
    r  = rng.integers(5, 18)
    c  = tuple(int(v) for v in rng.integers(80, 255, 3))
    cv2.circle(BG, (cx, cy), r, c, -1)
# ArUco-like squares for extra structure
for i in range(0, BG.shape[1], 120):
    for j in range(0, BG.shape[0], 120):
        cv2.rectangle(BG, (i+10, j+10), (i+50, j+50), (200, 200, 200), -1)
        cv2.rectangle(BG, (i+20, j+20), (i+40, j+40), (30, 30, 30),    -1)


def render_frame(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Warp background to simulate camera rotation."""
    # Approximate homography for small rotations using projective transform
    fx = W  # approximate intrinsics
    yaw_r   = math.radians(yaw_deg)
    pitch_r = math.radians(pitch_deg)
    roll_r  = math.radians(roll_deg)

    # Rotation matrix Ry * Rx * Rz
    cy, sy = math.cos(yaw_r),   math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cr, sr = math.cos(roll_r),  math.sin(roll_r)
    R = np.array([
        [cy*cr + sy*sp*sr, -cy*sr + sy*sp*cr,  sy*cp],
        [cp*sr,             cp*cr,             -sp],
        [-sy*cr + cy*sp*sr,  sy*sr + cy*sp*cr,  cy*cp],
    ])

    # Camera matrix
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    # Compute homography H = K * R * K^-1 (planar scene at Z=1)
    Kinv = np.linalg.inv(K)
    H_mat = K @ R @ Kinv

    # Offset into large background so we have room to pan
    ox, oy = BG.shape[1] // 2 - W // 2, BG.shape[0] // 2 - H // 2
    T_in  = np.array([[1, 0, -W/2], [0, 1, -H/2], [0, 0, 1]], dtype=np.float64)
    T_out = np.array([[1, 0, BG.shape[1]//2], [0, 1, BG.shape[0]//2], [0, 0, 1]], dtype=np.float64)
    H_full = T_out @ H_mat @ T_in

    warped_bg = cv2.warpPerspective(BG, H_full, (BG.shape[1], BG.shape[0]))
    crop = warped_bg[oy:oy+H, ox:ox+W]
    if crop.shape[:2] != (H, W):
        crop = cv2.resize(crop, (W, H))
    return crop


total_frames = DURATION_S * FPS
for i in range(total_frames):
    t = i / FPS
    # Motion profile:
    #   0–8s:  yaw  sweeps 0→+40→-40→0
    #   8–14s: pitch sweeps 0→+25→-25→0
    #   14–20s: roll  sweeps 0→+30→-30→0
    if t < 8:
        yaw   = 40.0 * math.sin(2 * math.pi * t / 8)
        pitch = 0.0
        roll  = 0.0
    elif t < 14:
        yaw   = 0.0
        pitch = 25.0 * math.sin(2 * math.pi * (t - 8) / 6)
        roll  = 0.0
    else:
        yaw   = 0.0
        pitch = 0.0
        roll  = 30.0 * math.sin(2 * math.pi * (t - 14) / 6)

    frame = render_frame(yaw, pitch, roll)

    # Annotate ground truth
    cv2.putText(frame, f"GT YAW  {yaw:+6.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"GT PIT  {pitch:+6.1f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"GT ROL  {roll:+6.1f}", (10, 71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"[gen] saved → {out_path}  ({total_frames} frames, {DURATION_S}s @ {FPS}fps)")
