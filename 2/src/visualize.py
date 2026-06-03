import cv2
import numpy as np

from draw_cv import render_text_lines, tr_value


def draw_pose_overlay(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    fps: float,
    inliers: int,
    npts: int,
    w: int,
    h: int,
    scene: str | None = None,
    scene_conf: float | None = None,
    cam_motion: str | None = None,
    flow: str | None = None,
    zoom_in: bool = False,
    depth_level: str | None = None,
    yaw_na: bool = False,
    show_fps: bool = True,
    show_stats: bool = True,
) -> None:
    """
    Draw yaw/pitch/roll (+ optional FPS / inlier stats) in the top-left corner.

    When scene/motion/depth values are supplied, extra lines are added:
        SCN <indoor/outdoor> <conf>
        MOV <cam_dir> / <flow_dir>[ ZOOM]
        DEP <near/mid/far>

    Single-image mode uses ``yaw_na=True`` (yaw unobservable; pitch/roll come
    from the horizon estimate) and ``show_fps=show_stats=False`` for a slim
    overlay.
    """
    cyan = (255, 220, 120)
    green = (0, 255, 0)
    grey = (200, 200, 200)

    # Each entry: (中文, English-fallback, colour)
    if yaw_na:
        lines = [("偏航  無", "YAW  N/A", cyan)]
    else:
        lines = [(f"偏航 {yaw:+6.1f}°", f"YAW {yaw:+6.1f}", green)]
    lines += [
        (f"俯仰 {pitch:+6.1f}°", f"PIT {pitch:+6.1f}", green),
        (f"側傾 {roll:+6.1f}°", f"ROL {roll:+6.1f}", green),
    ]
    if show_fps:
        lines.append((f"幀率 {fps:.1f}  {w}x{h}", f"FPS {fps:.1f} {w}x{h}", grey))
    if show_stats:
        lines.append((f"特徵 {npts}  內點 {inliers}", f"PTS {npts} INL {inliers}", grey))
    if scene is not None:
        conf = f" {scene_conf:.2f}" if scene_conf is not None else ""
        lines.append((f"場景 {tr_value(scene)}{conf}", f"SCN {scene}{conf}", cyan))
    if cam_motion is not None or flow is not None:
        zoom_zh = " 推近" if zoom_in else ""
        zoom_en = " ZOOM" if zoom_in else ""
        lines.append((
            f"運動 {tr_value(cam_motion or 'N/A')} / {tr_value(flow or 'N/A')}{zoom_zh}",
            f"MOV {cam_motion or '-'} / {flow or '-'}{zoom_en}", cyan))
    if depth_level is not None:
        lines.append((f"深度 {tr_value(depth_level)}", f"DEP {depth_level}", cyan))

    render_text_lines(frame, lines, x=10, y0=20, dy=30, size=22)


def draw_orientation_indicator(
    frame: np.ndarray,
    R_global: np.ndarray,
    scale: int = 55,
) -> None:
    """
    Draw an XYZ orientation indicator in the top-right corner.

    World-frame axes are rotated by R_global and projected orthographically
    (depth dimension dropped).  Red=X, Green=Y, Blue=Z.
    """
    h, w = frame.shape[:2]
    cx, cy = w - 80, 80

    # Background disc
    cv2.circle(frame, (cx, cy), scale + 6, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), scale + 6, (70, 70, 70),  1, cv2.LINE_AA)

    # Unit axes × scale, rotated into camera frame, then drop Z
    axes_3d = np.eye(3) * scale               # rows: X, Y, Z unit vectors
    rotated  = (R_global @ axes_3d.T).T       # (3, 3) — each row is a rotated axis

    colors = [(50, 50, 255), (50, 255, 50), (255, 130, 50)]  # BGR X/Y/Z
    labels = ["X", "Y", "Z"]

    for i in range(3):
        ex = int(cx + rotated[i, 0])
        ey = int(cy + rotated[i, 1])
        cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                        colors[i], 2, cv2.LINE_AA, tipLength=0.28)
        cv2.putText(frame, labels[i], (ex + 3, ey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colors[i], 1, cv2.LINE_AA)
