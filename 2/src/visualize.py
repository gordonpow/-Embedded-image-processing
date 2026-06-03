import cv2
import numpy as np


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
    yaw_line = ("YAW    N/A", cyan) if yaw_na else (f"YAW  {yaw:+7.1f} deg", (0, 255, 0))
    lines = [
        yaw_line,
        (f"PIT  {pitch:+7.1f} deg", (0, 255, 0)),
        (f"ROL  {roll:+7.1f} deg", (0, 255, 0)),
    ]
    if show_fps:
        lines.append((f"FPS  {fps:5.1f}  {w}x{h}", (200, 200, 200)))
    if show_stats:
        lines.append((f"PTS  {npts:4d}  INL {inliers:4d}", (200, 200, 200)))

    if scene is not None:
        conf_txt = f" {scene_conf:.2f}" if scene_conf is not None else ""
        lines.append((f"SCN  {scene}{conf_txt}", cyan))
    if cam_motion is not None or flow is not None:
        zoom_txt = " ZOOM" if zoom_in else ""
        lines.append((f"MOV  {cam_motion or '-'} / {flow or '-'}{zoom_txt}", cyan))
    if depth_level is not None:
        lines.append((f"DEP  {depth_level}", cyan))

    x, y0, dy = 10, 28, 24
    for i, (text, color) in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 3, cv2.LINE_AA)   # black stroke
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 1, cv2.LINE_AA)        # colored fill


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
