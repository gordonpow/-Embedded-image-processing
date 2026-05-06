import cv2
import numpy as np

_COLORS = {
    'fire':  (0,  69, 255),   # BGR: orange-red
    'smoke': (160, 160, 160), # BGR: gray
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bboxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame in-place."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name  = det['class_name']
        conf  = det['confidence']
        color = _COLORS.get(name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.55, 1)
        top = max(y1 - th - baseline - 2, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw, top + th + baseline + 2), color, -1)
        cv2.putText(frame, label, (x1, top + th + 1), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_fps_overlay(frame: np.ndarray, fps: float, resolution: tuple) -> np.ndarray:
    """Draw FPS and resolution info on top-left corner in-place."""
    w, h = resolution
    text = f"FPS:{fps:5.1f}  {w}x{h}"
    cv2.putText(frame, text, (8, 22), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, 22), _FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
