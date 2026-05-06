import cv2
import numpy as np

_COLORS = {
    'fire':  (0,  69, 255),   # BGR: orange-red
    'smoke': (160, 160, 160), # BGR: gray
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# HSV ranges for contour segmentation
_HSV_RANGES = {
    'fire': [
        ((0,   100, 100), (35,  255, 255)),   # yellow → orange
        ((160, 100, 100), (180, 255, 255)),   # wrapping red
    ],
    'smoke': [
        ((0, 0, 80), (180, 60, 255)),         # low-sat gray/white
    ],
}
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _label(frame, x1, y1, name, conf, color):
    text = f"{name} {conf:.2f}"
    (tw, th), bl = cv2.getTextSize(text, _FONT, 0.55, 1)
    top = max(y1 - th - bl - 2, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw, top + th + bl + 2), color, -1)
    cv2.putText(frame, text, (x1, top + th + 1), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bboxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame in-place."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name  = det['class_name']
        conf  = det['confidence']
        color = _COLORS.get(name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, x1, y1, name, conf, color)
    return frame


def draw_contours(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw HSV-segmented contours that follow the actual fire/smoke shape.
    Falls back to a bounding box when no mask pixels survive thresholding.
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        name  = det['class_name']
        conf  = det['confidence']
        color = _COLORS.get(name, (0, 255, 0))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _label(frame, x1, y1, name, conf, color)
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Build combined mask from all HSV ranges for this class
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _HSV_RANGES.get(name, []):
            mask |= cv2.inRange(hsv, lo, hi)

        # Morphological cleanup: close holes, remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No color match — fall back to plain bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            # Shift ROI-local contours back to full-frame coordinates
            shifted = [c + np.array([[[x1, y1]]]) for c in contours]
            cv2.drawContours(frame, shifted, -1, color, 2)

        _label(frame, x1, y1, name, conf, color)

    return frame


def draw_fps_overlay(frame: np.ndarray, fps: float, resolution: tuple) -> np.ndarray:
    """Draw FPS and resolution info on top-left corner in-place."""
    w, h = resolution
    text = f"FPS:{fps:5.1f}  {w}x{h}"
    cv2.putText(frame, text, (8, 22), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, 22), _FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
