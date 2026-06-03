"""
Indoor / outdoor scene classification.

Two interchangeable backends:

  1. DNN (preferred) — a Places365 ResNet18 run through OpenCV's ``cv2.dnn``
     module (see ``SceneClassifier``).  Still "OpenCV only" — no PyTorch at
     inference — yet it understands night / snow / bright-interior scenes that
     low-level cues cannot.  Run once every N frames on the Pi 4B.

  2. Heuristic (fallback) — classical CV cues in HSV space, used automatically
     when no model file is present (zero dependencies):
        * sky_frac    — top-third pixels that look like sky (bright + white/blue)
        * veg_frac    — green-vegetation fraction over the frame
        * bright_norm — overall brightness
     A weighted score vs a threshold decides indoor/outdoor.  Only "broadly
     correct": bright interiors and night/snow outdoor scenes are misjudged —
     which is exactly why the DNN backend exists.
"""
import os

import cv2
import numpy as np

# ImageNet normalisation used when the Places365 weights were trained.
_DNN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DNN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DNN_SIZE = 224

# Decision weights and threshold (tuned for "broadly correct" behaviour).
_W_SKY = 0.40
_W_VEG = 0.35
_W_BRIGHT = 0.25
_THRESHOLD = 0.35


def classify_indoor_outdoor(frame_bgr: np.ndarray, classifier=None) -> tuple[str, float]:
    """
    Classify a single BGR frame as ``"indoor"`` or ``"outdoor"``.

    Parameters
    ----------
    classifier : SceneClassifier | None
        If given (a loaded DNN backend), it is used; otherwise the classical
        heuristic below runs.

    Returns
    -------
    (label, confidence)
        label      : "indoor" | "outdoor"
        confidence : float in [0.0, 1.0]

    Raises
    ------
    ValueError / TypeError
        If ``frame_bgr`` is None or not a 3-channel image.
    """
    if classifier is not None:
        return classifier.predict(frame_bgr)

    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise TypeError("frame_bgr must be an HxWx3 BGR image")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h, w = frame_bgr.shape[:2]

    # --- Sky cue: bright pixels that are either whitish or blue, top third ---
    top = max(1, h // 3)
    sv, ss, sh = v_[:top], s_[:top], h_[:top]
    sky_mask = (sv > 150) & ((ss < 60) | ((sh >= 90) & (sh <= 140)))
    sky_frac = float(sky_mask.mean())

    # --- Vegetation cue: saturated green over the whole frame ---
    veg_mask = (h_ >= 35) & (h_ <= 85) & (s_ > 40) & (v_ > 40)
    veg_frac = float(veg_mask.mean())

    # --- Brightness cue ---
    bright_norm = float(np.clip((v_.mean() - 50.0) / 150.0, 0.0, 1.0))

    score = _W_SKY * sky_frac + _W_VEG * veg_frac + _W_BRIGHT * bright_norm

    if score >= _THRESHOLD:
        label = "outdoor"
        conf = 0.5 + 0.5 * min(1.0, (score - _THRESHOLD) / (1.0 - _THRESHOLD))
    else:
        label = "indoor"
        conf = 0.5 + 0.5 * min(1.0, (_THRESHOLD - score) / _THRESHOLD)

    return label, float(conf)


# ------------------------------------------------------------------
# DNN backend — Places365 ResNet18 via cv2.dnn
# ------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def aggregate_io(probs: np.ndarray, io_flags: np.ndarray) -> tuple[str, float]:
    """
    Collapse 365 per-category probabilities into indoor vs outdoor.

    Places365's ``IO_places365.txt`` tags every category 1=indoor / 2=outdoor.
    We sum the probability mass of each group; the larger group wins and its
    mass is the confidence.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    io_flags = np.asarray(io_flags).reshape(-1)
    indoor_p = float(probs[io_flags == 1].sum())
    outdoor_p = float(probs[io_flags == 2].sum())
    total = indoor_p + outdoor_p
    if total <= 0:
        return "indoor", 0.0
    if outdoor_p >= indoor_p:
        return "outdoor", outdoor_p / total
    return "indoor", indoor_p / total


class SceneClassifier:
    """
    Places365 ResNet18 indoor/outdoor backend, loaded through cv2.dnn.

    No PyTorch at runtime — only ``cv2.dnn`` — so it stays light and runs on a
    Raspberry Pi 4B (one forward pass ~0.6–0.9 s; call it every N frames).
    """

    def __init__(self, model_path: str, io_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        if not os.path.exists(io_path):
            raise FileNotFoundError(io_path)
        self._net = cv2.dnn.readNetFromONNX(model_path)
        with open(io_path, encoding="utf-8") as f:
            self._io = np.array([int(line.strip()) for line in f if line.strip()])

    @classmethod
    def try_load(cls, model_path: str, io_path: str):
        """Return a classifier, or None if the model files are unavailable."""
        try:
            return cls(model_path, io_path)
        except Exception as exc:   # missing files / unreadable model
            print(f"[scene] DNN backend unavailable ({exc}); using heuristic fallback")
            return None

    def predict(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (_DNN_SIZE, _DNN_SIZE)),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _DNN_MEAN) / _DNN_STD
        blob = rgb.transpose(2, 0, 1)[None]      # 1×3×224×224
        self._net.setInput(blob)
        logits = self._net.forward().reshape(-1)
        return aggregate_io(softmax(logits), self._io)
