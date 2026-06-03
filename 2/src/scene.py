"""
Indoor / outdoor scene classification — classical CV heuristics (no ML).

Designed to run per-frame on Raspberry Pi 4B at negligible cost.
The decision combines three cheap cues computed in HSV space:

  * sky_frac    — fraction of the top third that looks like sky
                  (bright + either low-saturation/white OR blue hue)
  * veg_frac    — fraction of the whole frame that looks like green vegetation
  * bright_norm — overall brightness, normalised to [0, 1]

A weighted score is compared against a threshold; outdoor scenes score high
(sky/foliage/bright), indoor scenes score low.  Accuracy is "broadly correct"
by design — boundary scenes (e.g. a bright window indoors) may be misjudged.
"""
import cv2
import numpy as np

# Decision weights and threshold (tuned for "broadly correct" behaviour).
_W_SKY = 0.40
_W_VEG = 0.35
_W_BRIGHT = 0.25
_THRESHOLD = 0.35


def classify_indoor_outdoor(frame_bgr: np.ndarray) -> tuple[str, float]:
    """
    Classify a single BGR frame as ``"indoor"`` or ``"outdoor"``.

    Returns
    -------
    (label, confidence)
        label      : "indoor" | "outdoor"
        confidence : float in [0.5, 1.0] — distance of the score from the
                     decision threshold (0.5 == right on the fence).

    Raises
    ------
    ValueError / TypeError
        If ``frame_bgr`` is None or not a 3-channel image.
    """
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
