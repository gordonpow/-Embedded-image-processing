import cv2
import math
import numpy as np


class PoseEstimator:
    """
    Monocular relative pose estimator using ORB + 5-point Essential Matrix.

    Accumulates cumulative rotation R_global from frame 0.
    Outputs yaw / pitch / roll (degrees) using YXZ Euler decomposition:
        R = Ry(yaw) * Rx(pitch) * Rz(roll)
    Camera frame: X-right, Y-down, Z-forward (OpenCV standard).
      Yaw   = pan left/right  (rotation around Y)
      Pitch = tilt up/down    (rotation around X)
      Roll  = lean clockwise  (rotation around Z)

    Scale ambiguity: translation from recoverPose is unit-length and NOT
    accumulated — only rotation is integrated.
    """

    def __init__(
        self,
        nfeatures: int = 500,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 1.0,
        keyframe_min_ratio: float = 0.5,
        keyframe_min_inliers: int = 60,
        max_angle_per_frame: float = 25.0,
        smooth_alpha: float = 0.4,
    ) -> None:
        # nlevels=4 reduces descriptor cost on Pi 4B (~30% faster than default 8)
        self._orb = cv2.ORB_create(nfeatures=nfeatures, nlevels=4)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._ratio_thresh = ratio_thresh
        self._ransac_thresh = ransac_thresh
        self._kf_min_ratio = keyframe_min_ratio
        self._kf_min_inliers = keyframe_min_inliers
        # Reject single-frame rotation larger than this (noisy E-matrix filter)
        self._max_angle_rad = math.radians(max_angle_per_frame)
        # EMA smoothing for output angles
        self._alpha = smooth_alpha

        self._R_global: np.ndarray = np.eye(3, dtype=np.float64)
        self._ref_kp = None
        self._ref_des = None
        self._K: np.ndarray | None = None

        # EMA state for output angles
        self._yaw_ema   = 0.0
        self._pitch_ema = 0.0
        self._roll_ema  = 0.0
        self._ema_init  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_K(self, K: np.ndarray) -> None:
        self._K = K.astype(np.float64)

    def reset(self) -> None:
        self._R_global = np.eye(3, dtype=np.float64)
        self._ref_kp = None
        self._ref_des = None
        self._yaw_ema = self._pitch_ema = self._roll_ema = 0.0
        self._ema_init = False

    @property
    def R_global(self) -> np.ndarray:
        return self._R_global

    def process(
        self, gray: np.ndarray
    ) -> tuple[float, float, float, int, int]:
        """
        Process one grayscale frame.

        Returns
        -------
        (yaw_deg, pitch_deg, roll_deg, inlier_count, nfeatures)
        inlier_count == -1 when this frame is used as a new reference.
        """
        kp, des = self._orb.detectAndCompute(gray, None)
        npts = len(kp)

        # Not enough descriptors — set as reference and return current pose
        if self._ref_kp is None or des is None or len(des) < 8:
            self._ref_kp, self._ref_des = kp, des
            yaw, pitch, roll = self._smoothed_angles(-1)
            return yaw, pitch, roll, -1, npts

        if self._ref_des is None or len(self._ref_des) < 8:
            self._ref_kp, self._ref_des = kp, des
            yaw, pitch, roll = self._smoothed_angles(-1)
            return yaw, pitch, roll, -1, npts

        # Lowe ratio test
        raw_matches = self._matcher.knnMatch(self._ref_des, des, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self._ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < 8:
            self._ref_kp, self._ref_des = kp, des
            yaw, pitch, roll = self._smoothed_angles(-1)
            return yaw, pitch, roll, -1, npts

        pts1 = np.float32([self._ref_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self._ransac_thresh,
        )

        if E is None or mask is None:
            self._ref_kp, self._ref_des = kp, des
            yaw, pitch, roll = self._smoothed_angles(-1)
            return yaw, pitch, roll, -1, npts

        inlier_count = int(mask.sum())
        inlier_ratio = inlier_count / len(good) if good else 0.0

        _, R, _t, _ = cv2.recoverPose(E, pts1, pts2, self._K, mask=mask)

        # --- Rotation magnitude gate ---
        # reject R if it implies an unrealistically large per-frame rotation
        angle_rad = _rot_angle(R)
        if angle_rad <= self._max_angle_rad:
            self._R_global = _orthonormalize(self._R_global @ R)

        # Spawn new keyframe when tracking quality degrades
        if inlier_ratio < self._kf_min_ratio or inlier_count < self._kf_min_inliers:
            self._ref_kp, self._ref_des = kp, des

        yaw, pitch, roll = self._smoothed_angles(inlier_count)
        return yaw, pitch, roll, inlier_count, npts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _smoothed_angles(self, inliers: int) -> tuple[float, float, float]:
        """Return EMA-smoothed yaw/pitch/roll from current R_global."""
        yaw, pitch, roll = _rot_to_ypr(self._R_global)
        if not self._ema_init:
            self._yaw_ema, self._pitch_ema, self._roll_ema = yaw, pitch, roll
            self._ema_init = True
        else:
            a = self._alpha
            self._yaw_ema   = a * yaw   + (1 - a) * self._yaw_ema
            self._pitch_ema = a * pitch + (1 - a) * self._pitch_ema
            self._roll_ema  = a * roll  + (1 - a) * self._roll_ema
        return self._yaw_ema, self._pitch_ema, self._roll_ema


# ------------------------------------------------------------------
# YXZ Euler decomposition — R = Ry(yaw) * Rx(pitch) * Rz(roll)
# ------------------------------------------------------------------

def _rot_to_ypr(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3×3 rotation matrix to (yaw, pitch, roll) in degrees.

    Decomposition: R = Ry(yaw) · Rx(pitch) · Rz(roll)
    Derived analytically from the product expansion:
        R[1,2] = -sin(pitch)
        R[0,2] = sin(yaw)*cos(pitch)   → yaw = atan2(R[0,2], R[2,2])
        R[1,0] = cos(pitch)*sin(roll)  → roll = atan2(R[1,0], R[1,1])
    """
    pitch_rad = math.asin(max(-1.0, min(1.0, float(-R[1, 2]))))
    cp = math.cos(pitch_rad)
    if abs(cp) > 1e-6:
        yaw_rad  = math.atan2(float(R[0, 2]), float(R[2, 2]))
        roll_rad = math.atan2(float(R[1, 0]), float(R[1, 1]))
    else:
        # Gimbal lock (pitch = ±90°)
        yaw_rad  = math.atan2(float(-R[2, 0]), float(R[0, 0]))
        roll_rad = 0.0
    return (
        math.degrees(yaw_rad),
        math.degrees(pitch_rad),
        math.degrees(roll_rad),
    )


def _rot_angle(R: np.ndarray) -> float:
    """Rotation angle (radians) of a rotation matrix via trace formula."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_theta)))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project R onto SO(3) via SVD to prevent numerical drift."""
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean
