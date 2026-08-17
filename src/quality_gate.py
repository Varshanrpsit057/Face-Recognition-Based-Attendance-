import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from config import cfg
from src.model_manager import get_model_manager
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class QualityResult:
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    face_width: int = 0
    face_height: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    eye_aspect_ratio: float = 0.0
    quality_score: float = 0.0
    ml_quality_score: Optional[float] = None  # CR-FIQA, 0..1, None if not computed

class QualityGate:
    def __init__(self, config=None):
        self.config = config if config else cfg.quality

    def assess(self, face_image: np.ndarray, bbox: List[float], landmarks: Optional[np.ndarray]) -> QualityResult:
        """Assess the quality of the given face region."""
        reasons = []
        
        x1, y1, x2, y2 = map(int, bbox)
        face_width = max(0, x2 - x1)
        face_height = max(0, y2 - y1)
        
        if face_width < self.config.min_face_width or face_height < self.config.min_face_height:
            reasons.append(f"Face too small ({face_width}x{face_height})")

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        
        # Blur check
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.config.min_laplacian_variance:
            reasons.append(f"Image too blurry ({blur_score:.2f})")

        # Brightness check
        brightness = np.mean(gray)
        if brightness < self.config.min_brightness or brightness > self.config.max_brightness:
            reasons.append(f"Bad brightness ({brightness:.2f})")

        # Pose & EAR check
        yaw, pitch, roll = 0.0, 0.0, 0.0
        ear = 1.0
        if landmarks is not None and len(landmarks) >= 5:
            le, re, nose = landmarks[0], landmarks[1], landmarks[2]
            eye_center = (np.array(le) + np.array(re)) / 2.0
            eye_dist = max(1.0, float(np.linalg.norm(np.array(re) - np.array(le))))

            yaw = float(np.degrees(np.arctan2(eye_center[0] - nose[0], eye_dist)))
            pitch = float(np.degrees(np.arctan2((eye_center[1] - nose[1]) + 0.45 * eye_dist, eye_dist)))
            roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

            if abs(yaw) > self.config.max_yaw:
                reasons.append(f"Yaw too large ({yaw:.1f}°)")
            if abs(pitch) > self.config.max_pitch:
                reasons.append(f"Pitch too large ({pitch:.1f}°)")
            if abs(roll) > self.config.max_roll:
                reasons.append(f"Roll too large ({roll:.1f}°)")

            ear = self._check_eye_aspect_ratio(landmarks)
            if ear < self.config.min_eye_aspect_ratio:
                reasons.append("Eyes closed (EAR too low)")

        # Generate final composite score
        score = self._compute_quality_score(blur_score, brightness, yaw, pitch, roll, face_width, face_height)

        return QualityResult(
            passed=len(reasons) == 0,
            rejection_reasons=reasons,
            face_width=face_width,
            face_height=face_height,
            blur_score=blur_score,
            brightness=brightness,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            eye_aspect_ratio=ear,
            quality_score=score
        )

    def _check_eye_aspect_ratio(self, landmarks: Optional[np.ndarray]) -> float:
        """Calculates a pseudo EAR for 5-point landmarks."""
        if landmarks is None or len(landmarks) < 5:
            return 1.0
            
        le, re, nose = landmarks[0], landmarks[1], landmarks[2]
        
        # Use horizontal distance between eyes and vertical distance from eyes to nose
        eye_dist = np.linalg.norm(np.array(re) - np.array(le))
        eye_center = (np.array(le) + np.array(re)) / 2.0
        nose_dist = np.linalg.norm(eye_center - np.array(nose))
        
        if nose_dist == 0:
            return 1.0
            
        pseudo_ear = eye_dist / (nose_dist + 1e-6)
        return float(pseudo_ear)

    def _compute_quality_score(self, blur: float, brightness: float, yaw: float, pitch: float, roll: float, w: int, h: int) -> float:
        """
        Computes a weighted quality score from 0.0 to 1.0:
        - Blur (30%)
        - Pose (30%)
        - Brightness (20%)
        - Size (20%)
        """
        norm_blur = min(1.0, blur / max(1.0, self.config.min_laplacian_variance * 3))
        
        # Penalize deviating from ideal midrange brightness (128)
        norm_brightness = max(0.0, 1.0 - abs(brightness - 128) / 128.0)
        
        max_angle = max(abs(yaw), abs(pitch), abs(roll))
        norm_pose = max(0.0, 1.0 - (max_angle / max(1.0, self.config.max_yaw)))
        
        ideal_size = max(self.config.min_face_width * 2, 200)
        norm_size = min(1.0, (w + h) / (2.0 * ideal_size))
        
        weighted_score = (
            0.30 * norm_blur +
            0.20 * norm_brightness +
            0.30 * norm_pose +
            0.20 * norm_size
        )
        return float(weighted_score)

    def fuse_ml_score(self, result: QualityResult, ml_score: float) -> QualityResult:
        """Blend a CR-FIQA learned quality score into an existing QualityResult.

        The ML score is informational/additive only — it re-weights
        `quality_score` (used e.g. to pick the best frame of a track
        for final recognition) but never overturns `passed`, since the
        model's raw output has no dataset-verified rejection threshold
        for this deployment. Degrades to the heuristic score untouched
        if ML scoring is disabled or fails upstream.
        """
        w = min(1.0, max(0.0, getattr(self.config, 'ml_quality_weight', 0.4)))
        result.ml_quality_score = float(ml_score)
        result.quality_score = float((1 - w) * result.quality_score + w * ml_score)
        return result


class MLQualityScorer:
    """CR-FIQA-based learned face image quality scorer.

    Reuses the exact 112x112 aligned tensor already produced for
    recognition (see FacePreprocessor.preprocess) so scoring a face
    costs one extra small-ResNet forward pass, not a second alignment.
    """

    def __init__(self, config=None):
        self.config = config if config else cfg.quality
        self.session = get_model_manager().get_session('cr_fiqa')
        self._input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self._quality_idx = 1
        for i, o in enumerate(outputs):
            if o.name.lower() == "quality_score":
                self._quality_idx = i
                break
        self.score_min = getattr(self.config, 'cr_fiqa_score_min', 0.0)
        self.score_max = getattr(self.config, 'cr_fiqa_score_max', 1.0)

    def raw_score(self, aligned_tensor: np.ndarray) -> float:
        """aligned_tensor: the (1,3,112,112) float32 tensor from FacePreprocessor.preprocess()."""
        outputs = self.session.run(None, {self._input_name: aligned_tensor})
        return float(outputs[self._quality_idx].flatten()[0])

    def score(self, aligned_tensor: np.ndarray) -> float:
        """Raw score normalized to 0..1 via the calibrated min/max bounds."""
        raw = self.raw_score(aligned_tensor)
        span = max(1e-6, self.score_max - self.score_min)
        return float(np.clip((raw - self.score_min) / span, 0.0, 1.0))


_ml_quality_scorer: Optional["MLQualityScorer"] = None


def get_ml_quality_scorer() -> Optional["MLQualityScorer"]:
    """Lazily-initialized singleton; returns None if the model can't load
    so callers can gracefully fall back to heuristic-only quality."""
    global _ml_quality_scorer
    if not getattr(cfg.quality, 'use_ml_quality', True):
        return None
    if _ml_quality_scorer is None:
        try:
            _ml_quality_scorer = MLQualityScorer()
        except Exception as e:
            logger.warning(f"CR-FIQA quality model unavailable, using heuristic quality only: {e}")
            return None
    return _ml_quality_scorer
