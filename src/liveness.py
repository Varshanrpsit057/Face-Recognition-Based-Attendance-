"""
Liveness / anti-spoofing gate — MiniFASNet-V2.

Applied only to the live camera feed. Never call this on dataset
enrollment photos: those are themselves static images, so a spoof
classifier would (correctly, but unhelpfully) flag them as fake.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from config import cfg
from src.model_manager import get_model_manager
from src.logger import get_logger

logger = get_logger(__name__)

# Class convention follows the reference MiniFASNet-V2 (Silent-Face-
# Anti-Spoofing) checkpoints this ONNX export derives from: index 1 is
# "real face"; 0 and 2 are distinct spoof categories (print / replay).
LIVE_CLASS_INDEX = 1


class LivenessDetector:
    """Wraps models/minifasnet_v2.onnx for single-face liveness scoring."""

    def __init__(self, config=None):
        self.config = config if config else cfg.quality
        self.session = get_model_manager().get_session('minifasnet_v2')
        self._input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # shape is [batch, 3, H, W]; fall back to 80x80 if dynamic/unreadable
        try:
            self.input_hw = (int(input_shape[2]), int(input_shape[3]))
        except Exception:
            self.input_hw = (80, 80)
        self.crop_scale = getattr(self.config, 'liveness_crop_scale', 2.7)
        self.threshold = getattr(self.config, 'liveness_threshold', 0.5)

    def _scaled_crop(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop a `crop_scale`x padded, aspect-preserving region around bbox,
        clamped to image bounds — matches the standard MiniFASNet crop
        recipe (pad around the face, don't just resize the tight box)."""
        src_h, src_w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x, y = float(x1), float(y1)
        w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))

        scale = min((src_h - 1) / h, min((src_w - 1) / w, self.crop_scale))
        new_w, new_h = w * scale, h * scale
        cx, cy = x + w / 2.0, y + h / 2.0
        lx, ly = cx - new_w / 2.0, cy - new_h / 2.0
        rx, ry = cx + new_w / 2.0, cy + new_h / 2.0

        if lx < 0:
            rx -= lx
            lx = 0
        if ly < 0:
            ry -= ly
            ly = 0
        if rx > src_w - 1:
            lx -= (rx - src_w + 1)
            rx = src_w - 1
        if ry > src_h - 1:
            ly -= (ry - src_h + 1)
            ry = src_h - 1

        lx, ly = max(0, int(lx)), max(0, int(ly))
        rx, ry = min(src_w, int(rx)), min(src_h, int(ry))
        return image[ly:ry, lx:rx]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / max(1e-9, e.sum())

    def predict(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """Returns (is_live, live_confidence in 0..1)."""
        crop = self._scaled_crop(image, bbox)
        if crop.size == 0:
            return False, 0.0

        resized = cv2.resize(crop, self.input_hw)
        # Reference implementation reads frames via cv2 (BGR) and only
        # scales to [0,1] — no channel swap, no mean/std normalization.
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run(None, {self._input_name: tensor})
        logits = outputs[0].flatten()
        probs = self._softmax(logits)
        live_score = float(probs[LIVE_CLASS_INDEX]) if len(probs) > LIVE_CLASS_INDEX else float(probs.max())
        return live_score >= self.threshold, live_score


_liveness_detector: Optional["LivenessDetector"] = None


def get_liveness_detector() -> Optional["LivenessDetector"]:
    """Lazily-initialized singleton; returns None if the model can't load
    so callers can gracefully skip the liveness gate rather than crash."""
    global _liveness_detector
    if not getattr(cfg.quality, 'liveness_enabled', True):
        return None
    if _liveness_detector is None:
        try:
            _liveness_detector = LivenessDetector()
        except Exception as e:
            logger.warning(f"MiniFASNet liveness model unavailable, liveness gate disabled: {e}")
            return None
    return _liveness_detector
