import cv2
import numpy as np
from typing import Tuple, List, Optional
from config import cfg
from src.logger import get_logger

logger = get_logger(__name__)

class FacePreprocessor:
    REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def __init__(self, config=None):
        self.config = config if config is not None else getattr(cfg, 'preprocessing', None)
        self.target_size = getattr(self.config, 'target_size', (112, 112))
        self.use_clahe = getattr(self.config, 'use_clahe', False)
        self.gamma = getattr(self.config, 'gamma', 1.0)
        self.mean = np.array(getattr(self.config, 'mean', [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(getattr(self.config, 'std', [0.5, 0.5, 0.5]), dtype=np.float32)

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        tform, _ = cv2.estimateAffinePartial2D(landmarks, self.REFERENCE_LANDMARKS, method=cv2.LMEDS)
        if tform is None:
            return self.crop_face(image, (0, 0, image.shape[1], image.shape[0]))
        aligned = cv2.warpAffine(image, tform, self.target_size, borderValue=0.0)
        return aligned

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        margin_w, margin_h = int(w * margin), int(h * margin)
        nx1 = max(0, x1 - margin_w)
        ny1 = max(0, y1 - margin_h)
        nx2 = min(image.shape[1], x2 + margin_w)
        ny2 = min(image.shape[0], y2 + margin_h)
        cropped = image[ny1:ny2, nx1:nx2]
        if cropped.size == 0:
            return cv2.resize(image, self.target_size)
        return cv2.resize(cropped, self.target_size)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32)
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        if mean.max() <= 1.0:
            img_float = img_float / 255.0
        img_norm = (img_float - mean) / std
        return np.transpose(img_norm, (2, 0, 1))

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma == 1.0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def preprocess(self, image: np.ndarray, bbox: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        if landmarks is not None and len(landmarks) == 5:
            face = self.align_face(image, landmarks)
        else:
            face = self.crop_face(image, bbox)

        if self.use_clahe:
            face = self.apply_clahe(face)
        if self.gamma != 1.0:
            face = self.apply_gamma(face, self.gamma)

        tensor = self.normalize(face)
        return np.expand_dims(tensor, axis=0)

    def letterbox(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        dw, dh = (tw - nw) // 2, (th - nh) // 2
        padded = cv2.copyMakeBorder(resized, dh, th - nh - dh, dw, tw - nw - dw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, scale, (dw, dh)

    def batch_preprocess(self, images: List[np.ndarray], bboxes: List[Tuple[int, int, int, int]], landmarks_list: Optional[List[Optional[np.ndarray]]] = None) -> np.ndarray:
        if landmarks_list is None:
            landmarks_list = [None] * len(images)
        tensors = []
        for img, bbox, lm in zip(images, bboxes, landmarks_list):
            tensors.append(self.preprocess(img, bbox, lm)[0])
        return np.array(tensors)
