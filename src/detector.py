import time
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
from config import cfg, DetectorBackend
from src.logger import get_logger
from src.model_manager import get_model_manager
from src.utils import nms_boxes

logger = get_logger(__name__)

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray]
    detection_time: float

class FaceDetectorInterface(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]: ...
    @abstractmethod
    def name(self) -> str: ...

def distance2bbox(points, distance, max_shape=None):
    points = np.atleast_2d(points)
    distance = np.atleast_2d(distance)
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    points = np.atleast_2d(points)
    distance = np.atleast_2d(distance)
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)

# SCRFD model variants available locally. "34g" (SCRFD-34GF) is a
# substantially deeper backbone than the default "10g" (SCRFD-10GF) and
# recovers more true positives on small/far faces — the exact failure
# mode of a single 8MP camera covering a 30x40 ft room — at roughly
# 2-3x the inference cost. This particular 34g export has no keypoint
# (landmark) head, so faces it finds fall back to margin-crop alignment
# instead of the 5-point warp; recall matters more than alignment
# precision at classroom distances, but "10g" remains available where
# landmark-accurate alignment is preferred (e.g. small rooms/webcams).
SCRFD_MODEL_KEYS = {"10g": "scrfd", "34g": "scrfd_34g"}


class SCRFDDetector(FaceDetectorInterface):
    def __init__(self, variant: Optional[str] = None):
        self.variant = variant or getattr(cfg.detection, 'scrfd_variant', '10g')
        model_key = SCRFD_MODEL_KEYS.get(self.variant, 'scrfd')
        self.session = get_model_manager().get_session(model_key)
        self.input_size = getattr(cfg.detection, 'input_size', (640, 640))
        self.conf_threshold = getattr(cfg.detection, 'confidence_threshold', 0.5)
        self.nms_threshold = getattr(cfg.detection, 'nms_threshold', 0.4)
        self.center_cache = {}
        # Detected lazily from the first inference: some exports (e.g.
        # SCRFD-34G here) omit the keypoint head entirely.
        self.has_kps: Optional[bool] = None

    def name(self) -> str: return f"SCRFD-{self.variant.upper()}"

    def detect(self, image: np.ndarray) -> List[Detection]:
        start_time = time.time()
        im_h, im_w, _ = image.shape

        scale = min(self.input_size[0] / im_w, self.input_size[1] / im_h)
        new_w, new_h = int(im_w * scale), int(im_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(det_img, 1.0/128, self.input_size, (127.5, 127.5, 127.5), swapRB=True)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})

        has_kps = len(outputs) in (9, 10, 15)
        self.has_kps = has_kps
        num_strides = 3
        scores_list, bboxes_list, kpss_list = [], [], []

        if has_kps or len(outputs) == 6:
            for idx, stride in enumerate([8, 16, 32]):
                scores = outputs[idx]
                if scores.ndim == 3:
                    scores = scores[0]
                bboxes = outputs[idx + num_strides]
                if bboxes.ndim == 3:
                    bboxes = bboxes[0]
                bboxes = bboxes * stride
                kpss = None
                if has_kps:
                    kpss = outputs[idx + num_strides * 2]
                    if kpss.ndim == 3:
                        kpss = kpss[0]
                    kpss = kpss * stride

                height, width = self.input_size[1] // stride, self.input_size[0] // stride
                key = (height, width, stride)
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape((-1, 2))
                    self.center_cache[key] = anchor_centers

                pos_inds = np.where(scores.flatten() >= self.conf_threshold)[0]
                if len(pos_inds) == 0:
                    continue

                bboxes_dec = distance2bbox(anchor_centers, bboxes)

                pos_scores = scores.flatten()[pos_inds]
                pos_bboxes = bboxes_dec[pos_inds]

                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                if has_kps:
                    kpss_dec = distance2kps(anchor_centers, kpss)
                    kpss_list.append(kpss_dec[pos_inds])

        if not scores_list:
            return []

        scores = np.concatenate(scores_list)
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list) if has_kps else None

        bboxes /= scale
        if has_kps:
            kpss /= scale

        keep = nms_boxes(bboxes, scores, self.nms_threshold)
        detections = []
        for i in keep:
            box = bboxes[i]
            score = scores[i]
            kp = kpss[i] if has_kps else None
            detections.append(Detection(
                bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                confidence=float(score),
                landmarks=kp,
                detection_time=time.time() - start_time
            ))
        return detections

class InsightFaceDetector(FaceDetectorInterface):
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def name(self) -> str: return "InsightFace"
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        start = time.time()
        faces = self.app.get(image)
        results = []
        for face in faces:
            results.append(Detection(
                bbox=(int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3])),
                confidence=float(face.det_score),
                landmarks=face.kps,
                detection_time=time.time() - start
            ))
        return results

class OpenCVDetector(FaceDetectorInterface):
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def name(self) -> str: return "OpenCV"
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        start = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        results = []
        for (x, y, w, h) in faces:
            results.append(Detection(
                bbox=(int(x), int(y), int(x+w), int(y+h)),
                confidence=1.0,
                landmarks=None,
                detection_time=time.time() - start
            ))
        return results

class TiledFaceDetector(FaceDetectorInterface):
    """Wraps a base detector with overlapping-tile detection for large frames.

    A single SCRFD pass letterboxes the whole frame down to `input_size`
    (640x640 by default). For an 8MP frame (e.g. 3840x2160) that shows a
    full 30x40 ft classroom, a student's face at the back of the room can
    shrink to only a few pixels after that downscale — well below what
    any detector or recognizer can work with. Splitting the native-
    resolution frame into overlapping tiles and running the base detector
    on each tile keeps far-row faces at a usable pixel size; results are
    remapped to full-frame coordinates and merged with NMS so a face
    caught in more than one overlapping tile is only reported once.
    """

    def __init__(self, base_detector: FaceDetectorInterface, config: Any = None):
        self.base = base_detector
        cfg_det = config or cfg.detection
        self.use_tiling = getattr(cfg_det, 'use_tiling', True)
        self.tile_grid = getattr(cfg_det, 'tile_grid', (3, 2))
        self.tile_overlap = getattr(cfg_det, 'tile_overlap', 0.20)
        self.tile_min_dimension = getattr(cfg_det, 'tile_min_dimension', 1280)
        self.merge_iou = getattr(cfg_det, 'tile_merge_iou', 0.3)
        self.max_faces = getattr(cfg_det, 'max_faces', 100)

    def name(self) -> str: return f"Tiled({self.base.name()})"

    def _tile_boxes(self, im_w: int, im_h: int) -> List[Tuple[int, int, int, int]]:
        cols, rows = self.tile_grid
        cols, rows = max(1, cols), max(1, rows)
        overlap = min(0.6, max(0.0, self.tile_overlap))

        tile_w = int(im_w / (cols - (cols - 1) * overlap)) if cols > 1 else im_w
        tile_h = int(im_h / (rows - (rows - 1) * overlap)) if rows > 1 else im_h
        tile_w, tile_h = min(tile_w, im_w), min(tile_h, im_h)
        step_x = max(1, int(tile_w * (1 - overlap))) if cols > 1 else im_w
        step_y = max(1, int(tile_h * (1 - overlap))) if rows > 1 else im_h

        boxes = []
        for r in range(rows):
            y1 = min(r * step_y, max(0, im_h - tile_h))
            for c in range(cols):
                x1 = min(c * step_x, max(0, im_w - tile_w))
                boxes.append((x1, y1, min(x1 + tile_w, im_w), min(y1 + tile_h, im_h)))
        return boxes

    def detect(self, image: np.ndarray) -> List[Detection]:
        im_h, im_w = image.shape[:2]

        if not self.use_tiling or max(im_h, im_w) <= self.tile_min_dimension:
            return self.base.detect(image)

        start_time = time.time()
        all_dets: List[Detection] = list(self.base.detect(image))  # full-frame pass catches large/near faces

        for (x1, y1, x2, y2) in self._tile_boxes(im_w, im_h):
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            try:
                tile_dets = self.base.detect(tile)
            except Exception as e:
                logger.warning(f"Tile detection failed at ({x1},{y1},{x2},{y2}): {e}")
                continue
            for d in tile_dets:
                bx1, by1, bx2, by2 = d.bbox
                offset_landmarks = d.landmarks + np.array([x1, y1], dtype=np.float32) if d.landmarks is not None else None
                all_dets.append(Detection(
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    confidence=d.confidence,
                    landmarks=offset_landmarks,
                    detection_time=d.detection_time,
                ))

        if not all_dets:
            return []

        boxes = np.array([d.bbox for d in all_dets], dtype=np.float64)
        scores = np.array([d.confidence for d in all_dets], dtype=np.float64)
        keep = nms_boxes(boxes, scores, self.merge_iou)
        merged = [all_dets[i] for i in keep]

        merged.sort(key=lambda d: d.confidence, reverse=True)
        merged = merged[: self.max_faces]

        elapsed = time.time() - start_time
        for d in merged:
            d.detection_time = elapsed
        return merged


class FaceDetectorFactory:
    @staticmethod
    def _maybe_tile(detector: FaceDetectorInterface) -> FaceDetectorInterface:
        if getattr(cfg.detection, 'use_tiling', True):
            return TiledFaceDetector(detector)
        return detector

    @staticmethod
    def create(backend: DetectorBackend = None) -> FaceDetectorInterface:
        if backend == DetectorBackend.SCRFD: return FaceDetectorFactory._maybe_tile(SCRFDDetector())
        if backend == DetectorBackend.INSIGHTFACE: return InsightFaceDetector()
        if backend == DetectorBackend.OPENCV: return OpenCVDetector()
        return FaceDetectorFactory._maybe_tile(SCRFDDetector())

    @staticmethod
    def create_with_fallback() -> FaceDetectorInterface:
        try:
            return FaceDetectorFactory._maybe_tile(SCRFDDetector())
        except Exception as e:
            logger.warning(f"Failed to load SCRFD, falling back to OpenCV. Error: {e}")
            return OpenCVDetector()
