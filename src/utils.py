import time
import hashlib
from pathlib import Path
from typing import Union, List, Tuple, Generator
import cv2
import numpy as np
from datetime import datetime

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk in BGR format. Returns None if loading fails."""
    try:
        path_str = str(path)
        img = cv2.imread(path_str)
        return img  # May be None if cv2 can't read it
    except Exception:
        return None

def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    cv2.imwrite(str(path), image)

def is_image_file(path: Union[str, Path]) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_image_files(directory: Path) -> List[Path]:
    return [p for p in directory.rglob('*') if p.is_file() and is_image_file(p)]

def get_student_dirs(dataset_dir: Path) -> List[Tuple[str, Path]]:
    result = []
    if not dataset_dir.exists():
        return result
    for p in dataset_dir.iterdir():
        if p.is_dir():
            result.append((p.name, p))
    return result

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    return f"{seconds:.2f}s"

def format_bytes(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}TB"

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def resize_with_pad(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2
    
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    padded[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
    return padded

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))

def chunk_list(lst: List, chunk_size: int) -> Generator:
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy NMS over axis-aligned boxes in (x1, y1, x2, y2) format.

    cv2.dnn.NMSBoxes expects (x, y, w, h) rects; SCRFD/tile boxes are
    already (x1, y1, x2, y2), so a plain numpy implementation avoids
    silently mis-scoring overlap and either over- or under-suppressing
    faces when many are close together (dense classroom scenes).
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou <= iou_threshold]

    return keep
