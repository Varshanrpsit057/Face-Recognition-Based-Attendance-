import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from config import cfg, RecognizerBackend
from src.model_manager import get_model_manager
from src.preprocessing import FacePreprocessor
from src.utils import l2_normalize, cosine_similarity

class RecognizerInterface(ABC):
    @abstractmethod
    def extract(self, face_image: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray: ...
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def embedding_dim(self) -> int: ...

class BaseRecognizer(RecognizerInterface):
    def __init__(self, model_key: str):
        self.session = get_model_manager().get_session(model_key)
        self.preprocessor = FacePreprocessor()
        self._input_name = self.session.get_inputs()[0].name
        # Newer exports (e.g. adaface_ir101.onnx, cr_fiqa_l.onnx) declare
        # multiple named outputs (embedding, norm, quality_score, ...).
        # Resolve the identity embedding by name so extra heads don't
        # get mistaken for it; single/legacy-output models fall back to
        # output 0 unchanged.
        self._emb_idx = 0
        for i, o in enumerate(self.session.get_outputs()):
            if o.name.lower() == "embedding":
                self._emb_idx = i
                break

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        if face_image.ndim == 4 and face_image.shape[1] == 3:
            tensor = face_image.astype(np.float32)
        elif face_image.ndim == 3 and face_image.shape[0] == 3 and face_image.dtype == np.float32:
            tensor = np.expand_dims(face_image, axis=0)
        else:
            h, w = face_image.shape[:2]
            tensor = self.preprocessor.preprocess(face_image, (0, 0, w, h))
        outputs = self.session.run(None, {self._input_name: tensor})
        return l2_normalize(outputs[self._emb_idx].flatten())

    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        if not face_images:
            return np.array([])
        tensors = []
        for img in face_images:
            if img.ndim == 4 and img.shape[1] == 3:
                tensors.append(img[0])
            elif img.ndim == 3 and img.shape[0] == 3 and img.dtype == np.float32:
                tensors.append(img)
            else:
                h, w = img.shape[:2]
                tensors.append(self.preprocessor.preprocess(img, (0, 0, w, h))[0])
        batch_tensor = np.array(tensors, dtype=np.float32)
        outputs = self.session.run(None, {self._input_name: batch_tensor})
        res = []
        for out in outputs[self._emb_idx]:
            res.append(l2_normalize(out.flatten()))
        return np.array(res)

class AdaFaceRecognizer(BaseRecognizer):
    def __init__(self): super().__init__('adaface')
    def name(self) -> str: return "AdaFace"
    def embedding_dim(self) -> int: return getattr(cfg.recognition, 'embedding_dim', 512)

class GhostFaceRecognizer(BaseRecognizer):
    def __init__(self): super().__init__('ghostfacenet')
    def name(self) -> str: return "GhostFaceNet"
    def embedding_dim(self) -> int: return getattr(cfg.recognition, 'embedding_dim', 512)

class MobileFaceRecognizer(BaseRecognizer):
    def __init__(self): super().__init__('mobilefacenet')
    def name(self) -> str: return "MobileFaceNet"
    def embedding_dim(self) -> int: return getattr(cfg.recognition, 'embedding_dim', 128)

class RecognizerFactory:
    @staticmethod
    def create(backend: RecognizerBackend = None) -> RecognizerInterface:
        if backend == RecognizerBackend.ADAFACE: return AdaFaceRecognizer()
        if backend == RecognizerBackend.GHOSTFACENET: return GhostFaceRecognizer()
        if backend == RecognizerBackend.MOBILEFACENET: return MobileFaceRecognizer()
        return AdaFaceRecognizer()

class FaceRecognitionPipeline:
    def __init__(self, recognizer: RecognizerInterface = None):
        self.recognizer = recognizer or RecognizerFactory.create()
        self.threshold = getattr(cfg.recognition, 'similarity_threshold', 0.5)
        
    def recognize(self, face_image: np.ndarray, gallery_embeddings: np.ndarray, gallery_labels: List[str], threshold: float = None) -> Tuple[Optional[str], float]:
        if gallery_embeddings is None or len(gallery_embeddings) == 0:
            return None, 0.0
        thresh = threshold if threshold is not None else self.threshold
        emb = self.recognizer.extract(face_image)
        sims = [cosine_similarity(emb, g_emb) for g_emb in gallery_embeddings]
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]
        if max_sim >= thresh:
            return gallery_labels[max_idx], float(max_sim)
        return None, 0.0

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(cosine_similarity(embedding1, embedding2))

    def get_recognizer(self) -> RecognizerInterface:
        return self.recognizer
