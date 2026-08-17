import time
import onnxruntime as ort
from typing import Dict, Optional, List, Tuple

from config import cfg, MODEL_REGISTRY, ModelInfo
from src.device_manager import DeviceManager
from src.downloader import ModelDownloader
from src.logger import get_logger
from pathlib import Path

logger = get_logger("ModelManager", "system")

class ModelManager:
    _instance = None

    def __new__(cls, device_manager: Optional[DeviceManager] = None, downloader: Optional[ModelDownloader] = None):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init(device_manager, downloader)
        return cls._instance

    def _init(self, device_manager: Optional[DeviceManager], downloader: Optional[ModelDownloader]):
        self.device_manager = device_manager or DeviceManager()
        self.downloader = downloader or ModelDownloader(Path(cfg.paths.models_dir))
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._load_times: Dict[str, float] = {}

    def get_session(self, model_key: str) -> ort.InferenceSession:
        if model_key in self._sessions:
            return self._sessions[model_key]

        path = self.downloader.get_model_path(model_key)
        if not path or not path.exists():
            logger.info(f"Model {model_key} not found locally, initiating download.")
            path = self.downloader.download_model(model_key)

        providers = self.device_manager.get_best_providers()
        options = self.device_manager.get_onnx_session_options()

        start_t = time.perf_counter()
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            self._sessions[model_key] = session
            self._load_times[model_key] = time.perf_counter() - start_t
            logger.info(f"Loaded {model_key} ONNX session on providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX session for {model_key}: {e}")
            raise

    def unload_model(self, model_key: str) -> None:
        if model_key in self._sessions:
            del self._sessions[model_key]
            logger.info(f"Unloaded model {model_key}")
        if model_key in self._load_times:
            del self._load_times[model_key]

    def unload_all(self) -> None:
        self._sessions.clear()
        self._load_times.clear()
        logger.info("Unloaded all models")

    def is_loaded(self, model_key: str) -> bool:
        return model_key in self._sessions

    def get_loaded_models(self) -> List[str]:
        return list(self._sessions.keys())

    def get_load_time(self, model_key: str) -> Optional[float]:
        return self._load_times.get(model_key)

    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        return MODEL_REGISTRY.get(model_key)

    def get_input_shape(self, model_key: str) -> Optional[Tuple]:
        sess = self.get_session(model_key)
        if sess:
            shape = sess.get_inputs()[0].shape
            return tuple(shape) if shape else None
        return None

    def get_output_shape(self, model_key: str) -> Optional[Tuple]:
        sess = self.get_session(model_key)
        if sess:
            shape = sess.get_outputs()[0].shape
            return tuple(shape) if shape else None
        return None

    def cleanup(self) -> None:
        self.unload_all()


def get_model_manager() -> ModelManager:
    return ModelManager()
