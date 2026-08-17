import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from src.logger import get_logger
from src.utils import compute_sha256, ensure_dir
from config import MODEL_REGISTRY

logger = get_logger("ModelDownloader", "system")

class ModelDownloader:
    def __init__(self, models_dir: Path, retry_count: int = 3, timeout: int = 30):
        self.models_dir = ensure_dir(models_dir)
        self.retry_count = retry_count
        self.timeout = timeout
        self._status: Dict[str, bool] = {}

    def download_file(self, url: str, dest: Path, sha256: str = '', description: str = '') -> bool:
        if dest.exists() and (not sha256 or compute_sha256(dest) == sha256):
            logger.info(f"File {dest.name} already exists and verified.")
            return True

        resume_byte_pos = 0
        if dest.exists():
            resume_byte_pos = dest.stat().st_size

        for attempt in range(self.retry_count):
            try:
                headers = {}
                if resume_byte_pos > 0:
                    headers['Range'] = f"bytes={resume_byte_pos}-"

                response = requests.get(url, stream=True, headers=headers, timeout=self.timeout)
                if response.status_code == 416: # Range Not Satisfiable
                    resume_byte_pos = 0
                    headers.pop('Range', None)
                    response = requests.get(url, stream=True, headers=headers, timeout=self.timeout)
                elif response.status_code not in (200, 206):
                    logger.warning(f"Failed to download from {url} (status {response.status_code})")
                    continue
                
                total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos
                mode = 'ab' if response.status_code == 206 else 'wb'

                if mode == 'wb':
                    resume_byte_pos = 0

                desc = description or dest.name
                with open(dest, mode) as f, tqdm(
                    desc=desc,
                    total=total_size,
                    initial=resume_byte_pos,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=8192):
                        size = f.write(data)
                        bar.update(size)

                if sha256 and compute_sha256(dest) != sha256:
                    logger.error(f"SHA256 mismatch for {dest.name}")
                    dest.unlink()
                    continue

                return True
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(f"Attempt {attempt+1}/{self.retry_count} failed: {e}")
                
        return False

    def download_model(self, model_key: str) -> Path:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_key} not in registry.")
        info = MODEL_REGISTRY[model_key]
        dest = self.models_dir / info.filename
        
        success = self.download_file(info.url, dest, info.sha256, description=f"Downloading {model_key}")
        self._status[model_key] = success
        if not success:
            raise RuntimeError(f"Failed to download model {model_key}")
        return dest

    def download_all_models(self) -> Dict[str, Path]:
        res = {}
        for k in MODEL_REGISTRY:
            res[k] = self.download_model(k)
        return res

    def verify_model(self, model_key: str) -> bool:
        if model_key not in MODEL_REGISTRY:
            return False
        info = MODEL_REGISTRY[model_key]
        dest = self.models_dir / info.filename
        if not dest.exists():
            return False
        if not info.sha256:
            return True
        return compute_sha256(dest) == info.sha256

    def is_model_available(self, model_key: str) -> bool:
        path = self.get_model_path(model_key)
        return path is not None and path.exists()

    def get_model_path(self, model_key: str) -> Optional[Path]:
        if model_key not in MODEL_REGISTRY:
            return None
        return self.models_dir / MODEL_REGISTRY[model_key].filename

    def get_download_status(self) -> Dict[str, bool]:
        return self._status
