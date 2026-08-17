"""
Vector Database — FAISS-backed (with numpy fallback) embedding storage
and nearest-neighbour search.
"""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import cfg
from src.logger import get_logger
from src.utils import cosine_similarity, l2_normalize

logger = get_logger("VectorDatabase", "system")

# Graceful FAISS import
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("FAISS not installed — using numpy fallback for vector search.")


class VectorDatabase:
    """FAISS (preferred) or NumPy-backed embedding vector database."""

    def __init__(self, config: Any = None) -> None:
        self.config = config or cfg.vector_db
        self._dim = getattr(cfg.recognition, "embedding_dim", 512)
        self._labels: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index: Any = None
        self._use_faiss = HAS_FAISS

        if self._use_faiss:
            self._index = faiss.IndexFlatIP(self._dim)

    # ── Index Management ───────────────────────────────────────────

    def build_index(self, embeddings: np.ndarray, labels: List[str]) -> None:
        """Clear and rebuild from scratch."""
        self.clear()
        self.add_embeddings(embeddings, labels)

    def add_embeddings(self, embeddings: np.ndarray, labels: List[str]) -> None:
        """Add new embeddings to the database."""
        if len(embeddings) == 0:
            return
        emb = embeddings.astype(np.float32)

        # Store raw embeddings
        if self._embeddings is None:
            self._embeddings = emb.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, emb])
        self._labels.extend(labels)

        # Update FAISS index
        if self._use_faiss:
            normed = emb.copy()
            faiss.normalize_L2(normed)
            self._index.add(normed)

    def remove_embedding(self, label: str) -> bool:
        """Remove all embeddings for a given label."""
        indices = [i for i, l in enumerate(self._labels) if l == label]
        if not indices:
            return False

        mask = np.ones(len(self._labels), dtype=bool)
        mask[indices] = False
        self._labels = [l for i, l in enumerate(self._labels) if mask[i]]
        self._embeddings = self._embeddings[mask] if self._embeddings is not None else None

        # Rebuild FAISS index
        if self._use_faiss and self._embeddings is not None and len(self._embeddings) > 0:
            self._index = faiss.IndexFlatIP(self._dim)
            normed = self._embeddings.astype(np.float32).copy()
            faiss.normalize_L2(normed)
            self._index.add(normed)
        elif self._use_faiss:
            self._index = faiss.IndexFlatIP(self._dim)

        return True

    # ── Search ─────────────────────────────────────────────────────

    def search(self, query: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Search for the top_k nearest embeddings to `query`.

        Args:
            query: 1-D embedding vector (dim,)
            top_k: number of results

        Returns:
            List of (label, similarity_score) tuples, descending by score.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        if self._use_faiss:
            q = query.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(q)
            k = min(top_k, self.size())
            distances, indices = self._index.search(q, k)
            results: List[Tuple[str, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    results.append((self._labels[idx], float(dist)))
            return results
        else:
            return self._numpy_search(query, top_k)

    def search_threshold(
        self, query: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Search and filter by similarity threshold."""
        results = self.search(query, top_k=self.size() or 1)
        return [(label, score) for label, score in results if score >= threshold]

    def _numpy_search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Fallback search using numpy cosine similarity."""
        sims = [cosine_similarity(query, emb) for emb in self._embeddings]
        indices = np.argsort(sims)[::-1][:top_k]
        return [(self._labels[i], float(sims[i])) for i in indices]

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save embeddings, labels, and FAISS index to disk."""
        base = Path(path) if path else Path(cfg.paths.faiss_index_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        pkl_path = base.with_suffix(".pkl")
        data = {"labels": self._labels, "embeddings": self._embeddings}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        if self._use_faiss and self._index is not None:
            faiss.write_index(self._index, str(base.with_suffix(".faiss")))

        logger.info(f"Saved VectorDB ({self.size()} embeddings) to {pkl_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load embeddings, labels, and FAISS index from disk."""
        base = Path(path) if path else Path(cfg.paths.faiss_index_path)
        pkl_path = base.with_suffix(".pkl")

        if not pkl_path.exists():
            logger.info(f"No vector DB found at {pkl_path}, starting fresh.")
            return False

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self._labels = data["labels"]
        self._embeddings = data["embeddings"]

        if self._use_faiss:
            faiss_path = base.with_suffix(".faiss")
            if faiss_path.exists():
                self._index = faiss.read_index(str(faiss_path))
            elif self._embeddings is not None and len(self._embeddings) > 0:
                self._index = faiss.IndexFlatIP(self._dim)
                normed = self._embeddings.astype(np.float32).copy()
                faiss.normalize_L2(normed)
                self._index.add(normed)

        logger.info(f"Loaded VectorDB ({self.size()} embeddings) from {pkl_path}")
        return True

    # ── Accessors ──────────────────────────────────────────────────

    def size(self) -> int:
        """Total number of embeddings."""
        return len(self._labels)

    def labels_list(self) -> List[str]:
        """Return unique enrolled labels."""
        return list(set(self._labels))

    def clear(self) -> None:
        """Reset the database."""
        self._labels = []
        self._embeddings = None
        if self._use_faiss:
            self._index = faiss.IndexFlatIP(self._dim)

    def get_embeddings_for_label(self, label: str) -> np.ndarray:
        """Get all embeddings for a specific label."""
        indices = [i for i, l in enumerate(self._labels) if l == label]
        if not indices or self._embeddings is None:
            return np.array([])
        return self._embeddings[indices]

    def get_all_embeddings(self) -> Tuple[Optional[np.ndarray], List[str]]:
        """Return (embeddings_array, labels_list)."""
        return self._embeddings, self._labels
