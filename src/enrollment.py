"""
Enrollment Engine — face detection, quality gating, embedding extraction,
and vector database management for student enrollment.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import cfg
from src.logger import get_logger
from src.utils import get_student_dirs, get_image_files, load_image
from src.detector import FaceDetectorFactory, FaceDetectorInterface
from src.preprocessing import FacePreprocessor
from src.quality_gate import QualityGate
from src.recognizer import RecognizerFactory, RecognizerInterface
from src.vector_db import VectorDatabase
from src.profiler import get_profiler

logger = get_logger(__name__, "system")
profiler = get_profiler()


class EnrollmentEngine:
    """Enroll students from image directories into the FAISS vector database."""

    def __init__(
        self,
        detector: Optional[FaceDetectorInterface] = None,
        preprocessor: Optional[FacePreprocessor] = None,
        quality_gate: Optional[QualityGate] = None,
        recognizer: Optional[RecognizerInterface] = None,
        vector_db: Optional[VectorDatabase] = None,
    ) -> None:
        self.detector = detector or FaceDetectorFactory.create_with_fallback()
        self.preprocessor = preprocessor or FacePreprocessor()
        self.quality_gate = quality_gate or QualityGate()
        self.recognizer = recognizer or RecognizerFactory.create()
        self.vector_db = vector_db or VectorDatabase()

    # ── Single Student ─────────────────────────────────────────────

    def enroll_student(self, student_id: str, image_dir: Path) -> Dict[str, Any]:
        """Enroll a single student from an image directory."""
        logger.info(f"Enrolling student: {student_id} from {image_dir}")
        image_files = get_image_files(image_dir)

        embeddings: List[np.ndarray] = []
        rejected: Dict[str, str] = {}

        for img_path in image_files:
            try:
                img = load_image(img_path)
                if img is None:
                    rejected[img_path.name] = "Failed to load image"
                    continue

                profiler.start("enroll_detect")
                detections = self.detector.detect(img)
                profiler.stop("enroll_detect")

                if not detections:
                    rejected[img_path.name] = "No face detected"
                    continue
                if len(detections) > 1:
                    rejected[img_path.name] = "Multiple faces detected"
                    continue

                det = detections[0]

                # Crop for quality check
                x1, y1, x2, y2 = det.bbox
                face_img = img[max(0, y1):y2, max(0, x1):x2]
                if face_img.size == 0:
                    rejected[img_path.name] = "Invalid face crop"
                    continue

                profiler.start("enroll_quality")
                qr = self.quality_gate.assess(face_img, det.bbox, det.landmarks)
                profiler.stop("enroll_quality")

                if not qr.passed:
                    rejected[img_path.name] = f"Quality: {', '.join(qr.rejection_reasons)}"
                    continue

                profiler.start("enroll_preprocess")
                tensor = self.preprocessor.preprocess(img, det.bbox, det.landmarks)
                profiler.stop("enroll_preprocess")

                profiler.start("enroll_extract")
                emb = self.recognizer.extract(tensor[0])
                profiler.stop("enroll_extract")

                if emb is not None:
                    embeddings.append(emb)
                else:
                    rejected[img_path.name] = "Embedding extraction failed"

            except Exception as exc:
                logger.error(f"Error processing {img_path}: {exc}")
                rejected[img_path.name] = str(exc)

        # Compute mean embedding and store
        mean_emb = None
        if embeddings:
            from src.utils import l2_normalize
            mean_emb = l2_normalize(np.mean(embeddings, axis=0))
            self.vector_db.add_embeddings(np.array([mean_emb]), [student_id])

        return {
            "student_id": student_id,
            "total_images": len(image_files),
            "enrolled_count": len(embeddings),
            "rejected_count": len(rejected),
            "rejected_reasons": rejected,
        }

    # ── Batch Enrollment ───────────────────────────────────────────

    def enroll_all(self, dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Enroll all students from the dataset directory."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        student_dirs = get_student_dirs(dataset_dir)

        stats: Dict[str, Any] = {
            "total_students_processed": 0,
            "successful_enrollments": 0,
            "failed_enrollments": 0,
            "details": {},
        }

        for student_id, s_dir in student_dirs:
            result = self.enroll_student(student_id, s_dir)
            stats["total_students_processed"] += 1
            if result["enrolled_count"] > 0:
                stats["successful_enrollments"] += 1
            else:
                stats["failed_enrollments"] += 1
            stats["details"][student_id] = result

        # Save the database
        try:
            db_path = Path(cfg.paths.faiss_index_path)
            self.vector_db.save(db_path)
            logger.info(f"Vector database saved to {db_path}")
        except Exception as exc:
            logger.error(f"Failed to save vector database: {exc}")

        return stats

    # ── Single Image Enrollment ────────────────────────────────────

    def enroll_single_image(self, student_id: str, image: np.ndarray) -> Dict[str, Any]:
        """Enroll a student from a single image."""
        detections = self.detector.detect(image)
        if not detections:
            return {"success": False, "reason": "No face detected"}

        det = detections[0]
        x1, y1, x2, y2 = det.bbox
        face_img = image[max(0, y1):y2, max(0, x1):x2]
        if face_img.size == 0:
            return {"success": False, "reason": "Invalid face crop"}

        qr = self.quality_gate.assess(face_img, det.bbox, det.landmarks)
        if not qr.passed:
            return {"success": False, "reason": f"Quality: {', '.join(qr.rejection_reasons)}"}

        tensor = self.preprocessor.preprocess(image, det.bbox, det.landmarks)
        emb = self.recognizer.extract(tensor[0])

        if emb is not None:
            self.vector_db.add_embeddings(np.array([emb]), [student_id])
            self.vector_db.save(Path(cfg.paths.faiss_index_path))
            return {"success": True}
        return {"success": False, "reason": "Embedding extraction failed"}

    # ── Database Management ────────────────────────────────────────

    def rebuild_index(self) -> int:
        """Clear DB and re-enroll all students."""
        self.vector_db.clear()
        stats = self.enroll_all()
        return stats["successful_enrollments"]

    def remove_student(self, student_id: str) -> bool:
        """Remove a student from the vector database."""
        success = self.vector_db.remove_embedding(student_id)
        if success:
            self.vector_db.save(Path(cfg.paths.faiss_index_path))
        return success

    def is_enrolled(self, student_id: str) -> bool:
        """Check if a student is enrolled."""
        return student_id in self.vector_db.labels_list()

    def get_enrolled_students(self) -> List[str]:
        """Get list of enrolled student IDs."""
        return self.vector_db.labels_list()

    def get_enrollment_stats(self) -> Dict[str, Any]:
        """Get current enrollment statistics."""
        return {
            "total_students": len(self.vector_db.labels_list()),
            "total_embeddings": self.vector_db.size(),
        }
