"""
Dataset Validator — check dataset integrity, detect issues, generate reports.
"""

from __future__ import annotations

import csv
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import cfg
from src.logger import get_logger
from src.utils import get_student_dirs, get_image_files, load_image
from src.detector import FaceDetectorFactory, FaceDetectorInterface

logger = get_logger(__name__, "system")


class DatasetValidator:
    """Validates dataset images for face detection quality and integrity."""

    def __init__(self, detector: Optional[FaceDetectorInterface] = None) -> None:
        try:
            self.detector = detector or FaceDetectorFactory.create_with_fallback()
        except Exception:
            self.detector = None
            logger.warning("Could not initialise face detector for validation.")

    def validate_dataset(self, dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Validate all images in the dataset directory."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        student_dirs = get_student_dirs(dataset_dir)

        report: Dict[str, Any] = {
            "total_students": len(student_dirs),
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": [],
            "per_student": {},
        }

        for student_id, s_dir in student_dirs:
            images = get_image_files(s_dir)
            student_valid = 0
            student_invalid: List[Dict[str, str]] = []

            for img_path in images:
                report["total_images"] += 1
                res = self.validate_image(img_path)
                if res["valid"]:
                    student_valid += 1
                    report["valid_images"] += 1
                else:
                    student_invalid.append({"path": str(img_path), "reason": res["reason"]})
                    report["invalid_images"].append({"path": str(img_path), "reason": res["reason"]})

            report["per_student"][student_id] = {
                "total": len(images),
                "valid": student_valid,
                "invalid": len(student_invalid),
                "issues": student_invalid,
            }

        return report

    def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate a single image for faces, corruption, blur, brightness."""
        img = load_image(image_path)
        if img is None:
            return {"valid": False, "reason": "Corrupted or unreadable image"}

        if img.shape[0] < 10 or img.shape[1] < 10:
            return {"valid": False, "reason": f"Image too small ({img.shape[1]}x{img.shape[0]})"}

        if self.detector is None:
            return {"valid": True, "reason": "OK (no detector available)"}

        try:
            detections = self.detector.detect(img)
        except Exception as exc:
            return {"valid": False, "reason": f"Detection error: {exc}"}

        if not detections:
            return {"valid": False, "reason": "No face detected"}
        if len(detections) > 1:
            return {"valid": False, "reason": f"Multiple faces detected ({len(detections)})"}

        # Check blur
        det = detections[0]
        x1, y1, x2, y2 = det.bbox
        face = img[max(0, y1):y2, max(0, x1):x2]
        if face.size == 0:
            return {"valid": False, "reason": "Empty face crop"}

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < cfg.quality.min_laplacian_variance:
            return {"valid": False, "reason": f"Too blurry (variance={blur:.1f})"}

        brightness = float(np.mean(gray))
        if brightness < cfg.quality.min_brightness or brightness > cfg.quality.max_brightness:
            return {"valid": False, "reason": f"Bad brightness ({brightness:.1f})"}

        return {"valid": True, "reason": "OK", "blur": blur, "brightness": brightness}

    def detect_duplicates(self, student_dir: Path) -> List[Tuple[Path, Path]]:
        """Detect visually similar images using perceptual hashing."""
        images = get_image_files(student_dir)
        hashes: Dict[str, Path] = {}
        duplicates: List[Tuple[Path, Path]] = []

        for img_path in images:
            img = load_image(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            avg = resized.mean()
            h = "".join(["1" if p > avg else "0" for p in resized.flatten()])
            if h in hashes:
                duplicates.append((img_path, hashes[h]))
            else:
                hashes[h] = img_path

        return duplicates

    def remove_invalid(self, dataset_dir: Optional[Path] = None, dry_run: bool = True) -> Dict[str, Any]:
        """Remove (or list) invalid images from the dataset."""
        report = self.validate_dataset(dataset_dir)
        removed: List[str] = []
        for inv in report["invalid_images"]:
            path = Path(inv["path"])
            if not dry_run and path.exists():
                path.unlink(missing_ok=True)
            removed.append(str(path))
        return {"dry_run": dry_run, "removed_count": len(removed), "removed": removed}

    def get_statistics(self, dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Compute per-student and overall dataset statistics."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        student_dirs = get_student_dirs(dataset_dir)

        stats: Dict[str, Any] = {
            "total_students": len(student_dirs),
            "total_images": 0,
            "per_student": {},
        }

        blur_scores: List[float] = []
        brightness_scores: List[float] = []
        resolutions: List[Tuple[int, int]] = []

        for student_id, s_dir in student_dirs:
            images = get_image_files(s_dir)
            count = len(images)
            stats["total_images"] += count

            s_blurs: List[float] = []
            s_brights: List[float] = []
            s_res: List[Tuple[int, int]] = []

            for img_path in images:
                img = load_image(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                s_res.append((w, h))
                resolutions.append((w, h))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                bright = float(np.mean(gray))
                s_blurs.append(blur)
                s_brights.append(bright)
                blur_scores.append(blur)
                brightness_scores.append(bright)

            avg_w = np.mean([r[0] for r in s_res]) if s_res else 0
            avg_h = np.mean([r[1] for r in s_res]) if s_res else 0

            stats["per_student"][student_id] = {
                "image_count": count,
                "avg_resolution": f"{int(avg_w)}x{int(avg_h)}",
                "avg_blur": float(np.mean(s_blurs)) if s_blurs else 0.0,
                "avg_brightness": float(np.mean(s_brights)) if s_brights else 0.0,
            }

        stats["avg_blur"] = float(np.mean(blur_scores)) if blur_scores else 0.0
        stats["avg_brightness"] = float(np.mean(brightness_scores)) if brightness_scores else 0.0
        return stats

    # ── Report Generation ──────────────────────────────────────────

    def generate_csv_report(self, results: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """Generate a CSV report of invalid images."""
        path = path or Path(cfg.paths.reports_dir) / "dataset_validation.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Student", "Image Path", "Reason"])
            for inv in results.get("invalid_images", []):
                parts = Path(inv["path"]).parts
                student = parts[-2] if len(parts) >= 2 else "unknown"
                writer.writerow([student, inv["path"], inv["reason"]])
        return path

    def generate_json_report(self, results: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """Generate a JSON report."""
        path = path or Path(cfg.paths.reports_dir) / "dataset_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        return path

    def generate_html_report(self, results: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """Generate an HTML validation report."""
        path = path or Path(cfg.paths.reports_dir) / "dataset_validation.html"
        path.parent.mkdir(parents=True, exist_ok=True)

        total = results.get("total_images", 0)
        valid = results.get("valid_images", 0)
        invalid_list = results.get("invalid_images", [])
        per_student = results.get("per_student", {})
        score = self.compute_dataset_health_score(results)

        rows = ""
        for inv in invalid_list:
            rows += f"<tr><td>{inv['path']}</td><td>{inv['reason']}</td></tr>\n"

        student_rows = ""
        for sid, info in per_student.items():
            student_rows += (
                f"<tr><td>{sid}</td><td>{info.get('total', 0)}</td>"
                f"<td>{info.get('valid', 0)}</td><td>{info.get('invalid', 0)}</td></tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Dataset Validation Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #eee; }}
h1 {{ color: #00C853; }} h2 {{ color: #29B6F6; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
th {{ background: #333; }}
.score {{ font-size: 2em; color: #00C853; font-weight: bold; }}
</style></head><body>
<h1>📊 Dataset Validation Report</h1>
<p>Total Students: {results.get('total_students', 0)} | Total Images: {total} | Valid: {valid} | Invalid: {len(invalid_list)}</p>
<p>Health Score: <span class='score'>{score:.1f}/100</span></p>
<h2>Per-Student Summary</h2>
<table><tr><th>Student ID</th><th>Total</th><th>Valid</th><th>Invalid</th></tr>
{student_rows}</table>
<h2>Invalid Images</h2>
<table><tr><th>Path</th><th>Reason</th></tr>
{rows}</table>
</body></html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    # ── Distribution Analysis ──────────────────────────────────────

    def get_face_size_distribution(self, dataset_dir: Optional[Path] = None) -> Dict[str, int]:
        """Get distribution of face sizes: small/medium/large."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        dist: Dict[str, int] = {"small": 0, "medium": 0, "large": 0}
        if self.detector is None:
            return dist
        for _, s_dir in get_student_dirs(dataset_dir):
            for img_path in get_image_files(s_dir):
                img = load_image(img_path)
                if img is None:
                    continue
                try:
                    dets = self.detector.detect(img)
                    if dets:
                        w = dets[0].bbox[2] - dets[0].bbox[0]
                        if w < 50:
                            dist["small"] += 1
                        elif w < 150:
                            dist["medium"] += 1
                        else:
                            dist["large"] += 1
                except Exception:
                    pass
        return dist

    def get_pose_distribution(self, dataset_dir: Optional[Path] = None) -> Dict[str, List[float]]:
        """Placeholder for pose distribution (requires landmark analysis)."""
        return {"yaw": [], "pitch": [], "roll": []}

    def get_blur_histogram(self, dataset_dir: Optional[Path] = None) -> List[float]:
        """Get blur scores for all images."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        scores: List[float] = []
        for _, s_dir in get_student_dirs(dataset_dir):
            for img_path in get_image_files(s_dir):
                img = load_image(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        return scores

    def get_brightness_histogram(self, dataset_dir: Optional[Path] = None) -> List[float]:
        """Get brightness values for all images."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        values: List[float] = []
        for _, s_dir in get_student_dirs(dataset_dir):
            for img_path in get_image_files(s_dir):
                img = load_image(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    values.append(float(np.mean(gray)))
        return values

    # ── Scoring & Recommendations ──────────────────────────────────

    def compute_dataset_health_score(self, results: Dict[str, Any]) -> float:
        """Compute a 0-100 health score for the dataset."""
        total = results.get("total_images", 0)
        invalid_count = len(results.get("invalid_images", []))
        if total == 0:
            return 0.0
        valid_ratio = (total - invalid_count) / total

        # Penalise low student count
        student_count = results.get("total_students", 0)
        student_penalty = min(student_count / 5.0, 1.0)  # Full score at ≥5 students

        # Penalise very few images per student
        per_student = results.get("per_student", {})
        if per_student:
            avg_images = np.mean([v.get("total", 0) for v in per_student.values()])
            image_penalty = min(avg_images / 5.0, 1.0)
        else:
            image_penalty = 0.0

        score = 100.0 * (0.5 * valid_ratio + 0.25 * student_penalty + 0.25 * image_penalty)
        return max(0.0, min(100.0, score))

    def get_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recs: List[str] = []
        total = results.get("total_images", 0)
        invalid_count = len(results.get("invalid_images", []))

        if total == 0:
            recs.append("📂 Add student images to the dataset/ directory.")
            return recs

        if invalid_count > 0:
            recs.append(f"🗑️ Remove {invalid_count} invalid image(s) to improve data quality.")

        per_student = results.get("per_student", {})
        for sid, info in per_student.items():
            count = info.get("total", 0)
            if count < 3:
                recs.append(f"📸 Add more images for student '{sid}' (currently {count}).")

        if results.get("total_students", 0) < 2:
            recs.append("👥 Add at least 2 students for meaningful evaluation.")

        invalid_ratio = invalid_count / total if total > 0 else 0
        if invalid_ratio > 0.3:
            recs.append("⚠️ Over 30% of images are invalid. Review lighting, angles, and image quality.")

        if not recs:
            recs.append("✅ Dataset looks healthy! No issues found.")

        return recs
