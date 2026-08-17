import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from config import cfg
from src.logger import get_logger

logger = get_logger(__name__)

class Visualizer:
    def __init__(self, output_dir: Path = None, dpi: int = None, formats: List[str] = None):
        self.output_dir = output_dir or Path(cfg.paths.outputs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi or cfg.evaluation.dpi
        self.formats = formats or cfg.evaluation.figure_formats
        plt.style.use('dark_background')

    def _save_figure(self, fig, name: str) -> List[Path]:
        paths = []
        for fmt in self.formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            paths.append(path)
        plt.close(fig)
        return paths

    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], title: str = '') -> List[Path]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title or "Confusion Matrix")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        return self._save_figure(fig, 'confusion_matrix')

    def plot_roc_curve(self, fpr, tpr, auc_score: float) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return self._save_figure(fig, 'roc_curve')

    def plot_precision_recall_curve(self, precision, recall) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label='PR curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        return self._save_figure(fig, 'pr_curve')

    def plot_det_curve(self, fpr, fnr) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, fnr, label='DET curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('False Negative Rate')
        ax.set_title('Detection Error Tradeoff')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        return self._save_figure(fig, 'det_curve')

    def plot_cmc_curve(self, cmc_dict: Dict[int, float]) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ranks = list(cmc_dict.keys())
        accs = list(cmc_dict.values())
        ax.plot(ranks, accs, marker='o')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Recognition Rate')
        ax.set_title('Cumulative Match Characteristic (CMC) Curve')
        return self._save_figure(fig, 'cmc_curve')

    def plot_similarity_histogram(self, genuine_scores, impostor_scores) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(genuine_scores, bins=50, color='green', label='Genuine', alpha=0.5, ax=ax)
        sns.histplot(impostor_scores, bins=50, color='red', label='Impostor', alpha=0.5, ax=ax)
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Count')
        ax.set_title('Similarity Score Distributions')
        ax.legend()
        return self._save_figure(fig, 'similarity_hist')

    def plot_embedding_distance_histogram(self, distances: np.ndarray) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(distances, bins=50, ax=ax)
        ax.set_title('Embedding Distance Distribution')
        return self._save_figure(fig, 'distance_hist')

    def plot_latency_histogram(self, latencies: Dict[str, List[float]]) -> List[Path]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for op, lats in latencies.items():
            sns.kdeplot(lats, label=op, ax=ax)
        ax.set_xlabel('Latency (s)')
        ax.set_title('Latency Distribution by Operation')
        ax.legend()
        return self._save_figure(fig, 'latency_hist')

    def plot_fps_timeline(self, fps_values: List[float]) -> List[Path]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fps_values)
        ax.set_xlabel('Frame')
        ax.set_ylabel('FPS')
        ax.set_title('FPS Timeline')
        return self._save_figure(fig, 'fps_timeline')

    def plot_confidence_histogram(self, confidences: List[float]) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(confidences, bins=30, ax=ax)
        ax.set_title('Confidence Score Distribution')
        return self._save_figure(fig, 'confidence_hist')

    def plot_far_vs_threshold(self, thresholds, far_values) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, far_values)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('FAR')
        ax.set_title('False Accept Rate vs Threshold')
        return self._save_figure(fig, 'far_vs_threshold')

    def plot_frr_vs_threshold(self, thresholds, frr_values) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, frr_values)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('FRR')
        ax.set_title('False Reject Rate vs Threshold')
        return self._save_figure(fig, 'frr_vs_threshold')

    def plot_threshold_sweep(self, sweep_results: Dict) -> List[Path]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(sweep_results['thresholds'], sweep_results['far'], label='FAR')
        ax.plot(sweep_results['thresholds'], sweep_results['frr'], label='FRR')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('FAR & FRR vs Threshold')
        ax.legend()
        return self._save_figure(fig, 'threshold_sweep')

    def plot_per_class_metrics(self, per_class: Dict[str, Dict[str, float]]) -> List[Path]:
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(per_class.keys())
        f1s = [v.get('f1-score', 0) for v in per_class.values()]
        sns.barplot(x=classes, y=f1s, ax=ax)
        ax.set_xticklabels(classes, rotation=90)
        ax.set_title('Per-Class F1 Score')
        return self._save_figure(fig, 'per_class_f1')

    def generate_all(self, results: Dict) -> Dict[str, List[Path]]:
        return {}
