import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, matthews_corrcoef, balanced_accuracy_score)
from config import cfg
from src.logger import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    @staticmethod
    def accuracy(y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred, average='macro') -> float:
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def recall(y_true, y_pred, average='macro') -> float:
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def f1_score(y_true, y_pred, average='macro') -> float:
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def specificity(y_true, y_pred) -> float:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0

    @staticmethod
    def balanced_accuracy(y_true, y_pred) -> float:
        return balanced_accuracy_score(y_true, y_pred)

    @staticmethod
    def matthews_corrcoef(y_true, y_pred) -> float:
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def roc_curve(y_true, y_scores) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return roc_curve(y_true, y_scores)

    @staticmethod
    def roc_auc(y_true, y_scores) -> float:
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.0

    @staticmethod
    def precision_recall_curve(y_true, y_scores) -> Tuple:
        return precision_recall_curve(y_true, y_scores)

    @staticmethod
    def equal_error_rate(y_true, y_scores) -> Tuple[float, float]:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[idx]
        eer_threshold = thresholds[idx]
        return float(eer), float(eer_threshold)

    @staticmethod
    def far_frr(y_true, y_scores, threshold) -> Tuple[float, float]:
        preds = (np.array(y_scores) >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            return float(far), float(frr)
        return 0.0, 0.0

    @staticmethod
    def det_curve(y_true, y_scores) -> Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        return fpr, fnr

    @staticmethod
    def cmc_curve(gallery_embeddings, gallery_labels, probe_embeddings, probe_labels, ranks) -> Dict[int, float]:
        cmc = {r: 0.0 for r in ranks}
        if not probe_embeddings or not gallery_embeddings:
            return cmc
            
        gallery_emb = np.array(gallery_embeddings)
        gallery_lab = np.array(gallery_labels)
        
        for i, p_emb in enumerate(probe_embeddings):
            dists = np.linalg.norm(gallery_emb - p_emb, axis=1)
            sorted_indices = np.argsort(dists)
            sorted_labels = gallery_lab[sorted_indices]
            
            p_lab = probe_labels[i]
            for r in ranks:
                if p_lab in sorted_labels[:r]:
                    cmc[r] += 1.0
                    
        num_probes = len(probe_labels)
        for r in ranks:
            cmc[r] = cmc[r] / num_probes if num_probes > 0 else 0.0
            
        return cmc

    @staticmethod
    def threshold_sweep(y_true, y_scores, start, end, steps) -> Dict[str, List]:
        thresholds = np.linspace(start, end, steps)
        far_list, frr_list, acc_list = [], [], []
        
        for t in thresholds:
            far, frr = MetricsCalculator.far_frr(y_true, y_scores, t)
            preds = (np.array(y_scores) >= t).astype(int)
            acc = accuracy_score(y_true, preds)
            far_list.append(far)
            frr_list.append(frr)
            acc_list.append(acc)
            
        return {
            'thresholds': thresholds.tolist(),
            'far': far_list,
            'frr': frr_list,
            'accuracy': acc_list
        }

    @staticmethod
    def per_class_metrics(y_true, y_pred, labels) -> Dict[str, Dict[str, float]]:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        return {str(k): v for k, v in report.items() if isinstance(v, dict)}

    @staticmethod
    def compute_all(y_true, y_pred, y_scores, labels, gallery_emb, gallery_lab, probe_emb, probe_lab) -> Dict[str, Any]:
        return {
            'accuracy': MetricsCalculator.accuracy(y_true, y_pred),
            'precision': MetricsCalculator.precision(y_true, y_pred),
            'recall': MetricsCalculator.recall(y_true, y_pred),
            'f1_score': MetricsCalculator.f1_score(y_true, y_pred),
            'balanced_accuracy': MetricsCalculator.balanced_accuracy(y_true, y_pred),
            'matthews_corrcoef': MetricsCalculator.matthews_corrcoef(y_true, y_pred)
        }
