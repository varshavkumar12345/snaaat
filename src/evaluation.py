"""
Evaluation metrics: Precision@K, Recall@K, F1@K, ROC-AUC, and ROC curves.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

def compute_precision_recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    Compute Precision@K, Recall@K, and F1@K.
    y_true: binary labels (1 for positive test edges, 0 for negative samples).
    y_scores: algorithm scores.
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0, 0.0, 0.0

    # Sort indices by score descending; if tie, preserve stable sort
    sorted_indices = np.argsort(-y_scores, kind="mergesort")
    top_k_indices = sorted_indices[:min(k, len(sorted_indices))]

    true_positives_at_k = np.sum(y_true[top_k_indices])
    total_positives = np.sum(y_true)

    precision_at_k = float(true_positives_at_k / len(top_k_indices))
    recall_at_k = float(true_positives_at_k / total_positives) if total_positives > 0 else 0.0

    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2.0 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0

    return precision_at_k, recall_at_k, f1_at_k

def evaluate_algorithm(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_list: List[int] = [5, 10, 20, 50]
) -> Dict[str, Any]:
    """Compute ROC-AUC, Precision@K, Recall@K, F1@K for a single algorithm."""
    results = {}

    # ROC-AUC calculation
    try:
        # Check if variance in scores exists
        if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
            auc = float(roc_auc_score(y_true, y_scores))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5
    results["ROC-AUC"] = round(auc, 4)

    for k in k_list:
        prec, rec, f1 = compute_precision_recall_at_k(y_true, y_scores, k)
        results[f"Precision@{k}"] = round(prec, 4)
        results[f"Recall@{k}"] = round(rec, 4)
        results[f"F1@{k}"] = round(f1, 4)

    return results

def compare_all_algorithms(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Create a structured comparison DataFrame from evaluation results."""
    records = []
    for algo_name, metrics in results_dict.items():
        row = {"Algorithm": algo_name}
        row.update(metrics)
        records.append(row)
    df = pd.DataFrame(records)
    return df

