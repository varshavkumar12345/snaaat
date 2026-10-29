"""
src/evaluation.py
------------------
Phase 6: Comprehensive Link Prediction Evaluator.

Metrics:
  - ROC-AUC
  - Average Precision (PR-AUC)
  - Precision@K (K=10, K=20)
  - Recall@K (K=10, K=20)
  - F1-Score (at 0.5 threshold)
  - Handles edge-cases, undefined metrics, and sparse partitions gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]
ScoreMap = Dict[Edge, float]
AllScores = Dict[str, ScoreMap]


class Evaluator:
    """Evaluates link-prediction algorithms across ranking and classification metrics."""

    def __init__(self, config: dict) -> None:
        self._config = config
        rec_cfg = config.get("recommendations", {})
        self._top_k_values: List[int] = rec_cfg.get("top_k_values", [10, 20])

    def evaluate(
        self,
        all_scores: AllScores,
        test_pos: List[Edge],
        test_neg: List[Edge],
    ) -> pd.DataFrame:
        """
        Evaluate all algorithms on the same test candidate set.

        Parameters
        ----------
        all_scores:
            Mapping of algorithm -> {(u, v): score}.
        test_pos:
            Positive held-out edges (y_true = 1).
        test_neg:
            Negative sample non-edges (y_true = 0).

        Returns
        -------
        pd.DataFrame
            Comparison table indexed by algorithm name.
        """
        rows = []
        test_pos_set = {tuple(sorted(p)) for p in test_pos}
        test_neg_set = {tuple(sorted(p)) for p in test_neg}
        all_test_pairs = list(test_pos_set) + list(test_neg_set)

        y_true = np.array([1] * len(test_pos_set) + [0] * len(test_neg_set))

        for algo, score_map in all_scores.items():
            y_scores = []
            for u, v in all_test_pairs:
                pair = tuple(sorted((u, v)))
                score = score_map.get(pair, score_map.get((v, u), 0.0))
                y_scores.append(score)

            y_scores = np.array(y_scores, dtype=float)
            metrics = self._compute_metrics_for_algo(y_true, y_scores, all_test_pairs, test_pos_set)
            metrics["algorithm"] = algo
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("algorithm")
        return df

    def _compute_metrics_for_algo(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        test_pairs: List[Edge],
        test_pos_set: set[Edge],
    ) -> Dict[str, Any]:
        """Compute AUC, F1, and Precision/Recall@K."""
        results: Dict[str, Any] = {
            "n_test_pos": int(sum(y_true == 1)),
            "n_test_neg": int(sum(y_true == 0)),
        }

        # 1. ROC-AUC and PR-AUC
        if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
            try:
                results["roc_auc"] = round(float(roc_auc_score(y_true, y_scores)), 4)
                results["pr_auc"] = round(float(average_precision_score(y_true, y_scores)), 4)
            except Exception:
                results["roc_auc"] = np.nan
                results["pr_auc"] = np.nan
        else:
            results["roc_auc"] = 0.5
            results["pr_auc"] = float(round(sum(y_true == 1) / len(y_true), 4)) if len(y_true) > 0 else np.nan

        # 2. Precision & Recall @ K
        # Sort candidates descending by score
        order = np.argsort(-y_scores)
        total_positives = len(test_pos_set)

        for k in self._top_k_values:
            actual_k = min(k, len(y_true))
            top_indices = order[:actual_k]
            hits = sum(y_true[top_indices] == 1)

            prec_k = hits / actual_k if actual_k > 0 else 0.0
            rec_k = hits / total_positives if total_positives > 0 else 0.0

            results[f"precision@{k}"] = round(float(prec_k), 4)
            results[f"recall@{k}"] = round(float(rec_k), 4)

        # 3. F1-score with min-max normalized threshold
        score_min, score_max = y_scores.min(), y_scores.max()
        if score_max > score_min:
            norm_scores = (y_scores - score_min) / (score_max - score_min)
            y_pred = (norm_scores >= 0.5).astype(int)
        else:
            y_pred = np.zeros_like(y_true)

        results["f1_score"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        return results
