"""
src/evaluation.py
------------------
Evaluate link-prediction algorithm performance.

Metrics computed
~~~~~~~~~~~~~~~~
* AUC-ROC  — area under the receiver operating characteristic curve
* Average Precision (AUC-PR)
* Precision @ threshold = 0.5 (after min-max normalisation)
* Recall @ threshold = 0.5
* F1 Score @ threshold = 0.5

Usage
~~~~~
    from src import load_config
    from src.evaluation import Evaluator

    cfg = load_config()
    evaluator = Evaluator(cfg)
    results_df = evaluator.evaluate(all_scores, test_pos, test_neg)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

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
    """Compute evaluation metrics for all link-prediction algorithms."""

    def __init__(self, config: dict) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        all_scores: AllScores,
        test_pos: List[Edge],
        test_neg: List[Edge],
    ) -> pd.DataFrame:
        """Evaluate all algorithms and return a comparison DataFrame.

        Parameters
        ----------
        all_scores:
            Output of ``LinkPredictor.score_all()``.
        test_pos:
            Held-out positive edges (label = 1).
        test_neg:
            Sampled negative edges (label = 0).

        Returns
        -------
        pd.DataFrame
            One row per algorithm with columns:
            algorithm, auc_roc, avg_precision, precision, recall, f1,
            n_test_pos, n_test_neg.
        """
        rows = []
        for algo, score_map in all_scores.items():
            metrics = self._compute_metrics(score_map, test_pos, test_neg)
            metrics["algorithm"] = algo
            rows.append(metrics)
            logger.info(
                "%-30s  AUC-ROC=%.4f  AP=%.4f  F1=%.4f",
                algo,
                metrics["auc_roc"],
                metrics["avg_precision"],
                metrics["f1"],
            )

        df = pd.DataFrame(rows).set_index("algorithm")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        score_map: ScoreMap,
        test_pos: List[Edge],
        test_neg: List[Edge],
    ) -> dict:
        """Compute all metrics for a single algorithm's score map."""
        y_true: List[int] = []
        y_score: List[float] = []

        for edge in test_pos:
            score = score_map.get(edge, score_map.get((edge[1], edge[0]), 0.0))
            y_true.append(1)
            y_score.append(score)

        for edge in test_neg:
            score = score_map.get(edge, score_map.get((edge[1], edge[0]), 0.0))
            y_true.append(0)
            y_score.append(score)

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score, dtype=float)

        # Min-max normalise scores to [0, 1] for threshold-based metrics
        score_min, score_max = y_score_arr.min(), y_score_arr.max()
        if score_max > score_min:
            y_score_norm = (y_score_arr - score_min) / (score_max - score_min)
        else:
            y_score_norm = y_score_arr

        y_pred = (y_score_norm >= 0.5).astype(int)

        # Guard against degenerate cases
        n_pos = int(y_true_arr.sum())
        n_neg = len(y_true_arr) - n_pos

        if n_pos == 0 or n_neg == 0:
            logger.warning("Degenerate test set (all same label). Skipping metrics.")
            return {
                "auc_roc": float("nan"),
                "avg_precision": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "n_test_pos": n_pos,
                "n_test_neg": n_neg,
            }

        auc = roc_auc_score(y_true_arr, y_score_arr)
        ap = average_precision_score(y_true_arr, y_score_arr)
        prec = precision_score(y_true_arr, y_pred, zero_division=0)
        rec = recall_score(y_true_arr, y_pred, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)

        return {
            "auc_roc": round(float(auc), 6),
            "avg_precision": round(float(ap), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "n_test_pos": n_pos,
            "n_test_neg": n_neg,
        }

