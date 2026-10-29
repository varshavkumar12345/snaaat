"""
src/recommendations.py
-----------------------
Generate Top-K collaboration recommendations for the seed developer.

Uses every configured link-prediction algorithm to score candidate
(seed_user, candidate) pairs, then aggregates into a ranked list.

Outputs
~~~~~~~
* ``recommendations_top{K}.csv`` in ``outputs/recommendations/``
* Returns a DataFrame for use in the notebook and report.

Usage
~~~~~
    from src import load_config
    from src.recommendations import RecommendationEngine

    cfg = load_config()
    engine = RecommendationEngine(cfg, predictor)
    recs = engine.generate(G_train, seed_user="varshavkumar12345")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]


class RecommendationEngine:
    """Generate and persist Top-K collaboration recommendations."""

    def __init__(self, config: dict, predictor) -> None:  # noqa: ANN001
        self._config = config
        self._predictor = predictor
        rec_cfg = config.get("recommendations", {})
        self._top_k_values: List[int] = rec_cfg.get("top_k_values", [10, 20])
        self._output_dir = Path(rec_cfg.get("output_dir", "outputs/recommendations"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self, G, seed_user: str  # noqa: ANN001
    ) -> Dict[int, pd.DataFrame]:
        """Generate recommendations for each Top-K value.

        Parameters
        ----------
        G:
            Collaboration graph (training graph preferred).
        seed_user:
            The GitHub username to generate recommendations for.

        Returns
        -------
        dict
            ``{k: DataFrame}`` where each DataFrame has columns:
            rank, algorithm, candidate, score.
        """
        max_k = max(self._top_k_values)
        raw: Dict[str, List[Tuple[Edge, float]]] = self._predictor.recommend(
            G, seed_user, top_k=max_k
        )

        if not raw:
            logger.warning("No recommendations generated for '%s'.", seed_user)
            return {}

        results: Dict[int, pd.DataFrame] = {}
        for k in self._top_k_values:
            df = self._build_dataframe(raw, seed_user, k)
            results[k] = df
            out_path = self._output_dir / f"recommendations_top{k}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Top-%d recommendations saved to %s", k, out_path)

        # Also save a combined summary
        self._save_combined(raw, seed_user, max_k)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dataframe(
        self,
        raw: Dict[str, List[Tuple[Edge, float]]],
        seed_user: str,
        k: int,
    ) -> pd.DataFrame:
        """Build a ranked DataFrame for a single k value."""
        rows = []
        for algo, ranked_pairs in raw.items():
            for rank, (edge, score) in enumerate(ranked_pairs[:k], start=1):
                candidate = edge[1] if edge[0] == seed_user else edge[0]
                rows.append(
                    {
                        "rank": rank,
                        "algorithm": algo,
                        "seed_user": seed_user,
                        "candidate": candidate,
                        "score": round(score, 6),
                    }
                )
        return pd.DataFrame(rows)

    def _save_combined(
        self,
        raw: Dict[str, List[Tuple[Edge, float]]],
        seed_user: str,
        max_k: int,
    ) -> None:
        """Save an ensemble summary sorted by mean rank across algorithms."""
        from collections import defaultdict

        candidate_scores: dict = defaultdict(list)
        for algo, ranked_pairs in raw.items():
            for rank, (edge, score) in enumerate(ranked_pairs[:max_k], start=1):
                candidate = edge[1] if edge[0] == seed_user else edge[0]
                candidate_scores[candidate].append(score)

        ensemble_rows = []
        for candidate, scores in candidate_scores.items():
            ensemble_rows.append(
                {
                    "candidate": candidate,
                    "mean_score": round(sum(scores) / len(scores), 6),
                    "n_algorithms": len(scores),
                }
            )

        ensemble_df = pd.DataFrame(ensemble_rows).sort_values(
            ["n_algorithms", "mean_score"], ascending=[False, False]
        )
        ensemble_df.insert(0, "rank", range(1, len(ensemble_df) + 1))

        out = self._output_dir / "recommendations_ensemble.csv"
        ensemble_df.to_csv(out, index=False)
        logger.info("Ensemble recommendations saved to %s", out)

