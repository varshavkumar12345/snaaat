"""
src/recommendations.py
-----------------------
Phase 6: Recommendation Engine for Developer Collaborations.

Generates:
  - Top-10 and Top-20 recommendations per algorithm.
  - Ensemble aggregated consensus ranking.
  - Complete structural feature explanations for each recommended developer.
  - Exports CSV and JSON artifacts.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates explainable Top-K developer recommendations."""

    def __init__(self, config: dict, predictor) -> None:
        self._config = config
        self._predictor = predictor
        rec_cfg = config.get("recommendations", {})
        self._top_k_values: List[int] = rec_cfg.get("top_k_values", [10, 20])
        self._output_dir = Path(rec_cfg.get("output_dir", "outputs/recommendations"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_recommendations(
        self,
        G,
        seed_user: str,
        exclude_users: set[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate, explain, and export Top-10 and Top-20 recommendations.

        Returns
        -------
        dict
            Contains DataFrames for top_10, top_20, ensemble, and json path.
        """
        seed_user = seed_user.lower().strip()
        max_k = max(self._top_k_values)

        raw_recs = self._predictor.recommend(
            G, seed_user, top_k=max_k, exclude_users=exclude_users
        )

        if not raw_recs:
            logger.warning("No recommendations could be generated for %s", seed_user)
            return {}

        results: Dict[str, Any] = {"raw": raw_recs}

        # 1. Build Top-K DataFrames and CSVs
        for k in self._top_k_values:
            rows = []
            for algo, rec_list in raw_recs.items():
                for item in rec_list[:k]:
                    rows.append(item)
            df_k = pd.DataFrame(rows)
            results[f"top_{k}"] = df_k
            csv_path = self._output_dir / f"recommendations_top{k}.csv"
            df_k.to_csv(csv_path, index=False)
            logger.info("Saved Top-%d recommendations to %s", k, csv_path)

        # 2. Build Ensemble Consensus Ranking
        ensemble_df = self._build_ensemble(raw_recs, max_k)
        results["ensemble"] = ensemble_df
        ens_csv = self._output_dir / "recommendations_ensemble.csv"
        ensemble_df.to_csv(ens_csv, index=False)

        # 3. Export Comprehensive JSON Bundle
        json_path = self._output_dir / "recommendations_explained.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_recs, f, indent=2)
        results["json_path"] = json_path

        return results

    def _build_ensemble(
        self, raw_recs: Dict[str, List[Dict[str, Any]]], max_k: int
    ) -> pd.DataFrame:
        """Combine rankings across all algorithms into a robust consensus ranking."""
        candidate_ranks: Dict[str, List[int]] = defaultdict(list)
        candidate_explanations: Dict[str, str] = {}

        for algo, recs in raw_recs.items():
            for item in recs:
                cand = item["candidate"]
                candidate_ranks[cand].append(item["rank"])
                if cand not in candidate_explanations:
                    candidate_explanations[cand] = item["explanation"]

        rows = []
        for cand, ranks in candidate_ranks.items():
            mean_rank = sum(ranks) / len(ranks)
            support_count = len(ranks)
            rows.append({
                "candidate": cand,
                "algorithms_supporting": support_count,
                "mean_rank": round(mean_rank, 2),
                "summary_rationale": candidate_explanations.get(cand, ""),
            })

        df = pd.DataFrame(rows).sort_values(
            ["algorithms_supporting", "mean_rank"], ascending=[False, True]
        )
        df.insert(0, "consensus_rank", range(1, len(df) + 1))
        return df.head(max_k)
