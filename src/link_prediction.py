"""
src/link_prediction.py
-----------------------
Phase 6: Topological Link-Prediction Algorithms and Recommendation Explainer.

Algorithms:
  1. Common Neighbors (CN)
  2. Jaccard Coefficient (JC)
  3. Adamic-Adar Index (AA)
  4. Preferential Attachment (PA)

Features:
  - Generates structural explanations for each recommended pair.
  - Excludes existing neighbors, self-loops, and bots.
  - Scores are treated as ranking heuristics, not calibrated probabilities.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]
ScoreMap = Dict[Edge, float]
AllScores = Dict[str, ScoreMap]


class LinkPredictor:
    """Compute link prediction scores and generate explainable recommendations."""

    def __init__(self, config: dict) -> None:
        self._config = config
        pred_cfg = config.get("prediction", {})
        self._algorithms: List[str] = pred_cfg.get(
            "algorithms",
            [
                "common_neighbors",
                "jaccard_coefficient",
                "adamic_adar",
                "preferential_attachment",
            ],
        )

    def score_candidates(
        self,
        G: nx.Graph,
        candidate_pairs: List[Edge],
    ) -> AllScores:
        """Score candidate pairs with all 4 algorithms on training graph G."""
        all_scores: AllScores = {}
        for algo in self._algorithms:
            scores = self._dispatch(algo, G, candidate_pairs)
            all_scores[algo] = scores
        return all_scores

    def recommend(
        self,
        G: nx.Graph,
        seed_user: str,
        top_k: int = 20,
        exclude_users: Optional[Set[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate ranked recommendations for seed_user excluding existing neighbors,
        self-loops, and excluded users/bots.
        """
        seed_user = seed_user.lower().strip()
        if seed_user not in G:
            logger.warning("Seed user '%s' not present in graph.", seed_user)
            return {}

        existing_neighbors = set(G.neighbors(seed_user)) | {seed_user}
        if exclude_users:
            existing_neighbors |= exclude_users

        # Candidate pool: nodes in G not already connected to seed_user
        candidate_nodes = [n for n in G.nodes() if n not in existing_neighbors]
        if not candidate_nodes:
            logger.warning("No candidate non-neighbors available for '%s'.", seed_user)
            return {}

        candidate_pairs = [(seed_user, target) for target in candidate_nodes]
        all_scores = self.score_candidates(G, candidate_pairs)

        recommendations: Dict[str, List[Dict[str, Any]]] = {}

        for algo, score_map in all_scores.items():
            sorted_pairs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            algo_recs = []
            for rank, ((u, v), score) in enumerate(sorted_pairs[:top_k], start=1):
                target = v if u == seed_user else u
                explanation = self._explain_prediction(G, seed_user, target, algo, score)
                algo_recs.append({
                    "rank": rank,
                    "algorithm": algo,
                    "seed_user": seed_user,
                    "candidate": target,
                    "score": round(float(score), 6),
                    "explanation": explanation,
                })
            recommendations[algo] = algo_recs

        return recommendations

    def _explain_prediction(
        self,
        G: nx.Graph,
        u: str,
        v: str,
        algo: str,
        score: float,
    ) -> str:
        """Produce graph-theoretic explanation for a recommendation."""
        cn = list(nx.common_neighbors(G, u, v)) if (u in G and v in G) else []
        deg_u = G.degree(u) if u in G else 0
        deg_v = G.degree(v) if v in G else 0

        if algo == "common_neighbors":
            return (
                f"Shares {len(cn)} mutual collaborator(s) "
                f"({', '.join(cn[:3]) + ('...' if len(cn) > 3 else '') if cn else 'none'})."
            )
        elif algo == "jaccard_coefficient":
            union_size = len(set(G.neighbors(u)) | set(G.neighbors(v))) if (u in G and v in G) else 0
            return (
                f"Relative neighborhood overlap of {score:.3f} "
                f"({len(cn)} shared out of {union_size} total distinct neighbors)."
            )
        elif algo == "adamic_adar":
            return (
                f"Connects via {len(cn)} shared collaborator(s) weighted by inverse log degree "
                f"(prioritizes niche, specific co-collaborators)."
            )
        elif algo == "preferential_attachment":
            return (
                f"High network visibility and activity: product of degrees "
                f"({deg_u} × {deg_v} = {deg_u * deg_v})."
            )
        return "Topological affinity in collaboration graph."

    def _dispatch(self, algorithm: str, G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        dispatch = {
            "common_neighbors": self._common_neighbors,
            "jaccard_coefficient": self._jaccard_coefficient,
            "adamic_adar": self._adamic_adar,
            "preferential_attachment": self._preferential_attachment,
        }
        fn = dispatch.get(algorithm)
        if not fn:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")
        return fn(G, candidates)

    @staticmethod
    def _common_neighbors(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        scores: ScoreMap = {}
        for u, v in candidates:
            if u in G and v in G:
                scores[(u, v)] = float(len(list(nx.common_neighbors(G, u, v))))
            else:
                scores[(u, v)] = 0.0
        return scores

    @staticmethod
    def _jaccard_coefficient(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        valid = [(u, v) for u, v in candidates if u in G and v in G]
        scores: ScoreMap = {
            (u, v): float(p)
            for u, v, p in nx.jaccard_coefficient(G, valid)
        }
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores

    @staticmethod
    def _adamic_adar(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        valid = [(u, v) for u, v in candidates if u in G and v in G]
        scores: ScoreMap = {
            (u, v): float(p)
            for u, v, p in nx.adamic_adar_index(G, valid)
        }
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores

    @staticmethod
    def _preferential_attachment(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        valid = [(u, v) for u, v in candidates if u in G and v in G]
        scores: ScoreMap = {
            (u, v): float(p)
            for u, v, p in nx.preferential_attachment(G, valid)
        }
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores
