"""
src/link_prediction.py
-----------------------
Four graph-theoretic link-prediction algorithms.

All algorithms operate on a NetworkX graph (the *training* graph) and return
scored candidate pairs, suitable for evaluation against held-out test edges.

Algorithms
~~~~~~~~~~
1. **Common Neighbors** — |N(u) ∩ N(v)|
2. **Jaccard Coefficient** — |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
3. **Adamic-Adar Index** — Σ_{w ∈ N(u)∩N(v)} 1 / log(|N(w)|)
4. **Preferential Attachment** — |N(u)| × |N(v)|

Usage
~~~~~
    from src import load_config
    from src.link_prediction import LinkPredictor

    cfg = load_config()
    predictor = LinkPredictor(cfg)
    scores = predictor.score_all(G_train, test_pos, test_neg)
    # scores is a dict: algorithm_name → {(u,v): score, …}
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Type aliases
Edge = Tuple[str, str]
ScoreMap = Dict[Edge, float]
AllScores = Dict[str, ScoreMap]


class LinkPredictor:
    """Compute link-prediction scores for a set of candidate edges."""

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_all(
        self,
        G: nx.Graph,
        test_pos: List[Edge],
        test_neg: List[Edge],
    ) -> AllScores:
        """Score all candidate edges with every configured algorithm.

        Parameters
        ----------
        G:
            Training graph (test edges removed).
        test_pos:
            Held-out positive (existing) edge pairs.
        test_neg:
            Sampled negative (non-existing) edge pairs.

        Returns
        -------
        dict
            ``{algorithm_name: {(u, v): score}}`` for every algorithm.
        """
        candidates: List[Edge] = test_pos + test_neg
        all_scores: AllScores = {}

        for algo in self._algorithms:
            logger.info("Running link-prediction algorithm: %s …", algo)
            try:
                scores = self._dispatch(algo, G, candidates)
                all_scores[algo] = scores
                logger.info("  → %d pairs scored.", len(scores))
            except Exception as exc:  # noqa: BLE001
                logger.error("Algorithm %s failed: %s", algo, exc)

        return all_scores

    def recommend(
        self,
        G: nx.Graph,
        seed_user: str,
        top_k: int = 20,
    ) -> Dict[str, List[Tuple[Edge, float]]]:
        """Generate Top-K collaboration recommendations for *seed_user*.

        Only considers non-neighbours of *seed_user* that are present in the
        graph (true link-prediction candidates).

        Parameters
        ----------
        G:
            Collaboration graph.
        seed_user:
            The user for whom to generate recommendations.
        top_k:
            Number of recommendations to return per algorithm.

        Returns
        -------
        dict
            ``{algorithm_name: [(edge, score), …]}`` sorted descending.
        """
        if seed_user not in G:
            logger.warning("Seed user '%s' not in graph.", seed_user)
            return {}

        # Candidate pairs: seed ↔ non-neighbour nodes
        neighbors = set(G.neighbors(seed_user)) | {seed_user}
        candidates: List[Edge] = [
            (seed_user, v) for v in G.nodes() if v not in neighbors
        ]

        if not candidates:
            logger.warning("No candidate pairs found for '%s'.", seed_user)
            return {}

        results: Dict[str, List[Tuple[Edge, float]]] = {}
        for algo in self._algorithms:
            scores = self._dispatch(algo, G, candidates)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results[algo] = ranked[:top_k]

        return results

    # ------------------------------------------------------------------
    # Algorithm dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, algorithm: str, G: nx.Graph, candidates: List[Edge]
    ) -> ScoreMap:
        """Route to the correct scoring function."""
        dispatch = {
            "common_neighbors": self._common_neighbors,
            "jaccard_coefficient": self._jaccard_coefficient,
            "adamic_adar": self._adamic_adar,
            "preferential_attachment": self._preferential_attachment,
        }
        fn = dispatch.get(algorithm)
        if fn is None:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")
        return fn(G, candidates)

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _common_neighbors(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        """|N(u) ∩ N(v)| for each candidate pair."""
        scores: ScoreMap = {}
        for u, v in candidates:
            if u in G and v in G:
                cn = len(list(nx.common_neighbors(G, u, v)))
                scores[(u, v)] = float(cn)
            else:
                scores[(u, v)] = 0.0
        return scores

    @staticmethod
    def _jaccard_coefficient(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        """Jaccard coefficient for each candidate pair.

        |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
        """
        scores: ScoreMap = {}
        # Use NetworkX built-in for correctness
        nx_scores = dict(nx.jaccard_coefficient(G, candidates))
        for pair, score in nx_scores.items():
            scores[pair] = score
        # Fill any missing pairs
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores

    @staticmethod
    def _adamic_adar(G: nx.Graph, candidates: List[Edge]) -> ScoreMap:
        """Adamic-Adar index for each candidate pair.

        Σ_{w ∈ N(u)∩N(v)} 1 / log(|N(w)|)
        """
        scores: ScoreMap = {}
        nx_scores = dict(nx.adamic_adar_index(G, candidates))
        for pair, score in nx_scores.items():
            scores[pair] = score
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores

    @staticmethod
    def _preferential_attachment(
        G: nx.Graph, candidates: List[Edge]
    ) -> ScoreMap:
        """Preferential attachment score: |N(u)| × |N(v)|."""
        scores: ScoreMap = {}
        nx_scores = dict(nx.preferential_attachment(G, candidates))
        for pair, score in nx_scores.items():
            scores[pair] = float(score)
        for pair in candidates:
            if pair not in scores:
                scores[pair] = 0.0
        return scores

