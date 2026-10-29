"""
tests/test_link_prediction.py
------------------------------
Unit tests for the LinkPredictor and Evaluator modules.

Uses small synthetic NetworkX graphs — no file I/O, no network calls.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.link_prediction import LinkPredictor
from src.evaluation import Evaluator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_config(algorithms=None) -> dict:
    return {
        "prediction": {
            "algorithms": algorithms or [
                "common_neighbors",
                "jaccard_coefficient",
                "adamic_adar",
                "preferential_attachment",
            ],
            "test_split_ratio": 0.2,
            "random_seed": 42,
            "negative_sampling_ratio": 1.0,
        }
    }


def _small_graph() -> nx.Graph:
    """Return a small graph suitable for testing:
         a - b - c - d
             |       |
             e ------+
    """
    G = nx.Graph()
    G.add_edges_from([
        ("a", "b"), ("b", "c"), ("c", "d"),
        ("b", "e"), ("d", "e"),
    ])
    return G


def _candidate_pairs(G: nx.Graph, n: int = 4):
    """Return a mix of existing and non-existing edges."""
    pos = list(G.edges())[:n]
    non_edges = list(nx.non_edges(G))[:n]
    return pos, non_edges


# ─── LinkPredictor tests ──────────────────────────────────────────────────────

class TestLinkPredictorScoreAll:
    def test_returns_dict_with_all_algorithms(self):
        G = _small_graph()
        test_pos, test_neg = _candidate_pairs(G)
        predictor = LinkPredictor(_make_config())
        scores = predictor.score_all(G, test_pos, test_neg)
        assert set(scores.keys()) == {
            "common_neighbors",
            "jaccard_coefficient",
            "adamic_adar",
            "preferential_attachment",
        }

    def test_each_algorithm_returns_scores_for_all_pairs(self):
        G = _small_graph()
        test_pos, test_neg = _candidate_pairs(G, n=3)
        predictor = LinkPredictor(_make_config())
        scores = predictor.score_all(G, test_pos, test_neg)
        n_pairs = len(test_pos) + len(test_neg)
        for algo, score_map in scores.items():
            assert len(score_map) == n_pairs, (
                f"{algo} returned {len(score_map)} scores, expected {n_pairs}"
            )

    def test_scores_are_non_negative(self):
        G = _small_graph()
        test_pos, test_neg = _candidate_pairs(G)
        predictor = LinkPredictor(_make_config())
        scores = predictor.score_all(G, test_pos, test_neg)
        for algo, score_map in scores.items():
            for (u, v), s in score_map.items():
                assert s >= 0.0, f"{algo}: score {s} for ({u},{v}) is negative."

    def test_invalid_algorithm_raises(self):
        G = _small_graph()
        test_pos, test_neg = _candidate_pairs(G)
        predictor = LinkPredictor(_make_config(algorithms=["nonexistent_algo"]))
        # score_all catches exceptions internally and logs them; result has no keys
        scores = predictor.score_all(G, test_pos, test_neg)
        # The algorithm fails, so it's missing from results (not in output dict)
        assert "nonexistent_algo" not in scores


class TestCommonNeighbors:
    def test_known_common_neighbors(self):
        G = nx.Graph()
        # a-b, a-c, b-c → a and b share neighbour c; a and c share b
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "c")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._common_neighbors(G, [("a", "b")])
        # a and b share c → 1 common neighbour
        assert scores[("a", "b")] == 1.0

    def test_no_common_neighbors(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("c", "d")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._common_neighbors(G, [("a", "c")])
        assert scores[("a", "c")] == 0.0

    def test_node_not_in_graph_returns_zero(self):
        G = _small_graph()
        predictor = LinkPredictor(_make_config())
        scores = predictor._common_neighbors(G, [("a", "MISSING")])
        assert scores[("a", "MISSING")] == 0.0


class TestJaccardCoefficient:
    def test_complete_overlap_is_one(self):
        # If N(u) == N(v), Jaccard = 1
        G = nx.Graph()
        G.add_edges_from([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._jaccard_coefficient(G, [("a", "b")])
        # Jaccard = |{x,y}| / |{x,y}| = 1.0
        assert abs(scores[("a", "b")] - 1.0) < 1e-6

    def test_no_overlap_is_zero(self):
        G = nx.Graph()
        G.add_edges_from([("a", "x"), ("b", "y")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._jaccard_coefficient(G, [("a", "b")])
        assert scores[("a", "b")] == 0.0

    def test_scores_between_0_and_1(self):
        G = _small_graph()
        pairs = list(nx.non_edges(G))[:10]
        predictor = LinkPredictor(_make_config())
        scores = predictor._jaccard_coefficient(G, pairs)
        for (u, v), s in scores.items():
            assert 0.0 <= s <= 1.0, f"Jaccard score {s} out of [0, 1]"


class TestAdamicAdar:
    def test_score_positive_for_shared_neighbours(self):
        G = nx.Graph()
        G.add_edges_from([("a", "c"), ("b", "c"), ("c", "d")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._adamic_adar(G, [("a", "b")])
        # c is a common neighbour with degree > 1
        assert scores[("a", "b")] > 0.0

    def test_score_zero_for_no_shared_neighbours(self):
        G = nx.Graph()
        G.add_edges_from([("a", "x"), ("b", "y")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._adamic_adar(G, [("a", "b")])
        assert scores[("a", "b")] == 0.0


class TestPreferentialAttachment:
    def test_hub_node_gets_higher_score(self):
        G = nx.star_graph(5)  # node 0 is hub with degree 5
        predictor = LinkPredictor(_make_config())
        # Pair involving hub (0): score = 5 * deg(other)
        # Pair not involving hub: score = 1 * 1 = 1
        scores_hub = predictor._preferential_attachment(G, [(0, 1)])
        scores_leaf = predictor._preferential_attachment(G, [(1, 2)])
        assert scores_hub[(0, 1)] >= scores_leaf[(1, 2)]

    def test_isolated_node_gives_zero(self):
        G = nx.Graph()
        G.add_node("isolated")
        G.add_edges_from([("a", "b")])
        predictor = LinkPredictor(_make_config())
        scores = predictor._preferential_attachment(G, [("a", "isolated")])
        assert scores[("a", "isolated")] == 0.0


# ─── Evaluator tests ──────────────────────────────────────────────────────────

class TestEvaluator:
    def _run_eval(self, G, algo="common_neighbors"):
        predictor = LinkPredictor(_make_config(algorithms=[algo]))
        pos = list(G.edges())[:4]
        neg = list(nx.non_edges(G))[:4]
        all_scores = predictor.score_all(G, pos, neg)
        evaluator = Evaluator(_make_config())
        return evaluator.evaluate(all_scores, pos, neg)

    def test_returns_dataframe_with_algorithm_index(self):
        G = _small_graph()
        df = self._run_eval(G)
        assert "common_neighbors" in df.index

    def test_auc_roc_in_valid_range(self):
        G = _small_graph()
        df = self._run_eval(G)
        row = df.loc["common_neighbors"]
        auc = row["auc_roc"]
        assert 0.0 <= float(auc) <= 1.0 or str(auc) == "nan"

    def test_all_metric_columns_present(self):
        G = _small_graph()
        df = self._run_eval(G)
        for col in ["auc_roc", "avg_precision", "precision", "recall", "f1"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_evaluates_multiple_algorithms(self):
        G = _small_graph()
        predictor = LinkPredictor(_make_config())
        pos = list(G.edges())[:4]
        neg = list(nx.non_edges(G))[:4]
        all_scores = predictor.score_all(G, pos, neg)
        evaluator = Evaluator(_make_config())
        df = evaluator.evaluate(all_scores, pos, neg)
        assert len(df) == 4  # one row per algorithm


class TestLinkPredictorRecommend:
    def test_returns_dict_with_all_algorithms(self):
        G = _small_graph()
        predictor = LinkPredictor(_make_config())
        recs = predictor.recommend(G, "a", top_k=3)
        assert set(recs.keys()) == {
            "common_neighbors",
            "jaccard_coefficient",
            "adamic_adar",
            "preferential_attachment",
        }

    def test_recommendations_are_non_neighbours(self):
        G = _small_graph()
        neighbours_of_a = set(G.neighbors("a")) | {"a"}
        predictor = LinkPredictor(_make_config())
        recs = predictor.recommend(G, "a", top_k=10)
        for algo, ranked in recs.items():
            for (u, v), score in ranked:
                other = v if u == "a" else u
                assert other not in neighbours_of_a, (
                    f"{algo}: recommended neighbour {other}"
                )

    def test_top_k_respected(self):
        G = nx.complete_graph(20)  # dense graph, many non-neighbours after removal
        G = nx.relabel_nodes(G, {i: f"u{i}" for i in range(20)})
        # Remove some edges to create non-edges
        edges = list(G.edges())
        G.remove_edges_from(edges[:50])

        predictor = LinkPredictor(_make_config())
        recs = predictor.recommend(G, "u0", top_k=5)
        for algo, ranked in recs.items():
            assert len(ranked) <= 5

    def test_unknown_seed_returns_empty(self):
        G = _small_graph()
        predictor = LinkPredictor(_make_config())
        recs = predictor.recommend(G, "NONEXISTENT_USER", top_k=5)
        assert recs == {}

