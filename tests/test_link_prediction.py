"""
tests/test_link_prediction.py
------------------------------
Unit tests for LinkPredictor and Evaluator in Phase 6.
"""

from __future__ import annotations

import networkx as nx
import pytest
from src.link_prediction import LinkPredictor
from src.evaluation import Evaluator


def _make_config():
    return {
        "prediction": {
            "algorithms": [
                "common_neighbors",
                "jaccard_coefficient",
                "adamic_adar",
                "preferential_attachment",
            ],
            "test_split_ratio": 0.2,
            "random_seed": 42,
        },
        "recommendations": {
            "top_k_values": [10, 20],
        }
    }


def _make_triangle_graph():
    G = nx.Graph()
    G.add_edges_from([
        ("a", "b"), ("b", "c"), ("c", "d"),
        ("b", "d"), ("d", "e"),
    ])
    return G


class TestLinkPredictionAlgorithms:
    def test_scores_all_algorithms(self):
        G = _make_triangle_graph()
        predictor = LinkPredictor(_make_config())
        candidates = [("a", "c"), ("a", "d"), ("c", "e")]
        scores = predictor.score_candidates(G, candidates)
        assert set(scores.keys()) == {
            "common_neighbors",
            "jaccard_coefficient",
            "adamic_adar",
            "preferential_attachment",
        }
        for algo, smap in scores.items():
            assert len(smap) == 3

    def test_recommend_excludes_existing_neighbors(self):
        G = _make_triangle_graph()
        predictor = LinkPredictor(_make_config())
        recs = predictor.recommend(G, "a", top_k=5)
        for algo, rec_list in recs.items():
            for item in rec_list:
                assert item["candidate"] != "b", "Recommended existing neighbor 'b'!"
                assert item["candidate"] != "a", "Recommended self!"
                assert "explanation" in item

    def test_evaluator_metrics(self):
        G = _make_triangle_graph()
        predictor = LinkPredictor(_make_config())
        evaluator = Evaluator(_make_config())
        pos = [("a", "c")]
        neg = [("a", "e")]
        scores = predictor.score_candidates(G, pos + neg)
        df = evaluator.evaluate(scores, pos, neg)
        assert "roc_auc" in df.columns
        assert "precision@10" in df.columns
        assert "f1_score" in df.columns
        assert len(df) == 4
