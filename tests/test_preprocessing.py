"""
tests/test_preprocessing.py
---------------------------
Unit tests for Preprocessor in Phase 5.
"""

from __future__ import annotations

import networkx as nx
import pytest
from src.preprocessing import Preprocessor


def _make_config():
    return {
        "prediction": {
            "test_split_ratio": 0.20,
            "random_seed": 42,
            "negative_sampling_ratio": 1.0,
        }
    }


class TestPreprocessor:
    def test_clean_graph_removes_self_loops(self):
        G = nx.Graph()
        G.add_edge("alice", "alice")
        G.add_edge("alice", "bob")
        prep = Preprocessor(_make_config())
        clean = prep.clean_graph(G)
        assert not clean.has_edge("alice", "alice")
        assert clean.has_edge("alice", "bob")

    def test_prepare_and_split_prevents_leakage(self):
        G = nx.complete_graph(10)
        mapping = {i: f"user_{i}" for i in range(10)}
        G = nx.relabel_nodes(G, mapping)
        prep = Preprocessor(_make_config())
        G_train, train_pos, test_pos, test_neg, tables = prep.prepare_and_split(G)

        # 1. Total positive edges preserved
        assert len(train_pos) + len(test_pos) == G.number_of_edges()

        # 2. Test edges must NOT be in G_train (zero leakage)
        for u, v in test_pos:
            assert not G_train.has_edge(u, v)

        # 3. Negative edges must NOT be in original G
        for u, v in test_neg:
            assert not G.has_edge(u, v)

        # 4. Tables exist
        assert "nodes" in tables
        assert "edges" in tables

    def test_sparse_graph_under_threshold_raises(self):
        G = nx.path_graph(3)  # only 2 edges
        prep = Preprocessor(_make_config())
        with pytest.raises(ValueError, match="Insufficient connectivity"):
            prep.prepare_and_split(G)

