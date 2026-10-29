"""
tests/test_graph_builder.py
----------------------------
Unit tests for GraphBuilder ego network construction and provenance tracking.
"""

from __future__ import annotations

import pandas as pd
import networkx as nx
import pytest
from src.graph_builder import GraphBuilder, compute_network_stats


def _make_config(username="alice", max_depth=2, max_nodes=50):
    return {
        "github": {"username": username},
        "graph": {
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "min_edge_weight": 1,
        },
    }


def _make_df():
    data = [
        {"source_id": "1", "source_login": "alice", "target_id": "2", "target_login": "bob",
         "relationship": "pr_review", "repo": "repoA", "event_id": "e1"},
        {"source_id": "2", "source_login": "bob", "target_id": "3", "target_login": "carol",
         "relationship": "issue_comment", "repo": "repoB", "event_id": "e2"},
        {"source_id": "3", "source_login": "carol", "target_id": "4", "target_login": "dave",
         "relationship": "pr_merge", "repo": "repoC", "event_id": "e3"},
    ]
    return pd.DataFrame(data)


class TestGraphBuilder:
    def test_one_hop_expansion(self):
        df = _make_df()
        builder = GraphBuilder(_make_config(username="alice", max_depth=1))
        G, status = builder.build_network(df)
        assert status["seed_found"] is True
        assert set(G.nodes()) == {"alice", "bob"}
        assert G.has_edge("alice", "bob")
        assert "pr_review" in G["alice"]["bob"]["relationship_types"]

    def test_two_hop_expansion(self):
        df = _make_df()
        builder = GraphBuilder(_make_config(username="alice", max_depth=2))
        G, status = builder.build_network(df)
        # alice -> bob (1-hop) -> carol (2-hop). dave (3-hop) excluded
        assert set(G.nodes()) == {"alice", "bob", "carol"}
        assert not G.has_node("dave")

    def test_node_cap_respected(self):
        df = _make_df()
        builder = GraphBuilder(_make_config(username="alice", max_depth=2, max_nodes=2))
        G, status = builder.build_network(df)
        assert G.number_of_nodes() <= 2

    def test_missing_seed_explanation(self):
        df = _make_df()
        builder = GraphBuilder(_make_config(username="nonexistent", max_depth=1))
        G, status = builder.build_network(df)
        assert G.number_of_nodes() == 0
        assert status["insufficient_data"] is True
        assert "nonexistent" in status["reason"]
