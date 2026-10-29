"""
tests/test_graph_builder.py
----------------------------
Unit tests for the GraphBuilder module.

Uses synthetic DataFrames (no file I/O, no network calls).
"""

from __future__ import annotations

import pandas as pd
import networkx as nx
import pytest

from src.graph_builder import GraphBuilder, compute_network_stats


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_config(
    max_nodes: int = 100,
    max_depth: int = 2,
    min_edge_weight: int = 1,
    username: str = "alice",
) -> dict:
    return {
        "github": {"username": username},
        "graph": {
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "min_edge_weight": min_edge_weight,
        },
    }


def _make_events_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal events DataFrame from a list of dicts."""
    import pandas as pd
    base = {
        "event_id": "",
        "event_type": "PushEvent",
        "actor": "",
        "repo_owner": "",
        "repo_name": "",
        "repo_full_name": "",
        "created_at": pd.Timestamp("2024-10-01T12:00:00Z"),
        "payload_action": "",
        "extra_actors": [],
    }
    data = []
    for i, row in enumerate(rows):
        rec = {**base, "event_id": str(i), **row}
        data.append(rec)
    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


SIMPLE_EVENTS = [
    # alice → PushEvent on bob's repo → alice–bob edge
    {"actor": "alice", "repo_owner": "bob", "extra_actors": []},
    # carol reviews alice's PR → carol–alice edge
    {"actor": "carol", "repo_owner": "alice", "extra_actors": ["alice"]},
    # alice comments on dave's issue → alice–dave edge
    {"actor": "alice", "repo_owner": "dave", "extra_actors": ["dave"]},
]


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestGraphBuilderBasic:
    def test_builds_non_empty_graph(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config())
        G = builder.build(df)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_returns_undirected_graph(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config())
        G = builder.build(df)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_seed_user_in_graph(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        assert "alice" in G

    def test_alice_bob_edge_exists(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        assert G.has_edge("alice", "bob") or G.has_edge("bob", "alice")

    def test_edge_has_weight_attribute(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 1

    def test_empty_df_returns_empty_graph(self):
        df = pd.DataFrame()
        builder = GraphBuilder(_make_config())
        G = builder.build(df)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_max_nodes_respected(self):
        # Generate 20 distinct actors pushing to 20 distinct repos
        events = [
            {"actor": f"user{i}", "repo_owner": f"owner{i}", "extra_actors": []}
            for i in range(20)
        ]
        df = _make_events_df(events)
        builder = GraphBuilder(_make_config(max_nodes=5, username="user0"))
        G = builder.build(df)
        assert G.number_of_nodes() <= 5

    def test_min_edge_weight_filters_weak_edges(self):
        # alice→bob appears once; alice→carol appears three times
        events = [
            {"actor": "alice", "repo_owner": "bob", "extra_actors": []},
            {"actor": "alice", "repo_owner": "carol", "extra_actors": []},
            {"actor": "alice", "repo_owner": "carol", "extra_actors": []},
            {"actor": "alice", "repo_owner": "carol", "extra_actors": []},
        ]
        df = _make_events_df(events)
        builder = GraphBuilder(_make_config(min_edge_weight=2, username="alice"))
        G = builder.build(df)
        # alice–bob (weight 1) should be filtered out
        assert not G.has_edge("alice", "bob")
        # alice–carol (weight 3) should remain
        assert G.has_edge("alice", "carol")

    def test_no_self_loops(self):
        events = [
            {"actor": "alice", "repo_owner": "alice", "extra_actors": ["alice"]},
        ]
        df = _make_events_df(events)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        assert not G.has_edge("alice", "alice")

    def test_nodes_have_degree_centrality(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        for node in G.nodes():
            assert "degree_centrality" in G.nodes[node]

    def test_nodes_have_clustering_coefficient(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        for node in G.nodes():
            assert "clustering_coefficient" in G.nodes[node]

    def test_seed_marked(self):
        df = _make_events_df(SIMPLE_EVENTS)
        builder = GraphBuilder(_make_config(username="alice"))
        G = builder.build(df)
        if "alice" in G:
            assert G.nodes["alice"].get("is_seed") is True


class TestComputeNetworkStats:
    def test_empty_graph_returns_empty_dict(self):
        G = nx.Graph()
        stats = compute_network_stats(G)
        assert stats == {}

    def test_stats_contain_expected_keys(self):
        G = nx.path_graph(5)
        stats = compute_network_stats(G)
        assert "n_nodes" in stats
        assert "n_edges" in stats
        assert "density" in stats
        assert "avg_degree" in stats
        assert "n_components" in stats

    def test_n_nodes_correct(self):
        G = nx.path_graph(7)
        stats = compute_network_stats(G)
        assert stats["n_nodes"] == 7

    def test_n_edges_correct(self):
        G = nx.path_graph(5)
        stats = compute_network_stats(G)
        assert stats["n_edges"] == 4

    def test_density_range(self):
        G = nx.complete_graph(4)
        stats = compute_network_stats(G)
        assert 0.0 <= stats["density"] <= 1.0

