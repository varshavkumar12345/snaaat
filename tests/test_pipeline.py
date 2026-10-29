"""
Unit tests and pipeline validation using synthetic fixtures and dataset records.
"""
import pytest
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

from src.graph_builder import (
    is_bot, clean_interactions, aggregate_edges,
    resolve_starting_developer, construct_graph, compute_network_statistics
)
from src.link_prediction import (
    train_test_split_edges,
    compute_common_neighbors, compute_jaccard_coefficient,
    compute_adamic_adar, compute_preferential_attachment,
    score_pairs
)
from src.evaluation import compute_precision_recall_at_k, evaluate_algorithm
from src.recommendation import generate_collaborator_recommendations
from src.config import DELIVERABLES_DIR, PROCESSED_DATA_DIR

@pytest.fixture
def synthetic_graph():
    """
    Synthetic graph with known topology:
    Nodes: A, B, C, D, E
    Edges: (A, B), (A, C), (B, C), (B, D), (C, D), (D, E)
    """
    G = nx.Graph()
    edges = [
        ("A", "B", {"weight": 5}),
        ("A", "C", {"weight": 3}),
        ("B", "C", {"weight": 2}),
        ("B", "D", {"weight": 4}),
        ("C", "D", {"weight": 1}),
        ("D", "E", {"weight": 2})
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    return G

def test_bot_filtering():
    """Test that bot usernames are detected and filtered, while human users are preserved."""
    assert is_bot("github-actions[bot]") is True
    assert is_bot("dependabot") is True
    assert is_bot("codecov[bot]") is True
    assert is_bot("regro-cf-autotick-bot") is True
    assert is_bot("web-flow") is True
    assert is_bot(None) is True
    assert is_bot("") is True
    # Human maintainers
    assert is_bot("jreback") is False
    assert is_bot("wesm") is False
    assert is_bot("thomasjpfan") is False

def test_clean_interactions():
    """Test hygiene filtering removes self-loops, bots, and empty records."""
    raw_data = [
        {"source": "alice", "target": "bob", "interaction_type": "pr_review"},
        {"source": "alice", "target": "alice", "interaction_type": "pr_comment"},  # self-loop
        {"source": "github-actions[bot]", "target": "bob", "interaction_type": "pr_merge"},  # bot
        {"source": "alice", "target": "codecov[bot]", "interaction_type": "pr_comment"},  # bot
        {"source": "bob", "target": "charlie", "interaction_type": "issue_comment"}
    ]
    df = pd.DataFrame(raw_data)
    cleaned = clean_interactions(df)
    assert len(cleaned) == 2
    assert set(cleaned["source"]) == {"alice", "bob"}
    assert set(cleaned["target"]) == {"bob", "charlie"}

def test_link_prediction_algorithms_on_fixture(synthetic_graph):
    """
    Test 4 link prediction heuristics against analytical expectations on synthetic graph.
    Nodes: A, B, C, D, E
    Pair (A, D): non-edge.
      N(A) = {B, C}
      N(D) = {B, C, E}
      Common neighbors = {B, C} -> count = 2
      deg(B) = 3 (A, C, D), deg(C) = 3 (A, B, D)
      AA(A, D) = 1/log(3) + 1/log(3) = 2 / log(3) ≈ 1.8205
      deg(A) = 2, deg(D) = 3 -> PA(A, D) = 6
      Union = {B, C, E} -> size 3 -> JC(A, D) = 2 / 3 ≈ 0.6667
    """
    G = synthetic_graph
    cn = compute_common_neighbors(G, "A", "D")
    assert cn == 2.0

    jc = compute_jaccard_coefficient(G, "A", "D")
    assert pytest.approx(jc, 0.01) == 2.0 / 3.0

    aa = compute_adamic_adar(G, "A", "D")
    import math
    expected_aa = 2.0 / math.log(3)
    assert pytest.approx(aa, 0.01) == expected_aa

    pa = compute_preferential_attachment(G, "A", "D")
    assert pa == 6.0

def test_train_test_split_integrity(synthetic_graph):
    """
    Verify strict train-test separation:
    - Test edges removed from G_train to prevent data leakage.
    - Negative edges have zero overlap with observed edges and no self-loops.
    """
    G = synthetic_graph
    G_train, train_pos, test_pos, train_neg, test_neg = train_test_split_edges(
        G, test_ratio=0.33, seed=42
    )
    # Check partition
    assert len(train_pos) + len(test_pos) == G.number_of_edges()
    assert G_train.number_of_edges() == len(train_pos)

    # Check zero data leakage: NO test edge is present in G_train
    for u, v in test_pos:
        assert not G_train.has_edge(u, v)

    # Check negative samples
    for u, v in test_neg:
        assert u != v
        assert not G.has_edge(u, v)

def test_evaluation_metrics():
    """Verify calculation of Precision@K, Recall@K, and ROC-AUC."""
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.2, 0.1])
    prec, rec, f1 = compute_precision_recall_at_k(y_true, y_scores, k=2)
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0

    metrics = evaluate_algorithm(y_true, y_scores, k_list=[2])
    assert metrics["ROC-AUC"] == 1.0
    assert metrics["Precision@2"] == 1.0

def test_developer_resolution():
    """Test resolution of valid developer vs non-existent developer."""
    df_edges = pd.DataFrame([
        {"source": "alice", "target": "bob", "weight": 5},
        {"source": "alice", "target": "charlie", "weight": 2}
    ])
    res_valid = resolve_starting_developer(df_edges, "alice")
    assert res_valid["resolved"] is True
    assert res_valid["degree"] == 2
    assert res_valid["total_weight"] == 7

    res_invalid = resolve_starting_developer(df_edges, "nonexistent_dev")
    assert res_invalid["resolved"] is False
    assert res_invalid["degree"] == 0

def test_recommendation_filtering(synthetic_graph):
    """
    Verify recommendation engine excludes self, existing neighbors, and bots.
    Target: A
    Existing neighbors: B, C
    Candidate: D (common neighbors B, C), E (no common neighbors)
    """
    G = synthetic_graph
    recs = generate_collaborator_recommendations(G, "A", top_k=5, primary_algo="Adamic-Adar")
    recommended_users = [r["candidate_user"] for r in recs]
    assert "A" not in recommended_users  # no self
    assert "B" not in recommended_users  # no existing neighbor
    assert "C" not in recommended_users  # no existing neighbor
    assert "D" in recommended_users      # valid candidate
    # D should be ranked #1 because it has common neighbors {B, C}
    assert recs[0]["candidate_user"] == "D"
    assert recs[0]["common_neighbors_count"] == 2

def test_deliverables_exist():
    """Verify all primary deliverables are generated on disk."""
    required_files = [
        "developer_network.graphml",
        "interactive_network.html",
        "link_prediction_evaluation.csv",
        "network_ego.png",
        "network_ego.svg",
        "network_overall.png",
        "network_overall.svg",
        "network_predictions.png",
        "network_predictions.svg",
        "roc_curves.png",
        "roc_curves.svg",
        "NumFOCUS_Collaboration_Report.pdf",
        "NumFOCUS_Collaboration_Presentation.pptx",
        "NumFOCUS_Collaboration_Presentation.pdf",
        "viva_questions_and_answers.md"
    ]
    for filename in required_files:
        fpath = DELIVERABLES_DIR / filename
        assert fpath.exists(), f"Deliverable missing: {filename}"
        assert fpath.stat().st_size > 0, f"Deliverable is empty: {filename}"

