"""
Train-test graph partitioning, negative edge sampling, and link prediction algorithms.
"""
import math
import random
from typing import List, Tuple, Set, Dict, Any, Optional
import networkx as nx
import numpy as np

def train_test_split_edges(
    G: nx.Graph,
    test_ratio: float = 0.20,
    seed: int = 42,
    ensure_connected: bool = False
) -> Tuple[nx.Graph, List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split positive edges into 80% train and 20% test.
    Construct training graph G_train without test edges to strictly prevent data leakage.
    Generate equal number of negative samples (non-edges) for train and test sets.
    """
    rng = random.Random(seed)
    edges = list(G.edges())
    # Sort for deterministic order before shuffling
    edges.sort()
    rng.shuffle(edges)

    num_test = int(len(edges) * test_ratio)
    test_edges_pos = edges[:num_test]
    train_edges_pos = edges[num_test:]

    # Construct training graph G_train
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())  # preserve all nodes
    for u, v in train_edges_pos:
        edge_data = G.get_edge_data(u, v, default={})
        G_train.add_edge(u, v, **edge_data)

    # Negative sampling: pairs (u, v) with u < v that are NOT in G
    existing_edges = set()
    for u, v in G.edges():
        existing_edges.add(tuple(sorted([u, v])))

    nodes = sorted(list(G.nodes()))
    num_nodes = len(nodes)
    total_non_edges_possible = (num_nodes * (num_nodes - 1)) // 2 - len(existing_edges)

    # Required negatives: train negatives and test negatives
    needed_negatives = len(train_edges_pos) + len(test_edges_pos)
    needed_negatives = min(needed_negatives, total_non_edges_possible)

    negatives_set: Set[Tuple[str, str]] = set()
    max_attempts = needed_negatives * 30
    attempts = 0

    while len(negatives_set) < needed_negatives and attempts < max_attempts:
        attempts += 1
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        pair = tuple(sorted([u, v]))
        if pair not in existing_edges and pair not in negatives_set:
            negatives_set.add(pair)

    negatives_list = list(negatives_set)
    rng.shuffle(negatives_list)

    num_test_neg = min(len(test_edges_pos), len(negatives_list) // 2) if negatives_list else 0
    test_edges_neg = negatives_list[:num_test_neg]
    train_edges_neg = negatives_list[num_test_neg:num_test_neg + len(train_edges_pos)]

    return G_train, train_edges_pos, test_edges_pos, train_edges_neg, test_edges_neg

def compute_common_neighbors(G: nx.Graph, u: str, v: str) -> float:
    """Calculate Common Neighbors score: |N(u) ∩ N(v)|."""
    if u not in G or v not in G:
        return 0.0
    u_nbrs = set(G.neighbors(u))
    v_nbrs = set(G.neighbors(v))
    return float(len(u_nbrs.intersection(v_nbrs)))

def compute_jaccard_coefficient(G: nx.Graph, u: str, v: str) -> float:
    """Calculate Jaccard Coefficient: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|."""
    if u not in G or v not in G:
        return 0.0
    u_nbrs = set(G.neighbors(u))
    v_nbrs = set(G.neighbors(v))
    union_len = len(u_nbrs.union(v_nbrs))
    if union_len == 0:
        return 0.0
    return float(len(u_nbrs.intersection(v_nbrs)) / union_len)

def compute_adamic_adar(G: nx.Graph, u: str, v: str) -> float:
    """
    Calculate Adamic-Adar Index: sum_{w in N(u) ∩ N(v)} 1 / log(deg(w)).
    Ignores common neighbors with degree <= 1 to avoid division by zero or negative values.
    """
    if u not in G or v not in G:
        return 0.0
    u_nbrs = set(G.neighbors(u))
    v_nbrs = set(G.neighbors(v))
    common = u_nbrs.intersection(v_nbrs)
    score = 0.0
    for w in common:
        deg = G.degree(w)
        if deg > 1:
            score += 1.0 / math.log(deg)
    return float(score)

def compute_preferential_attachment(G: nx.Graph, u: str, v: str) -> float:
    """Calculate Preferential Attachment score: deg(u) * deg(v)."""
    if u not in G or v not in G:
        return 0.0
    return float(G.degree(u) * G.degree(v))

ALGORITHMS = {
    "Common Neighbors": compute_common_neighbors,
    "Jaccard Coefficient": compute_jaccard_coefficient,
    "Adamic-Adar": compute_adamic_adar,
    "Preferential Attachment": compute_preferential_attachment
}

def score_pairs(G: nx.Graph, pairs: List[Tuple[str, str]], algorithm: str) -> np.ndarray:
    """Compute scores for a list of node pairs using the specified heuristic."""
    func = ALGORITHMS.get(algorithm)
    if not func:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    scores = np.array([func(G, u, v) for u, v in pairs], dtype=float)
    return scores

