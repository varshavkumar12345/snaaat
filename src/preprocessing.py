"""
src/preprocessing.py
---------------------
Phase 5: Data Preparation, Network Cleaning, and Train-Test Splitting.

Guarantees:
  - Clean and normalized developer IDs & edge records.
  - Zero data leakage: positive test edges are strictly removed from G_train.
  - Reproducible negative sampling: uniform non-edges excluding all existing edges & self-loops.
  - Handles sparse networks and small components gracefully.
  - Exports node tables, edge tables, and split files.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]


class Preprocessor:
    """Prepares and splits the collaboration network for link prediction."""

    def __init__(self, config: dict) -> None:
        self._config = config
        pred_cfg = config.get("prediction", {})
        self._test_ratio: float = pred_cfg.get("test_split_ratio", 0.20)
        self._seed: int = pred_cfg.get("random_seed", 42)
        self._neg_ratio: float = pred_cfg.get("negative_sampling_ratio", 1.0)
        self._splits_dir = Path("data/splits")
        self._processed_dir = Path("data/processed")
        self._splits_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_graph(self, G: nx.Graph) -> nx.Graph:
        """
        Clean graph by removing self-loops, isolated nodes, and ensuring undirected edges.
        """
        G_clean = G.copy()
        # 1. Remove self-loops
        self_loops = list(nx.selfloop_edges(G_clean))
        if self_loops:
            G_clean.remove_edges_from(self_loops)
            logger.info("Removed %d self-loop(s).", len(self_loops))

        # 2. Canonicalize node string logins (lowercase, stripped)
        mapping = {n: str(n).lower().strip() for n in G_clean.nodes()}
        G_clean = nx.relabel_nodes(G_clean, mapping)

        return G_clean

    def prepare_and_split(
        self, G: nx.Graph
    ) -> Tuple[nx.Graph, List[Edge], List[Edge], List[Edge], Dict[str, pd.DataFrame]]:
        """
        Execute 80/20 train-test split and negative edge sampling.

        Returns
        -------
        G_train : nx.Graph
            Training graph with test edges removed.
        train_pos : list[tuple[str, str]]
            80% positive training edges.
        test_pos : list[tuple[str, str]]
            20% positive held-out test edges.
        test_neg : list[tuple[str, str]]
            Negative samples (non-edges) for testing.
        tables : dict[str, DataFrame]
            DataFrames for 'nodes', 'edges', 'splits'.
        """
        G_clean = self.clean_graph(G)
        edges = [tuple(sorted((u, v))) for u, v in G_clean.edges()]
        # Deduplicate edges
        unique_edges: List[Edge] = sorted(list(set(edges)))

        if len(unique_edges) < 5:
            raise ValueError(
                f"Graph contains only {len(unique_edges)} edge(s). "
                "Insufficient connectivity for a train-test split (minimum 5 required)."
            )

        # 1. Split positive edges (80% train, 20% test)
        if len(unique_edges) >= 10:
            train_pos, test_pos = train_test_split(
                unique_edges,
                test_size=self._test_ratio,
                random_state=self._seed,
                shuffle=True,
            )
        else:
            # Fallback for very small test graphs
            split_idx = max(1, int(len(unique_edges) * (1.0 - self._test_ratio)))
            train_pos = unique_edges[:split_idx]
            test_pos = unique_edges[split_idx:]

        # 2. Construct G_train without test edges (Prevent Data Leakage)
        G_train = G_clean.copy()
        G_train.remove_edges_from(test_pos)

        # 3. Generate negative samples (reproducible non-edges)
        n_neg = max(len(test_pos), int(len(test_pos) * self._neg_ratio))
        test_neg = self._sample_negative_edges(G_clean, n_neg)

        # 4. Generate structured summary tables
        nodes_df = self._build_nodes_table(G_clean)
        edges_df = self._build_edges_table(G_clean)

        tables = {
            "nodes": nodes_df,
            "edges": edges_df,
            "train_pos": pd.DataFrame(train_pos, columns=["u", "v"]),
            "test_pos": pd.DataFrame(test_pos, columns=["u", "v"]),
            "test_neg": pd.DataFrame(test_neg, columns=["u", "v"]),
        }

        # 5. Save all artifacts to disk
        self._save_artifacts(G_clean, G_train, train_pos, test_pos, test_neg, tables)

        return G_train, train_pos, test_pos, test_neg, tables

    def _sample_negative_edges(self, G: nx.Graph, n_samples: int) -> List[Edge]:
        """
        Sample non-edges uniformly at random without self-loops or existing connections.
        """
        rng = np.random.default_rng(self._seed)
        nodes = sorted(list(G.nodes()))
        num_nodes = len(nodes)

        existing_edges: Set[Edge] = set()
        for u, v in G.edges():
            existing_edges.add(tuple(sorted((u, v))))

        negative_edges: Set[Edge] = set()
        max_possible_edges = (num_nodes * (num_nodes - 1)) // 2
        max_possible_negatives = max_possible_edges - len(existing_edges)
        actual_samples_target = min(n_samples, max_possible_negatives)

        max_attempts = actual_samples_target * 50
        attempts = 0

        while len(negative_edges) < actual_samples_target and attempts < max_attempts:
            attempts += 1
            i, j = rng.choice(num_nodes, size=2, replace=False)
            u, v = nodes[i], nodes[j]
            pair = tuple(sorted((u, v)))
            if pair not in existing_edges and pair not in negative_edges:
                negative_edges.add(pair)

        return sorted(list(negative_edges))

    def _build_nodes_table(self, G: nx.Graph) -> pd.DataFrame:
        rows = []
        for n, data in G.nodes(data=True):
            rows.append({
                "username": n,
                "github_id": data.get("github_id", ""),
                "account_type": data.get("account_type", "User"),
                "is_seed": data.get("is_seed", False),
                "degree": G.degree(n),
                "qualifying_interactions": data.get("qualifying_interactions", 0),
            })
        return pd.DataFrame(rows).sort_values("degree", ascending=False)

    def _build_edges_table(self, G: nx.Graph) -> pd.DataFrame:
        rows = []
        for u, v, data in G.edges(data=True):
            rows.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "relationships": data.get("relationship_types", ""),
                "repositories": data.get("shared_repositories", ""),
            })
        return pd.DataFrame(rows).sort_values("weight", ascending=False)

    def _sanitize_for_graphml(self, G: nx.Graph) -> nx.Graph:
        """Create a copy of G with all attributes converted to GraphML-compatible primitives."""
        G_out = nx.Graph()
        # Copy graph-level attributes (exclude lists/dicts)
        for k, v in G.graph.items():
            if isinstance(v, (str, int, float, bool)):
                G_out.graph[k] = v

        for n, data in G.nodes(data=True):
            clean_data = {}
            for k, v in data.items():
                if isinstance(v, (list, set, tuple)):
                    clean_data[k] = ", ".join(map(str, v))
                elif isinstance(v, (str, int, float, bool)):
                    clean_data[k] = v
                else:
                    clean_data[k] = str(v)
            G_out.add_node(n, **clean_data)

        for u, v, data in G.edges(data=True):
            clean_data = {}
            for k, val in data.items():
                if isinstance(val, (list, set, tuple)):
                    clean_data[k] = ", ".join(map(str, val))
                elif isinstance(val, (str, int, float, bool)):
                    clean_data[k] = val
                else:
                    clean_data[k] = str(val)
            G_out.add_edge(u, v, **clean_data)

        return G_out

    def _save_artifacts(
        self,
        G_full: nx.Graph,
        G_train: nx.Graph,
        train_pos: List[Edge],
        test_pos: List[Edge],
        test_neg: List[Edge],
        tables: Dict[str, pd.DataFrame],
    ) -> None:
        """Persist graph models and tabular data to disk."""
        # Save node and edge tables
        tables["nodes"].to_csv(self._processed_dir / "nodes.csv", index=False)
        tables["edges"].to_csv(self._processed_dir / "edges.csv", index=False)

        # Save train/test split CSVs
        tables["train_pos"].to_csv(self._splits_dir / "train_pos.csv", index=False)
        tables["test_pos"].to_csv(self._splits_dir / "test_pos.csv", index=False)
        tables["test_neg"].to_csv(self._splits_dir / "test_neg.csv", index=False)

        # Save GraphML files safely
        G_full_s = self._sanitize_for_graphml(G_full)
        G_train_s = self._sanitize_for_graphml(G_train)
        nx.write_graphml(G_full_s, str(self._processed_dir / "network_full.graphml"))
        nx.write_graphml(G_train_s, str(self._splits_dir / "network_train.graphml"))

        # Save pickle dictionary
        splits_bundle = {
            "train_pos": train_pos,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "seed": self._seed,
            "test_ratio": self._test_ratio,
        }
        with open(self._splits_dir / "splits_bundle.pkl", "wb") as f:
            pickle.dump(splits_bundle, f)

        logger.info("Phase 5 splits and tables saved to %s and %s", self._splits_dir, self._processed_dir)
