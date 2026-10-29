"""
src/preprocessing.py
---------------------
Graph preprocessing, cleaning, and train/test splitting for link prediction.

Responsibilities
~~~~~~~~~~~~~~~~
* Remove self-loops and isolated nodes (optional).
* Filter out weak edges below a configurable weight threshold.
* Produce positive edge samples (existing edges) and negative edge samples
  (non-existing node pairs) for evaluation.
* Split edges into train and test sets (stratified by weight bucket).
* Persist splits to disk for reproducibility.

Usage
~~~~~
    from src import load_config
    from src.preprocessing import Preprocessor

    cfg = load_config()
    prep = Preprocessor(cfg)
    G_train, test_pos, test_neg = prep.split(G)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]


class Preprocessor:
    """Prepare the collaboration graph for link-prediction evaluation."""

    def __init__(self, config: dict) -> None:
        self._config = config
        pred_cfg = config.get("prediction", {})
        self._test_ratio: float = pred_cfg.get("test_split_ratio", 0.20)
        self._seed: int = pred_cfg.get("random_seed", 42)
        self._neg_ratio: float = pred_cfg.get("negative_sampling_ratio", 1.0)
        self._splits_dir = Path("data/splits")
        self._splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clean(self, G: nx.Graph) -> nx.Graph:
        """Remove self-loops and return a cleaned copy of *G*.

        Parameters
        ----------
        G:
            Input collaboration graph.

        Returns
        -------
        nx.Graph
            Graph with self-loops removed.
        """
        G2 = G.copy()
        self_loops = list(nx.selfloop_edges(G2))
        if self_loops:
            G2.remove_edges_from(self_loops)
            logger.info("Removed %d self-loop(s).", len(self_loops))
        return G2

    def split(
        self, G: nx.Graph
    ) -> Tuple[nx.Graph, List[Edge], List[Edge]]:
        """Create a train/test edge split for link-prediction evaluation.

        1. All existing edges are the **positive** sample.
        2. A random sample of non-edges is drawn as the **negative** sample.
        3. ``test_split_ratio`` fraction of positives (and matching negatives)
           are held out as test; the remainder form the training graph.

        Parameters
        ----------
        G:
            Cleaned collaboration graph.

        Returns
        -------
        G_train : nx.Graph
            Graph with test edges removed (used for feature computation).
        test_pos : list[tuple[str, str]]
            Held-out positive edges.
        test_neg : list[tuple[str, str]]
            Held-out negative edges (non-edges sampled from the full graph).
        """
        G = self.clean(G)

        all_edges: List[Edge] = list(G.edges())
        if len(all_edges) < 10:
            raise ValueError(
                f"Graph has only {len(all_edges)} edges — too few to split. "
                "Widen the date range or lower min_edge_weight."
            )

        # Positive split
        train_pos, test_pos = train_test_split(
            all_edges,
            test_size=self._test_ratio,
            random_state=self._seed,
        )

        # Negative sampling
        n_neg = int(len(all_edges) * self._neg_ratio)
        test_neg = self._sample_negative_edges(G, n_neg)

        # Training graph = full graph minus held-out test edges
        G_train = G.copy()
        G_train.remove_edges_from(test_pos)

        logger.info(
            "Split → train_pos=%d  test_pos=%d  test_neg=%d",
            len(train_pos),
            len(test_pos),
            len(test_neg),
        )

        self._save_splits(train_pos, test_pos, test_neg)
        return G_train, test_pos, test_neg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_negative_edges(
        self, G: nx.Graph, n: int
    ) -> List[Edge]:
        """Sample *n* non-edges from *G* uniformly at random.

        Uses a rejection-sampling approach; terminates early if the graph is
        too dense to find enough non-edges.
        """
        rng = np.random.default_rng(self._seed)
        nodes = list(G.nodes())
        edge_set = set(G.edges()) | {(v, u) for u, v in G.edges()}

        negatives: List[Edge] = []
        max_attempts = n * 20  # give up after this many tries

        attempts = 0
        while len(negatives) < n and attempts < max_attempts:
            u, v = rng.choice(nodes, size=2, replace=False)
            if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
                negatives.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
            attempts += 1

        if len(negatives) < n:
            logger.warning(
                "Only found %d negative edges (requested %d). "
                "Graph may be too dense.",
                len(negatives),
                n,
            )
        return negatives

    def _save_splits(
        self,
        train_pos: List[Edge],
        test_pos: List[Edge],
        test_neg: List[Edge],
    ) -> None:
        """Persist edge splits to disk for reproducibility."""
        splits = {
            "train_pos": train_pos,
            "test_pos": test_pos,
            "test_neg": test_neg,
        }
        out = self._splits_dir / "edge_splits.pkl"
        with out.open("wb") as fh:
            pickle.dump(splits, fh)
        logger.info("Edge splits saved to %s", out)

        # Also save as CSV for human inspection
        pd.DataFrame(train_pos, columns=["u", "v"]).to_csv(
            self._splits_dir / "train_pos.csv", index=False
        )
        pd.DataFrame(test_pos, columns=["u", "v"]).to_csv(
            self._splits_dir / "test_pos.csv", index=False
        )
        pd.DataFrame(test_neg, columns=["u", "v"]).to_csv(
            self._splits_dir / "test_neg.csv", index=False
        )

    def load_splits(self) -> Tuple[List[Edge], List[Edge], List[Edge]]:
        """Load previously saved edge splits from disk."""
        pkl = self._splits_dir / "edge_splits.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"No saved splits found at {pkl}")
        with pkl.open("rb") as fh:
            splits = pickle.load(fh)
        return splits["train_pos"], splits["test_pos"], splits["test_neg"]

