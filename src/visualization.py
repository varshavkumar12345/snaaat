"""
src/visualization.py
---------------------
Generate all visualizations required by the project.

Outputs
~~~~~~~
1. Static network plot (Matplotlib PNG)
2. Degree distribution plot (Matplotlib PNG)
3. Algorithm comparison bar chart (Matplotlib PNG)
4. Interactive HTML network (PyVis)
5. Cytoscape-compatible GraphML export

Usage
~~~~~
    from src import load_config
    from src.visualization import Visualizer

    cfg = load_config()
    viz = Visualizer(cfg)
    viz.plot_network(G)
    viz.plot_degree_distribution(G)
    viz.plot_evaluation(eval_df)
    viz.interactive_network(G)
    viz.export_graphml(G)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / notebooks
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Produce all required project visualizations."""

    def __init__(self, config: dict) -> None:
        self._config = config
        viz_cfg = config.get("visualization", {})
        self._output_dir = Path(viz_cfg.get("output_dir", "outputs/visualizations"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        static_cfg = viz_cfg.get("static", {})
        self._dpi: int = static_cfg.get("dpi", 150)
        self._fmt: str = static_cfg.get("format", "png")

        interactive_cfg = viz_cfg.get("interactive", {})
        self._iv_height: str = interactive_cfg.get("height", "750px")
        self._iv_width: str = interactive_cfg.get("width", "100%")

        self._seed_user: str = config.get("github", {}).get("username", "")

    # ------------------------------------------------------------------
    # 1. Static network plot
    # ------------------------------------------------------------------

    def plot_network(
        self,
        G: nx.Graph,
        filename: str = "network_static",
        title: str = "Developer Collaboration Network",
        max_nodes_display: int = 150,
    ) -> Path:
        """Draw a static force-directed network layout.

        Parameters
        ----------
        G:
            The collaboration graph.
        filename:
            Output file base name (extension added automatically).
        title:
            Plot title.
        max_nodes_display:
            For very large graphs, sample this many nodes for clarity.

        Returns
        -------
        Path
            Path to the saved image file.
        """
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph — skipping network plot.")
            return self._output_dir / f"{filename}.{self._fmt}"

        # Subsample if needed
        if G.number_of_nodes() > max_nodes_display:
            logger.info(
                "Graph has %d nodes; sampling %d for static plot.",
                G.number_of_nodes(), max_nodes_display,
            )
            nodes = list(G.nodes())
            # Prioritise the seed user and its neighbours
            seed_nbrs = (
                list(G.neighbors(self._seed_user))
                if self._seed_user in G
                else []
            )
            priority = (
                ([self._seed_user] + seed_nbrs)[:max_nodes_display]
                if self._seed_user in G
                else []
            )
            remaining = [n for n in nodes if n not in priority]
            np.random.seed(42)
            sampled = priority + list(
                np.random.choice(
                    remaining,
                    size=min(max_nodes_display - len(priority), len(remaining)),
                    replace=False,
                )
            )
            G = G.subgraph(sampled)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Layout
        try:
            pos = nx.spring_layout(G, seed=42, k=0.5)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        # Node sizes by degree
        degrees = dict(G.degree())
        node_sizes = [max(100, degrees.get(n, 1) * 80) for n in G.nodes()]

        # Node colours: seed = red, others by degree
        node_colours = []
        for n in G.nodes():
            if n == self._seed_user:
                node_colours.append("#e74c3c")
            else:
                node_colours.append("#3498db")

        # Edge widths by weight
        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 2.5 * (w / max_w) for w in weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.4, edge_color="#aaaaaa"
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color=node_colours, alpha=0.85
        )

        # Labels only for high-degree nodes
        degree_threshold = sorted(degrees.values(), reverse=True)
        label_threshold = degree_threshold[min(30, len(degree_threshold) - 1)]
        labels = {
            n: n
            for n in G.nodes()
            if degrees.get(n, 0) >= label_threshold or n == self._seed_user
        }
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax, font_size=7, font_color="#222222"
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.axis("off")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=10, label=f"Seed: {self._seed_user}"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                   markersize=8, label="Other developer"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

        out = self._output_dir / f"{filename}.{self._fmt}"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Static network plot saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 2. Degree distribution
    # ------------------------------------------------------------------

    def plot_degree_distribution(
        self,
        G: nx.Graph,
        filename: str = "degree_distribution",
    ) -> Path:
        """Plot the degree distribution of the collaboration graph.

        Parameters
        ----------
        G:
            Collaboration graph.
        filename:
            Output file base name.

        Returns
        -------
        Path
        """
        degrees = sorted(dict(G.degree()).values(), reverse=True)
        degree_counts: dict = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        x = list(degree_counts.keys())
        y = list(degree_counts.values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale histogram
        axes[0].bar(x, y, color="#3498db", edgecolor="white", linewidth=0.5)
        axes[0].set_xlabel("Degree", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Degree Distribution (Linear)", fontsize=12)

        # Log-log scale
        axes[1].scatter(x, y, color="#e74c3c", s=30, alpha=0.7)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Degree (log)", fontsize=11)
        axes[1].set_ylabel("Count (log)", fontsize=11)
        axes[1].set_title("Degree Distribution (Log-Log)", fontsize=12)

        fig.suptitle("Developer Collaboration Network — Degree Distribution",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

        out = self._output_dir / f"{filename}.{self._fmt}"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Degree distribution plot saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 3. Algorithm evaluation comparison
    # ------------------------------------------------------------------

    def plot_evaluation(
        self,
        eval_df: pd.DataFrame,
        filename: str = "algorithm_comparison",
    ) -> Path:
        """Bar chart comparing algorithm metrics.

        Parameters
        ----------
        eval_df:
            DataFrame produced by ``Evaluator.evaluate()``.
        filename:
            Output file base name.

        Returns
        -------
        Path
        """
        metrics = ["auc_roc", "avg_precision", "precision", "recall", "f1"]
        metric_labels = ["AUC-ROC", "Avg Precision", "Precision", "Recall", "F1"]

        n_algos = len(eval_df)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = 0.8 / n_algos

        fig, ax = plt.subplots(figsize=(13, 6))
        colours = plt.cm.Set2(np.linspace(0, 1, n_algos))  # type: ignore[attr-defined]

        for i, (algo, row) in enumerate(eval_df.iterrows()):
            vals = [row.get(m, 0.0) for m in metrics]
            offset = (i - n_algos / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, label=algo,
                          color=colours[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Link Prediction Algorithm Comparison", fontsize=13,
                     fontweight="bold")
        ax.legend(title="Algorithm", loc="upper right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out = self._output_dir / f"{filename}.{self._fmt}"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Evaluation comparison plot saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 4. Interactive HTML network (PyVis)
    # ------------------------------------------------------------------

    def interactive_network(
        self,
        G: nx.Graph,
        filename: str = "network_interactive",
        max_nodes: int = 200,
    ) -> Optional[Path]:
        """Generate an interactive HTML network visualization using PyVis.

        Parameters
        ----------
        G:
            Collaboration graph.
        filename:
            Output HTML file base name.
        max_nodes:
            Cap on nodes for performance.

        Returns
        -------
        Path | None
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis is not installed. Run: pip install pyvis")
            return None

        # Subsample if needed
        if G.number_of_nodes() > max_nodes:
            logger.info(
                "Sampling %d nodes for interactive plot (graph has %d).",
                max_nodes, G.number_of_nodes(),
            )
            seed_nbrs = (
                list(G.neighbors(self._seed_user))
                if self._seed_user in G
                else []
            )
            priority = (
                ([self._seed_user] + seed_nbrs)[:max_nodes]
                if self._seed_user in G
                else []
            )
            remaining = [n for n in G.nodes() if n not in priority]
            np.random.seed(42)
            extra = min(max_nodes - len(priority), len(remaining))
            sampled = priority + list(np.random.choice(remaining, extra, replace=False))
            G = G.subgraph(sampled)

        viz_cfg = self._config.get("visualization", {}).get("interactive", {})
        net = Network(
            height=viz_cfg.get("height", "750px"),
            width=viz_cfg.get("width", "100%"),
            bgcolor=viz_cfg.get("bgcolor", "#ffffff"),
            font_color=viz_cfg.get("font_color", "#000000"),
            notebook=False,
        )
        net.force_atlas_2based()

        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1

        for node in G.nodes():
            deg = degrees.get(node, 1)
            size = 10 + 30 * (deg / max_deg)
            colour = "#e74c3c" if node == self._seed_user else "#3498db"
            title = f"<b>{node}</b><br>Degree: {deg}"
            attrs = G.nodes[node]
            if "degree_centrality" in attrs:
                title += f"<br>Degree Centrality: {attrs['degree_centrality']:.4f}"
            if "clustering_coefficient" in attrs:
                title += f"<br>Clustering: {attrs['clustering_coefficient']:.4f}"
            net.add_node(node, label=node, size=size, color=colour, title=title)

        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1)
            net.add_edge(u, v, value=w, title=f"Weight: {w}")

        out = self._output_dir / f"{filename}.html"
        net.write_html(str(out))
        logger.info("Interactive network saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 5. GraphML export
    # ------------------------------------------------------------------

    def export_graphml(
        self, G: nx.Graph, filename: str = "collaboration_network"
    ) -> Path:
        """Export the graph in Cytoscape-compatible GraphML format.

        Parameters
        ----------
        G:
            Collaboration graph.
        filename:
            Output file base name.

        Returns
        -------
        Path
        """
        out = self._output_dir / f"{filename}.graphml"
        nx.write_graphml(G, str(out))
        logger.info("GraphML exported: %s", out)
        return out

