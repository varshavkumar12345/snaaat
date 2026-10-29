"""
src/visualization.py
---------------------
Phase 7: Comprehensive Graph Visualization Suite.

Outputs:
  - Overall network layout (PNG & SVG)
  - Starting-node ego network (PNG & SVG)
  - Top-10 predicted links visualization (distinguishing observed vs predicted edges)
  - Interactive HTML visualization (PyVis)
  - Cytoscape-compatible GraphML export
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates all static, vector, interactive, and GraphML visualizations."""

    def __init__(self, config: dict) -> None:
        self._config = config
        viz_cfg = config.get("visualization", {})
        self._output_dir = Path(viz_cfg.get("output_dir", "outputs/visualizations"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._dpi: int = viz_cfg.get("static", {}).get("dpi", 200)

    def plot_overall_network(
        self,
        G: nx.Graph,
        seed_user: str,
        filename: str = "network_overall",
    ) -> Dict[str, Path]:
        """Draw full collaboration graph layout in PNG and SVG."""
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=0.35)

        degrees = dict(G.degree())
        node_sizes = [max(80, degrees.get(n, 1) * 35) for n in G.nodes()]
        node_colors = ["#e74c3c" if n == seed_user else "#3498db" for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, edge_color="#7f8c8d", width=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.85)

        # Label hub nodes and seed
        high_deg_threshold = np.percentile(list(degrees.values()), 80) if degrees else 0
        labels = {n: n for n in G.nodes() if degrees[n] >= high_deg_threshold or n == seed_user}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

        ax.set_title("Developer Collaboration Network (Global Layout)", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")

        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label=f"Seed: {seed_user}"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="Developer Node"),
            Line2D([0], [0], color="#7f8c8d", lw=1.5, label="Observed Collaboration"),
        ]
        ax.legend(handles=legend_items, loc="upper right", framealpha=0.9)

        png_path = self._output_dir / f"{filename}.png"
        svg_path = self._output_dir / f"{filename}.svg"
        fig.savefig(png_path, dpi=self._dpi, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        return {"png": png_path, "svg": svg_path}

    def plot_ego_network(
        self,
        G: nx.Graph,
        seed_user: str,
        filename: str = "network_ego",
    ) -> Dict[str, Path]:
        """Draw 1-hop and 2-hop ego network centered on starting developer."""
        neighbors = set(G.neighbors(seed_user)) | {seed_user} if seed_user in G else set(G.nodes())
        G_ego = G.subgraph(neighbors).copy()

        fig, ax = plt.subplots(figsize=(12, 9))
        pos = nx.spring_layout(G_ego, seed=42, k=0.5)

        degrees = dict(G_ego.degree())
        node_sizes = [400 if n == seed_user else max(150, degrees.get(n, 1) * 70) for n in G_ego.nodes()]
        node_colors = ["#e74c3c" if n == seed_user else "#2ecc71" for n in G_ego.nodes()]

        nx.draw_networkx_edges(G_ego, pos, ax=ax, alpha=0.5, edge_color="#34495e", width=1.5)
        nx.draw_networkx_nodes(G_ego, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G_ego, pos, ax=ax, font_size=8, font_weight="bold")

        ax.set_title(f"Ego-Network Centered on '{seed_user}' (1-Hop Collaborators)", fontsize=13, fontweight="bold", pad=12)
        ax.axis("off")

        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=12, label="Starting Developer"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=9, label="Direct Collaborator"),
            Line2D([0], [0], color="#34495e", lw=1.5, label="Direct Interaction Edge"),
        ]
        ax.legend(handles=legend_items, loc="upper right", framealpha=0.9)

        png_path = self._output_dir / f"{filename}.png"
        svg_path = self._output_dir / f"{filename}.svg"
        fig.savefig(png_path, dpi=self._dpi, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        return {"png": png_path, "svg": svg_path}

    def plot_top10_predicted_links(
        self,
        G_train: nx.Graph,
        seed_user: str,
        predicted_edges: List[Tuple[str, str, float]],
        filename: str = "network_top10_predicted",
    ) -> Dict[str, Path]:
        """
        Visualize Top-10 predicted collaboration links.
        Observed edges are drawn solid gray/blue; predicted edges are dashed red.
        """
        pred_nodes = {seed_user}
        for u, v, _ in predicted_edges[:10]:
            pred_nodes.add(u)
            pred_nodes.add(v)

        # Include direct neighbors for structural context
        direct_nbrs = set(G_train.neighbors(seed_user)) if seed_user in G_train else set()
        subgraph_nodes = pred_nodes | set(list(direct_nbrs)[:10])
        G_sub = G_train.subgraph(subgraph_nodes).copy()

        fig, ax = plt.subplots(figsize=(13, 10))
        pos = nx.spring_layout(G_sub, seed=42, k=0.6)

        # 1. Draw observed edges (solid slate)
        nx.draw_networkx_edges(G_sub, pos, ax=ax, edge_color="#95a5a6", width=1.5, alpha=0.6)

        # 2. Draw predicted edges (dashed red)
        pred_lines = []
        for u, v, score in predicted_edges[:10]:
            if u in pos and v in pos:
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                line, = ax.plot(x, y, color="#e74c3c", linestyle="--", linewidth=2.2, alpha=0.9, zorder=3)
                pred_lines.append(line)

        # Draw nodes
        node_colors = []
        for n in G_sub.nodes():
            if n == seed_user:
                node_colors.append("#8e44ad")  # purple seed
            elif n in {v for _, v, _ in predicted_edges[:10]}:
                node_colors.append("#e67e22")  # orange candidate
            else:
                node_colors.append("#3498db")  # blue neighbor

        nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_color=node_colors, node_size=350, alpha=0.95)
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=8, font_weight="bold")

        ax.set_title(f"Top-10 Predicted Collaboration Links for '{seed_user}'", fontsize=13, fontweight="bold", pad=12)
        ax.axis("off")

        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#8e44ad", markersize=11, label=f"Seed: {seed_user}"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=9, label="Predicted Candidate"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="Existing Collaborator"),
            Line2D([0], [0], color="#95a5a6", lw=1.5, label="Observed Edge (Solid)"),
            Line2D([0], [0], color="#e74c3c", lw=2, linestyle="--", label="Predicted Edge (Dashed Red)"),
        ]
        ax.legend(handles=legend_items, loc="upper right", framealpha=0.9)

        png_path = self._output_dir / f"{filename}.png"
        svg_path = self._output_dir / f"{filename}.svg"
        fig.savefig(png_path, dpi=self._dpi, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        return {"png": png_path, "svg": svg_path}

    def interactive_network_html(
        self,
        G: nx.Graph,
        seed_user: str,
        predicted_edges: Optional[List[Tuple[str, str, float]]] = None,
        filename: str = "network_interactive",
    ) -> Path:
        """Create interactive PyVis visualization with physics and node inspect."""
        from pyvis.network import Network

        net = Network(height="750px", width="100%", bgcolor="#f8f9fa", font_color="#2c3e50")
        net.force_atlas_2based()

        # Add nodes
        for n, data in G.nodes(data=True):
            deg = G.degree(n)
            color = "#e74c3c" if n == seed_user else "#3498db"
            size = 25 if n == seed_user else max(10, min(30, deg * 3))
            title = f"<b>User:</b> {n}<br><b>Degree:</b> {deg}<br><b>Interactions:</b> {data.get('qualifying_interactions', 0)}"
            net.add_node(n, label=n, title=title, color=color, size=size)

        # Add observed edges
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, color="#bdc3c7", value=data.get("weight", 1), title=f"Weight: {data.get('weight', 1)}")

        # Add predicted edges in dashed red
        if predicted_edges:
            for u, v, score in predicted_edges[:10]:
                if u in G and v in G:
                    net.add_edge(u, v, color="#e74c3c", dashes=True, value=2, title=f"Predicted Link (Score: {score:.4f})")

        out_html = self._output_dir / f"{filename}.html"
        net.write_html(str(out_html))
        logger.info("Interactive visualization saved to %s", out_html)
        return out_html

    def export_graphml(self, G: nx.Graph, filename: str = "cytoscape_network.graphml") -> Path:
        """Export clean GraphML file ready for Cytoscape."""
        from src.preprocessing import Preprocessor
        prep = Preprocessor(self._config)
        clean_G = prep._sanitize_for_graphml(G)
        out_path = self._output_dir / filename
        nx.write_graphml(clean_G, str(out_path))
        logger.info("GraphML exported to %s", out_path)
        return out_path
