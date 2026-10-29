"""
Network visualization generators: Overall network, ego-network, predicted links, ROC curves, and PyVis HTML.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from pyvis.network import Network
from src.config import FIGURES_DIR, DELIVERABLES_DIR

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_overall_network(
    G: nx.Graph,
    out_png: Optional[Path] = None,
    out_svg: Optional[Path] = None,
    max_nodes: int = 120
) -> None:
    """Plot global network layout (subsampled to top degree nodes if large for clarity)."""
    if out_png is None:
        out_png = FIGURES_DIR / "network_overall.png"
    if out_svg is None:
        out_svg = FIGURES_DIR / "network_overall.svg"

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        H = G.subgraph(top_nodes).copy()
    else:
        H = G.copy()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    pos = nx.spring_layout(H, seed=42, k=0.35)

    degrees = dict(H.degree())
    node_sizes = [max(80, degrees[n] * 25) for n in H.nodes()]
    node_colors = [degrees[n] for n in H.nodes()]

    # Draw edges
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, edge_color="#718096", width=0.8)

    # Draw nodes
    sc = nx.draw_networkx_nodes(
        H, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        cmap=plt.cm.viridis, alpha=0.88, edgecolors="white", linewidths=1.2
    )

    # Top degree labels
    top_label_nodes = sorted(H.nodes(), key=lambda n: degrees[n], reverse=True)[:15]
    labels = {n: n for n in top_label_nodes}
    nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=8, font_weight="bold", font_color="#1a202c")

    plt.colorbar(sc, ax=ax, label="Developer Degree", fraction=0.035, pad=0.02)
    ax.set_title("NumFOCUS Developer Collaboration Network (Global Topology)", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def plot_ego_network(
    G: nx.Graph,
    target_user: str,
    out_png: Optional[Path] = None,
    out_svg: Optional[Path] = None
) -> None:
    """Plot 1-hop ego network centered on the starting developer."""
    if out_png is None:
        out_png = FIGURES_DIR / "network_ego.png"
    if out_svg is None:
        out_svg = FIGURES_DIR / "network_ego.svg"

    if target_user not in G:
        return

    neighbors = list(G.neighbors(target_user))
    ego_nodes = [target_user] + neighbors
    H = G.subgraph(ego_nodes).copy()

    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    pos = nx.spring_layout(H, seed=42, k=0.4)

    # Edge weights
    weights = [H[u][v].get("weight", 1) for u, v in H.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [1.0 + (w / max_w) * 3.5 for w in weights]

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.4, edge_color="#4a5568", width=edge_widths)

    # Colors and sizes
    node_colors = []
    node_sizes = []
    for n in H.nodes():
        if n == target_user:
            node_colors.append("#e53e3e")  # Crimson for target user
            node_sizes.append(650)
        else:
            node_colors.append("#3182ce")  # Blue for neighbors
            node_sizes.append(max(120, H.degree(n) * 30))

    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors="white", linewidths=1.5)

    # Labels for target and key collaborators
    labels = {target_user: f"[TARGET] {target_user}"}
    for n in sorted(neighbors, key=lambda x: H[target_user][x].get("weight", 1), reverse=True)[:12]:
        labels[n] = n

    nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=8, font_weight="bold", font_color="#1a202c")

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e53e3e', label=f'Target Developer ({target_user})'),
        mpatches.Patch(color='#3182ce', label=f'Direct Collaborator ({len(neighbors)})')
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    ax.set_title(f"Ego Collaboration Network for @{target_user} (1-Hop Neighborhood)", fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def plot_predicted_links_network(
    G: nx.Graph,
    target_user: str,
    recommendations: List[Dict[str, Any]],
    out_png: Optional[Path] = None,
    out_svg: Optional[Path] = None,
    top_k: int = 10
) -> None:
    """
    Plot ego network combined with predicted links for top K recommendations.
    Observed edges are drawn solid gray; predicted links are drawn with visually distinct dashed red/orange lines.
    """
    if out_png is None:
        out_png = FIGURES_DIR / "network_predictions.png"
    if out_svg is None:
        out_svg = FIGURES_DIR / "network_predictions.svg"

    top_recs = recommendations[:top_k]
    rec_nodes = [r["candidate_user"] for r in top_recs]
    neighbors = list(G.neighbors(target_user)) if target_user in G else []

    # Display subset of nodes to keep visualization crisp
    display_neighbors = sorted(neighbors, key=lambda n: G.degree(n), reverse=True)[:15]
    sub_nodes = list(set([target_user] + display_neighbors + rec_nodes))
    H = G.subgraph(sub_nodes).copy()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    pos = nx.spring_layout(H, seed=42, k=0.45)

    # 1. Draw observed edges (solid slate)
    observed_edges = list(H.edges())
    nx.draw_networkx_edges(H, pos, edgelist=observed_edges, ax=ax, style="solid", alpha=0.35, edge_color="#718096", width=1.2)

    # 2. Draw PREDICTED edges (distinct dashed orange/red lines)
    predicted_edges = [(target_user, r["candidate_user"]) for r in top_recs if r["candidate_user"] in pos]
    nx.draw_networkx_edges(H, pos, edgelist=predicted_edges, ax=ax, style="dashed", alpha=0.9, edge_color="#dd6b20", width=2.5)

    # Node styling
    node_colors = []
    node_sizes = []
    for n in H.nodes():
        if n == target_user:
            node_colors.append("#e53e3e")  # Target: Crimson
            node_sizes.append(700)
        elif n in rec_nodes:
            node_colors.append("#dd6b20")  # Recommended: Orange
            node_sizes.append(500)
        else:
            node_colors.append("#3182ce")  # Observed neighbor: Blue
            node_sizes.append(250)

    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.92, edgecolors="white", linewidths=1.5)

    # Labels
    labels = {target_user: f"[TARGET] {target_user}"}
    for r in top_recs:
        labels[r["candidate_user"]] = f"#{r['rank']} {r['candidate_user']}"
    for n in display_neighbors[:8]:
        if n not in labels:
            labels[n] = n

    nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=8, font_weight="bold", font_color="#1a202c")

    legend_elements = [
        mpatches.Patch(color='#e53e3e', label=f'Target Developer ({target_user})'),
        mpatches.Patch(color='#3182ce', label='Observed Collaborator'),
        mpatches.Patch(color='#dd6b20', label=f'Top-{top_k} Recommended Collaborator'),
        plt.Line2D([0], [0], color='#718096', lw=1.5, linestyle='solid', label='Observed Collaboration Edge'),
        plt.Line2D([0], [0], color='#dd6b20', lw=2.5, linestyle='dashed', label='Predicted Potential Link (Top-K)')
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    ax.set_title(f"Link Prediction: Top-{top_k} Recommended Collaborators for @{target_user}", fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curves(
    eval_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    out_png: Optional[Path] = None,
    out_svg: Optional[Path] = None
) -> None:
    """Plot comparative ROC curves for all algorithms."""
    if out_png is None:
        out_png = FIGURES_DIR / "roc_curves.png"
    if out_svg is None:
        out_svg = FIGURES_DIR / "roc_curves.svg"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    colors = {
        "Common Neighbors": "#3182ce",
        "Jaccard Coefficient": "#38a169",
        "Adamic-Adar": "#e53e3e",
        "Preferential Attachment": "#805ad5"
    }

    for algo_name, (fpr, tpr, auc) in eval_curves.items():
        c = colors.get(algo_name, "#4a5568")
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{algo_name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="#a0aec0", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves Comparison: Link Prediction Algorithms", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def generate_interactive_pyvis(
    G: nx.Graph,
    target_user: str,
    recommendations: List[Dict[str, Any]],
    out_html: Optional[Path] = None,
    top_k: int = 15
) -> Path:
    """Generate interactive PyVis HTML visualization."""
    if out_html is None:
        out_html = DELIVERABLES_DIR / "interactive_network.html"

    top_recs = recommendations[:top_k]
    rec_dict = {r["candidate_user"]: r for r in top_recs}
    neighbors = list(G.neighbors(target_user)) if target_user in G else []
    
    sub_nodes = list(set([target_user] + neighbors[:25] + list(rec_dict.keys())))
    H = G.subgraph(sub_nodes).copy()

    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#2d3748", directed=False)
    net.force_atlas_2based()

    # Add nodes
    for n in H.nodes():
        if n == target_user:
            net.add_node(
                n, label=f"★ {n} (Target)", color="#e53e3e", size=32,
                title=f"Target Developer: {n}<br>Degree: {G.degree(n)}"
            )
        elif n in rec_dict:
            rec = rec_dict[n]
            net.add_node(
                n, label=f"#{rec['rank']} {n}", color="#dd6b20", size=24,
                title=(
                    f"Recommended: {n} (Rank {rec['rank']})<br>"
                    f"AA Score: {rec['adamic_adar_score']}<br>"
                    f"Common Neighbors: {rec['common_neighbors_count']}<br>"
                    f"Degree: {rec['candidate_degree']}"
                )
            )
        else:
            net.add_node(
                n, label=n, color="#3182ce", size=15,
                title=f"Collaborator: {n}<br>Degree: {G.degree(n)}"
            )

    # Add observed edges
    for u, v in H.edges():
        w = H[u][v].get("weight", 1)
        net.add_edge(u, v, value=w, color="#a0aec0", title=f"Interactions: {w}")

    # Add predicted edges (dashed orange)
    for cand_user, rec in rec_dict.items():
        if cand_user in H.nodes():
            net.add_edge(
                target_user, cand_user,
                color="#dd6b20", dashes=True, width=3,
                title=f"PREDICTED LINK<br>Adamic-Adar: {rec['adamic_adar_score']}<br>CN: {rec['common_neighbors_count']}"
            )

    net.save_graph(str(out_html))
    return out_html
