"""
src/graph_builder.py
---------------------
Construct a Developer Collaboration Network centered on a seed user with configurable
expansion depth (1-hop, 2-hop), deterministic node caps, and rich node/edge attributes.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a NetworkX graph centered on a starting developer."""

    def __init__(self, config: dict) -> None:
        self._config = config
        graph_cfg = config.get("graph", {})
        self._max_nodes: int = graph_cfg.get("max_nodes", 500)
        self._max_depth: int = graph_cfg.get("max_depth", 2)
        self._min_weight: int = graph_cfg.get("min_edge_weight", 1)
        self._seed_user: str = config.get("github", {}).get("username", "").lower().strip()

    def build_network(
        self,
        interactions_df: pd.DataFrame,
        seed_metadata: Optional[Dict[str, Any]] = None,
        api_collaborators: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """Construct ego-network with deterministic BFS expansion up to max_depth hops."""
        G_full = nx.Graph()

        # Step 1: Accumulate interaction provenance into G_full
        for _, row in interactions_df.iterrows():
            u = row["source_login"]
            v = row["target_login"]
            if not u or not v or u == v:
                continue

            for user, uid in [(u, row.get("source_id")), (v, row.get("target_id"))]:
                if user not in G_full:
                    G_full.add_node(user, github_id=uid or "", username=user,
                                    account_type="User", qualifying_interactions=0)
                elif uid and not G_full.nodes[user].get("github_id"):
                    G_full.nodes[user]["github_id"] = uid

            G_full.nodes[u]["qualifying_interactions"] += 1
            G_full.nodes[v]["qualifying_interactions"] += 1

            if G_full.has_edge(u, v):
                G_full[u][v]["weight"] += 1
                G_full[u][v]["relationships"].add(row["relationship"])
                G_full[u][v]["repos"].add(row["repo"])
                G_full[u][v]["event_ids"].append(row["event_id"])
            else:
                G_full.add_edge(
                    u, v,
                    weight=1,
                    relationships={row["relationship"]},
                    repos={row["repo"]},
                    event_ids=[row["event_id"]],
                )

        # Step 2: Inject direct collaborators from API if available
        if api_collaborators:
            for c in api_collaborators:
                v = c.get("login", "").lower().strip()
                if not v or v == self._seed_user:
                    continue
                if self._seed_user not in G_full:
                    G_full.add_node(self._seed_user, github_id="", username=self._seed_user,
                                    account_type="User", qualifying_interactions=0)
                if v not in G_full:
                    G_full.add_node(v, github_id=str(c.get("id", "")), username=v,
                                    account_type="User", qualifying_interactions=0)
                
                rel = c.get("relationship", "api_collaborator")
                repo = c.get("repo", "api_enrichment")
                if G_full.has_edge(self._seed_user, v):
                    G_full[self._seed_user][v]["weight"] += 1
                    G_full[self._seed_user][v]["relationships"].add(rel)
                    G_full[self._seed_user][v]["repos"].add(repo)
                else:
                    G_full.add_edge(self._seed_user, v, weight=1,
                                    relationships={rel}, repos={repo}, event_ids=["api_bootstrap"])

        # Filter by min_weight
        weak_edges = [(u, v) for u, v, d in G_full.edges(data=True) if d["weight"] < self._min_weight]
        G_full.remove_edges_from(weak_edges)

        status_info = {
            "seed_user": self._seed_user,
            "seed_found": self._seed_user in G_full,
            "max_depth_requested": self._max_depth,
            "max_nodes_cap": self._max_nodes,
            "insufficient_data": False,
            "reason": "",
        }

        if not self._seed_user:
            status_info["insufficient_data"] = True
            status_info["reason"] = "No seed username was configured."
            return nx.Graph(), status_info

        if self._seed_user not in G_full:
            status_info["insufficient_data"] = True
            status_info["reason"] = (
                f"Starting developer '{self._seed_user}' has 0 qualifying public interactions "
                f"in the examined dataset. A network cannot be constructed without seed activity."
            )
            return nx.Graph(), status_info

        # Step 3: Expand ego network deterministically via BFS
        selected_nodes: Set[str] = {self._seed_user}
        current_layer: List[str] = [self._seed_user]
        
        for depth in range(1, self._max_depth + 1):
            next_layer: List[str] = []
            for node in current_layer:
                nbrs = sorted(
                    G_full.neighbors(node),
                    key=lambda n: (-G_full[node][n]["weight"], n)
                )
                for nbr in nbrs:
                    if nbr not in selected_nodes:
                        selected_nodes.add(nbr)
                        next_layer.append(nbr)
                        if len(selected_nodes) >= self._max_nodes:
                            break
                if len(selected_nodes) >= self._max_nodes:
                    break
            current_layer = next_layer
            if len(selected_nodes) >= self._max_nodes or not current_layer:
                break

        G = G_full.subgraph(selected_nodes).copy()

        # Update node metadata & SNA metrics
        if seed_metadata and self._seed_user in G:
            G.nodes[self._seed_user]["github_id"] = str(seed_metadata.get("id", ""))
            G.nodes[self._seed_user]["account_type"] = seed_metadata.get("type", "User")
            G.nodes[self._seed_user]["name"] = seed_metadata.get("name", "")

        for n in G.nodes():
            G.nodes[n]["is_seed"] = (n == self._seed_user)
            G.nodes[n]["num_collaborators"] = G.degree(n)

        # Convert set attributes to sorted comma-delimited strings for GraphML compatibility
        for u, v in G.edges():
            rels = G[u][v].get("relationships", set())
            repos = G[u][v].get("repos", set())
            G[u][v]["relationship_types"] = ", ".join(sorted(list(rels)))
            G[u][v]["shared_repositories"] = ", ".join(sorted(list(repos)))

        status_info["nodes_built"] = G.number_of_nodes()
        status_info["edges_built"] = G.number_of_edges()
        return G, status_info

    def build(self, events_df: pd.DataFrame) -> nx.Graph:
        """Compatibility wrapper for pipeline integration."""
        G, _ = self.build_network(events_df)
        return G


def compute_network_stats(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "avg_degree": round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 4),
        "n_components": nx.number_connected_components(G),
        "largest_component_size": len(max(nx.connected_components(G), key=len)),
    }
    if stats["largest_component_size"] > 1:
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        try:
            stats["avg_clustering"] = round(nx.average_clustering(lcc), 6)
        except Exception:
            pass
    return stats
