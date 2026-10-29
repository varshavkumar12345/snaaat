"""
src/graph_builder.py
---------------------
Construct a weighted, undirected developer collaboration graph from parsed
GitHub Archive events, optionally enriched via the GitHub REST API.

Graph semantics
~~~~~~~~~~~~~~~
* **Nodes** — GitHub usernames (developers).
* **Edges** — an edge (u, v) exists when u and v appear in the same event
  (e.g. actor and PR author, or two contributors to the same repo).
* **Edge weight** — the number of distinct events in which u and v co-appear.

The graph is built around an *ego network* seeded with the configured
starting username, then expanded up to ``graph.max_depth`` hops and
capped at ``graph.max_nodes`` nodes.

Usage
~~~~~
    from src import load_config
    from src.graph_builder import GraphBuilder

    cfg = load_config()
    builder = GraphBuilder(cfg)
    G = builder.build(events_df)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a NetworkX collaboration graph from a parsed events DataFrame."""

    def __init__(self, config: dict) -> None:
        self._config = config
        graph_cfg = config.get("graph", {})
        self._max_nodes: int = graph_cfg.get("max_nodes", 500)
        self._max_depth: int = graph_cfg.get("max_depth", 2)
        self._min_weight: int = graph_cfg.get("min_edge_weight", 1)
        self._seed_user: str = config.get("github", {}).get("username", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, events_df: pd.DataFrame) -> nx.Graph:
        """Build and return a weighted undirected collaboration graph.

        Parameters
        ----------
        events_df:
            DataFrame produced by ``EventParser.parse_files()``.

        Returns
        -------
        nx.Graph
            Nodes carry attribute ``degree_centrality``; edges carry
            attribute ``weight`` (co-occurrence count).
        """
        if events_df.empty:
            logger.warning("Events DataFrame is empty — returning empty graph.")
            return nx.Graph()

        # Step 1: accumulate raw edge weights
        edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
        self._accumulate_edges(events_df, edge_weights)

        # Step 2: build initial graph from all edges
        G_full = nx.Graph()
        for (u, v), w in edge_weights.items():
            if w >= self._min_weight:
                G_full.add_edge(u, v, weight=w)

        logger.info(
            "Full graph: %d nodes, %d edges (min_weight=%d).",
            G_full.number_of_nodes(),
            G_full.number_of_edges(),
            self._min_weight,
        )

        # Step 3: ego-network expansion from seed user
        G = self._expand_ego_network(G_full)

        # Step 4: annotate nodes with basic SNA metrics
        self._annotate(G)

        logger.info(
            "Final graph: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate_edges(
        self,
        events_df: pd.DataFrame,
        edge_weights: Dict[Tuple[str, str], int],
    ) -> None:
        """Populate *edge_weights* from actor ↔ extra_actor pairs."""
        for _, row in events_df.iterrows():
            actor: str = row["actor"]
            extra_actors: List[str] = row.get("extra_actors", [])
            repo_owner: str = row.get("repo_owner", "")

            # Actor ↔ each extra actor (direct interaction)
            for other in extra_actors:
                if other and other != actor:
                    key = tuple(sorted([actor, other]))
                    edge_weights[key] += 1  # type: ignore[index]

            # Actor ↔ repo owner (contributed to someone's repo)
            if repo_owner and repo_owner != actor:
                key = tuple(sorted([actor, repo_owner]))
                edge_weights[key] += 1  # type: ignore[index]

    def _expand_ego_network(self, G_full: nx.Graph) -> nx.Graph:
        """Return a subgraph of *G_full* rooted at the seed user.

        Performs BFS up to ``max_depth`` hops from the seed.  Falls back to
        the largest connected component if the seed is not in the graph.
        """
        seed = self._seed_user

        if seed and seed in G_full:
            visited: Set[str] = set()
            frontier = {seed}
            for depth in range(self._max_depth):
                next_frontier: Set[str] = set()
                for node in frontier:
                    for nbr in G_full.neighbors(node):
                        if nbr not in visited and nbr not in frontier:
                            next_frontier.add(nbr)
                visited |= frontier
                frontier = next_frontier

                if len(visited) + len(frontier) >= self._max_nodes:
                    logger.info(
                        "Node cap (%d) reached at depth %d.",
                        self._max_nodes,
                        depth + 1,
                    )
                    break

            all_nodes = visited | frontier
            all_nodes = set(list(all_nodes)[: self._max_nodes])
            subgraph = G_full.subgraph(all_nodes).copy()

        elif G_full.number_of_nodes() > 0:
            logger.warning(
                "Seed user '%s' not found in the graph. "
                "Using the largest connected component instead.",
                seed,
            )
            largest_cc = max(nx.connected_components(G_full), key=len)
            nodes = set(list(largest_cc)[: self._max_nodes])
            subgraph = G_full.subgraph(nodes).copy()

        else:
            subgraph = nx.Graph()

        return subgraph

    def _annotate(self, G: nx.Graph) -> None:
        """Annotate nodes with centrality metrics in-place."""
        if G.number_of_nodes() == 0:
            return

        degree_cent = nx.degree_centrality(G)
        nx.set_node_attributes(G, degree_cent, "degree_centrality")

        # Betweenness is expensive; skip for large graphs
        if G.number_of_nodes() <= 300:
            betweenness = nx.betweenness_centrality(G, weight="weight")
            nx.set_node_attributes(G, betweenness, "betweenness_centrality")

        # Clustering coefficient
        clustering = nx.clustering(G, weight="weight")
        nx.set_node_attributes(G, clustering, "clustering_coefficient")

        # Mark the seed user
        if self._seed_user in G:
            G.nodes[self._seed_user]["is_seed"] = True


def compute_network_stats(G: nx.Graph) -> dict:
    """Compute and return a summary dict of network-level SNA statistics.

    Parameters
    ----------
    G:
        The collaboration graph.

    Returns
    -------
    dict
        Keys: n_nodes, n_edges, density, avg_degree, avg_clustering,
        n_components, largest_component_size, is_connected,
        avg_shortest_path (if small enough), diameter (if small enough).
    """
    if G.number_of_nodes() == 0:
        return {}

    stats: dict = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "avg_degree": round(
            sum(d for _, d in G.degree()) / G.number_of_nodes(), 4
        ),
        "avg_clustering": round(nx.average_clustering(G, weight="weight"), 6),
        "n_components": nx.number_connected_components(G),
        "largest_component_size": len(max(nx.connected_components(G), key=len)),
        "is_connected": nx.is_connected(G),
    }

    # Diameter and average path length only on the LCC and if not too large
    lcc_nodes = max(nx.connected_components(G), key=len)
    lcc = G.subgraph(lcc_nodes)
    if lcc.number_of_nodes() <= 500:
        try:
            stats["diameter"] = nx.diameter(lcc)
            stats["avg_shortest_path"] = round(
                nx.average_shortest_path_length(lcc), 4
            )
        except nx.NetworkXError:
            pass

    return stats

