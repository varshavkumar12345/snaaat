"""
Graph construction, cleaning, ego-network extraction, and network statistics calculation.
"""
import re
from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import networkx as nx
from src.config import BOT_IDENTIFIERS, PROCESSED_DATA_DIR, DELIVERABLES_DIR

def is_bot(username: Optional[str]) -> bool:
    """Return True if username matches bot patterns."""
    if not username or not isinstance(username, str):
        return True
    u_lower = username.lower().strip()
    if u_lower.endswith("[bot]") or u_lower.endswith("-bot"):
        return True
    for bot_id in BOT_IDENTIFIERS:
        if bot_id in u_lower:
            return True
    return False

def clean_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean interaction records:
    - Remove self-loops (source == target)
    - Filter bots from source and target
    - Remove invalid or empty usernames
    - Deduplicate identical action records if needed
    """
    if interactions_df.empty:
        return interactions_df

    df = interactions_df.copy()
    # Normalize usernames
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    # Filter out empty or None
    df = df[(df["source"] != "") & (df["target"] != "") & (df["source"] != "None") & (df["target"] != "None")]

    # Filter out self-loops
    df = df[df["source"] != df["target"]]

    # Filter out bots
    df = df[~df["source"].apply(is_bot)]
    df = df[~df["target"].apply(is_bot)]

    return df

def aggregate_edges(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate interactions into weighted undirected edges between developers.
    Each pair (u, v) is stored canonically with u < v.
    """
    if cleaned_df.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "interaction_types", "repositories", "first_seen", "last_seen"])

    records = []
    for _, row in cleaned_df.iterrows():
        u, v = sorted([row["source"], row["target"]])
        records.append({
            "source": u,
            "target": v,
            "interaction_type": row.get("interaction_type", "interaction"),
            "repository": row.get("repository", "unknown"),
            "timestamp": row.get("timestamp")
        })

    canonical_df = pd.DataFrame(records)

    def agg_group(grp):
        types_count = grp["interaction_type"].value_counts().to_dict()
        types_str = "; ".join([f"{k}:{v}" for k, v in types_count.items()])
        repos = sorted(list(set(grp["repository"].dropna().unique())))
        repos_str = "; ".join(repos[:5]) + ("..." if len(repos) > 5 else "")
        timestamps = grp["timestamp"].dropna().sort_values()
        first_seen = timestamps.iloc[0] if not timestamps.empty else None
        last_seen = timestamps.iloc[-1] if not timestamps.empty else None
        return pd.Series({
            "weight": len(grp),
            "interaction_types": types_str,
            "repositories": repos_str,
            "first_seen": first_seen,
            "last_seen": last_seen
        })

    agg_df = canonical_df.groupby(["source", "target"]).apply(agg_group).reset_index()
    return agg_df

def resolve_starting_developer(edge_df: pd.DataFrame, username: str) -> Dict[str, Any]:
    """
    Resolve the given starting username against available records.
    Returns developer existence, degree, total interactions, and collaborator list.
    """
    u_clean = username.strip()
    u_lower = u_clean.lower()

    # Find matching nodes
    all_users = set(edge_df["source"]).union(set(edge_df["target"]))
    user_map = {u.lower(): u for u in all_users}

    if u_lower not in user_map:
        # Check closest partial matches
        candidates = [user_map[k] for k in user_map if u_lower in k or k in u_lower][:5]
        return {
            "resolved": False,
            "query": username,
            "message": f"Starting developer '{username}' not found in dataset records.",
            "candidates": candidates,
            "collaborators": [],
            "degree": 0,
            "total_weight": 0
        }

    canonical_user = user_map[u_lower]
    user_edges = edge_df[(edge_df["source"] == canonical_user) | (edge_df["target"] == canonical_user)]
    collaborators = []
    total_weight = 0

    for _, row in user_edges.iterrows():
        neighbor = row["target"] if row["source"] == canonical_user else row["source"]
        collaborators.append((neighbor, row["weight"]))
        total_weight += row["weight"]

    collaborators.sort(key=lambda x: x[1], reverse=True)

    return {
        "resolved": True,
        "query": username,
        "canonical_username": canonical_user,
        "degree": len(collaborators),
        "total_weight": total_weight,
        "collaborators": collaborators,
        "top_collaborators": collaborators[:10]
    }

def construct_graph(edge_df: pd.DataFrame, starting_user: Optional[str] = None, hops: int = 2) -> nx.Graph:
    """
    Construct NetworkX graph from aggregated edges.
    If starting_user is provided and found, extracts an ego network within specified hops.
    """
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=int(row["weight"]),
            interaction_types=row.get("interaction_types", ""),
            repositories=row.get("repositories", ""),
            first_seen=str(row.get("first_seen", "")),
            last_seen=str(row.get("last_seen", ""))
        )

    if starting_user and starting_user in G:
        ego_nodes = nx.single_source_shortest_path_length(G, starting_user, cutoff=hops).keys()
        G_sub = G.subgraph(ego_nodes).copy()
        return G_sub

    return G

def compute_network_statistics(G: nx.Graph) -> Dict[str, Any]:
    """Calculate basic and structural network statistics."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"num_nodes": 0, "num_edges": 0}

    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees) if degrees else 0

    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    largest_cc = max(connected_components, key=len) if connected_components else set()
    largest_cc_size = len(largest_cc)

    # Transitivity and average clustering
    transitivity = nx.transitivity(G)
    avg_clustering = nx.average_clustering(G)

    # Subgraph for diameter if reasonable size
    diameter = None
    avg_shortest_path = None
    if largest_cc_size > 1 and largest_cc_size <= 2500:
        sub_largest = G.subgraph(largest_cc)
        try:
            diameter = nx.diameter(sub_largest)
            avg_shortest_path = nx.average_shortest_path_length(sub_largest)
        except Exception:
            pass

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 4),
        "max_degree": max_degree,
        "num_components": num_components,
        "largest_cc_size": largest_cc_size,
        "largest_cc_ratio": round(largest_cc_size / num_nodes, 4) if num_nodes else 0,
        "transitivity": round(transitivity, 4),
        "avg_clustering": round(avg_clustering, 4),
        "diameter": diameter,
        "avg_shortest_path": round(avg_shortest_path, 4) if avg_shortest_path else None
    }

def export_network_tables(G: nx.Graph, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export node and edge tables to CSV and GraphML."""
    out_path = PROCESSED_DATA_DIR if output_dir is None else output_dir

    # Node table
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G)
    
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            "developer": node,
            "degree": degrees.get(node, 0),
            "weighted_degree": weighted_degrees.get(node, 0),
            "clustering_coefficient": round(clustering.get(node, 0.0), 4)
        })
    node_df = pd.DataFrame(nodes_data).sort_values("degree", ascending=False)
    node_df.to_csv(out_path / "nodes.csv", index=False)

    # Edge table
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
            "interaction_types": data.get("interaction_types", ""),
            "repositories": data.get("repositories", ""),
            "first_seen": data.get("first_seen", ""),
            "last_seen": data.get("last_seen", "")
        })
    edge_df = pd.DataFrame(edges_data).sort_values("weight", ascending=False)
    edge_df.to_csv(out_path / "edges.csv", index=False)

    # GraphML export for Cytoscape
    graphml_path = DELIVERABLES_DIR / "developer_network.graphml"
    nx.write_graphml(G, graphml_path)

    return node_df, edge_df
