"""
Collaborator recommendation engine with multi-algorithm ranking and feature explanations.
"""
from typing import List, Dict, Any, Optional
import networkx as nx
import pandas as pd
from src.graph_builder import is_bot
from src.link_prediction import (
    compute_common_neighbors,
    compute_jaccard_coefficient,
    compute_adamic_adar,
    compute_preferential_attachment
)
from src.config import PROCESSED_DATA_DIR, DELIVERABLES_DIR

def generate_collaborator_recommendations(
    G: nx.Graph,
    target_user: str,
    top_k: int = 20,
    primary_algo: str = "Adamic-Adar"
) -> List[Dict[str, Any]]:
    """
    Generate potential collaborator recommendations for target_user:
    - Filters out existing neighbors, self, and bots.
    - Computes all 4 topological metrics (CN, JC, AA, PA).
    - Sorts by primary algorithm.
    - Provides rich explanatory feature attribution.
    """
    if target_user not in G:
        return []

    existing_neighbors = set(G.neighbors(target_user))
    candidates = []

    for candidate in G.nodes():
        if candidate == target_user:
            continue
        if candidate in existing_neighbors:
            continue
        if is_bot(candidate):
            continue

        cn = compute_common_neighbors(G, target_user, candidate)
        jc = compute_jaccard_coefficient(G, target_user, candidate)
        aa = compute_adamic_adar(G, target_user, candidate)
        pa = compute_preferential_attachment(G, target_user, candidate)

        # Retrieve shared neighbors
        shared = sorted(list(set(G.neighbors(target_user)).intersection(set(G.neighbors(candidate)))))

        candidates.append({
            "target_user": target_user,
            "candidate_user": candidate,
            "candidate_degree": G.degree(candidate),
            "common_neighbors_count": int(cn),
            "common_neighbors_list": shared,
            "jaccard_coefficient": round(jc, 4),
            "adamic_adar_score": round(aa, 4),
            "preferential_attachment": int(pa)
        })

    # Sort based on primary algorithm
    key_map = {
        "Adamic-Adar": lambda x: (x["adamic_adar_score"], x["common_neighbors_count"], x["preferential_attachment"]),
        "Common Neighbors": lambda x: (x["common_neighbors_count"], x["adamic_adar_score"], x["preferential_attachment"]),
        "Jaccard Coefficient": lambda x: (x["jaccard_coefficient"], x["adamic_adar_score"], x["preferential_attachment"]),
        "Preferential Attachment": lambda x: (x["preferential_attachment"], x["common_neighbors_count"])
    }
    sort_func = key_map.get(primary_algo, key_map["Adamic-Adar"])
    candidates.sort(key=sort_func, reverse=True)

    top_candidates = candidates[:top_k]

    # Generate explanations
    for rank, item in enumerate(top_candidates, 1):
        item["rank"] = rank
        shared_str = ", ".join(item["common_neighbors_list"][:3])
        if len(item["common_neighbors_list"]) > 3:
            shared_str += f" (+{len(item['common_neighbors_list']) - 3} others)"

        if item["common_neighbors_count"] > 0:
            item["explanation"] = (
                f"Shares {item['common_neighbors_count']} mutual collaborator(s) [{shared_str}]. "
                f"Adamic-Adar: {item['adamic_adar_score']}, Jaccard: {item['jaccard_coefficient']}, "
                f"Preferential Attachment: {item['preferential_attachment']} (deg={item['candidate_degree']})."
            )
        else:
            item["explanation"] = (
                f"High network prominence (degree={item['candidate_degree']}) yielding "
                f"Preferential Attachment score {item['preferential_attachment']}. Potential cross-project bridge."
            )

    return top_candidates

def export_recommendations(
    recommendations: List[Dict[str, Any]],
    target_user: str,
    top_k_values: List[int] = [10, 20]
) -> Dict[int, pd.DataFrame]:
    """Export recommendations table to CSV and JSON in deliverables folder."""
    result = {}
    for k in top_k_values:
        sub_recs = recommendations[:k]
        rows = []
        for r in sub_recs:
            rows.append({
                "Rank": r["rank"],
                "Target Developer": r["target_user"],
                "Recommended Collaborator": r["candidate_user"],
                "Candidate Degree": r["candidate_degree"],
                "Common Neighbors": r["common_neighbors_count"],
                "Adamic-Adar": r["adamic_adar_score"],
                "Jaccard": r["jaccard_coefficient"],
                "Preferential Attachment": r["preferential_attachment"],
                "Mutual Collaborators": "; ".join(r["common_neighbors_list"][:5]),
                "Explanation": r["explanation"]
            })
        df = pd.DataFrame(rows)
        out_csv = DELIVERABLES_DIR / f"recommendations_top_{k}_{target_user}.csv"
        df.to_csv(out_csv, index=False)
        result[k] = df

    return result

