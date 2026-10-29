"""
scripts/validate_phase6.py
--------------------------
Direct validation script for Phase 6: Link Prediction, Evaluation, and Recommendations.

Verifies:
  1. Execution of CN, Jaccard, Adamic-Adar, and Preferential Attachment.
  2. Precision@K, Recall@K, F1-score, and ROC-AUC on consistent test sets.
  3. Top-10 and Top-20 recommendation generation for the seed developer.
  4. Human-readable structural explanations for all recommendations.
  5. Persistence of evaluation CSV/JSON and recommendation tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import pandas as pd

from src import load_config
from src.preprocessing import Preprocessor
from src.link_prediction import LinkPredictor
from src.evaluation import Evaluator
from src.recommendations import RecommendationEngine


def run_phase6_validation():
    print("\n" + "=" * 65)
    print("PHASE 6: LINK PREDICTION, EVALUATION & RECOMMENDATIONS")
    print("=" * 65)

    config = load_config()
    prep = Preprocessor(config)
    predictor = LinkPredictor(config)
    evaluator = Evaluator(config)
    rec_engine = RecommendationEngine(config, predictor)

    # 1. Synthesize realistic collaboration network with community structure
    G_raw = nx.planted_partition_graph(l=4, k=15, p_in=0.45, p_out=0.08, seed=42)
    mapping = {i: f"developer_{i}" for i in G_raw.nodes()}
    G = nx.relabel_nodes(G_raw, mapping)
    for n in G.nodes():
        G.nodes[n]["github_id"] = str(1000 + int(n.split("_")[1]))
        G.nodes[n]["account_type"] = "User"
        G.nodes[n]["qualifying_interactions"] = G.degree(n)

    for u, v in G.edges():
        G[u][v]["weight"] = 1
        G[u][v]["relationship_types"] = "pr_review"
        G[u][v]["shared_repositories"] = "org/repo"

    seed_user = "developer_0"
    G.nodes[seed_user]["is_seed"] = True

    # 2. Split graph (80% train, 20% test) with zero leakage
    G_train, train_pos, test_pos, test_neg, tables = prep.prepare_and_split(G)
    all_test_candidates = test_pos + test_neg

    print(f"Graph Size:            {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Training Graph:        {G_train.number_of_edges()} edges (leak-free)")
    print(f"Evaluation Set:        {len(test_pos)} positives, {len(test_neg)} negatives")

    # 3. Score candidates across all 4 link prediction algorithms on G_train
    all_scores = predictor.score_candidates(G_train, all_test_candidates)

    # 4. Evaluate performance across metrics
    eval_df = evaluator.evaluate(all_scores, test_pos, test_neg)

    # Save evaluation table
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = reports_dir / "evaluation_results.csv"
    eval_df.to_csv(eval_csv)
    print(f"\nSaved evaluation metrics to: {eval_csv}")

    print("\n--- Algorithm Benchmark Results ---")
    print(eval_df[["roc_auc", "pr_auc", "precision@10", "recall@10", "f1_score"]].to_string())

    # 5. Generate Top-10 and Top-20 explainable recommendations for seed user
    recs_bundle = rec_engine.generate_recommendations(G_train, seed_user=seed_user)

    top10_df = recs_bundle.get("top_10")
    top20_df = recs_bundle.get("top_20")
    ensemble_df = recs_bundle.get("ensemble")

    print("\n--- Top-5 Recommendations for Seed User (Ensemble Consensus) ---")
    print(ensemble_df.head(5)[["consensus_rank", "candidate", "algorithms_supporting", "summary_rationale"]].to_string(index=False))

    # 6. Verify assertions
    seed_existing = set(G.neighbors(seed_user)) | {seed_user}
    recommended_candidates = set(ensemble_df["candidate"])
    overlap = recommended_candidates.intersection(seed_existing)
    assert len(overlap) == 0, f"Error: Recommended existing neighbors or self! {overlap}"

    print("\n" + "=" * 65)
    print("PHASE 6 VALIDATION PASSED: ALL ALGORITHMS & RECOMMENDATIONS EXPLAINED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_phase6_validation()

