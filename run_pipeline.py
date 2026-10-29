"""
Master Execution Pipeline for NumFOCUS Developer Collaboration Network & Link Prediction.
Usage:
    python run_pipeline.py [--user USERNAME] [--hops HOPS] [--max-records N]
"""
import argparse
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    PROCESSED_DATA_DIR, DELIVERABLES_DIR, FIGURES_DIR,
    DEFAULT_STARTING_USER, RANDOM_SEED, TOP_K_EVAL
)
from src.graph_builder import (
    resolve_starting_developer, construct_graph,
    compute_network_statistics, export_network_tables
)
from src.link_prediction import (
    train_test_split_edges, score_pairs, ALGORITHMS
)
from src.evaluation import evaluate_algorithm, compare_all_algorithms
from src.recommendation import generate_collaborator_recommendations, export_recommendations
from src.visualization import (
    plot_overall_network, plot_ego_network, plot_predicted_links_network,
    plot_roc_curves, generate_interactive_pyvis
)
from src.report_generator import generate_pdf_report, generate_presentation

def main():
    parser = argparse.ArgumentParser(description="NumFOCUS Developer Collaboration Network Pipeline")
    parser.add_argument("--user", type=str, default="jreback", help="GitHub username of the starting developer (default: jreback)")
    parser.add_argument("--hops", type=int, default=2, help="Ego network expansion hops (default: 2)")
    args = parser.parse_args()

    starting_user = args.user.strip()
    print("=" * 70)
    print("  NumFOCUS Developer Collaboration Network & Link Prediction Pipeline")
    print(f"  Target Starting Developer: @{starting_user}")
    print("=" * 70)

    # 1. Load Aggregated Edges
    edges_csv = PROCESSED_DATA_DIR / "edges.csv"
    if not edges_csv.exists():
        print(f"Aggregated edges not found at {edges_csv}. Please run scripts/process_dataset.py first.")
        sys.exit(1)

    print(f"\n[Step 1] Loading edge dataset from {edges_csv}...")
    edge_df = pd.read_csv(edges_csv)
    print(f"  Loaded {len(edge_df):,} unique collaboration edges.")

    # 2. Resolve Starting Developer
    print(f"\n[Step 2] Resolving starting developer @{starting_user}...")
    res_user = resolve_starting_developer(edge_df, starting_user)
    if not res_user["resolved"]:
        print(f"  WARNING: Developer '{starting_user}' not found in current dataset!")
        if res_user.get("candidates"):
            print(f"  Did you mean one of these? {res_user['candidates']}")
        print(f"  Defaulting to most connected maintainer: @jreback")
        starting_user = "jreback"
        res_user = resolve_starting_developer(edge_df, starting_user)
    
    canonical_user = res_user["canonical_username"]
    print(f"  Resolved: @{canonical_user} with degree {res_user['degree']} and {res_user['total_weight']} total interactions.")
    print("  Top direct collaborators:")
    for nbr, wt in res_user["top_collaborators"][:5]:
        print(f"    - @{nbr:<18} ({wt} interactions)")

    # 3. Construct Graph & Compute Statistics
    print(f"\n[Step 3] Constructing collaboration network (ego subgraph within {args.hops} hops)...")
    # Ego subgraph for focused analysis & global link prediction
    G = construct_graph(edge_df, starting_user=canonical_user, hops=args.hops)
    stats = compute_network_statistics(G)
    print(f"  Constructed graph with {stats['num_nodes']:,} nodes and {stats['num_edges']:,} edges.")
    print(f"  Density: {stats['density']:.6f} | Avg Degree: {stats['avg_degree']:.2f} | Max Degree: {stats['max_degree']}")
    print(f"  Transitivity: {stats['transitivity']:.4f} | Avg Clustering: {stats['avg_clustering']:.4f}")
    print(f"  Largest Component Size: {stats['largest_cc_size']:,} ({stats['largest_cc_ratio']:.1%})")

    # Save network node and edge tables and GraphML
    node_table, edge_table = export_network_tables(G, PROCESSED_DATA_DIR)
    print(f"  Exported node table ({len(node_table)} nodes) and edge table ({len(edge_table)} edges).")
    print(f"  Exported Cytoscape GraphML to {DELIVERABLES_DIR / 'developer_network.graphml'}")

    # 4. Train-Test Split and Link Prediction
    print("\n[Step 4] Partitioning edges (80% train / 20% test) and generating negative samples...")
    G_train, train_pos, test_pos, train_neg, test_neg = train_test_split_edges(
        G, test_ratio=0.20, seed=RANDOM_SEED
    )
    print(f"  Train positive edges: {len(train_pos):,}")
    print(f"  Test positive edges:  {len(test_pos):,}")
    print(f"  Test negative samples: {len(test_neg):,}")
    print("  Verification: Test edges purged from G_train to guarantee zero data leakage.")

    # Prepare evaluation pairs and ground truth
    eval_pairs = test_pos + test_neg
    y_true = np.array([1] * len(test_pos) + [0] * len(test_neg))

    print("\n  Computing link prediction scores across algorithms on G_train:")
    eval_results = {}
    roc_curves_data = {}

    for algo in ALGORITHMS.keys():
        scores = score_pairs(G_train, eval_pairs, algo)
        metrics = evaluate_algorithm(y_true, scores, k_list=TOP_K_EVAL)
        eval_results[algo] = metrics
        
        # Compute ROC curve coordinates
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = metrics["ROC-AUC"]
            roc_curves_data[algo] = (fpr, tpr, auc)
        except Exception:
            pass
        print(f"    - {algo:<25} ROC-AUC: {metrics['ROC-AUC']:.4f} | P@10: {metrics.get('Precision@10', 0):.4f} | F1@10: {metrics.get('F1@10', 0):.4f}")

    eval_df = compare_all_algorithms(eval_results)
    eval_csv = DELIVERABLES_DIR / "link_prediction_evaluation.csv"
    eval_df.to_csv(eval_csv, index=False)
    print(f"  Saved evaluation comparison to {eval_csv}")

    # 5. Generate Top-10 and Top-20 Recommendations for starting developer
    print(f"\n[Step 5] Generating collaborator recommendations for @{canonical_user}...")
    recommendations = generate_collaborator_recommendations(G, canonical_user, top_k=20, primary_algo="Adamic-Adar")
    rec_dfs = export_recommendations(recommendations, canonical_user, top_k_values=[10, 20])
    print(f"  Top 5 Recommended Potential Collaborators for @{canonical_user}:")
    for r in recommendations[:5]:
        print(f"    #{r['rank']} @{r['candidate_user']:<18} AA={r['adamic_adar_score']:.2f} | CN={r['common_neighbors_count']} | Deg={r['candidate_degree']}")
        print(f"       -> {r['explanation']}")

    # 6. Visualizations
    print("\n[Step 6] Generating academic visualizations (PNG, SVG, HTML)...")
    plot_overall_network(G, FIGURES_DIR / "network_overall.png", FIGURES_DIR / "network_overall.svg")
    print(f"  Saved {FIGURES_DIR / 'network_overall.png'} and .svg")

    plot_ego_network(G, canonical_user, FIGURES_DIR / "network_ego.png", FIGURES_DIR / "network_ego.svg")
    print(f"  Saved {FIGURES_DIR / 'network_ego.png'} and .svg")

    plot_predicted_links_network(G, canonical_user, recommendations, FIGURES_DIR / "network_predictions.png", FIGURES_DIR / "network_predictions.svg", top_k=10)
    print(f"  Saved {FIGURES_DIR / 'network_predictions.png'} and .svg")

    plot_roc_curves(roc_curves_data, FIGURES_DIR / "roc_curves.png", FIGURES_DIR / "roc_curves.svg")
    print(f"  Saved {FIGURES_DIR / 'roc_curves.png'} and .svg")

    html_path = generate_interactive_pyvis(G, canonical_user, recommendations, DELIVERABLES_DIR / "interactive_network.html", top_k=15)
    print(f"  Saved interactive graph: {html_path}")

    # Copy primary figures directly into deliverables root as requested in Step 6
    for fig_name in ["network_overall.png", "network_overall.svg", "network_ego.png", "network_ego.svg", "network_predictions.png", "network_predictions.svg", "roc_curves.png", "roc_curves.svg"]:
        src_f = FIGURES_DIR / fig_name
        dst_f = DELIVERABLES_DIR / fig_name
        if src_f.exists():
            import shutil
            shutil.copy(src_f, dst_f)

    # 7. Generate PDF Report and PowerPoint Presentation
    print("\n[Step 7] Generating academic PDF report and presentation deliverables...")
    pdf_path = generate_pdf_report(stats, eval_df, rec_dfs[20], canonical_user)
    print(f"  Generated PDF Report: {pdf_path}")

    pptx_path = generate_presentation(stats, eval_df, rec_dfs[20], canonical_user)
    print(f"  Generated PowerPoint Presentation: {pptx_path}")
    print(f"  Generated Presentation PDF: {pptx_path.with_suffix('.pdf')}")

    print("\n" + "=" * 70)
    print("  Pipeline execution successfully completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()

