"""
scripts/generate_all_deliverables.py
-----------------------------------
End-to-End master runner that generates every project deliverable:
  - Overall network (PNG & SVG)
  - Starting-node ego network (PNG & SVG)
  - Top-10 predicted links (PNG & SVG, distinguishing observed vs predicted edges)
  - Interactive HTML visualization (PyVis)
  - Cytoscape-compatible GraphML
  - Formatted academic PDF report (ReportLab)
  - PowerPoint presentation deck (python-pptx)
  - Recommendation CSVs and JSONs
  - Full file existence and size validation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import pandas as pd

from src import load_config
from src.archive_collector import ArchiveCollector
from src.event_parser import EventParser
from src.graph_builder import GraphBuilder, compute_network_stats
from src.preprocessing import Preprocessor
from src.link_prediction import LinkPredictor
from src.evaluation import Evaluator
from src.recommendations import RecommendationEngine
from src.visualization import Visualizer
from src.report_generator import ReportGenerator


def main():
    print("\n" + "=" * 65)
    print("EXECUTING FULL PIPELINE & GENERATING ALL DELIVERABLES")
    print("=" * 65)

    config = load_config()

    # 1. Ingest cached archive
    collector = ArchiveCollector(config)
    archive_paths = collector.list_cached()
    if not archive_paths:
        print("Downloading sample archive file...")
        archive_paths = collector.collect(limit=1)

    print(f"Archive file: {archive_paths[0].name}")

    # 2. Parse direct interactions
    parser = EventParser(config)
    df_events = parser.parse_files(archive_paths)
    print(f"Direct dyadic interactions extracted: {len(df_events):,}")

    # Pick developer with highest collaborator diversity (distinct collaborators)
    collab_counts = df_events.groupby("source_login")["target_login"].nunique().sort_values(ascending=False)
    # Filter out any bots that slipped past
    candidates = [u for u in collab_counts.index if not parser.should_exclude(u)]
    top_user = candidates[0] if candidates else collab_counts.index[0]
    config["github"]["username"] = top_user
    print(f"Active seed developer selected for network construction: '{top_user}' ({collab_counts[top_user]} direct collaborators)")

    # 3. Build network
    builder = GraphBuilder(config)
    builder._seed_user = top_user
    builder._max_depth = 2
    G, status = builder.build_network(df_events)
    stats = compute_network_stats(G)
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 4. Train-Test Split (zero leakage)
    prep = Preprocessor(config)
    G_train, train_pos, test_pos, test_neg, tables = prep.prepare_and_split(G)
    print(f"Train/Test split: {len(train_pos)} train edges, {len(test_pos)} test edges, {len(test_neg)} negative non-edges")

    # 5. Link Prediction & Evaluation
    predictor = LinkPredictor(config)
    evaluator = Evaluator(config)
    all_candidates = test_pos + test_neg
    scores = predictor.score_candidates(G_train, all_candidates)
    eval_df = evaluator.evaluate(scores, test_pos, test_neg)

    # 6. Explainable Recommendations
    rec_engine = RecommendationEngine(config, predictor)
    recs = rec_engine.generate_recommendations(G_train, seed_user=top_user)

    # 7. Visualizations (PNG, SVG, Interactive HTML, GraphML)
    viz = Visualizer(config)
    img_overall = viz.plot_overall_network(G, seed_user=top_user)
    img_ego = viz.plot_ego_network(G, seed_user=top_user)

    # Format top predicted edges for top10 plot
    top10_raw = recs["raw"].get("common_neighbors", [])[:10]
    top10_pred_edges = [(r["seed_user"], r["candidate"], r["score"]) for r in top10_raw]
    img_pred = viz.plot_top10_predicted_links(G_train, seed_user=top_user, predicted_edges=top10_pred_edges)

    html_path = viz.interactive_network_html(G_train, seed_user=top_user, predicted_edges=top10_pred_edges)
    graphml_path = viz.export_graphml(G, "cytoscape_network.graphml")

    # 8. PDF Report & PowerPoint Presentation
    rep_gen = ReportGenerator(config)
    image_paths_for_pdf = [img_overall["png"], img_ego["png"], img_pred["png"]]

    pdf_path = rep_gen.generate_pdf_report(
        network_stats=stats,
        eval_df=eval_df,
        recommendations_df=recs["ensemble"],
        image_paths=image_paths_for_pdf,
    )

    pptx_path = rep_gen.generate_presentation(
        network_stats=stats,
        eval_df=eval_df,
        recommendations_df=recs["ensemble"],
        image_paths=image_paths_for_pdf,
    )

    # 9. Deliverables Verification
    print("\n" + "=" * 65)
    print("DELIVERABLES VERIFICATION SUMMARY")
    print("=" * 65)

    deliverables = [
        ("Overall Network (PNG)", img_overall["png"]),
        ("Overall Network (SVG)", img_overall["svg"]),
        ("Ego Network (PNG)", img_ego["png"]),
        ("Ego Network (SVG)", img_ego["svg"]),
        ("Top-10 Predicted Links (PNG)", img_pred["png"]),
        ("Top-10 Predicted Links (SVG)", img_pred["svg"]),
        ("Interactive HTML Network", html_path),
        ("Cytoscape GraphML", graphml_path),
        ("Academic Evaluation PDF Report", pdf_path),
        ("Academic Presentation PPTX", pptx_path),
        ("Top-10 Recommendations CSV", Path("outputs/recommendations/recommendations_top10.csv")),
        ("Top-20 Recommendations CSV", Path("outputs/recommendations/recommendations_top20.csv")),
        ("Ensemble Consensus CSV", Path("outputs/recommendations/recommendations_ensemble.csv")),
        ("Recommendations JSON (Explained)", Path("outputs/recommendations/recommendations_explained.json")),
        ("Algorithm Evaluation CSV", Path("outputs/reports/evaluation_results.csv")),
        ("Viva Voce Q&A Document", Path("docs/viva_questions_and_answers.md")),
        ("Executable Jupyter Notebook", Path("notebooks/github_link_prediction.ipynb")),
    ]

    all_exist = True
    for label, path in deliverables:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            print(f"[OK] {label:<35} -> {str(p)} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"[FAILED] {label:<35} -> Missing or Empty!")
            all_exist = False

    print("=" * 65)
    if all_exist:
        print("ALL DELIVERABLES SUCCESSFULLY GENERATED AND VERIFIED!")
    else:
        print("WARNING: Some deliverables failed to generate.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
