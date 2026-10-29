"""
src/pipeline.py
----------------
Orchestrates the full end-to-end pipeline.

Can be run as a module::

    python -m src.pipeline
    python -m src.pipeline --username torvalds
    python -m src.pipeline --config path/to/config.yaml

Each phase validates the previous before proceeding.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Developer Collaboration Network & Link Prediction Pipeline"
    )
    parser.add_argument(
        "--username",
        help="GitHub username to use as the seed node (overrides config.yaml).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help=(
            "Run only up to this phase (0 = all phases). "
            "Phases: 1=setup 2=collect 3=parse 4=graph 5=predict 6=recommend "
            "7=visualize 8=report"
        ),
    )
    return parser.parse_args()


def run(config: dict, max_phase: int = 0) -> None:  # noqa: C901
    """Execute the pipeline phases in order.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.
    max_phase:
        Stop after this phase (0 = run all).
    """
    from src.archive_collector import ArchiveCollector
    from src.event_parser import EventParser
    from src.graph_builder import GraphBuilder, compute_network_stats
    from src.preprocessing import Preprocessor
    from src.link_prediction import LinkPredictor
    from src.evaluation import Evaluator
    from src.recommendations import RecommendationEngine
    from src.visualization import Visualizer
    from src.report_generator import ReportGenerator

    seed_user: str = config["github"]["username"]
    logger.info("=" * 60)
    logger.info("GitHub Collaboration Network Pipeline")
    logger.info("Seed user: %s", seed_user)
    logger.info("=" * 60)

    # ── Phase 2: Data Collection ───────────────────────────────────────
    if max_phase == 0 or max_phase >= 2:
        logger.info("\n[Phase 2] Downloading GitHub Archive files …")
        collector = ArchiveCollector(config)
        archive_files = collector.collect()
        if not archive_files:
            logger.error("No archive files downloaded. Check your date range.")
            sys.exit(1)
        logger.info("[Phase 2] ✓ %d archive files ready.", len(archive_files))
    else:
        archive_files = []

    # ── Phase 3: Event Parsing ─────────────────────────────────────────
    if max_phase == 0 or max_phase >= 3:
        logger.info("\n[Phase 3] Parsing events …")
        parser = EventParser(config)
        events_df = parser.parse_files(archive_files)
        if events_df.empty:
            logger.error("No events parsed. Try wider date range or more archive files.")
            sys.exit(1)

        # Cache processed events
        processed_path = Path("data/processed/events.parquet")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_parquet(str(processed_path), index=False)
        logger.info("[Phase 3] ✓ %d events parsed and cached.", len(events_df))
    else:
        import pandas as pd
        events_df = pd.read_parquet("data/processed/events.parquet")

    # ── Phase 4: Graph Construction ────────────────────────────────────
    if max_phase == 0 or max_phase >= 4:
        logger.info("\n[Phase 4] Building collaboration graph …")
        builder = GraphBuilder(config)
        G = builder.build(events_df)
        if G.number_of_nodes() == 0:
            logger.error(
                "Graph is empty. No collaboration events found for '%s'. "
                "Try a more active user or wider date range.",
                seed_user,
            )
            sys.exit(1)

        network_stats = compute_network_stats(G)
        logger.info("[Phase 4] Network stats: %s", network_stats)
    else:
        import networkx as nx
        G = nx.read_graphml("outputs/visualizations/collaboration_network.graphml")
        network_stats = compute_network_stats(G)

    # ── Phase 5: Link Prediction & Evaluation ─────────────────────────
    if max_phase == 0 or max_phase >= 5:
        logger.info("\n[Phase 5] Running link prediction …")
        preprocessor = Preprocessor(config)

        try:
            G_train, test_pos, test_neg = preprocessor.split(G)
        except ValueError as exc:
            logger.error("Cannot split graph: %s", exc)
            sys.exit(1)

        predictor = LinkPredictor(config)
        all_scores = predictor.score_all(G_train, test_pos, test_neg)

        evaluator = Evaluator(config)
        eval_df = evaluator.evaluate(all_scores, test_pos, test_neg)

        # Save evaluation results
        eval_out = Path("outputs/reports/evaluation_results.csv")
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(str(eval_out))
        logger.info("[Phase 5] ✓ Evaluation complete. Results: %s", eval_out)
    else:
        import pandas as pd
        import networkx as nx
        G_train = G
        predictor = LinkPredictor(config)
        eval_df = pd.DataFrame()
        test_pos, test_neg = [], []

    # ── Phase 6: Recommendations ───────────────────────────────────────
    if max_phase == 0 or max_phase >= 6:
        logger.info("\n[Phase 6] Generating recommendations …")
        engine = RecommendationEngine(config, predictor)
        rec_dfs = engine.generate(G_train, seed_user)
        logger.info("[Phase 6] ✓ Recommendations generated.")
    else:
        rec_dfs = {}

    # ── Phase 7: Visualizations ────────────────────────────────────────
    if max_phase == 0 or max_phase >= 7:
        logger.info("\n[Phase 7] Generating visualizations …")
        viz = Visualizer(config)
        img_paths = []

        p1 = viz.plot_network(G_train)
        img_paths.append(p1)

        p2 = viz.plot_degree_distribution(G_train)
        img_paths.append(p2)

        if not eval_df.empty:
            p3 = viz.plot_evaluation(eval_df)
            img_paths.append(p3)

        viz.interactive_network(G_train)
        viz.export_graphml(G_train)
        logger.info("[Phase 7] ✓ All visualizations saved.")
    else:
        img_paths = []

    # ── Phase 8: Reports ───────────────────────────────────────────────
    if max_phase == 0 or max_phase >= 8:
        logger.info("\n[Phase 8] Generating reports …")
        gen = ReportGenerator(config)
        gen.generate_pdf(network_stats, eval_df, rec_dfs, img_paths)
        gen.generate_pptx(network_stats, eval_df, rec_dfs, img_paths)
        logger.info("[Phase 8] ✓ PDF report and PowerPoint saved.")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete. Outputs in: outputs/")
    logger.info("=" * 60)


def main() -> None:
    """Entry point for ``python -m src.pipeline``."""
    args = parse_args()

    from src import load_config
    config = load_config(args.config)

    # CLI username overrides config file
    if args.username:
        config.setdefault("github", {})["username"] = args.username
        logger.info("Username overridden via CLI: %s", args.username)

    # Set up file logging
    log_cfg = config.get("logging", {})
    log_file = Path(log_cfg.get("log_file", "outputs/pipeline.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    run(config, max_phase=args.phase)


if __name__ == "__main__":
    main()

