"""
scripts/run_phase2.py
---------------------
Execute Phase 2: Understand GitHub Archive Data.
Downloads a sample archive file, inspects real schemas, identifies direct interactions,
evaluates data coverage and seed user presence, and generates validation deliverables.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config
from src.archive_collector import ArchiveCollector
from src.archive_inspector import ArchiveInspector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2")


def main():
    cfg = load_config()
    logger.info("Initializing Phase 2: GitHub Archive Data Understanding & Coverage Analysis")
    
    # 1. Download / cache real archive file(s)
    collector = ArchiveCollector(cfg)
    # Collect 1 hour file to ensure fast laptop execution while inspecting authentic data
    archive_files = collector.collect(limit=1)
    
    if not archive_files:
        logger.error("No archive files could be collected. Please check internet connection or dates.")
        sys.exit(1)
        
    logger.info("Examining %d archive file(s): %s", len(archive_files), [f.name for f in archive_files])
    
    # 2. Run deep inspection
    inspector = ArchiveInspector(cfg)
    summary = inspector.inspect_archives(archive_files)
    
    # 3. Export JSON report
    json_path = Path("outputs/reports/phase2_coverage_summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON summary to %s", json_path)
    
    # 4. Export Markdown report
    md_path = Path("data/coverage_summary.md")
    inspector.generate_markdown_report(summary, md_path)
    logger.info("Saved Markdown coverage report to %s", md_path)
    
    # 5. Print Summary
    print("\n" + "=" * 60)
    print("PHASE 2 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Archives Examined:             {summary['archives_examined']} ({', '.join(summary['archive_files'])})")
    print(f"Total Events Scanned:          {summary['total_events_scanned']:,}")
    print(f"Unique Repositories:           {summary['unique_repositories']:,}")
    print(f"Unique Human Developers:       {summary['unique_human_actors']:,}")
    print(f"Bot Accounts Filtered:         {summary['unique_bot_actors']:,}")
    print(f"Qualifying Direct Interactions:{summary['qualifying_direct_interactions']:,}")
    print("Direct Interaction Types:")
    for k, v in summary["direct_interaction_types"].items():
        print(f"  - {k}: {v:,}")
    print(f"Seed User:                     '{summary['seed_user']}'")
    print(f"Seed User Events in Archive:   {summary['seed_user_events_found']}")
    if summary['seed_user_events_found'] == 0:
        print("\n[NOTE ON SEED USER ACTIVITY]")
        print(f"Seed user '{summary['seed_user']}' had 0 events in this 1-hour archive window.")
        print("This is expected for quiet/student accounts in narrow public slices.")
        print("The pipeline incorporates GitHub REST API enrichment to harvest the seed user's")
        print("repositories and collaborators to bootstrap the ego-network.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

