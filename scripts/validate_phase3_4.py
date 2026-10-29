"""
scripts/validate_phase3_4.py
----------------------------
Direct test script for Phase 3 and Phase 4.
Validates all required scenarios:
  1. Active user with qualifying public interactions.
  2. Valid user with sparse / insufficient interactions.
  3. Invalid or unresolvable username.
  4. Configurable expansion depth (1-hop vs 2-hop).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config
from src.archive_collector import ArchiveCollector
from src.collaborator_discovery import CollaboratorDiscoveryEngine


def run_tests():
    config = load_config()
    collector = ArchiveCollector(config)
    archive_paths = collector.list_cached()
    if not archive_paths:
        archive_paths = collector.collect(limit=1)

    engine = CollaboratorDiscoveryEngine(config)
    print("Parsing archive records once for test suite...")
    events_df = engine.parser.parse_files(archive_paths)

    print("\n" + "=" * 65)
    print("PHASE 3 & 4: COLLABORATION NETWORK DISCOVERY VALIDATION")
    print("=" * 65)

    test_cases = [
        ("Valid Active User (hamishwillee) [1-hop]", "hamishwillee", 1),
        ("Valid Active User (hamishwillee) [2-hop]", "hamishwillee", 2),
        ("Quiet / Insufficient User (varshavkumar12345)", "varshavkumar12345", 1),
        ("Invalid Username (non_existent_dev_999999xyz)", "non_existent_dev_999999xyz", 1),
    ]

    for label, uname, depth in test_cases:
        print(f"\n--- Testing Scenario: {label} ---")
        G, report = engine.discover_and_build(uname, archive_paths, expand_depth=depth, pre_parsed_df=events_df)
        print(f"Status:             {report['status']}")
        if report["resolved_user"]:
            u = report["resolved_user"]
            print(f"Resolved Identity:  {u['login']} (GitHub ID: {u['id']}, Repos: {u['public_repos']})")
        if G is not None:
            print(f"Network Nodes:      {report['graph_nodes']}")
            print(f"Network Edges:      {report['graph_edges']}")
            print(f"Direct Collabs:     {len(report['direct_collaborators'])} ({', '.join(report['direct_collaborators'][:5])}...)")
            print(f"Stats:              {report.get('network_stats')}")
        else:
            print(f"Explanation:        {report['error_or_reason']}")

    print("\n" + "=" * 65)
    print("ALL PHASE 3 & PHASE 4 VALIDATIONS COMPLETED.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tests()
