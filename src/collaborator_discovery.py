"""
src/collaborator_discovery.py
------------------------------
Core engine for Phase 3 and Phase 4.
Accepts a starting developer, resolves user identity, searches archive interactions,
boosts with API collaboration links if needed, and builds the verifiable developer graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.github_client import GitHubClient
from src.event_parser import EventParser
from src.graph_builder import GraphBuilder, compute_network_stats

logger = logging.getLogger(__name__)


class CollaboratorDiscoveryEngine:
    """Discovers collaborators and constructs the Phase 4 collaboration graph."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = GitHubClient(config)
        self.parser = EventParser(config)
        self.builder = GraphBuilder(config)

    def discover_and_build(
        self,
        username: str,
        archive_paths: List[Path],
        expand_depth: int = 1,
        pre_parsed_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[nx.Graph], Dict[str, Any]]:
        """
        Accept starting developer, resolve identity, extract direct interactions,
        and build 1-hop or 2-hop network.
        """
        username = username.strip()
        report: Dict[str, Any] = {
            "query_username": username,
            "status": "INIT",
            "resolved_user": None,
            "archive_interactions_found": 0,
            "api_collaborators_found": 0,
            "direct_collaborators": [],
            "graph_nodes": 0,
            "graph_edges": 0,
            "error_or_reason": "",
        }

        # 1. Parse direct interactions (use cache/pre-parsed if provided for speed)
        events_df = pre_parsed_df if pre_parsed_df is not None else self.parser.parse_files(archive_paths)
        report["total_archive_records_parsed"] = len(events_df)
        canonical_login = username.lower()

        # 2. Resolve user identity via GitHub API (with graceful fallback to archive identity)
        user_meta = self.client.get_user(username)
        if not user_meta or "login" not in user_meta:
            local_user = events_df[
                (events_df["source_login"] == canonical_login) |
                (events_df["target_login"] == canonical_login)
            ]
            if not local_user.empty:
                uid = local_user.iloc[0]["source_id"] if local_user.iloc[0]["source_login"] == canonical_login else local_user.iloc[0]["target_id"]
                user_meta = {
                    "login": username,
                    "id": uid or "archive_id",
                    "name": username,
                    "public_repos": "N/A",
                    "type": "User",
                }
            else:
                report["status"] = "UNRESOLVED_USER"
                report["error_or_reason"] = (
                    f"Username '{username}' could not be resolved. The user does not exist on GitHub, "
                    "or has 0 interactions in the archive slice and public API access was unavailable."
                )
                return None, report
        report["total_archive_records_parsed"] = len(events_df)

        seed_interactions = events_df[
            (events_df["source_login"] == canonical_login) |
            (events_df["target_login"] == canonical_login)
        ]
        report["archive_interactions_found"] = len(seed_interactions)

        # 3. If archive activity is 0, query public repository contributors/PRs as bootstrap
        api_collaborators = []
        if seed_interactions.empty:
            logger.info("No archive interactions for %s. Querying public repos for collaborators...", canonical_login)
            repos = self.client.get_user_repos(canonical_login, max_pages=1)
            for repo in repos[:3]:
                rname = repo.get("name")
                owner = canonical_login
                prs = self.client.get_repo_pull_requests(owner, rname, max_pages=1)
                for pr in prs:
                    pr_author = (pr.get("user") or {}).get("login")
                    if pr_author and pr_author.lower() != canonical_login:
                        api_collaborators.append({
                            "login": pr_author,
                            "id": (pr.get("user") or {}).get("id"),
                            "relationship": "pr_participation",
                            "repo": f"{owner}/{rname}",
                        })
                contribs = self.client.get_repo_contributors(owner, rname, max_pages=1)
                for c in contribs:
                    clogin = c.get("login")
                    if clogin and clogin.lower() != canonical_login:
                        api_collaborators.append({
                            "login": clogin,
                            "id": c.get("id"),
                            "relationship": "co_contributor",
                            "repo": f"{owner}/{rname}",
                        })

            report["api_collaborators_found"] = len(api_collaborators)

        # 4. Check for insufficient activity
        if seed_interactions.empty and not api_collaborators:
            report["status"] = "INSUFFICIENT_ACTIVITY"
            report["error_or_reason"] = (
                f"Valid user '{canonical_login}' was resolved, but no qualifying public development "
                f"interactions (PR reviews, review comments, issue discussions, or co-contributions) "
                f"were found within the selected period or public repos. Graph cannot be constructed."
            )
            return None, report

        # 5. Build Graph
        self.builder._seed_user = canonical_login
        self.builder._max_depth = expand_depth
        G, status = self.builder.build_network(events_df, user_meta, api_collaborators)

        if G.number_of_nodes() <= 1:
            report["status"] = "INSUFFICIENT_COLLABORATORS"
            report["error_or_reason"] = (
                f"User '{canonical_login}' has zero collaborators in the graph after filtering."
            )
            return None, report

        direct_collabs = list(G.neighbors(canonical_login))
        report["status"] = "SUCCESS"
        report["direct_collaborators"] = direct_collabs
        report["graph_nodes"] = G.number_of_nodes()
        report["graph_edges"] = G.number_of_edges()
        report["network_stats"] = compute_network_stats(G)
        return G, report
