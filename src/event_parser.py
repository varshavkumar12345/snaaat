"""
src/event_parser.py
--------------------
Parse raw GitHub Archive records into verifiable, provenance-tracked direct interactions.

Strict Interaction Rules:
  - Do NOT create edges between every pair who touched the same repo.
  - Only direct dyadic actions:
      1. PR Reviews (reviewer -> PR author)
      2. PR Review Comments (inline code commenter -> PR author)
      3. Issue Comments (commenter -> issue author)
      4. PR Merges / Closures (maintainer -> PR author)
      5. Member Additions (admin -> member)
      6. Issue/PR Assignment (assigner -> assignee)
  - Numeric GitHub ID resolution preferred where available.
  - Complete provenance tracked (source event_id, event_type, repository, timestamp, relationship).
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

COLLABORATION_EVENTS = {
    "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent",
    "IssueCommentEvent",
    "PullRequestEvent",
    "IssuesEvent",
    "MemberEvent",
}


class EventParser:
    """Parse archive files into direct dyadic developer interactions with full provenance."""

    def __init__(self, config: dict) -> None:
        self._config = config
        filtering = config.get("filtering", {})
        self._exclude_bots: bool = filtering.get("exclude_bots", True)
        self._auto_detect_bots: bool = filtering.get("auto_detect_bots", True)
        self._extra_exclusions: Set[str] = set(filtering.get("extra_exclusions", []))

    def parse_files(self, paths: List[Path]) -> pd.DataFrame:
        """Parse archive files and return direct dyadic developer interactions."""
        all_interactions: List[Dict[str, Any]] = []

        for path in tqdm(paths, desc="Parsing interactions", unit="file"):
            interactions = self._parse_file(path)
            all_interactions.extend(interactions)

        if not all_interactions:
            logger.warning("No qualifying direct interactions found.")
            return pd.DataFrame(
                columns=[
                    "source_id", "source_login", "target_id", "target_login",
                    "relationship", "repo", "event_id", "created_at"
                ]
            )

        df = pd.DataFrame(all_interactions)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["source_login", "target_login"])
        df = df.drop_duplicates(subset=["source_login", "target_login", "event_id", "relationship"])
        return df

    def _parse_file(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with gzip.open(path, "rb") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    interactions = self.extract_direct_interactions(event)
                    records.extend(interactions)
        except Exception as exc:
            logger.error("Error reading %s: %s", path, exc)
        return records

    def extract_direct_interactions(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract verified dyadic interactions from an event dict."""
        etype = event.get("type", "")
        if etype not in COLLABORATION_EVENTS:
            return []

        actor = event.get("actor") or {}
        actor_login = actor.get("login") if isinstance(actor, dict) else str(actor)
        actor_id = actor.get("id") if isinstance(actor, dict) else None

        if not actor_login or self.should_exclude(actor_login):
            return []

        repo = (event.get("repo") or {}).get("name", "")
        event_id = str(event.get("id", ""))
        created_at = event.get("created_at", "")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            return []

        results: List[Dict[str, Any]] = []

        # 1. PullRequestReviewEvent
        if etype == "PullRequestReviewEvent":
            pr = payload.get("pull_request") or {}
            target = pr.get("user") or {}
            target_login = target.get("login")
            target_id = target.get("id")
            if target_login and target_login != actor_login and not self.should_exclude(target_login):
                results.append(self._make_rec(actor_id, actor_login, target_id, target_login,
                                              "pr_review", repo, event_id, created_at))

        # 2. PullRequestReviewCommentEvent
        elif etype == "PullRequestReviewCommentEvent":
            pr = payload.get("pull_request") or {}
            target = pr.get("user") or {}
            target_login = target.get("login")
            target_id = target.get("id")
            if target_login and target_login != actor_login and not self.should_exclude(target_login):
                results.append(self._make_rec(actor_id, actor_login, target_id, target_login,
                                              "pr_review_comment", repo, event_id, created_at))

        # 3. IssueCommentEvent
        elif etype == "IssueCommentEvent":
            issue = payload.get("issue") or {}
            target = issue.get("user") or {}
            target_login = target.get("login")
            target_id = target.get("id")
            if target_login and target_login != actor_login and not self.should_exclude(target_login):
                results.append(self._make_rec(actor_id, actor_login, target_id, target_login,
                                              "issue_comment", repo, event_id, created_at))

        # 4. PullRequestEvent
        elif etype == "PullRequestEvent":
            action = payload.get("action", "")
            pr = payload.get("pull_request") or {}
            pr_user = pr.get("user") or {}
            target_login = pr_user.get("login")
            target_id = pr_user.get("id")
            merged_by = pr.get("merged_by") or {}
            merger_login = merged_by.get("login")
            merger_id = merged_by.get("id")

            if action == "closed" and merger_login and target_login and merger_login != target_login:
                if not self.should_exclude(merger_login) and not self.should_exclude(target_login):
                    results.append(self._make_rec(merger_id, merger_login, target_id, target_login,
                                                  "pr_merge", repo, event_id, created_at))

            for assignee in pr.get("assignees") or []:
                a_login = assignee.get("login")
                a_id = assignee.get("id")
                if a_login and a_login != actor_login and not self.should_exclude(a_login):
                    results.append(self._make_rec(actor_id, actor_login, a_id, a_login,
                                                  "pr_assignee", repo, event_id, created_at))

        # 5. MemberEvent
        elif etype == "MemberEvent":
            if payload.get("action") == "added":
                member = payload.get("member") or {}
                m_login = member.get("login")
                m_id = member.get("id")
                if m_login and m_login != actor_login and not self.should_exclude(m_login):
                    results.append(self._make_rec(actor_id, actor_login, m_id, m_login,
                                                  "member_added", repo, event_id, created_at))

        # 6. IssuesEvent (Assignee)
        elif etype == "IssuesEvent":
            issue = payload.get("issue") or {}
            for assignee in issue.get("assignees") or []:
                a_login = assignee.get("login")
                a_id = assignee.get("id")
                if a_login and a_login != actor_login and not self.should_exclude(a_login):
                    results.append(self._make_rec(actor_id, actor_login, a_id, a_login,
                                                  "issue_assignee", repo, event_id, created_at))

        return results

    @staticmethod
    def _make_rec(source_id, source_login, target_id, target_login, rel, repo, event_id, created_at):
        return {
            "source_id": str(source_id) if source_id else None,
            "source_login": source_login.lower().strip(),
            "target_id": str(target_id) if target_id else None,
            "target_login": target_login.lower().strip(),
            "relationship": rel,
            "repo": repo,
            "event_id": event_id,
            "created_at": created_at,
        }

    def should_exclude(self, login: str) -> bool:
        if not login:
            return True
        if not self._exclude_bots:
            return login in self._extra_exclusions
        if login in self._extra_exclusions:
            return True
        if self._auto_detect_bots:
            lower = login.lower()
            if lower.endswith("[bot]") or lower.endswith("-bot"):
                return True
            if "bot" in lower and ("github" in lower or "actions" in lower or "renovate" in lower or "dependa" in lower):
                return True
        return False
