"""
src/event_parser.py
--------------------
Parse raw GitHub Archive JSON-lines files into structured event records.

GitHub Archive stores one JSON object per line inside each .json.gz file.
This module inspects the actual event schema documented at:
  https://docs.github.com/en/rest/using-the-rest-api/github-event-types

Supported event types that signal developer collaboration:

| Event type          | Collaboration signal                             |
|---------------------|--------------------------------------------------|
| PushEvent           | Actor pushed commits to a repo (may list authors)|
| PullRequestEvent    | Actor opened/merged/closed a PR                  |
| PullRequestReviewEvent | Actor reviewed someone else's PR              |
| IssuesEvent         | Actor opened/closed an issue                     |
| IssueCommentEvent   | Actor commented on an issue                      |
| ForkEvent           | Actor forked a repo (interest signal)            |
| WatchEvent          | Actor starred a repo                             |
| CreateEvent         | Actor created a branch/tag/repo                  |
| CommitCommentEvent  | Actor commented on a specific commit             |

Usage
~~~~~
    from src import load_config
    from src.event_parser import EventParser

    cfg = load_config()
    parser = EventParser(cfg)
    events_df = parser.parse_files(archive_paths)
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Event types that encode a collaboration signal
COLLABORATION_EVENTS = {
    "PushEvent",
    "PullRequestEvent",
    "PullRequestReviewEvent",
    "IssuesEvent",
    "IssueCommentEvent",
    "ForkEvent",
    "WatchEvent",
    "CreateEvent",
    "CommitCommentEvent",
    "PullRequestReviewCommentEvent",
    "MemberEvent",
}


class EventParser:
    """Parse GitHub Archive .json.gz files into a tidy DataFrame of events."""

    def __init__(self, config: dict) -> None:
        self._config = config
        filtering = config.get("filtering", {})
        self._exclude_bots: bool = filtering.get("exclude_bots", True)
        self._auto_detect_bots: bool = filtering.get("auto_detect_bots", True)
        self._extra_exclusions: set[str] = set(
            filtering.get("extra_exclusions", [])
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse_files(self, paths: List[Path]) -> pd.DataFrame:
        """Parse a list of archive files and return a consolidated DataFrame.

        Parameters
        ----------
        paths:
            Local paths to ``.json.gz`` archive files.

        Returns
        -------
        pd.DataFrame
            Columns: event_id, event_type, actor, repo_owner, repo_name,
            repo_full_name, created_at, payload_action, extra_actors.
        """
        all_records: List[Dict[str, Any]] = []

        for path in tqdm(paths, desc="Parsing archives", unit="file"):
            records = self._parse_file(path)
            all_records.extend(records)
            logger.debug("  %s → %d events", path.name, len(records))

        if not all_records:
            logger.warning("No collaboration events found in supplied archives.")
            return pd.DataFrame(
                columns=[
                    "event_id", "event_type", "actor", "repo_owner",
                    "repo_name", "repo_full_name", "created_at",
                    "payload_action", "extra_actors",
                ]
            )

        df = pd.DataFrame(all_records)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["actor", "created_at"])
        df = df.drop_duplicates(subset=["event_id"])

        logger.info(
            "Parsed %d collaboration events from %d files  "
            "(%d unique actors, %d unique repos).",
            len(df),
            len(paths),
            df["actor"].nunique(),
            df["repo_full_name"].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path) -> List[Dict[str, Any]]:
        """Parse a single archive file; return a list of event dicts."""
        records: List[Dict[str, Any]] = []

        try:
            with gzip.open(path, "rb") as fh:
                for raw_line in fh:
                    if not raw_line.strip():
                        continue
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    record = self._extract_record(event)
                    if record is not None:
                        records.append(record)

        except (OSError, gzip.BadGzipFile) as exc:
            logger.error("Cannot read %s: %s", path, exc)

        return records

    def _extract_record(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a normalised record from a raw GitHub Archive event.

        Returns
        -------
        dict | None
            A structured record, or None if the event should be skipped.
        """
        event_type: str = event.get("type", "")
        if event_type not in COLLABORATION_EVENTS:
            return None

        actor: Optional[str] = self._extract_actor_login(event)
        if actor is None:
            return None

        if self._should_exclude(actor):
            return None

        repo: Dict[str, Any] = event.get("repo", {})
        repo_full = repo.get("name", "")  # GitHub Archive uses "name" for full_name
        repo_parts = repo_full.split("/", 1)
        repo_owner = repo_parts[0] if len(repo_parts) == 2 else ""
        repo_name = repo_parts[1] if len(repo_parts) == 2 else repo_full

        payload: Dict[str, Any] = event.get("payload", {})
        payload_action: str = payload.get("action", "")

        # Collect secondary actors mentioned in the payload
        extra_actors = self._extract_extra_actors(event_type, payload)
        extra_actors = [a for a in extra_actors if not self._should_exclude(a)]

        return {
            "event_id": str(event.get("id", "")),
            "event_type": event_type,
            "actor": actor,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "repo_full_name": repo_full,
            "created_at": event.get("created_at", ""),
            "payload_action": payload_action,
            "extra_actors": extra_actors,
        }

    @staticmethod
    def _extract_actor_login(event: Dict[str, Any]) -> Optional[str]:
        """Return the primary actor's login from an event dict."""
        actor = event.get("actor")
        if isinstance(actor, dict):
            return actor.get("login")
        if isinstance(actor, str):
            return actor
        return None

    def _extract_extra_actors(
        self, event_type: str, payload: Dict[str, Any]
    ) -> List[str]:
        """Extract secondary actor logins from the payload.

        These are the other side of a collaboration edge:
        - For PullRequestEvent: the PR author when reviewed, or assignees.
        - For IssueCommentEvent: the issue author.
        - For MemberEvent: the added member.
        """
        extras: List[str] = []

        try:
            if event_type == "PullRequestEvent":
                pr = payload.get("pull_request", {})
                if user := pr.get("user", {}).get("login"):
                    extras.append(user)
                if merged_by := (pr.get("merged_by") or {}).get("login"):
                    extras.append(merged_by)
                for assignee in pr.get("assignees", []):
                    if login := assignee.get("login"):
                        extras.append(login)

            elif event_type == "PullRequestReviewEvent":
                pr = payload.get("pull_request", {})
                if user := pr.get("user", {}).get("login"):
                    extras.append(user)

            elif event_type in ("IssuesEvent", "IssueCommentEvent",
                                 "CommitCommentEvent",
                                 "PullRequestReviewCommentEvent"):
                issue = payload.get("issue", {})
                if user := issue.get("user", {}).get("login"):
                    extras.append(user)

            elif event_type == "ForkEvent":
                forkee = payload.get("forkee", {})
                if owner := forkee.get("owner", {}).get("login"):
                    extras.append(owner)

            elif event_type == "MemberEvent":
                if member := payload.get("member", {}).get("login"):
                    extras.append(member)

        except (AttributeError, TypeError):
            pass

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for login in extras:
            if login and login not in seen:
                seen.add(login)
                unique.append(login)
        return unique

    def _should_exclude(self, login: str) -> bool:
        """Return True if *login* should be filtered from the graph."""
        if not self._exclude_bots:
            return login in self._extra_exclusions

        if login in self._extra_exclusions:
            return True

        if self._auto_detect_bots:
            lower = login.lower()
            if lower.endswith("[bot]") or lower.endswith("-bot"):
                return True
            if "bot" in lower and ("github" in lower or "actions" in lower):
                return True

        return False

