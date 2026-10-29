"""
tests/test_event_parser.py
---------------------------
Unit tests for the EventParser module.

Tests use synthetic fixture data that mirrors the actual GitHub Archive
JSON event schema (https://docs.github.com/en/rest/using-the-rest-api/github-event-types)
to avoid any real network calls.
"""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from src.event_parser import EventParser, COLLABORATION_EVENTS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_config(exclude_bots: bool = True) -> dict:
    return {
        "filtering": {
            "exclude_bots": exclude_bots,
            "auto_detect_bots": True,
            "extra_exclusions": ["dependabot[bot]"],
        }
    }


def _write_archive(events: list, path: Path) -> None:
    """Write a list of event dicts as a gzip-compressed JSON-lines file."""
    with gzip.open(path, "wb") as fh:
        for event in events:
            fh.write((json.dumps(event) + "\n").encode())


PUSH_EVENT = {
    "id": "1000000001",
    "type": "PushEvent",
    "actor": {"login": "alice"},
    "repo": {"name": "alice/myrepo"},
    "created_at": "2024-10-01T12:00:00Z",
    "payload": {"action": "", "commits": []},
}

PR_EVENT = {
    "id": "1000000002",
    "type": "PullRequestEvent",
    "actor": {"login": "bob"},
    "repo": {"name": "alice/myrepo"},
    "created_at": "2024-10-01T13:00:00Z",
    "payload": {
        "action": "opened",
        "pull_request": {
            "user": {"login": "alice"},
            "merged_by": None,
            "assignees": [],
        },
    },
}

BOT_EVENT = {
    "id": "1000000003",
    "type": "PushEvent",
    "actor": {"login": "dependabot[bot]"},
    "repo": {"name": "alice/myrepo"},
    "created_at": "2024-10-01T14:00:00Z",
    "payload": {},
}

UNKNOWN_EVENT = {
    "id": "1000000004",
    "type": "UnknownEventType",
    "actor": {"login": "carol"},
    "repo": {"name": "carol/proj"},
    "created_at": "2024-10-01T15:00:00Z",
    "payload": {},
}

ISSUE_COMMENT_EVENT = {
    "id": "1000000005",
    "type": "IssueCommentEvent",
    "actor": {"login": "dave"},
    "repo": {"name": "alice/myrepo"},
    "created_at": "2024-10-01T16:00:00Z",
    "payload": {
        "action": "created",
        "issue": {"user": {"login": "alice"}},
    },
}


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCollaborationEvents:
    def test_collaboration_event_types_non_empty(self):
        assert len(COLLABORATION_EVENTS) > 0

    def test_push_event_in_collaboration_events(self):
        assert "PushEvent" in COLLABORATION_EVENTS

    def test_pull_request_event_in_collaboration_events(self):
        assert "PullRequestEvent" in COLLABORATION_EVENTS

    def test_issue_comment_event_in_collaboration_events(self):
        assert "IssueCommentEvent" in COLLABORATION_EVENTS


class TestEventParserBasic:
    def test_parse_push_event(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([PUSH_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "PushEvent"
        assert df.iloc[0]["actor"] == "alice"
        assert df.iloc[0]["repo_owner"] == "alice"
        assert df.iloc[0]["repo_name"] == "myrepo"

    def test_parse_pull_request_event_extracts_extra_actor(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([PR_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert len(df) == 1
        row = df.iloc[0]
        assert row["actor"] == "bob"
        assert "alice" in row["extra_actors"]

    def test_bot_events_excluded(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([BOT_EVENT], archive)

        parser = EventParser(_make_config(exclude_bots=True))
        df = parser.parse_files([archive])

        assert len(df) == 0, "Bot events should be excluded."

    def test_bot_events_included_when_disabled(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([BOT_EVENT], archive)

        parser = EventParser(_make_config(exclude_bots=False))
        df = parser.parse_files([archive])

        # dependabot[bot] is in extra_exclusions → still excluded
        assert len(df) == 0

    def test_unknown_event_type_excluded(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([UNKNOWN_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert len(df) == 0, "Unknown event types should be ignored."

    def test_multiple_events_parsed(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([PUSH_EVENT, PR_EVENT, ISSUE_COMMENT_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert len(df) == 3

    def test_created_at_is_datetime(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([PUSH_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        import pandas as pd
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_empty_archive_returns_empty_df(self, tmp_path):
        archive = tmp_path / "empty.json.gz"
        _write_archive([], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert df.empty

    def test_corrupted_line_skipped(self, tmp_path):
        archive = tmp_path / "corrupt.json.gz"
        with gzip.open(archive, "wb") as fh:
            fh.write(b"NOT_VALID_JSON\n")
            fh.write((json.dumps(PUSH_EVENT) + "\n").encode())

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        # Corrupted line skipped; valid line parsed
        assert len(df) == 1

    def test_missing_actor_skipped(self, tmp_path):
        event = {**PUSH_EVENT, "actor": None}
        archive = tmp_path / "test.json.gz"
        _write_archive([event], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert df.empty

    def test_duplicate_event_ids_deduplicated(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([PUSH_EVENT, PUSH_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        assert len(df) == 1, "Duplicate event IDs should be deduplicated."

    def test_issue_comment_extra_actors(self, tmp_path):
        archive = tmp_path / "test.json.gz"
        _write_archive([ISSUE_COMMENT_EVENT], archive)

        parser = EventParser(_make_config())
        df = parser.parse_files([archive])

        row = df.iloc[0]
        assert "alice" in row["extra_actors"]


class TestBotDetection:
    def setup_method(self):
        self._parser = EventParser(_make_config())

    def test_bracket_bot_suffix(self):
        assert self._parser._should_exclude("renovate[bot]")

    def test_dash_bot_suffix(self):
        assert self._parser._should_exclude("snyk-bot")

    def test_normal_user_not_excluded(self):
        assert not self._parser._should_exclude("torvalds")

    def test_extra_exclusion_list(self):
        assert self._parser._should_exclude("dependabot[bot]")

