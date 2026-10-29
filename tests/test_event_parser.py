"""
tests/test_event_parser.py
---------------------------
Unit tests for EventParser with direct dyadic interactions and provenance.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import pytest
from src.event_parser import EventParser, COLLABORATION_EVENTS


def _make_config():
    return {
        "filtering": {
            "exclude_bots": True,
            "auto_detect_bots": True,
            "extra_exclusions": ["dependabot[bot]"],
        }
    }


def _write_archive(events: list, path: Path) -> None:
    with gzip.open(path, "wb") as fh:
        for event in events:
            fh.write((json.dumps(event) + "\n").encode())


SAMPLE_PR_REVIEW = {
    "id": "10001",
    "type": "PullRequestReviewEvent",
    "actor": {"id": 101, "login": "alice"},
    "repo": {"name": "org/repo1"},
    "created_at": "2024-10-01T12:00:00Z",
    "payload": {
        "action": "submitted",
        "pull_request": {"user": {"id": 102, "login": "bob"}}
    }
}

SAMPLE_ISSUE_COMMENT = {
    "id": "10002",
    "type": "IssueCommentEvent",
    "actor": {"id": 103, "login": "carol"},
    "repo": {"name": "org/repo1"},
    "created_at": "2024-10-01T12:05:00Z",
    "payload": {
        "action": "created",
        "issue": {"user": {"id": 101, "login": "alice"}}
    }
}


class TestEventParserDirect:
    def test_extracts_pr_review_dyad(self, tmp_path):
        p = tmp_path / "test.json.gz"
        _write_archive([SAMPLE_PR_REVIEW], p)
        parser = EventParser(_make_config())
        df = parser.parse_files([p])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["source_login"] == "alice"
        assert row["target_login"] == "bob"
        assert row["source_id"] == "101"
        assert row["target_id"] == "102"
        assert row["relationship"] == "pr_review"

    def test_extracts_issue_comment_dyad(self, tmp_path):
        p = tmp_path / "test.json.gz"
        _write_archive([SAMPLE_ISSUE_COMMENT], p)
        parser = EventParser(_make_config())
        df = parser.parse_files([p])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["source_login"] == "carol"
        assert row["target_login"] == "alice"
        assert row["relationship"] == "issue_comment"

    def test_filters_bots(self, tmp_path):
        bot_event = {
            "id": "10003",
            "type": "IssueCommentEvent",
            "actor": {"id": 999, "login": "dependabot[bot]"},
            "repo": {"name": "org/repo1"},
            "created_at": "2024-10-01T12:10:00Z",
            "payload": {"issue": {"user": {"id": 101, "login": "alice"}}}
        }
        p = tmp_path / "test.json.gz"
        _write_archive([bot_event], p)
        parser = EventParser(_make_config())
        df = parser.parse_files([p])
        assert df.empty
