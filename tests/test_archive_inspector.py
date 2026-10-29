"""
tests/test_archive_inspector.py
--------------------------------
Unit tests for ArchiveInspector.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.archive_inspector import ArchiveInspector


def _make_config(seed_user="alice"):
    return {
        "github": {"username": seed_user},
        "filtering": {
            "exclude_bots": True,
            "auto_detect_bots": True,
            "extra_exclusions": ["dependabot[bot]"],
        },
    }


def _write_archive(events: list, path: Path) -> None:
    with gzip.open(path, "wb") as fh:
        for event in events:
            fh.write((json.dumps(event) + "\n").encode())


SAMPLE_REVIEW_EVENT = {
    "id": "2000000001",
    "type": "PullRequestReviewEvent",
    "actor": {"login": "bob"},
    "repo": {"name": "alice/awesome-repo"},
    "created_at": "2024-10-01T10:00:00Z",
    "payload": {
        "action": "submitted",
        "pull_request": {
            "user": {"login": "alice"},
        },
    },
}

SAMPLE_PUSH_UNRESOLVED = {
    "id": "2000000002",
    "type": "PushEvent",
    "actor": {"login": "bob"},
    "repo": {"name": "bob/tools"},
    "created_at": "2024-10-01T10:05:00Z",
    "payload": {
        "commits": [
            {"author": {"name": "Charlie Developer", "email": "charlie@corp.com"}},
        ]
    },
}


class TestArchiveInspector:
    def test_bot_detection(self):
        inspector = ArchiveInspector(_make_config())
        assert inspector.is_bot("github-actions[bot]")
        assert inspector.is_bot("renovate-bot")
        assert not inspector.is_bot("torvalds")

    def test_inspect_archives(self, tmp_path):
        archive = tmp_path / "test_sample.json.gz"
        _write_archive([SAMPLE_REVIEW_EVENT, SAMPLE_PUSH_UNRESOLVED], archive)

        inspector = ArchiveInspector(_make_config(seed_user="alice"))
        summary = inspector.inspect_archives([archive])

        assert summary["archives_examined"] == 1
        assert summary["total_events_scanned"] == 2
        assert "PullRequestReviewEvent" in summary["event_types_found"]
        assert summary["qualifying_direct_interactions"] == 1
        assert summary["direct_interaction_types"].get("pr_review") == 1
        assert summary["seed_user_events_found"] == 1
        assert summary["unresolved_identities"].get("PushEvent_commit_author_lacks_login") == 1

    def test_generate_markdown_report(self, tmp_path):
        archive = tmp_path / "test_sample.json.gz"
        _write_archive([SAMPLE_REVIEW_EVENT], archive)

        inspector = ArchiveInspector(_make_config())
        summary = inspector.inspect_archives([archive])

        out_md = tmp_path / "coverage_summary.md"
        inspector.generate_markdown_report(summary, out_md)

        assert out_md.exists()
        content = out_md.read_text(encoding="utf-8")
        assert "Phase 2: GitHub Archive Data Understanding" in content
        assert "PullRequestReviewEvent" in content

