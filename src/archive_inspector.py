"""
src/archive_inspector.py
-------------------------
Inspect real GitHub Archive files, analyse event schemas, identify interaction
evidence, detect missing/unresolved fields, and produce a data coverage summary.

Fulfills Phase 2 requirements:
  - Inspects real archive records.
  - Documents available event types and fields.
  - Categorises direct dyadic vs repository co-contribution interactions.
  - Tracks missing actor logins (e.g., git commit author emails).
  - Evaluates whether sufficient activity exists for the target seed user.
  - Produces a comprehensive validation summary report.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ArchiveInspector:
    """Inspect and analyse real GitHub Archive records."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._seed_user = config.get("github", {}).get("username", "")
        filtering = config.get("filtering", {})
        self._exclude_bots = filtering.get("exclude_bots", True)
        self._auto_detect_bots = filtering.get("auto_detect_bots", True)
        self._extra_exclusions = set(filtering.get("extra_exclusions", []))

    def is_bot(self, login: str) -> bool:
        """Check if a login represents a known or detected bot account."""
        if not login:
            return False
        if login in self._extra_exclusions:
            return True
        if self._auto_detect_bots:
            lower = login.lower()
            if lower.endswith("[bot]") or lower.endswith("-bot"):
                return True
            if "bot" in lower and ("github" in lower or "actions" in lower or "renovate" in lower or "dependa" in lower):
                return True
        return False

    def inspect_archives(
        self,
        archive_paths: List[Path],
        sample_limit_per_file: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform deep inspection of archive files.

        Parameters
        ----------
        archive_paths:
            List of downloaded .json.gz file paths.
        sample_limit_per_file:
            Optional maximum number of events to process per file (useful for fast inspection).

        Returns
        -------
        dict
            Comprehensive inspection report.
        """
        event_type_counts: Counter[str] = Counter()
        event_schemas: Dict[str, Set[str]] = defaultdict(set)
        payload_schemas: Dict[str, Set[str]] = defaultdict(set)
        
        all_actors: Set[str] = set()
        bot_actors: Set[str] = set()
        human_actors: Set[str] = set()
        repositories: Set[str] = set()

        # Direct interactions: (actor, recipient, interaction_type, repo)
        direct_interactions: List[Dict[str, str]] = []
        # Missing/unresolved identities (e.g., commits with author email but no GitHub login)
        unresolved_identities: Counter[str] = Counter()
        
        # Track seed user occurrences
        seed_events: List[Dict[str, Any]] = []

        total_lines = 0

        for path in archive_paths:
            logger.info("Inspecting archive: %s", path.name)
            try:
                with gzip.open(path, "rb") as fh:
                    file_line_count = 0
                    for raw_line in fh:
                        if not raw_line.strip():
                            continue
                        file_line_count += 1
                        if sample_limit_per_file and file_line_count > sample_limit_per_file:
                            break
                        
                        try:
                            event = json.loads(raw_line)
                        except Exception:
                            continue

                        total_lines += 1
                        event_type = event.get("type", "Unknown")
                        event_type_counts[event_type] += 1
                        
                        # Top-level keys
                        for k in event.keys():
                            event_schemas[event_type].add(k)

                        actor_dict = event.get("actor") or {}
                        actor_login = actor_dict.get("login") if isinstance(actor_dict, dict) else None
                        
                        repo_dict = event.get("repo") or {}
                        repo_name = repo_dict.get("name", "")
                        if repo_name:
                            repositories.add(repo_name)

                        if actor_login:
                            all_actors.add(actor_login)
                            if self.is_bot(actor_login):
                                bot_actors.add(actor_login)
                            else:
                                human_actors.add(actor_login)

                        if self._seed_user and (actor_login == self._seed_user or self._seed_user in repo_name):
                            seed_events.append({
                                "id": event.get("id"),
                                "type": event_type,
                                "actor": actor_login,
                                "repo": repo_name,
                                "created_at": event.get("created_at"),
                            })

                        # Payload analysis
                        payload = event.get("payload") or {}
                        if isinstance(payload, dict):
                            for pk in payload.keys():
                                payload_schemas[event_type].add(pk)

                        # Extract direct interaction evidence
                        interactions, unresolved = self._extract_interactions(event, actor_login, repo_name)
                        direct_interactions.extend(interactions)
                        for u_reason in unresolved:
                            unresolved_identities[u_reason] += 1

            except Exception as exc:
                logger.error("Failed inspecting %s: %s", path, exc)

        summary = {
            "archives_examined": len(archive_paths),
            "archive_files": [p.name for p in archive_paths],
            "total_events_scanned": total_lines,
            "event_types_found": dict(event_type_counts.most_common()),
            "unique_actors_total": len(all_actors),
            "unique_human_actors": len(human_actors),
            "unique_bot_actors": len(bot_actors),
            "unique_repositories": len(repositories),
            "qualifying_direct_interactions": len(direct_interactions),
            "direct_interaction_types": dict(Counter(i["type"] for i in direct_interactions).most_common()),
            "unresolved_identities": dict(unresolved_identities.most_common()),
            "seed_user": self._seed_user,
            "seed_user_events_found": len(seed_events),
            "seed_events_sample": seed_events[:10],
            "event_schemas": {k: sorted(list(v)) for k, v in event_schemas.items()},
            "payload_schemas": {k: sorted(list(v)) for k, v in payload_schemas.items()},
        }
        return summary

    def _extract_interactions(
        self,
        event: Dict[str, Any],
        actor: Optional[str],
        repo_name: str,
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Identify direct dyadic developer interactions in an event."""
        interactions: List[Dict[str, str]] = []
        unresolved: List[str] = []

        if not actor or self.is_bot(actor):
            return interactions, unresolved

        etype = event.get("type", "")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            return interactions, unresolved

        # 1. PullRequestReviewEvent: Reviewer directly reviews PR author
        if etype == "PullRequestReviewEvent":
            pr = payload.get("pull_request") or {}
            pr_author = (pr.get("user") or {}).get("login")
            if pr_author and pr_author != actor and not self.is_bot(pr_author):
                interactions.append({
                    "actor": actor,
                    "target": pr_author,
                    "type": "pr_review",
                    "repo": repo_name,
                })
            elif not pr_author:
                unresolved.append("PullRequestReviewEvent_missing_pr_author")

        # 2. PullRequestReviewCommentEvent: Commenter comments on PR author's code
        elif etype == "PullRequestReviewCommentEvent":
            pr = payload.get("pull_request") or {}
            pr_author = (pr.get("user") or {}).get("login")
            if pr_author and pr_author != actor and not self.is_bot(pr_author):
                interactions.append({
                    "actor": actor,
                    "target": pr_author,
                    "type": "pr_review_comment",
                    "repo": repo_name,
                })
            elif not pr_author:
                unresolved.append("PullRequestReviewCommentEvent_missing_pr_author")

        # 3. IssueCommentEvent: Commenter interacts with issue creator
        elif etype == "IssueCommentEvent":
            issue = payload.get("issue") or {}
            issue_author = (issue.get("user") or {}).get("login")
            if issue_author and issue_author != actor and not self.is_bot(issue_author):
                interactions.append({
                    "actor": actor,
                    "target": issue_author,
                    "type": "issue_comment",
                    "repo": repo_name,
                })
            elif not issue_author:
                unresolved.append("IssueCommentEvent_missing_issue_author")

        # 4. PullRequestEvent: Merged or closed by someone other than author
        elif etype == "PullRequestEvent":
            action = payload.get("action", "")
            pr = payload.get("pull_request") or {}
            pr_author = (pr.get("user") or {}).get("login")
            merged_by = (pr.get("merged_by") or {}).get("login")

            if action == "closed" and merged_by and pr_author and merged_by != pr_author:
                if not self.is_bot(merged_by) and not self.is_bot(pr_author):
                    interactions.append({
                        "actor": merged_by,
                        "target": pr_author,
                        "type": "pr_merge",
                        "repo": repo_name,
                    })

            # Check assignees
            for assignee in pr.get("assignees") or []:
                a_login = assignee.get("login")
                if a_login and a_login != actor and not self.is_bot(a_login):
                    interactions.append({
                        "actor": actor,
                        "target": a_login,
                        "type": "pr_assignee",
                        "repo": repo_name,
                    })

        # 5. MemberEvent: Developer added as collaborator by repo admin
        elif etype == "MemberEvent":
            action = payload.get("action", "")
            member = (payload.get("member") or {}).get("login")
            if action == "added" and member and member != actor and not self.is_bot(member):
                interactions.append({
                    "actor": actor,
                    "target": member,
                    "type": "member_added",
                    "repo": repo_name,
                })

        # 6. PushEvent: Unresolved Git Author Email vs GitHub Username
        elif etype == "PushEvent":
            commits = payload.get("commits") or []
            for c in commits:
                c_author = c.get("author") or {}
                # GitHub Archive only provides 'email' and 'name' inside commits, NOT 'login'
                author_email = c_author.get("email")
                author_name = c_author.get("name")
                if author_name and author_name != actor:
                    unresolved.append("PushEvent_commit_author_lacks_login")

        return interactions, unresolved

    def generate_markdown_report(self, summary: Dict[str, Any], output_path: Path) -> Path:
        """Generate a structured Markdown report documenting findings and schema analysis."""
        lines = []
        lines.append("# Phase 2: GitHub Archive Data Understanding & Coverage Report\n")
        lines.append(f"**Generated:** {summary.get('archives_examined')} archive files examined.\n")
        lines.append(f"**Seed Developer:** `{summary.get('seed_user')}`\n")
        lines.append("---\n")
        
        lines.append("## 1. Summary of Archives Examined\n")
        lines.append(f"- **Files Processed:** {summary.get('archives_examined')} hourly archive(s)")
        for fn in summary.get("archive_files", []):
            lines.append(f"  - `{fn}`")
        lines.append(f"- **Total Events Scanned:** {summary.get('total_events_scanned'):,}")
        lines.append(f"- **Unique Repositories:** {summary.get('unique_repositories'):,}")
        lines.append(f"- **Total Unique Actors:** {summary.get('unique_actors_total'):,}")
        lines.append(f"  - **Human Actors:** {summary.get('unique_human_actors'):,}")
        lines.append(f"  - **Bot/Automated Accounts Flagged:** {summary.get('unique_bot_actors'):,}\n")

        lines.append("## 2. Event Types & Collaboration Potential\n")
        lines.append("| Event Type | Count | % of Total | Direct Interaction Potential | Identifiable Fields |")
        lines.append("|---|---|---|---|---|")
        
        tot = max(summary.get("total_events_scanned", 1), 1)
        for etype, cnt in summary.get("event_types_found", {}).items():
            pct = (cnt / tot) * 100
            if etype in ("PullRequestReviewEvent", "PullRequestReviewCommentEvent", "IssueCommentEvent"):
                pot = "High (Dyadic: Reviewer/Commenter -> Author)"
            elif etype in ("PullRequestEvent", "IssuesEvent", "MemberEvent"):
                pot = "Medium (Actor -> PR/Issue Creator/Assignee)"
            elif etype in ("PushEvent", "ForkEvent", "WatchEvent"):
                pot = "Low/Indirect (Repo co-contribution / Interest)"
            else:
                pot = "None/Incidental"
            fields = ", ".join(summary.get("event_schemas", {}).get(etype, [])[:5])
            lines.append(f"| `{etype}` | {cnt:,} | {pct:.1f}% | {pot} | {fields} |")
        lines.append("\n")

        lines.append("## 3. Direct Dyadic Interaction Evidence\n")
        lines.append(f"- **Total Qualifying Direct Interactions:** {summary.get('qualifying_direct_interactions'):,}\n")
        lines.append("| Interaction Type | Count | Description |")
        lines.append("|---|---|---|")
        for itype, count in summary.get("direct_interaction_types", {}).items():
            desc = {
                "pr_review": "Code reviewer submitted review to PR author",
                "pr_review_comment": "Inline code comment directed to PR author",
                "issue_comment": "Issue discussion comment directed to issue author",
                "pr_merge": "Repository maintainer merged developer's PR",
                "pr_assignee": "PR creator assigned task to developer",
                "member_added": "Organization/Repo maintainer added developer as collaborator",
            }.get(itype, itype)
            lines.append(f"| `{itype}` | {count:,} | {desc} |")
        lines.append("\n")

        lines.append("## 4. Missing Information & GitHub API Enrichment Requirements\n")
        lines.append("### Unresolved Identities in GitHub Archive\n")
        for u_type, u_count in summary.get("unresolved_identities", {}).items():
            lines.append(f"- `{u_type}`: {u_count:,} occurrences")
        lines.append("\n")

        lines.append("### Key Limitations & API Enrichment Strategy\n")
        lines.append("1. **Commit Author Login Resolution (`PushEvent`):**\n"
                     "   GitHub Archive's `PushEvent` payload provides git author `email` and `name`, but **does not resolve them to GitHub usernames (`login`)**. To prevent attributing commits to unverified strings, developer edges must be enriched using GitHub REST API commit lookups `/repos/{owner}/{repo}/commits/{sha}`.\n")
        lines.append("2. **Historical Review Threads:**\n"
                     "   Single hourly archives only capture reviews happening in that 60-minute window. To obtain all collaborators on a repo, REST API `/repos/{owner}/{repo}/pulls?state=all` and `/repos/{owner}/{repo}/contributors` can be leveraged.\n")
        lines.append("3. **Ego Network Sparsity & Seed User Activity:**\n")
        seed = summary.get("seed_user")
        n_seed = summary.get("seed_user_events_found", 0)
        lines.append(f"   In the sampled archive period, the configured seed user `{seed}` had **{n_seed} events**.\n")
        if n_seed == 0:
            lines.append(f"   > **Evaluation Finding:** Seed user `{seed}` had no public activity in this specific sample window. "
                         f"To build a meaningful collaboration network for `{seed}`, our pipeline utilizes the dual-source strategy: "
                         f"querying GitHub REST API for `{seed}`'s public repositories, contributors, stargazers, and forks, and intersecting with GitHub Archive data. "
                         f"Alternatively, the user can configure an active public developer (e.g. `torvalds`, `tiangolo`, `octocat`) or expand the collection window.\n")
        else:
            lines.append(f"   > **Evaluation Finding:** Seed user `{seed}` has active records in the archive. Sufficient data exists for graph construction.\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown coverage summary written to: %s", output_path)
        return output_path

