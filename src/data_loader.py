"""
Dataset loading and inspection utilities for NumFOCUS contributor activity records.
"""
import os
import json
import zipfile
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional
from datetime import datetime
from src.config import RAW_DATA_DIR, ACTIVITIES_ZIP, ACTIONS_ZIP, ACTION_SCHEMA, ACTIVITY_SCHEMA

def inspect_schemas() -> Dict[str, Any]:
    """Inspect and return schema properties for actions and activities."""
    action_schema_path = RAW_DATA_DIR / ACTION_SCHEMA
    activity_schema_path = RAW_DATA_DIR / ACTIVITY_SCHEMA
    
    result = {}
    if action_schema_path.exists():
        with open(action_schema_path, "r", encoding="utf-8") as f:
            result["action_schema"] = json.load(f)
    if activity_schema_path.exists():
        with open(activity_schema_path, "r", encoding="utf-8") as f:
            result["activity_schema"] = json.load(f)
    return result

def list_zip_contents(zip_filename: str = ACTIVITIES_ZIP) -> List[str]:
    """List members of the specified dataset zip archive."""
    zip_path = RAW_DATA_DIR / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        return z.namelist()

def stream_activities(zip_filename: str = ACTIVITIES_ZIP, max_records: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Stream activities line-by-line from JSON/JSONL files inside the zip archive without extracting entirely to disk.
    Yields parsed activity dictionaries.
    """
    zip_path = RAW_DATA_DIR / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    count = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(('.json', '.jsonl')):
                with z.open(file_info) as f:
                    # Could be JSON array or JSON lines
                    first_char = f.peek(1)[:1]
                    if first_char == b'[':
                        # Full JSON array
                        records = json.load(f)
                        for rec in records:
                            yield rec
                            count += 1
                            if max_records and count >= max_records:
                                return
                    else:
                        # Line-delimited JSON
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                yield rec
                                count += 1
                                if max_records and count >= max_records:
                                    return
                            except json.JSONDecodeError:
                                continue

def extract_interactions_from_activity(activity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract direct developer-to-developer interactions from an activity record.
    Verifiable interaction links:
    1. Reviewer -> PR Author (reviewing someone's pull request)
    2. Commenter -> PR Author (commenting on someone's pull request)
    3. Commenter -> Issue Author (commenting on someone's issue)
    4. PR Merger / Closer -> PR Author
    """
    interactions = []
    actor = activity.get("actor", {})
    actor_login = actor.get("login") if isinstance(actor, dict) else None
    if not actor_login:
        return interactions

    repo_info = activity.get("repository", {})
    repo_name = repo_info.get("name") if isinstance(repo_info, dict) else ""
    org_name = repo_info.get("organisation") if isinstance(repo_info, dict) else ""
    if org_name and repo_name:
        if repo_name.startswith(org_name + "/"):
            full_repo = repo_name
        else:
            full_repo = f"{org_name}/{repo_name}"
    else:
        full_repo = repo_name or org_name or "unknown"
    act_type = activity.get("activity", "Unknown")
    start_date = activity.get("start_date") or activity.get("end_date")

    actions = activity.get("actions", [])
    if not actions and isinstance(activity.get("details"), dict):
        actions = [{"action": act_type, "date": start_date, "details": activity["details"]}]

    for action in actions:
        action_name = action.get("action", act_type)
        action_date = action.get("date") or start_date
        details = action.get("details", {})
        if not isinstance(details, dict):
            continue

        target_author = None
        interaction_type = None

        # Check PR details
        pr = details.get("pull_request")
        if isinstance(pr, dict):
            pr_author_obj = pr.get("author")
            if isinstance(pr_author_obj, dict):
                target_author = pr_author_obj.get("login")
            if "review" in details or "Review" in action_name or "Review" in act_type:
                interaction_type = "pr_review"
            elif "comment" in details or "Comment" in action_name:
                interaction_type = "pr_comment"
            elif "Merge" in action_name or "Merge" in act_type:
                interaction_type = "pr_merge"
            else:
                interaction_type = "pr_interaction"

        # Check Issue details
        issue = details.get("issue")
        if isinstance(issue, dict):
            issue_author_obj = issue.get("author")
            if isinstance(issue_author_obj, dict):
                target_author = issue_author_obj.get("login")
            if "comment" in details or "Comment" in action_name:
                interaction_type = "issue_comment"
            else:
                interaction_type = "issue_interaction"

        if target_author and target_author != actor_login:
            interactions.append({
                "source": actor_login,
                "target": target_author,
                "interaction_type": interaction_type or "collaboration",
                "repository": full_repo,
                "timestamp": action_date,
                "event_id": action.get("event_id")
            })

    return interactions
