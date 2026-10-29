"""
src/github_client.py
--------------------
Thin wrapper around the GitHub REST API (v3).

Responsibilities
~~~~~~~~~~~~~~~~
* Authenticated and unauthenticated requests with automatic rate-limit
  handling (exponential back-off + Retry-After header respect).
* Fetch a user's public repos, stargazers, forks, pull-request authors,
  issue commenters, and follower/following lists — all sources of
  collaboration signal.
* Return raw JSON; leave interpretation to event_parser / graph_builder.

Usage
~~~~~
    from src import load_config
    from src.github_client import GitHubClient

    cfg = load_config()
    client = GitHubClient(cfg)
    repos  = client.get_user_repos("torvalds")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import requests

logger = logging.getLogger(__name__)

# GitHub REST API v3 base URL
_BASE_URL = "https://api.github.com"

# Maximum pages to fetch per paginated endpoint (safety cap)
_MAX_PAGES = 10


class GitHubClient:
    """Authenticated GitHub REST API client with rate-limit awareness."""

    def __init__(self, config: dict) -> None:
        self._config = config
        token: str = config.get("github", {}).get("token", "")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            }
        )
        self._per_page = 100  # maximum allowed by GitHub

    # ------------------------------------------------------------------
    # Core request helper
    # ------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        retries: int = 4,
    ) -> Any:
        """Execute a GET request with exponential back-off on rate limits.

        Parameters
        ----------
        endpoint:
            URL path after the base URL, e.g. ``"/users/torvalds/repos"``.
        params:
            Query-string parameters.
        retries:
            Number of retry attempts on 429 / 403 rate-limit responses.

        Returns
        -------
        Parsed JSON response (dict or list).
        """
        url = f"{_BASE_URL}{endpoint}"
        delay = 2.0

        for attempt in range(retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.warning("Request failed (%s): %s", type(exc).__name__, exc)
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code in (403, 429):
                reset_ts = response.headers.get("X-RateLimit-Reset")
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                elif reset_ts:
                    wait = max(int(reset_ts) - int(time.time()) + 1, delay)
                else:
                    wait = delay
                logger.warning(
                    "Rate-limited by GitHub (HTTP %s). Waiting %ss …",
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                delay *= 2
                continue

            if response.status_code == 404:
                logger.debug("Not found: %s", url)
                return {}

            logger.warning(
                "Unexpected HTTP %s from %s", response.status_code, url
            )
            return {}

        return {}

    def _paginate(
        self,
        endpoint: str,
        params: dict | None = None,
        max_pages: int = _MAX_PAGES,
    ) -> Generator[dict, None, None]:
        """Yield all items from a paginated GitHub endpoint.

        Parameters
        ----------
        endpoint:
            API path, e.g. ``"/repos/torvalds/linux/stargazers"``.
        params:
            Extra query parameters (merged with ``per_page``).
        max_pages:
            Safety cap on the number of pages fetched.
        """
        params = dict(params or {})
        params["per_page"] = self._per_page
        page = 1

        while page <= max_pages:
            params["page"] = page
            data = self._get(endpoint, params=params)

            if not data:
                break

            items = data if isinstance(data, list) else data.get("items", [])
            if not items:
                break

            yield from items
            page += 1

            # GitHub returns fewer items than per_page on the last page
            if len(items) < self._per_page:
                break

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_user(self, username: str) -> dict:
        """Return public profile data for *username*."""
        return self._get(f"/users/{username}")

    def get_user_repos(
        self, username: str, max_pages: int = _MAX_PAGES
    ) -> list[dict]:
        """Return public repositories for *username* (sorted by push date)."""
        return list(
            self._paginate(
                f"/users/{username}/repos",
                params={"type": "public", "sort": "pushed"},
                max_pages=max_pages,
            )
        )

    def get_repo_contributors(
        self, owner: str, repo: str, max_pages: int = 5
    ) -> list[dict]:
        """Return contributors for *owner/repo*."""
        return list(
            self._paginate(
                f"/repos/{owner}/{repo}/contributors",
                max_pages=max_pages,
            )
        )

    def get_repo_stargazers(
        self, owner: str, repo: str, max_pages: int = 3
    ) -> list[dict]:
        """Return stargazers for *owner/repo*."""
        return list(
            self._paginate(
                f"/repos/{owner}/{repo}/stargazers",
                max_pages=max_pages,
            )
        )

    def get_repo_forks(
        self, owner: str, repo: str, max_pages: int = 3
    ) -> list[dict]:
        """Return fork metadata for *owner/repo*."""
        return list(
            self._paginate(
                f"/repos/{owner}/{repo}/forks",
                max_pages=max_pages,
            )
        )

    def get_repo_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_pages: int = 5,
    ) -> list[dict]:
        """Return pull requests for *owner/repo*."""
        return list(
            self._paginate(
                f"/repos/{owner}/{repo}/pulls",
                params={"state": state},
                max_pages=max_pages,
            )
        )

    def get_repo_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_pages: int = 5,
    ) -> list[dict]:
        """Return issues (excluding PRs) for *owner/repo*."""
        issues = list(
            self._paginate(
                f"/repos/{owner}/{repo}/issues",
                params={"state": state},
                max_pages=max_pages,
            )
        )
        # GitHub issues endpoint returns PRs too; filter them out
        return [i for i in issues if "pull_request" not in i]

    def get_repo_issue_comments(
        self, owner: str, repo: str, max_pages: int = 5
    ) -> list[dict]:
        """Return all issue comments for *owner/repo*."""
        return list(
            self._paginate(
                f"/repos/{owner}/{repo}/issues/comments",
                max_pages=max_pages,
            )
        )

    def get_user_followers(
        self, username: str, max_pages: int = 3
    ) -> list[dict]:
        """Return followers for *username*."""
        return list(
            self._paginate(
                f"/users/{username}/followers",
                max_pages=max_pages,
            )
        )

    def get_user_following(
        self, username: str, max_pages: int = 3
    ) -> list[dict]:
        """Return accounts that *username* follows."""
        return list(
            self._paginate(
                f"/users/{username}/following",
                max_pages=max_pages,
            )
        )

    def check_rate_limit(self) -> dict:
        """Return current rate-limit status."""
        return self._get("/rate_limit")

