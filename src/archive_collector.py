"""
src/archive_collector.py
-------------------------
Downloads and caches hourly GitHub Archive files.

GitHub Archive (https://www.gharchive.org/) publishes one gzip-compressed
JSON-lines file per hour: ``https://data.gharchive.org/YYYY-MM-DD-H.json.gz``

Responsibilities
~~~~~~~~~~~~~~~~
* Enumerate all hourly file URLs within the configured date range.
* Download files that are not yet cached; skip files that exist locally.
* Respect ``max_archive_files`` to avoid exhausting disk / bandwidth.
* Return a list of local file paths ready for parsing.

Usage
~~~~~
    from src import load_config
    from src.archive_collector import ArchiveCollector

    cfg = load_config()
    collector = ArchiveCollector(cfg)
    files = collector.collect()   # → list[Path]
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_ARCHIVE_BASE = "https://data.gharchive.org"


class ArchiveCollector:
    """Download and cache GitHub Archive hourly files."""

    def __init__(self, config: dict) -> None:
        self._config = config
        collection_cfg = config.get("collection", {})

        self._start_date = datetime.strptime(
            collection_cfg.get("start_date", "2024-10-01"), "%Y-%m-%d"
        )
        self._end_date = datetime.strptime(
            collection_cfg.get("end_date", "2024-10-07"), "%Y-%m-%d"
        )
        self._max_files: int = collection_cfg.get("max_archive_files", 24)
        self._cache_dir = (
            Path(config.get("collection", {}).get("cache_dir", "data/raw"))
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect(self) -> List[Path]:
        """Download (or retrieve from cache) all archive files in the range.

        Returns
        -------
        list[Path]
            Sorted list of local ``.json.gz`` file paths.
        """
        urls = self._enumerate_urls()
        logger.info(
            "Archive date range: %s → %s  |  %d URLs  |  cap=%d",
            self._start_date.date(),
            self._end_date.date(),
            len(urls),
            self._max_files,
        )
        urls = urls[: self._max_files]

        downloaded: List[Path] = []
        for url in tqdm(urls, desc="Archive files", unit="file"):
            local_path = self._local_path(url)
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.debug("Cache hit: %s", local_path.name)
            else:
                success = self._download(url, local_path)
                if not success:
                    continue
            downloaded.append(local_path)

        logger.info("Archive collection complete: %d files ready.", len(downloaded))
        return sorted(downloaded)

    def list_cached(self) -> List[Path]:
        """Return already-downloaded archive files in cache_dir."""
        return sorted(self._cache_dir.glob("*.json.gz"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enumerate_urls(self) -> List[str]:
        """Build the list of hourly archive URLs for the date range."""
        urls: List[str] = []
        current = self._start_date
        end = self._end_date + timedelta(days=1)  # inclusive end date

        while current < end:
            for hour in range(24):
                date_str = current.strftime("%Y-%m-%d")
                urls.append(f"{_ARCHIVE_BASE}/{date_str}-{hour}.json.gz")
            current += timedelta(days=1)

        return urls

    def _local_path(self, url: str) -> Path:
        """Derive a local cache path from a remote URL."""
        filename = url.rsplit("/", 1)[-1]
        return self._cache_dir / filename

    def _download(self, url: str, dest: Path) -> bool:
        """Stream-download *url* to *dest*.

        Returns
        -------
        bool
            True if the download succeeded; False on error.
        """
        try:
            logger.debug("Downloading %s …", url)
            with requests.get(url, stream=True, timeout=120) as resp:
                if resp.status_code == 404:
                    logger.debug("Archive not yet available: %s", url)
                    return False
                resp.raise_for_status()

                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)

            logger.debug("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return True

        except requests.RequestException as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            if dest.exists():
                dest.unlink()  # remove partial file
            return False

