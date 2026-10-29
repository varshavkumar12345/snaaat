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
    """Download and cache GitHub Archive hourly files with resume and tracking."""

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
        self.stats: dict = {
            "requested_files": 0,
            "cache_hits": 0,
            "downloaded": 0,
            "not_found_404": 0,
            "failed_errors": 0,
            "error_details": [],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect(self, limit: int | None = None) -> List[Path]:
        """Download (or retrieve from cache) all archive files in the range.

        Parameters
        ----------
        limit:
            Optional override for maximum files to collect.

        Returns
        -------
        list[Path]
            Sorted list of local ``.json.gz`` file paths.
        """
        urls = self._enumerate_urls()
        cap = limit if limit is not None else self._max_files
        logger.info(
            "Archive date range: %s → %s  |  %d total hours available  |  cap=%d",
            self._start_date.date(),
            self._end_date.date(),
            len(urls),
            cap,
        )
        urls = urls[:cap]
        self.stats["requested_files"] = len(urls)

        downloaded: List[Path] = []
        for url in tqdm(urls, desc="Archive files", unit="file"):
            local_path = self._local_path(url)
            if self._is_valid_cache(local_path):
                logger.debug("Cache hit: %s", local_path.name)
                self.stats["cache_hits"] += 1
                downloaded.append(local_path)
            else:
                success = self._download_resumable(url, local_path)
                if success:
                    self.stats["downloaded"] += 1
                    downloaded.append(local_path)

        logger.info(
            "Archive collection summary: %d files ready (hits=%d, downloaded=%d, 404s=%d, errors=%d)",
            len(downloaded),
            self.stats["cache_hits"],
            self.stats["downloaded"],
            self.stats["not_found_404"],
            self.stats["failed_errors"],
        )
        return sorted(downloaded)

    def list_cached(self) -> List[Path]:
        """Return already-downloaded archive files in cache_dir."""
        return sorted([p for p in self._cache_dir.glob("*.json.gz") if self._is_valid_cache(p)])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_cache(self, path: Path) -> bool:
        """Verify that a cached file exists and is a valid non-empty gzip file."""
        if not path.exists() or path.stat().st_size == 0:
            return False
        import gzip
        try:
            with gzip.open(path, "rb") as f:
                f.read(100)
            return True
        except Exception:
            return False

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

    def _download_resumable(self, url: str, dest: Path) -> bool:
        """Stream-download *url* to *dest* with resume support via .part file.

        Returns
        -------
        bool
            True if the download succeeded; False on error.
        """
        part_path = dest.with_suffix(".part")
        headers = {}
        initial_bytes = 0

        if part_path.exists():
            initial_bytes = part_path.stat().st_size
            if initial_bytes > 0:
                headers["Range"] = f"bytes={initial_bytes}-"
                logger.debug("Resuming %s from byte %d", dest.name, initial_bytes)

        try:
            logger.debug("Downloading %s …", url)
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 404:
                    logger.debug("Archive not yet available: %s", url)
                    self.stats["not_found_404"] += 1
                    if part_path.exists():
                        part_path.unlink()
                    return False

                if resp.status_code == 416:
                    # Requested range not satisfiable - file might already be complete
                    part_path.replace(dest)
                    if self._is_valid_cache(dest):
                        return True
                    part_path.unlink(missing_ok=True)
                    initial_bytes = 0
                    headers = {}
                    # Re-download from start
                    resp = requests.get(url, stream=True, timeout=120)

                resp.raise_for_status()

                mode = "ab" if (resp.status_code == 206 and initial_bytes > 0) else "wb"
                with part_path.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)

            # Move part file to final destination once complete
            if part_path.exists():
                part_path.replace(dest)

            if self._is_valid_cache(dest):
                logger.debug("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
                return True
            else:
                logger.warning("Corrupted download for %s, discarding.", dest.name)
                dest.unlink(missing_ok=True)
                self.stats["failed_errors"] += 1
                return False

        except requests.RequestException as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            self.stats["failed_errors"] += 1
            self.stats["error_details"].append({"url": url, "error": str(exc)})
            return False

