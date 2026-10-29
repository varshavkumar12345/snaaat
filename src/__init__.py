"""
src/__init__.py
---------------
Package initialiser for the GitHub Collaboration Network pipeline.

Exposes a convenience `load_config` function so every module can obtain the
shared configuration object with a single import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Package-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config path (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the YAML configuration as a plain Python dictionary.

    Parameters
    ----------
    path:
        Explicit path to a config file.  Defaults to the project-root
        ``config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please copy config.yaml from the project root."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Apply environment-variable overrides (e.g. GITHUB_TOKEN)
    if token := os.environ.get("GITHUB_TOKEN"):
        config.setdefault("github", {})["token"] = token

    if username := os.environ.get("GITHUB_USERNAME"):
        config.setdefault("github", {})["username"] = username

    # Configure logging level from config
    log_level = config.get("logging", {}).get("level", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    logger.debug("Configuration loaded from %s", config_path)
    return config

