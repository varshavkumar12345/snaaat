"""
Configuration parameters for the NumFOCUS Developer Collaboration Network & Link Prediction pipeline.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
REPORTS_DIR = DELIVERABLES_DIR / "reports"
FIGURES_DIR = DELIVERABLES_DIR / "figures"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DELIVERABLES_DIR, REPORTS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset URLs & Filenames
ZENODO_RECORD_ID = "14230406"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
ACTIVITIES_ZIP = "NumFocus_Jan22-Dec24_GH_Activities.zip"
ACTIONS_ZIP = "NumFocus_Jan22-Dec24_GH_Actions.zip"
ACTION_SCHEMA = "action-schema.json"
ACTIVITY_SCHEMA = "activity-schema.json"

# Bot filtering keywords
BOT_IDENTIFIERS = {
    "bot", "github-actions", "dependabot", "dependabot[bot]", "github-actions[bot]",
    "codecov", "codecov[bot]", "pre-commit-ci", "pre-commit-ci[bot]", "stale", "stale[bot]",
    "gitter-badger", "numfocus-bot", "conda-forge-linter", "regro-cf-autotick-bot",
    "pep8speaks", "coveralls", "imgbot[bot]", "greenkeeper[bot]", "renovate[bot]",
    "web-flow"
}

# Link Prediction parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20
TOP_K_EVAL = [5, 10, 20, 50]

# Candidate starting users (NumFOCUS prominent core developers/maintainers)
# e.g., 'wesm' (Pandas/Arrow), 'mrocklin' (Dask), 'stefanv' (scikit-image), 'choldgraf' (Jupyter)
DEFAULT_STARTING_USER = "wesm"

