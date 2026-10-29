# GitHub Developer Collaboration Network & Link Prediction

> **Academic Social Network Analysis project** — constructs a developer
> collaboration graph from public GitHub activity, applies four link-prediction
> algorithms, evaluates them, and generates a full set of deliverables
> (interactive HTML visualisation, GraphML, PDF report, and PowerPoint deck).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Running the Notebook](#7-running-the-notebook)
8. [Running the Tests](#8-running-the-tests)
9. [Outputs](#9-outputs)
10. [Pipeline Phases](#10-pipeline-phases)
11. [Algorithms](#11-algorithms)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

The pipeline:

1. Accepts a **GitHub username** (configured in `config.yaml`).
2. Downloads hourly **GitHub Archive** `.json.gz` files for the configured date
   range and extracts collaboration events (PushEvent, PullRequestEvent,
   IssuesEvent, IssueCommentEvent, ForkEvent, WatchEvent, CreateEvent).
3. Optionally enriches the network via the **GitHub REST API**.
4. Constructs a **weighted undirected collaboration graph** (NetworkX).
5. Computes standard **SNA metrics** (degree, betweenness, clustering, etc.).
6. Runs **four link-prediction algorithms** and evaluates them with AUC-ROC,
   Precision, Recall, F1, and Average Precision.
7. Generates **Top-10 / Top-20 collaboration recommendations**.
8. Produces: interactive HTML, static PNG plots, Cytoscape GraphML, a PDF
   report, and a PowerPoint presentation.

---

## 2. Directory Structure

```
github-collaboration-prediction/
├── README.md
├── requirements.txt
├── config.yaml                  ← All tunable parameters
├── .gitignore
├── data/
│   ├── raw/                     ← Downloaded .json.gz archive files
│   ├── processed/               ← Parsed events as Parquet / CSV
│   ├── splits/                  ← Train / test edge splits
│   └── README.md
├── notebooks/
│   └── github_link_prediction.ipynb
├── src/
│   ├── github_client.py         ← GitHub REST API wrapper
│   ├── archive_collector.py     ← GitHub Archive downloader / cache manager
│   ├── event_parser.py          ← Raw JSON → structured events
│   ├── graph_builder.py         ← Events → NetworkX graph
│   ├── preprocessing.py         ← Filtering, normalisation, train/test split
│   ├── link_prediction.py       ← Four link-prediction algorithms
│   ├── evaluation.py            ← Metrics: AUC-ROC, P, R, F1, AP
│   ├── recommendations.py       ← Top-K recommendation engine
│   ├── visualization.py         ← Static + interactive + GraphML outputs
│   └── report_generator.py      ← PDF + PowerPoint generation
├── tests/
│   ├── test_event_parser.py
│   ├── test_graph_builder.py
│   ├── test_link_prediction.py
│   └── fixtures/
└── outputs/
    ├── visualizations/
    ├── recommendations/
    ├── reports/
    └── presentation/
```

---

## 3. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 also tested |
| pip | 23+ | `python -m pip install --upgrade pip` |
| Internet | — | Required for GitHub Archive & API |
| RAM | 4 GB | 8 GB recommended for larger networks |
| Disk | 2 GB free | Archive files can be large |

---

## 4. Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd github-collaboration-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Verify imports
python -c "import src.graph_builder; print('All modules importable.')"
```

> **GitHub Token (recommended)**
> Unauthenticated GitHub API calls are limited to 60 requests per hour.
> Create a **fine-grained Personal Access Token** with *Public Repositories*
> read access at <https://github.com/settings/tokens> and add it to
> `config.yaml` → `github.token`.

---

## 5. Configuration

Edit **`config.yaml`** before running the pipeline:

```yaml
github:
  username: "varshavkumar12345"   # ← change to your seed user
  token:    ""                    # ← paste your PAT here

collection:
  start_date: "2024-10-01"
  end_date:   "2024-10-07"
  max_archive_files: 24           # keep low on a laptop

graph:
  max_nodes: 500
  max_depth: 2
```

All other parameters (bot exclusion, algorithm list, Top-K values, etc.) are
documented inline in `config.yaml`.

---

## 6. Running the Pipeline

```bash
# Run the full pipeline (all phases)
python -m src.pipeline

# Override the username at the command line
python -m src.pipeline --username torvalds

# Override multiple settings
python -m src.pipeline --username torvalds --config config.yaml
```

---

## 7. Running the Notebook

```bash
jupyter notebook notebooks/github_link_prediction.ipynb
```

The notebook mirrors the full pipeline and is self-contained. Run all cells
top-to-bottom. Each phase prints a validation summary before proceeding.

---

## 8. Running the Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 9. Outputs

| File | Location | Description |
|---|---|---|
| `network_interactive.html` | `outputs/visualizations/` | PyVis interactive graph |
| `network_static.png` | `outputs/visualizations/` | Matplotlib network plot |
| `degree_distribution.png` | `outputs/visualizations/` | Degree distribution chart |
| `collaboration_network.graphml` | `outputs/visualizations/` | Cytoscape-ready export |
| `recommendations_top10.csv` | `outputs/recommendations/` | Top-10 link predictions |
| `recommendations_top20.csv` | `outputs/recommendations/` | Top-20 link predictions |
| `evaluation_results.csv` | `outputs/reports/` | Algorithm comparison table |
| `report.pdf` | `outputs/reports/` | Full academic PDF report |
| `presentation.pptx` | `outputs/presentation/` | Auto-generated slide deck |
| `pipeline.log` | `outputs/` | Full execution log |

---

## 10. Pipeline Phases

| Phase | Module(s) | Description |
|---|---|---|
| 1 | — | Project setup & architecture (this phase) |
| 2 | `archive_collector`, `github_client` | Data collection from GitHub Archive & API |
| 3 | `event_parser`, `preprocessing` | Event parsing & graph preprocessing |
| 4 | `graph_builder` | Graph construction & SNA metrics |
| 5 | `link_prediction`, `evaluation` | Link prediction & evaluation |
| 6 | `recommendations` | Collaboration recommendations |
| 7 | `visualization` | Interactive & static visualizations |
| 8 | `report_generator` | PDF report & PowerPoint generation |

---

## 11. Algorithms

| Algorithm | Description |
|---|---|
| **Common Neighbors** | Counts shared neighbours between two nodes |
| **Jaccard Coefficient** | Normalises common neighbours by union of neighbours |
| **Adamic-Adar** | Weights common neighbours by inverse log degree |
| **Preferential Attachment** | Product of node degrees (rich-get-richer) |

---

## 12. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and re-run `pip install -r requirements.txt` |
| GitHub API 403 / rate limit | Add a PAT to `config.yaml → github.token` |
| Archive download timeout | Reduce `collection.max_archive_files` or widen the date range |
| Empty graph | Widen date range, lower `graph.min_edge_weight`, or increase `graph.max_nodes` |
| Memory error | Reduce `collection.max_archive_files` and `graph.max_nodes` |

