# GitHub Developer Collaboration Network & Link Prediction

> **Academic Social Network Analysis (SNA) Project**  
> Constructs a verifiable developer collaboration graph from public GitHub Archive activity, performs zero-leakage link prediction, evaluates four classical algorithms, and generates full academic deliverables (ReportLab PDF, PowerPoint slide deck, PyVis interactive network, Cytoscape GraphML, and explainable recommendations).

---

## 1. Project Overview

This application accepts any GitHub username (configured in `config.yaml` or provided via CLI), queries public GitHub event streams, isolates direct dyadic collaboration actions, and predicts future developer connections.

### Key Capabilities
- **Strict Interaction Provenance:** Rejects naive "same-repo" commits. Edges represent direct code review actions (`PullRequestReviewEvent`, `PullRequestReviewCommentEvent`), issue discussions (`IssueCommentEvent`), PR merges (`PullRequestEvent`), and collaborator additions (`MemberEvent`).
- **Deterministic Multi-Hop Ego Expansion:** Expands 1-hop and 2-hop networks with deterministic BFS node priority capping.
- **Zero-Leakage Train/Test Split:** 80% train / 20% test split on positive edges with matching negative non-edges; test edges are strictly purged from the training graph prior to topological feature calculation.
- **Topological Link Prediction:** Benchmarks **Common Neighbors**, **Jaccard Coefficient**, **Adamic–Adar**, and **Preferential Attachment**.
- **Structural Explanations:** Explains recommended connections using concrete graph features (e.g. mutual collaborators, degree products).
- **Academic Deliverables:** Generates actual, verifiable PDF reports, PowerPoint presentations, interactive HTML graphs, and Cytoscape GraphML exports.

---

## 2. Directory Structure

```
snaaat/
├── config.yaml                         # Configurable hyperparameters & settings
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation & execution guide
├── data/
│   ├── raw/                            # Cached GitHub Archive dumps (.json.gz)
│   ├── processed/                      # Node tables, edge tables, full graphml
│   └── splits/                         # Train/test edge splits & negative samples
├── docs/
│   └── viva_questions_and_answers.md   # Viva voce exam Q&A guide
├── notebooks/
│   └── github_link_prediction.ipynb    # Executable step-by-step Jupyter Notebook
├── outputs/
│   ├── visualizations/
│   │   ├── network_overall.png (.svg)  # Full layout visualization
│   │   ├── network_ego.png (.svg)      # Seed ego-network visualization
│   │   ├── network_top10_predicted.png # Top-10 predicted edges overlay
│   │   ├── network_interactive.html    # PyVis interactive 3D/physics viewer
│   │   └── cytoscape_network.graphml   # Cytoscape-compatible GraphML
│   ├── recommendations/
│   │   ├── recommendations_top10.csv   # Top-10 recommendations per algorithm
│   │   ├── recommendations_top20.csv   # Top-20 recommendations per algorithm
│   │   ├── recommendations_ensemble.csv# Consensus ranked recommendations
│   │   └── recommendations_explained.json
│   ├── reports/
│   │   ├── collaboration_network_report.pdf # Formatted academic PDF report
│   │   └── evaluation_results.csv      # Algorithm benchmark comparison table
│   └── presentation/
│       └── project_presentation.pptx   # PowerPoint slide deck
├── scripts/
│   ├── run_phase2.py                   # Archive inspection & coverage analysis
│   ├── validate_phase3_4.py            # Ego discovery & 4 validation cases
│   ├── validate_phase5.py              # Zero-leakage split validation
│   ├── validate_phase6.py              # Algorithm evaluation validation
│   └── generate_all_deliverables.py    # Master runner generating all artifacts
├── src/
│   ├── __init__.py                     # Config loader & logging initialization
│   ├── github_client.py                # GitHub REST API client with backoff
│   ├── archive_collector.py            # Hourly archive downloader & cache manager
│   ├── archive_inspector.py            # Schema inspector & coverage analyzer
│   ├── event_parser.py                 # Direct interaction parser & provenance
│   ├── graph_builder.py                # NetworkX builder & BFS ego expansion
│   ├── collaborator_discovery.py       # Seed discovery engine & edge case handler
│   ├── preprocessing.py                # Cleaning, 80/20 split & negative sampling
│   ├── link_prediction.py              # CN, Jaccard, Adamic-Adar, PA & explainer
│   ├── evaluation.py                   # ROC-AUC, PR-AUC, Prec@K, Recall@K, F1
│   ├── recommendations.py              # Top-K & ensemble consensus engine
│   ├── visualization.py                # Static, vector, PyVis & GraphML suite
│   └── report_generator.py             # ReportLab PDF & python-pptx generator
└── tests/                              # Automated unit tests (pytest)
```

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd snaaat

# Install dependencies
pip install -r requirements.txt
```

### Running the Complete Pipeline
To run the entire pipeline and generate all deliverables:
```bash
python scripts/generate_all_deliverables.py
```

### Running the Step-by-Step Validation Scripts
```bash
# Phase 2: Inspect archive schemas and generate coverage report
python scripts/run_phase2.py

# Phase 3 & 4: Validate seed user discovery, 1-hop/2-hop expansion, and edge cases
python scripts/validate_phase3_4.py

# Phase 5: Validate 80/20 train/test splitting and zero leakage
python scripts/validate_phase5.py

# Phase 6: Validate link prediction algorithms, metrics, and explanations
python scripts/validate_phase6.py
```

### Running Unit Tests
```bash
pytest tests/ -v
```

---

## 4. Generated Deliverables Summary

| Deliverable | Location | Description |
|---|---|---|
| **Academic Evaluation Report** | `outputs/reports/collaboration_network_report.pdf` | Formatted multi-page PDF report with embedded figures, topology metrics, and benchmark tables. |
| **PowerPoint Presentation** | `outputs/presentation/project_presentation.pptx` | 16:9 widescreen academic slide deck ready for project defense. |
| **Interactive Network Graph** | `outputs/visualizations/network_interactive.html` | PyVis interactive network visualization with physics and node inspection. |
| **Cytoscape GraphML** | `outputs/visualizations/cytoscape_network.graphml` | Typed GraphML file ready for import into Cytoscape. |
| **Predicted Links Overlay** | `outputs/visualizations/network_top10_predicted.png` (`.svg`) | Visualizes observed edges (solid slate) alongside Top-10 predicted edges (dashed red). |
| **Global & Ego Network Views** | `outputs/visualizations/network_overall.png`, `network_ego.png` | PNG & SVG vector renders of global and ego-level network topologies. |
| **Algorithm Benchmark Table** | `outputs/reports/evaluation_results.csv` | ROC-AUC, PR-AUC, Precision@10, Recall@10, and F1 across all 4 algorithms. |
| **Explainable Recommendations** | `outputs/recommendations/recommendations_top10.csv`, `top20.csv`, `ensemble.csv` | Ranked candidate collaborators with structural feature explanations. |
| **Viva Voce Exam Guide** | `docs/viva_questions_and_answers.md` | Comprehensive Q&A covering methodology, graph theory, and limitations. |
| **Executable Jupyter Notebook** | `notebooks/github_link_prediction.ipynb` | End-to-end interactive notebook. |

---

## 5. Limitations & Ethical Data Notes
1. **Temporal Slicing:** GitHub Archive captures hourly slices. An individual student account with no activity in a specific 1-hour window will show zero interactions without historical API enrichment.
2. **Commit Email Privacy:** `PushEvent` commit authors are excluded from edge creation unless verified through dyadic GitHub platform interactions (PR review, issue comments) to prevent false email attribution.
3. **Score Interpretation:** Link prediction scores are comparative topological ranking heuristics and should not be interpreted as absolute probabilities of collaboration.
