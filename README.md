# NumFOCUS Developer Collaboration Network & Link Prediction

Empirical mining, network analysis, and topological link prediction across the **58 open-source scientific computing projects** supported by the **NumFOCUS** foundation (e.g., NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Julia, Spyder).

Based on the research dataset:  
*A Dataset of Contributor Activities in the NumFocus Open-Source Community*  
**Authors:** Youness Hourri, Alexandre Decan, Tom Mens (MSR 2025)  
**Zenodo DOI:** [10.5281/zenodo.14230406](https://doi.org/10.5281/zenodo.14230406)

---

## 📌 Executive Summary

This project constructs an empirical collaboration network of open-source scientific software maintainers and evaluates unsupervised link prediction heuristics (Common Neighbors, Jaccard Coefficient, Adamic–Adar, and Preferential Attachment) under strict anti-leakage evaluation protocols. It produces candidate collaborator recommendations with structural explainability, high-resolution figures (PNG, SVG), interactive PyVis graphs, Cytoscape-compatible GraphML models, an academic PDF report, and a PowerPoint slide deck.

---

## 🏗️ Project Architecture

```
snaaat/
├── data/
│   ├── raw/                           # Downloaded Zenodo files & schemas
│   │   ├── action-schema.json
│   │   ├── activity-schema.json
│   │   └── NumFocus_Jan22-Dec24_GH_Activities.zip
│   └── processed/                     # Extracted and cleaned tables
│       ├── cleaned_interactions.csv   # 211,872 verified human interactions
│       ├── edges.csv                  # 40,001 unique weighted undirected edges
│       └── nodes.csv                  # Developer node topological features
├── deliverables/                      # Generated academic deliverables
│   ├── figures/                       # High-resolution (300 DPI) PNG & SVG figures
│   │   ├── network_overall.png / .svg
│   │   ├── network_ego.png / .svg
│   │   ├── network_predictions.png / .svg
│   │   └── roc_curves.png / .svg
│   ├── developer_network.graphml      # Cytoscape-compatible graph export
│   ├── interactive_network.html       # Standalone interactive PyVis 3D force graph
│   ├── link_prediction_evaluation.csv # Benchmark metrics (ROC-AUC, Precision@K, F1@K)
│   ├── NumFOCUS_Collaboration_Report.pdf        # Formatted academic publication PDF
│   ├── NumFOCUS_Collaboration_Presentation.pptx # Professional PowerPoint slide deck
│   ├── NumFOCUS_Collaboration_Presentation.pdf  # Slide deck exported as landscape PDF
│   ├── recommendations_top_10_jreback.csv       # Top-10 recommended collaborators
│   ├── recommendations_top_20_jreback.csv       # Top-20 recommended collaborators
│   └── viva_questions_and_answers.md            # 10 rigorous viva questions & answers
├── notebooks/
│   └── numfocus_collaboration_network.ipynb     # Fully executed Jupyter Notebook
├── src/                               # Modular Python pipeline components
│   ├── __init__.py
│   ├── config.py                      # Global configuration and constants
│   ├── data_loader.py                 # Schema inspection & streaming activities parser
│   ├── graph_builder.py               # Hygiene filters, edge weighting & graph stats
│   ├── link_prediction.py             # 80/20 train-test split & 4 heuristics
│   ├── evaluation.py                  # ROC-AUC, Precision@K, Recall@K, F1@K metrics
│   ├── recommendation.py              # Candidate ranking & structural explanation engine
│   ├── visualization.py               # Matplotlib & PyVis network plotters
│   └── report_generator.py            # ReportLab PDF & python-pptx presentation engine
├── tests/
│   └── test_pipeline.py               # Comprehensive pytest test suite (8 tests)
├── run_pipeline.py                    # Master CLI pipeline runner
└── requirements.txt                   # Dependency definitions
```

---

## ⚙️ Setup & Execution Instructions

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Dataset Processing
```powershell
# Download Zenodo dataset & extract verified developer interactions
python scripts/process_dataset.py
```

### 3. Master Pipeline Execution
```powershell
# Run the complete end-to-end pipeline for a designated starting maintainer
python run_pipeline.py --user jreback --hops 2
```
*Note: Any GitHub username can be passed to `--user` (e.g., `mroeschke`, `rgommers`, `thomasjpfan`, `wesm`). If a username is missing or lacks sufficient interactions, the system reports diagnostic feedback and suggests candidate contributors.*

### 4. Running the Jupyter Notebook
```powershell
# Open or execute the notebook
jupyter notebook notebooks/numfocus_collaboration_network.ipynb
```

### 5. Running Automated Tests
```powershell
pytest tests/test_pipeline.py -v
```

---

## 📊 Empirical Results Summary

### Network Topology (2-Hop Ego Network centered on `@jreback`)
- **Total Nodes ($|V|$):** 2,591 developers
- **Total Edges ($|E|$):** 9,588 weighted collaboration ties
- **Network Density:** 0.002858 (characteristic sparse scale-free OSS structure)
- **Average Node Degree:** 7.40
- **Maximum Node Degree:** 562 (`@mroeschke`)
- **Average Clustering Coefficient:** 0.3998
- **Transitivity:** 0.0741
- **Largest Component:** 100.0% of nodes connected in primary community

### Link Prediction Performance (80% Train / 20% Test, $N=1,917$ positive test edges, $N=1,917$ negative samples)
*Strict Anti-Leakage Protocol: Test edges are completely removed from $G_{train}$ before computing topological metrics.*

| Algorithm | ROC-AUC | Precision@10 | Recall@10 | F1@10 | Precision@20 | Recall@20 | F1@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adamic–Adar (AA)** | **0.8615** | **1.0000** | 0.0052 | 0.0104 | **1.0000** | 0.0104 | 0.0207 |
| **Common Neighbors (CN)** | 0.8567 | **1.0000** | 0.0052 | 0.0104 | **1.0000** | 0.0104 | 0.0207 |
| **Preferential Attachment (PA)** | 0.8591 | **1.0000** | 0.0052 | 0.0104 | **1.0000** | 0.0104 | 0.0207 |
| **Jaccard Coefficient (JC)** | 0.8107 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

*Finding:* **Adamic–Adar** achieves the highest ROC-AUC (0.8615) by dampening mega-hub maintainers through logarithmic inverse-degree weighting ($\frac{1}{\log deg(w)}$).

---

## 🔍 Top Collaborator Recommendations for `@jreback`

*All existing neighbors, self-loops, and bots are excluded.*

1. **`@thomasjpfan` (Thomas Fan, scikit-learn core dev):**
   - *Adamic–Adar Score:* 8.39 | *Common Neighbors:* 20 | *Degree:* 119
   - *Key Mutual Collaborators:* `CloseChoice`, `Diwakar-Gupta`, `Jakobhenningjensen` (+17 others).
2. **`@seberg` (Sebastian Berg, NumPy core dev):**
   - *Adamic–Adar Score:* 6.94 | *Common Neighbors:* 19 | *Degree:* 122
   - *Key Mutual Collaborators:* `Apteryks`, `DWesl`, `Howard-O-Neil` (+16 others).
3. **`@samukweku` (NumPy / SciPy ecosystem):**
   - *Adamic–Adar Score:* 6.61 | *Common Neighbors:* 11 | *Degree:* 15
   - *Key Mutual Collaborators:* `ELHoussineT`, `Gabriel-ROBIN`, `MichaelTiemannOSC` (+8 others).
4. **`@jklymak` (Jody Klymak, Matplotlib lead maintainer):**
   - *Adamic–Adar Score:* 6.36 | *Common Neighbors:* 17 | *Degree:* 67
   - *Key Mutual Collaborators:* `Apteryks`, `DWesl`, `alexlenail` (+14 others).
5. **`@rgommers` (Ralf Gommers, NumPy / SciPy lead maintainer):**
   - *Adamic–Adar Score:* 6.09 | *Common Neighbors:* 19 | *Degree:* 123
   - *Key Mutual Collaborators:* `DWesl`, `TomAugspurger`, `WillAyd` (+16 others).

---

## ⚠️ Limitations & Threats to Validity

1. **Observation Horizon:** The dataset spans January 2022 to December 2024. Collaborations established strictly prior to 2022 are not recorded.
2. **Platform Scope:** Interactions occurring outside GitHub (mailing lists, Discord, Slack, Gitter, PyData conferences) cannot be observed.
3. **Cold Start Effect:** New or isolated developers lack triadic closure, causing neighborhood-based heuristics (CN, JC, AA) to yield 0. This is mitigated by falling back to Preferential Attachment.

