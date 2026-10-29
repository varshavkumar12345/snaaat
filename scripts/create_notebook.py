"""
Script to create the complete Jupyter Notebook for NumFOCUS Collaboration Network & Link Prediction.
"""
import nbformat as nbf
from pathlib import Path

def build_notebook():
    project_root = Path(__file__).resolve().parent.parent
    nb = nbf.v4.new_notebook()

    cells = []

    # Title & Metadata Markdown
    cells.append(nbf.v4.new_markdown_cell("""# NumFOCUS Developer Collaboration Network & Link Prediction
## Empirical Mining and Structural Modeling of Open-Source Scientific Computing

**Dataset:** *A Dataset of Contributor Activities in the NumFocus Open-Source Community* (Zenodo DOI: [10.5281/zenodo.14230406](https://doi.org/10.5281/zenodo.14230406))  
**Authors:** Youness Hourri, Alexandre Decan, Tom Mens (IEEE/ACM MSR 2025)  
**Primary Focus:** Scientific Python and NumFOCUS ecosystem (NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Julia)

---
### Workflow Steps Covered:
1. **Project Setup & Dataset Inspection:** Inspect schema, activity types, and interaction links.
2. **Starting Developer Resolution:** Configurable starting maintainer resolution.
3. **Data Cleaning & Network Construction:** Bot removal, self-loop pruning, edge aggregation, and NetworkX modeling.
4. **Train-Test Splitting & Link Prediction:** 80/20 split without data leakage; 4 heuristic algorithms (CN, JC, AA, PA).
5. **Evaluation & Recommendations:** ROC-AUC, Precision@K, Recall@K, F1@K, and Top-K collaborator recommendations.
6. **Visualizations & Deliverables:** Network plots, interactive PyVis HTML, PDF report, and PPTX presentation.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path

# Set project root
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DELIVERABLES_DIR, FIGURES_DIR
from src.data_loader import inspect_schemas, stream_activities
from src.graph_builder import (
    resolve_starting_developer, construct_graph,
    compute_network_statistics, export_network_tables
)
from src.link_prediction import train_test_split_edges, score_pairs, ALGORITHMS
from src.evaluation import evaluate_algorithm, compare_all_algorithms
from src.recommendation import generate_collaborator_recommendations, export_recommendations
from src.visualization import (
    plot_overall_network, plot_ego_network, plot_predicted_links_network,
    plot_roc_curves, generate_interactive_pyvis
)

print("Environment initialized successfully. Matplotlib backend ready.")
"""))

    # Section 1: Dataset Inspection
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Project Setup and Dataset Inspection
We inspect the schema definitions provided with Zenodo record 14230406. The dataset provides a two-tier hierarchy:
- **Actions:** Low-level operations.
- **Activities:** 21 synthesized tasks (e.g., `ReviewPullRequest`, `CommentPullRequest`, `CommentIssue`).
Direct developer interactions are extracted from PR code reviews, comments, and issue discussions.
"""))

    cells.append(nbf.v4.new_code_cell("""schemas = inspect_schemas()
print("Action Schema Title:", schemas.get("action_schema", {}).get("title"))
print("Activity Schema Title:", schemas.get("activity_schema", {}).get("title"))
print("Key Activity Properties:", list(schemas.get("activity_schema", {}).get("properties", {}).keys()))
"""))

    # Section 2: Loading & Cleaning
    cells.append(nbf.v4.new_markdown_cell("""## Step 2 & 3: Data Cleaning, Starting Developer & Network Construction
We load the pre-aggregated clean interactions and edges extracted from the 500,000+ activity records.
We resolve our designated starting maintainer (`@jreback`, lead maintainer of pandas) and extract an ego network.
"""))

    cells.append(nbf.v4.new_code_cell("""edge_csv = PROCESSED_DATA_DIR / "edges.csv"
edge_df = pd.read_csv(edge_csv)
print(f"Total Unique Undirected Edges in Dataset: {len(edge_df):,}")

# Resolve Starting Developer
starting_user = "jreback"
res_user = resolve_starting_developer(edge_df, starting_user)
print(f"\\nDeveloper Resolution: @{res_user['canonical_username']}")
print(f"Degree: {res_user['degree']} | Total Verified Interactions: {res_user['total_weight']:,}")
print("Top 5 Collaborators:")
for nbr, wt in res_user["top_collaborators"][:5]:
    print(f"  - @{nbr:<18}: {wt} interactions")
"""))

    cells.append(nbf.v4.new_code_cell("""# Construct 2-hop ego network
G = construct_graph(edge_df, starting_user=starting_user, hops=2)
stats = compute_network_statistics(G)

stats_df = pd.DataFrame([
    {"Metric": "Nodes (|V|)", "Value": f"{stats['num_nodes']:,}"},
    {"Metric": "Edges (|E|)", "Value": f"{stats['num_edges']:,}"},
    {"Metric": "Density", "Value": f"{stats['density']:.6f}"},
    {"Metric": "Average Degree", "Value": f"{stats['avg_degree']:.2f}"},
    {"Metric": "Max Degree", "Value": str(stats['max_degree'])},
    {"Metric": "Transitivity", "Value": f"{stats['transitivity']:.4f}"},
    {"Metric": "Avg Clustering Coeff", "Value": f"{stats['avg_clustering']:.4f}"},
    {"Metric": "Largest Component Ratio", "Value": f"{stats['largest_cc_ratio']:.1%}"}
])
stats_df
"""))

    # Section 3: Train-Test Split & Link Prediction
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Train-Test Split and Link Prediction
To prevent **data leakage**, positive edges are split into 80% training ($G_{train}$) and 20% testing ($E_{test}$).
All test edges are strictly purged from $G_{train}$ before calculating topological scores.
Negative non-edges are uniformly sampled.
"""))

    cells.append(nbf.v4.new_code_cell("""G_train, train_pos, test_pos, train_neg, test_neg = train_test_split_edges(
    G, test_ratio=0.20, seed=42
)
print(f"Train Positive Edges (G_train): {len(train_pos):,}")
print(f"Test Positive Edges:            {len(test_pos):,}")
print(f"Test Negative Samples:          {len(test_neg):,}")

# Test edge leakage check
leakage = any(G_train.has_edge(u, v) for u, v in test_pos)
print(f"Data Leakage Verification: {'LEAKAGE DETECTED!' if leakage else 'PASSED (0 test edges in G_train)'}")
"""))

    # Section 4: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Algorithmic Evaluation & Metric Comparison
We evaluate four core heuristics:
1. **Common Neighbors (CN):** $|N(u) \\cap N(v)|$
2. **Jaccard Coefficient (JC):** $\\frac{|N(u) \\cap N(v)|}{|N(u) \\cup N(v)|}$
3. **Adamic–Adar (AA):** $\\sum_{w \\in N(u) \\cap N(v)} \\frac{1}{\\log |N(w)|}$
4. **Preferential Attachment (PA):** $|N(u)| \\cdot |N(v)|$
"""))

    cells.append(nbf.v4.new_code_cell("""eval_pairs = test_pos + test_neg
y_true = np.array([1] * len(test_pos) + [0] * len(test_neg))

eval_results = {}
for algo in ALGORITHMS.keys():
    scores = score_pairs(G_train, eval_pairs, algo)
    metrics = evaluate_algorithm(y_true, scores, k_list=[10, 20])
    eval_results[algo] = metrics

eval_df = compare_all_algorithms(eval_results)
eval_df
"""))

    # Section 5: Recommendations
    cells.append(nbf.v4.new_markdown_cell("""## Step 5 (Cont.): Collaborator Recommendations for `@jreback`
Using Adamic–Adar as the primary scoring mechanism, we rank candidate developers.
Existing neighbors, self-loops, and bots are filtered out.
"""))

    cells.append(nbf.v4.new_code_cell("""recommendations = generate_collaborator_recommendations(
    G, starting_user, top_k=20, primary_algo="Adamic-Adar"
)
rec_df = pd.DataFrame([{
    "Rank": r["rank"],
    "Candidate": r["candidate_user"],
    "Candidate Degree": r["candidate_degree"],
    "Common Neighbors": r["common_neighbors_count"],
    "Adamic-Adar": r["adamic_adar_score"],
    "Jaccard": r["jaccard_coefficient"],
    "Preferential Attachment": r["preferential_attachment"],
    "Explanation": r["explanation"]
} for r in recommendations])
rec_df.head(10)
"""))

    # Section 6: Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Visualizations
Visualizations of the collaboration topology, 1-hop ego network, and link prediction overlays.
"""))

    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

# Display generated high-resolution visualizations
print("1. Ego Network:")
display(Image(filename=str(DELIVERABLES_DIR / "network_ego.png"), width=650))

print("2. Link Prediction Overlay (Predicted edges in dashed orange):")
display(Image(filename=str(DELIVERABLES_DIR / "network_predictions.png"), width=650))

print("3. Comparative ROC Curves:")
display(Image(filename=str(DELIVERABLES_DIR / "roc_curves.png"), width=550))
"""))

    # Section 7: Deliverables & Summary
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Academic Deliverables Summary
All deliverables have been compiled and exported to the `deliverables/` directory:
- **Cytoscape GraphML:** `deliverables/developer_network.graphml`
- **Interactive Network Visualization:** `deliverables/interactive_network.html`
- **Academic PDF Report:** `deliverables/NumFOCUS_Collaboration_Report.pdf`
- **Presentation Deck:** `deliverables/NumFOCUS_Collaboration_Presentation.pptx`
- **Presentation PDF:** `deliverables/NumFOCUS_Collaboration_Presentation.pdf`
- **Viva Questions & Answers:** `deliverables/viva_questions_and_answers.md`
"""))

    nb.cells = cells
    
    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / "numfocus_collaboration_network.ipynb"
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"Jupyter Notebook generated at: {nb_path}")
    return nb_path

if __name__ == "__main__":
    build_notebook()
