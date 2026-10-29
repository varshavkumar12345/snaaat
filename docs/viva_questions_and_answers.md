# Academic Viva Voce: Questions & Answers

## Topic: GitHub Developer Collaboration Network & Link Prediction

---

### Q1: What constitutes a node and an edge in your developer collaboration graph?
**Answer:**
- **Nodes** represent identifiable individual human developers on GitHub. Each node carries attributes including canonical username, persistent numeric GitHub ID (when available), account type (`User`), qualifying interaction count, and degree. Automated bot accounts (e.g. `dependabot[bot]`, `renovate-bot`) are filtered out.
- **Edges** represent direct dyadic peer collaboration events: Pull Request code reviews, inline review comments, issue discussion threads, pull request merges, and repository collaborator additions.
- Edges are weighted by interaction frequency and annotated with full provenance (source event ID, timestamp, and repository name).

---

### Q2: Why don't you create edges between any two developers who pushed code to the same repository?
**Answer:**
In large repositories (e.g., Linux, Kubernetes, VS Code), thousands of independent developers commit code without interacting or even knowing each other. Treating same-repo commits as a collaboration edge introduces severe transitive noise and dense false-positive cliques. Restricting edges strictly to dyadic interactions (reviewing, commenting, assigning, merging) ensures our graph represents genuine peer collaboration.

---

### Q3: How do you prevent data leakage during train/test splitting for link prediction?
**Answer:**
We perform an 80/20 split on existing positive edges. All held-out 20% test edges are **strictly removed** from the training graph $G_{\text{train}}$. All topological features (such as common neighbors, neighborhood degrees, and paths) are computed **exclusively** on $G_{\text{train}}$. Computing topological metrics on the full graph prior to removing test edges would cause data leakage and artificially inflate evaluation metrics.

---

### Q4: How are negative samples generated, and why is negative sampling critical?
**Answer:**
Link prediction is a binary classification/ranking task where positive instances are observed edges. Negative samples are non-edges: pairs of developers $(u, v)$ with no observed edge in the ground-truth network.
- We generate negative samples uniformly at random from the complement set of all possible non-edges, with an explicit check ensuring zero overlap with observed edges and zero self-loops $(u, u)$.
- Negative sampling is crucial because real social networks are extremely sparse; without a controlled negative sample (or evaluating against the entire non-edge candidate space), true negatives would overwhelm the evaluation.

---

### Q5: Explain the mathematical intuition behind the four link-prediction algorithms.
**Answer:**
1. **Common Neighbors (CN):**
   $$\text{score}(u, v) = |N(u) \cap N(v)|$$
   Measures triadic closure: two developers sharing many mutual collaborators are likely to collaborate.
2. **Jaccard Coefficient (JC):**
   $$\text{score}(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$
   Normalizes common neighbors by the total size of their combined neighborhoods to avoid bias toward high-degree hub nodes.
3. **Adamic–Adar (AA):**
   $$\text{score}(u, v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\log |N(w)|}$$
   Penalizes high-degree shared neighbors. Sharing a niche, specialized collaborator provides stronger evidence of collaboration than sharing a universal maintainer who collaborates with thousands of people.
4. **Preferential Attachment (PA):**
   $$\text{score}(u, v) = |N(u)| \times |N(v)|$$
   Models the scale-free rich-get-richer phenomenon: highly active developers have a higher probability of acquiring new links.

---

### Q6: Why do we evaluate with both ROC-AUC and Precision@K / PR-AUC?
**Answer:**
- **ROC-AUC** evaluates the algorithm's global ranking capability across all false-positive rates. However, in sparse link prediction problems, ROC-AUC can be overly optimistic because of the large pool of easy true negatives.
- **Precision@K (K=10, 20)** reflects practical utility: when presenting a developer with a list of "Top 10 recommended collaborators", what proportion of the top-ranked candidates are true collaborators?
- **PR-AUC (Average Precision)** focuses strictly on the positive class without being inflated by true negative counts.

---

### Q7: How do you handle a "cold-start" or sparse user (e.g. a student account with few public records)?
**Answer:**
In a narrow 1-hour or 1-day GitHub Archive slice, student or low-activity accounts may have zero events. Rather than inventing interactions, our pipeline:
1. Resolves the user identity via the GitHub REST API.
2. Checks public repositories for recent contributor and PR activity.
3. If activity remains zero, it reports an **honest insufficiency diagnosis** (`INSUFFICIENT_ACTIVITY`) and suggests expanding the time window or specifying an active open-source seed developer (e.g., `hamishwillee`, `torvalds`).

---

### Q8: What are the primary threats to validity in this analysis?
**Answer:**
1. **Temporal Truncation:** Public archives capture slices of time. Relationships formed before or after the window are invisible without full historical API re-crawls.
2. **Unresolved Git Emails:** Commit records in `PushEvent` specify git author emails rather than GitHub handles. Without API commit lookups, these cannot be attributed to GitHub user nodes.
3. **Bot Distortions:** Automated bots (Dependabot, Renovate) perform thousands of daily PR actions. Without bot filtering, graph centrality metrics become heavily skewed toward bot accounts.

