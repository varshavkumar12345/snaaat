# NumFOCUS Developer Collaboration Network & Link Prediction
## Academic Viva Questions and Comprehensive Answers

---

### Q1: What is the NumFOCUS dataset, and how does it differ from raw GitHub event logs?
**Answer:**
The dataset (*"A Dataset of Contributor Activities in the NumFocus Open-Source Community"*, Zenodo DOI: 10.5281/zenodo.14230406; Hourri, Decan, & Mens, MSR 2025) captures three years (2022–2024) of contributor actions across 58 prominent scientific computing projects (NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Julia, etc.).
Raw GitHub event dumps (e.g., GHArchive) contain immense transactional noise: raw Git push webhooks, automated bot pings, branch renames, and multi-event batches. The NumFOCUS dataset introduces a structured two-tier abstraction using `ghmap`:
1. **Actions (Low-level):** Granular atomic operations.
2. **Activities (High-level):** 21 synthesized intent-driven tasks (e.g., `ReviewPullRequest`, `CommentPullRequest`, `MergePullRequest`, `CommentIssue`).
This abstraction filters webhook noise and allows modeling authentic human collaboration.

---

### Q2: How are developer collaboration edges defined and verified in your network?
**Answer:**
Edges represent verifiable direct developer-to-developer interactions rather than passive co-presence. An undirected, weighted edge $(u, v)$ is established when developer $u$ interacts directly with developer $v$'s artifacts:
1. **Pull Request Code Reviews:** Developer $u$ submits a formal review (`ReviewPullRequest`) on a PR authored by developer $v$.
2. **Pull Request Comments:** Developer $u$ posts technical comments on developer $v$'s PR.
3. **Issue Conversations:** Developer $u$ discusses or responds to an issue opened by developer $v$.
4. **Pull Request Merges:** Maintainer $u$ merges code authored by developer $v$.

Edges are weighted by the total cumulative frequency of qualifying interactions over time. Automated bot accounts (`dependabot`, `github-actions`, etc.), invalid logins, and self-loops ($u = v$) are systematically excised.

---

### Q3: What is "data leakage" in link prediction, and how did you prevent it?
**Answer:**
Data leakage occurs when information from the evaluation/test set is inadvertently accessible to the prediction model during training or feature scoring. In graph link prediction, if topological proximity metrics (like Common Neighbors or Adamic–Adar) are computed on the full graph $G = (V, E)$, paths formed by positive test edges $E_{test}$ would artificially inflate common neighbor counts and proximity scores, generating spuriously high evaluation metrics.

**Prevention Protocol:**
1. Positive edges $E$ are split into 80% training ($E_{train}$) and 20% testing ($E_{test}$).
2. The training graph $G_{train} = (V, E_{train})$ is explicitly constructed with all test edges removed.
3. All graph features, neighbor sets $N(u)$, and degrees are queried strictly from $G_{train}$.
4. Negative test samples $E_{test}^{neg}$ are drawn uniformly at random from true non-edges ($(u, v) \notin E$).

---

### Q4: Explain the mathematical intuition behind the four link prediction heuristics implemented.
**Answer:**
1. **Common Neighbors (CN):**
   $$Score_{CN}(u, v) = |N(u) \cap N(v)|$$
   *Intuition:* Based on triadic closure: two developers sharing many mutual colleagues are more likely to interact.
2. **Jaccard Coefficient (JC):**
   $$Score_{JC}(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$
   *Intuition:* Normalizes mutual neighbors by total combined neighborhood size, reducing bias towards high-degree nodes.
3. **Adamic–Adar Index (AA):**
   $$Score_{AA}(u, v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\log |N(w)|}$$
   *Intuition:* Weights common neighbors inversely by their degree. A mutual collaborator $w$ who interacts with only 3 people provides a much stronger signal of specific shared interest than a mega-maintainer who interacts with 1,000 developers.
4. **Preferential Attachment (PA):**
   $$Score_{PA}(u, v) = |N(u)| \cdot |N(v)|$$
   *Intuition:* Rooted in scale-free network dynamics ("rich get richer"): popular, highly connected maintainers are globally more likely to form new connections regardless of local community structure.

---

### Q5: Why does Adamic–Adar typically outperform Common Neighbors and Preferential Attachment in software engineering networks?
**Answer:**
In open-source software collaboration graphs, certain maintainers act as global "mega-hubs" (e.g., release managers, triage leads) who comment on hundreds of disparate pull requests across many repositories. Under simple Common Neighbors, two unrelated contributors might appear highly similar simply because both interacted with the same triage lead.
Adamic–Adar applies sub-linear logarithmic damping ($\frac{1}{\log deg(w)}$), drastically down-weighting ubiquitous hub nodes and amplifying the signal of tightly knit, specialized domain collaborations. This leads to higher Precision@K and cleaner ranking of genuine potential collaborators.

---

### Q6: How does network sparsity affect link prediction, and why are both ROC-AUC and Precision@K evaluated?
**Answer:**
Developer collaboration networks are extremely sparse; density is typically on the order of $10^{-4}$ to $10^{-3}$. In a network with $|V|$ nodes, the number of candidate non-edges is $\mathcal{O}(|V|^2)$, while actual edges scale as $\mathcal{O}(|V|)$.
- **ROC-AUC:** Evaluates the rank-order probability that a randomly chosen true positive edge scores higher than a randomly chosen negative sample across all classification thresholds. It is threshold-independent and robust to class imbalance.
- **Precision@K:** Evaluates whether the top-ranked predictions (e.g., Top-10 or Top-20) presented to a developer are actually relevant. In practical recommendation systems, a user will only review the top few suggestions, making Precision@K and Recall@K critical pragmatic metrics alongside global ROC-AUC.

---

### Q7: What is the "Cold Start" problem in developer recommendation, and how is it mitigated?
**Answer:**
The cold start problem occurs when a developer is new or isolated (degree 0 or 1), having zero common neighbors ($|N(u) \cap N(v)| = 0$) with other nodes. Triadic closure heuristics (CN, JC, AA) produce zero scores.
**Mitigation strategies:**
1. **Preferential Attachment Fallback:** Recommends globally prominent maintainers who actively review and mentor across the project.
2. **Bipartite Repository Projection:** Incorporates repository-level co-activity or programming language affinity to bridge disjoint components.
3. **Explicit Reporting:** The system detects when a user cannot be resolved or has insufficient local degree and reports explanatory diagnostic feedback.

---

### Q8: How were recommendations for the starting developer filtered and made explainable?
**Answer:**
Recommendations are strictly post-filtered to guarantee actionable quality:
1. **Self-Loop Exclusion:** Developer cannot be recommended to themselves.
2. **Existing Neighbor Exclusion:** All developers with prior direct interactions ($v \in N(u)$) are removed, ensuring novelty.
3. **Bot Sanitization:** CI/CD bots, issue triagers, and automated linters are pruned.

**Explainability:**
Rather than outputting black-box scores, each recommendation is paired with human-readable structural features:
- Number and names of mutual collaborators.
- Candidate developer degree and network prominence.
- Comparative algorithm rankings (AA vs. JC vs. PA).
- Primary shared sub-ecosystem context.

---

### Q9: Could temporal link prediction be conducted with this dataset?
**Answer:**
Yes. Because every action record in the NumFOCUS dataset carries an ISO-8601 UTC timestamp (`start_date`, `end_date`), interactions can be partitioned chronologically. For example, edges formed between January 2022 and December 2023 serve as $G_{train}$, and links that emerge for the first time in 2024 serve as future test targets. This simulates real-world forecasting where future collaborations are predicted from past behavior.

---

### Q10: What are the main threats to validity and limitations of this study?
**Answer:**
1. **Construct Validity:** Collaboration is defined through GitHub PR reviews, comments, and issues. Non-GitHub interactions (conferences, Discord, Slack, mailing lists) are unobserved.
2. **Temporal Window:** Data is bounded between 2022 and 2024; historical partnerships forged before 2022 might be categorized as "new" links if their first recorded 2022+ interaction falls in the test partition.
3. **Bot Heuristics:** Although extensive regex and keyword lists identify automated bots, some bots using standard human GitHub accounts might evade detection without NLP inspection.

