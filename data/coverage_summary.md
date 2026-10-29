# Phase 2: GitHub Archive Data Understanding & Coverage Report

**Generated:** 1 archive files examined.

**Seed Developer:** `varshavkumar12345`

---

## 1. Summary of Archives Examined

- **Files Processed:** 1 hourly archive(s)
  - `2024-10-01-0.json.gz`
- **Total Events Scanned:** 186,403
- **Unique Repositories:** 77,901
- **Total Unique Actors:** 38,476
  - **Human Actors:** 37,897
  - **Bot/Automated Accounts Flagged:** 579

## 2. Event Types & Collaboration Potential

| Event Type | Count | % of Total | Direct Interaction Potential | Identifiable Fields |
|---|---|---|---|---|
| `PushEvent` | 115,562 | 62.0% | Low/Indirect (Repo co-contribution / Interest) | actor, created_at, id, org, payload |
| `CreateEvent` | 23,269 | 12.5% | None/Incidental | actor, created_at, id, org, payload |
| `PullRequestEvent` | 15,790 | 8.5% | Medium (Actor -> PR/Issue Creator/Assignee) | actor, created_at, id, org, payload |
| `IssueCommentEvent` | 7,209 | 3.9% | High (Dyadic: Reviewer/Commenter -> Author) | actor, created_at, id, org, payload |
| `WatchEvent` | 6,547 | 3.5% | Low/Indirect (Repo co-contribution / Interest) | actor, created_at, id, org, payload |
| `DeleteEvent` | 5,945 | 3.2% | None/Incidental | actor, created_at, id, org, payload |
| `PullRequestReviewEvent` | 3,944 | 2.1% | High (Dyadic: Reviewer/Commenter -> Author) | actor, created_at, id, org, payload |
| `IssuesEvent` | 2,285 | 1.2% | Medium (Actor -> PR/Issue Creator/Assignee) | actor, created_at, id, org, payload |
| `PullRequestReviewCommentEvent` | 1,537 | 0.8% | High (Dyadic: Reviewer/Commenter -> Author) | actor, created_at, id, org, payload |
| `ForkEvent` | 1,412 | 0.8% | Low/Indirect (Repo co-contribution / Interest) | actor, created_at, id, org, payload |
| `ReleaseEvent` | 1,224 | 0.7% | None/Incidental | actor, created_at, id, org, payload |
| `MemberEvent` | 743 | 0.4% | Medium (Actor -> PR/Issue Creator/Assignee) | actor, created_at, id, org, payload |
| `PublicEvent` | 474 | 0.3% | None/Incidental | actor, created_at, id, org, payload |
| `GollumEvent` | 337 | 0.2% | None/Incidental | actor, created_at, id, org, payload |
| `CommitCommentEvent` | 125 | 0.1% | None/Incidental | actor, created_at, id, org, payload |


## 3. Direct Dyadic Interaction Evidence

- **Total Qualifying Direct Interactions:** 5,316

| Interaction Type | Count | Description |
|---|---|---|
| `issue_comment` | 1,643 | Issue discussion comment directed to issue author |
| `pr_review` | 1,525 | Code reviewer submitted review to PR author |
| `pr_review_comment` | 777 | Inline code comment directed to PR author |
| `member_added` | 633 | Organization/Repo maintainer added developer as collaborator |
| `pr_merge` | 617 | Repository maintainer merged developer's PR |
| `pr_assignee` | 121 | PR creator assigned task to developer |


## 4. Missing Information & GitHub API Enrichment Requirements

### Unresolved Identities in GitHub Archive

- `PushEvent_commit_author_lacks_login`: 71,405 occurrences


### Key Limitations & API Enrichment Strategy

1. **Commit Author Login Resolution (`PushEvent`):**
   GitHub Archive's `PushEvent` payload provides git author `email` and `name`, but **does not resolve them to GitHub usernames (`login`)**. To prevent attributing commits to unverified strings, developer edges must be enriched using GitHub REST API commit lookups `/repos/{owner}/{repo}/commits/{sha}`.

2. **Historical Review Threads:**
   Single hourly archives only capture reviews happening in that 60-minute window. To obtain all collaborators on a repo, REST API `/repos/{owner}/{repo}/pulls?state=all` and `/repos/{owner}/{repo}/contributors` can be leveraged.

3. **Ego Network Sparsity & Seed User Activity:**

   In the sampled archive period, the configured seed user `varshavkumar12345` had **0 events**.

   > **Evaluation Finding:** Seed user `varshavkumar12345` had no public activity in this specific sample window. To build a meaningful collaboration network for `varshavkumar12345`, our pipeline utilizes the dual-source strategy: querying GitHub REST API for `varshavkumar12345`'s public repositories, contributors, stargazers, and forks, and intersecting with GitHub Archive data. Alternatively, the user can configure an active public developer (e.g. `torvalds`, `tiangolo`, `octocat`) or expand the collection window.
