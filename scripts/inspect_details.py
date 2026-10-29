import json

with open('data/raw/action-schema.json', 'r', encoding='utf-8') as f:
    act = json.load(f)

defs = act.get("$defs", {})
for target in ["PullRequest", "Review", "Issue", "Comment", "PullRequestDetails", "ReviewDetails", "IssueDetails"]:
    if target in defs:
        print(f"=== {target} ===")
        print(json.dumps(defs[target], indent=2))

