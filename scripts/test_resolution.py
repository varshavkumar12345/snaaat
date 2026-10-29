import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.graph_builder import resolve_starting_developer

edges_df = pd.read_csv('data/processed/edges.csv')
test_users = ['jreback', 'mroeschke', 'rgommers', 'thomasjpfan', 'wesm', 'unknown_user_test']
for user in test_users:
    res = resolve_starting_developer(edges_df, user)
    print(f"User @{user}: resolved={res['resolved']}, degree={res['degree']}, total_weight={res['total_weight']}")
    if not res['resolved']:
        print(f"  Message: {res['message']}")
        print(f"  Candidate suggestions: {res.get('candidates')}")
    else:
        print(f"  Top 3 collaborators: {res['top_collaborators'][:3]}")
