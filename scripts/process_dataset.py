"""
Dataset extraction and processing script:
Reads NumFOCUS activities, extracts verifiable developer interactions, cleans and aggregates edges.
"""
import zipfile
import json
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ACTIVITIES_ZIP
from src.data_loader import extract_interactions_from_activity
from src.graph_builder import is_bot, clean_interactions, aggregate_edges

def run_extraction(max_records: int = 500000) -> None:
    zip_path = RAW_DATA_DIR / ACTIVITIES_ZIP
    print(f"Reading up to {max_records} activity records from {zip_path.name}...")
    t0 = time.time()

    interactions = []
    processed_count = 0
    activity_counts = {}

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('NumFocus_Jan22-Dec24_GH_Activities.jsonl') as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                rec = json.loads(line)
                act = rec.get("activity", "Unknown")
                activity_counts[act] = activity_counts.get(act, 0) + 1
                
                extracted = extract_interactions_from_activity(rec)
                for item in extracted:
                    if not is_bot(item["source"]) and not is_bot(item["target"]):
                        interactions.append(item)
                
                processed_count += 1
                if processed_count % 100000 == 0:
                    print(f"  Processed {processed_count:,} records, extracted {len(interactions):,} raw interactions...")

    elapsed = time.time() - t0
    print(f"Finished extraction: {processed_count:,} records processed in {elapsed:.2f}s.")
    print(f"Total raw interactions extracted: {len(interactions):,}")

    # Convert to DataFrame
    df_raw = pd.DataFrame(interactions)
    print("Cleaning interactions (removing bots, self-loops, duplicates)...")
    df_clean = clean_interactions(df_raw)
    print(f"Cleaned interactions: {len(df_clean):,}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_csv = PROCESSED_DATA_DIR / "cleaned_interactions.csv"
    df_clean.to_csv(clean_csv, index=False)
    print(f"Saved cleaned interactions to {clean_csv}")

    print("Aggregating interactions into weighted undirected edges...")
    df_edges = aggregate_edges(df_clean)
    print(f"Aggregated unique edges: {len(df_edges):,}")
    
    edges_csv = PROCESSED_DATA_DIR / "edges.csv"
    df_edges.to_csv(edges_csv, index=False)
    print(f"Saved aggregated edges to {edges_csv}")

    # Print summary of top developers by interaction frequency
    all_users = pd.concat([df_edges["source"], df_edges["target"]])
    user_counts = all_users.value_counts()
    print("\nTop 15 most connected developers in dataset:")
    for user, deg in user_counts.head(15).items():
        print(f"  @{user:<20} Degree: {deg}")

if __name__ == "__main__":
    run_extraction(max_records=500000)
