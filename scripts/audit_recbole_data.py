import pandas as pd
import numpy as np
import json
from pathlib import Path

def audit_recbole_data():
    base_dir = Path("data/recbole_dataset/obd_sample")
    sample_dir = Path("data/sample")
    
    print(f"## Dataset Contract (Post-Filtering Audit)")
    print(f"**Date**: 2026-02-18")
    
    # 1. Original Stats
    meta_path = sample_dir / "metadata_sample.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_rounds_orig = meta["n_rounds"]
    n_items_orig = meta["n_actions"]
    
    # 2. Check Raw Interactions (Impression Log)
    inter_path = sample_dir / "obd_time_sample.inter"
    df_orig = pd.read_csv(inter_path, sep="\t")
    n_users_orig = df_orig["user_id:token"].nunique()
    
    print(f"\n### Original (Metadata & Raw Data)")
    print(f"- n_rounds: {n_rounds_orig}")
    print(f"- n_users (from inter): {n_users_orig}")
    print(f"- n_items: {n_items_orig}")
    n_inter_orig = len(df_orig)
    n_clicks_orig = df_orig["click"].sum()
    n_impressions_removed = n_inter_orig - n_clicks_orig
    
    print(f"\n### Filtering Impact")
    print(f"- Original Interactions: {n_inter_orig}")
    print(f"- Positive Clicks (click=1): {n_clicks_orig}")
    print(f"- Removed (click=0): {n_impressions_removed} ({n_impressions_removed/n_inter_orig:.1%})")
    
    # 3. Final Split Stats
    modes = ["train", "valid", "test"]
    split_stats = {}
    users_union = set()
    items_union = set()
    total_inter = 0
    
    for mode in modes:
        path = base_dir / f"obd_sample.{mode}.inter"
        if not path.exists():
            print(f"Error: Missing {path}")
            continue
            
        df = pd.read_csv(path, sep="\t")
        n = len(df)
        u_unique = df["user_id:token"].nunique()
        i_unique = df["item_id:token"].nunique()
        
        split_stats[mode] = {"n": n, "u": u_unique, "i": i_unique}
        users_union.update(df["user_id:token"].unique())
        items_union.update(df["item_id:token"].unique())
        total_inter += n
        
    print(f"\n### Final Split Statistics (Warm-Start & Click-Only)")
    print(f"| Split | Interactions | Users | Items |")
    print(f"|---|---|---|---|")
    for mode in modes:
        s = split_stats[mode]
        print(f"| {mode.capitalize()} | {s['n']} | {s['u']} | {s['i']} |")
    
    print(f"\n- **Total Interactions**: {total_inter}")
    print(f"- **Unique Users (Union)**: {len(users_union)}")
    print(f"- **Unique Items (Union)**: {len(items_union)}")
    
    # 4. Users Removed (Dense / Cold)
    # n_clicks_orig is users with at least 1 click? No, interactions.
    # To find exactly how many users removed due to density/cold-start:
    # Users in original click=1 set
    users_with_clicks = df_orig[df_orig["click"]==1]["user_id:token"].nunique()
    users_removed = users_with_clicks - len(users_union)
    
    print(f"\n### User Filtering")
    print(f"- Users with >0 clicks (Original): {users_with_clicks}")
    print(f"- Users in Final Dataset: {len(users_union)}")
    print(f"- Users Removed (Dense/Cold-Start): {users_removed}")

if __name__ == "__main__":
    audit_recbole_data()
