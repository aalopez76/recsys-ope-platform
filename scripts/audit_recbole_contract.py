import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def audit_contract():
    base_dir = Path("data/recbole_dataset/obd_sample")
    sample_dir = Path("data/recbole_atomic_large")
    
    print("-" * 50)
    print("## 2. Dataset Contract (Post-Filtering)")
    print("-" * 50)
    
    # 1. Original
    meta_path = sample_dir / "metadata_sample.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n_rounds = meta.get("n_rounds", 0)
        n_items_orig = meta.get("n_actions", 0)
    else:
        n_rounds = 0
        n_items_orig = 0
        
    print(f"**Original Source**: `data/sample`")
    print(f"- n_rounds: {n_rounds}")
    print(f"- n_items (catalog): {n_items_orig}")
    
    # 2. Raw Interactions (for click=0 counting)
    inter_path = sample_dir / "obd_time_sample.inter"
    # 2. Raw Interactions (for click=0 counting)
    inter_path = sample_dir / "obd_time_sample.inter"
    if inter_path.exists():
        df_raw = pd.read_csv(inter_path, sep="\t")
        n_raw = len(df_raw)
        
        # Check for click or rating (which is click)
        if "click" in df_raw.columns:
            n_clicks = df_raw["click"].sum()
        elif "rating:float" in df_raw.columns:
             # rating is 1.0 or 0.0
             n_clicks = (df_raw["rating:float"] == 1.0).sum()
        elif "rating" in df_raw.columns:
             n_clicks = (df_raw["rating"] == 1).sum()
        else:
             n_clicks = 0
             
        n_removed_clicks = n_raw - n_clicks
        print(f"- n_interactions (raw): {n_raw}")
        print(f"- Removed (click=0): {n_removed_clicks} ({n_removed_clicks/n_raw:.1%})")
        print(f"- Retained (click=1): {n_clicks}")
    else:
        print("Error: Missing obd_time_sample.inter")

    # 3. Final Splits
    modes = ["train", "valid", "test"]
    split_data = {}
    users_final = set()
    items_final = set()
    
    # Atomic item file
    item_atomic_path = base_dir / "obd_sample.item"
    if item_atomic_path.exists():
         df_items = pd.read_csv(item_atomic_path, sep="\t")
         n_items_final = len(df_items)
    else:
         n_items_final = 0

    print(f"\n**Final Dataset** (Warm-Start & Density Filtered):")
    print(f"- n_items (final): {n_items_final} (Should be 80)")
    
    for mode in modes:
        fpath = base_dir / f"obd_sample.{mode}.inter"
        if fpath.exists():
            df = pd.read_csv(fpath, sep="\t")
            n = len(df)
            u = df["user_id:token"].nunique()
            i = df["item_id:token"].nunique()
            users_final.update(df["user_id:token"].unique())
            items_final.update(df["item_id:token"].unique())
            split_data[mode] = {"n": n, "u": u, "i": i}
        else:
            split_data[mode] = {"n": 0, "u": 0, "i": 0}

    # Table
    print("\n| Split | Interactions | Users | Items |")
    print("|---|---|---|---|")
    for mode in modes:
        d = split_data[mode]
        print(f"| {mode.capitalize()} | {d['n']} | {d['u']} | {d['i']} |")
        
    print(f"\n- **Total Interactions**: {sum(d['n'] for d in split_data.values())}")
    print(f"- **Unique Users (Global)**: {len(users_final)}")
    print(f"- **Unique Items (Global)**: {len(items_final)}")
    
    # 4. Removed Users Calculation
    # Users with at least 1 click in RAW:
    if inter_path.exists():
        if "click" in df_raw.columns:
             users_raw_click = df_raw[df_raw["click"] == 1]["user_id:token"].nunique()
        elif "rating:float" in df_raw.columns:
             users_raw_click = df_raw[df_raw["rating:float"] == 1.0]["user_id:token"].nunique()
        elif "rating" in df_raw.columns:
             users_raw_click = df_raw[df_raw["rating"] == 1]["user_id:token"].nunique()
        else:
             users_raw_click = 0
        removed_users = users_raw_click - len(users_final)
        print(f"\n### User Filtering Analysis")
        print(f"- Users with clicks (Raw): {users_raw_click}")
        print(f"- Users in Final Train/Val/Test: {len(users_final)}")
        print(f"- Users Removed (Density/Cold-Start): {removed_users}")

if __name__ == "__main__":
    audit_contract()
