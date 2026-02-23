
import pandas as pd
import hashlib
from pathlib import Path

def get_md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "MISSING"

def audit():
    base_dir = Path("data/recbole_dataset/obd_sample")
    print(f"Auditing {base_dir}...")
    
    files = ["train", "valid", "test"]
    for split in files:
        p = base_dir / f"obd_sample.{split}.inter"
        if not p.exists():
            print(f"{split}: MISSING")
            continue
            
        df = pd.read_csv(p, sep="\t")
        n_inter = len(df)
        n_users = df["user_id:token"].nunique() if "user_id:token" in df.columns else 0
        n_items = df["item_id:token"].nunique() if "item_id:token" in df.columns else 0
        md5 = get_md5(p)
        
        print(f"Split: {split.upper()}")
        print(f"  Interactions: {n_inter}")
        print(f"  Users: {n_users}")
        print(f"  Items: {n_items}")
        print(f"  MD5: {md5}")
        print("-" * 20)

if __name__ == "__main__":
    audit()
