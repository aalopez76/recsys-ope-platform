import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import shutil

def upsample_data(input_path: str, output_path: str, n_rounds: int, seed: int = 42):
    """Upsample CSV data with replacement."""
    print(f"Reading from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original size: {len(df)}")
    
    # Resample
    print(f"Upsampling to {n_rounds} rows (seed={seed})...")
    df_large = df.sample(n=n_rounds, replace=True, random_state=seed)
    
    # Reset timestamp to be sequential 0..N-1?
    # OBD timestamp is usually just index or integer.
    # If we shuffle, we disturb temporal order.
    # But for RecSys training (collaborative filtering), exact time might not matter as much as sequence?
    # Actually, RecBole 'timestamp' column is often used for splitting 'time'.
    # If we sample with replacement, we might have duplicate timestamps if we keep original.
    # It's better to regenerate sequential timestamps/indices to mimic a longer log.
    
    # However, 'timestamp' column in raw OBP might not exist or be named differently.
    # Let's check columns. Usually: timestamp, item_id, position, click, pscore...
    # In OBP: timestamp, item_id, position, click, pscore, context...
    
    # We will just write it out. The build pipeline processes it.
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all other files from source directory to output directory
    source_dir = Path(input_path).parent
    print(f"Copying auxiliary files from {source_dir} to {output_dir}")
    for item in source_dir.iterdir():
        if item.is_file() and item.name != Path(input_path).name:
            shutil.copy2(item, output_dir / item.name)
            print(f"Copied {item.name}")
    
    print(f"Writing to {output_path}")
    df_large.to_csv(output_path, index=False)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/obd/random/all/all.csv")
    parser.add_argument("--output", default="data/raw_large/obd/random/all/all.csv")
    parser.add_argument("--n-rounds", type=int, default=300000)
    args = parser.parse_args()
    
    upsample_data(args.input, args.output, args.n_rounds)
    
if __name__ == "__main__":
    main()
