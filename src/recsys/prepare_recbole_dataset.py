"""
Prepare RecBole dataset from OBD sample artifacts.

This script copies the atomic user/item files and splits the interaction file
into train/valid/test based on the deterministic split manifest.
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_recbole_dir(
    source_dir: Path,
    out_dir: Path,
    dataset_name: str,
    active_users: set = None,
    active_items: set = None,
) -> None:
    """Create output directory and copy atomic files, filtering for active IDs."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing dataset '{dataset_name}' in {out_dir}")

    # 1. Copy obd.user -> dataset_name.user (Filtered)
    # Check for obd.user or obd_sample.user
    user_src = source_dir / "obd.user"
    if not user_src.exists():
        user_src = source_dir / "obd_sample.user"

    user_dst = out_dir / f"{dataset_name}.user"
    if not user_src.exists():
        raise FileNotFoundError(f"Missing artifact: {user_src}")

    if active_users is not None:
        logger.info(f"Filtering users to {len(active_users)} active IDs")
        df_user = pd.read_csv(user_src, sep="\t")

        # Identify user_id column (usually user_id or user_id:token)
        user_col = next((c for c in df_user.columns if c.startswith("user_id")), "user_id")

        if user_col not in df_user.columns:
            logger.warning(
                f"Could not find user_id column in {user_src.name}. Columns: {df_user.columns}"
            )
            # Fallback to copy if checking fails? Or raise error?
            # Raising error is better to catch issues early.
            raise KeyError(f"Missing user_id column in {user_src.name}")

        # Filter
        original_count = len(df_user)
        df_user = df_user[df_user[user_col].isin(active_users)]
        logger.info(f"Users filtered ({user_col}): {original_count} -> {len(df_user)}")
        df_user.to_csv(user_dst, sep="\t", index=False)
    else:
        shutil.copy(user_src, user_dst)
        logger.info(f"Copied user file: {user_dst.name}")

    # 2. Copy obd.item -> dataset_name.item (ALL ITEMS)
    item_src = source_dir / "obd.item"
    if not item_src.exists():
        item_src = source_dir / "obd_sample.item"

    item_dst = out_dir / f"{dataset_name}.item"
    if not item_src.exists():
        raise FileNotFoundError(f"Missing artifact: {item_src}")

    # We enforce copying ALL items to ensure negative sampling space exists
    # If we filter items to only those with clicks, we shrink the space too much.
    shutil.copy(item_src, item_dst)
    logger.info(f"Copied item file (full): {item_dst.name}")


def create_splits(source_dir: Path, out_dir: Path, dataset_name: str) -> tuple[set, set]:
    """Read interaction file and split manifest, write train/valid/test .inter files. Returns active users/items."""
    # Try generic names first, then sample names
    inter_src = source_dir / "obd_time.inter"
    if not inter_src.exists():
        inter_src = source_dir / "obd_time_sample.inter"

    manifest_src = source_dir / "split_manifest.npz"
    if not manifest_src.exists():
        manifest_src = source_dir / "split_manifest_sample.npz"

    metadata_src = source_dir / "metadata.json"
    if not metadata_src.exists():
        metadata_src = source_dir / "metadata_sample.json"

    if not inter_src.exists():
        raise FileNotFoundError(f"Missing artifact: {inter_src}")
    if not manifest_src.exists():
        raise FileNotFoundError(f"Missing artifact: {manifest_src}")

    # Load interactions efficiently (just reading to split)
    # Note: RecBole atomic files usually have a header. we keep it.
    logger.info(f"Reading interactions from {inter_src}")

    # We use csv engine for tab separated
    df = pd.read_csv(inter_src, sep="\t")

    # Identify columns
    user_col = next((c for c in df.columns if c.startswith("user_id")), "user_id")
    item_col = next((c for c in df.columns if c.startswith("item_id")), "item_id")

    if user_col not in df.columns or item_col not in df.columns:
        raise KeyError(
            f"Missing user/item columns in {inter_src.name}. Vars: {user_col}, {item_col}"
        )

    # FIX: Check for rating column (with type suffix) if click is missing
    # Pandas reads 'rating:float' as column name.
    next((c for c in df.columns if c.startswith("rating")), None)

    # if "click" not in df.columns and rating_col:
    #     logger.info(f"Renaming '{rating_col}' to 'click' for compatibility.")
    #     df = df.rename(columns={rating_col: "click"})

    # Load manifest
    logger.info(f"Loading split manifest from {manifest_src}")
    manifest = np.load(manifest_src)
    train_idx = manifest["train_idx"]
    val_idx = manifest["val_idx"]
    test_idx = manifest["test_idx"]

    n_rounds = len(df)
    total_split = len(train_idx) + len(val_idx) + len(test_idx)

    if n_rounds != total_split:
        raise ValueError(
            f"Mismatch: inter file has {n_rounds} rows, manifest has {total_split} indices"
        )

    # Split based on original indices
    # df is full original dataframe here
    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # FILTER: RecBole implicit feedback models (Pop, BPR, LightGCN) treat all rows as positive.
    # We must filter for positive feedback (click=1 or rating=1) only.
    if "click" in df.columns:
        logger.info("Filtering splits for click=1 (Strict OPE Alignment)...")
        train_df = train_df[train_df["click"] == 1]
        valid_df = valid_df[valid_df["click"] == 1]
        test_df = test_df[test_df["click"] == 1]
        logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    else:
        logger.warning("No 'click' column found in splits. Implicit feedback training on all rows!")

    # Enforce Cold-Start constraint: Valid/Test users must be in Train
    train_users = set(train_df[user_col].unique())

    # Strict Warm-Start
    valid_df = valid_df[valid_df[user_col].isin(train_users)]
    test_df = test_df[test_df[user_col].isin(train_users)]

    # Check if we emptied the sets (Should not happen with large sample)
    if len(valid_df) == 0:
        logger.warning("Warm-start filtering emptied Validation set! This is critical for OPE.")
    if len(test_df) == 0:
        logger.warning("Warm-start filtering emptied Test set! This is critical for OPE.")

    logger.info(f"Filtered Valid (Cold Start): {len(valid_df)}")
    logger.info(f"Filtered Test (Cold Start): {len(test_df)}")

    # New Fix: Prevent "Interacted with ALL items" error
    # We must remove users who interacted with ALL items in the training set?
    # Or in the global set? RecBole sampler works on dataset level.
    # If a user in TRAIN set has interacted with all items available in TRAIN set (or global item set).

    # Let's count interactions per user in Train
    train_inter_counts = train_df[user_col].value_counts()

    # Get total item count (active items from full df? or atomic item file?)
    # We are using atomic item file (obd_sample.item) which has 80 items.
    # But effectively, RecBole only knows about items in the interaction file + atomic file.
    # If atomic file provided, n_items = 80.

    # We assume 80 items. If user has >= 80 interactions, DROP THEM.
    # Ideally we check against actual unique items in dataset.
    all_known_items = set(pd.concat([train_df, valid_df, test_df])[item_col].unique())
    len(all_known_items)  # Or 80 if we load atomic file.
    # To be safe, filter users with >= n_items_global interactions. (Or even n_items_global - 1 to be safe for sampling)

    # Actually, RecBole sampler needs at least 1 negative. So count < n_items.
    # If we load atomic file, n_items is fixed (80).
    # Let's use a safe threshold.

    # We'll use 80 as hard limit (or detect from item file if possible, but 80 is known).
    # Since we don't read item file here easily (we could), we use unique items in interactions as lower bound.
    n_items_visible = len(all_known_items)

    oversaturated_users = train_inter_counts[train_inter_counts >= n_items_visible].index
    if len(oversaturated_users) > 0:
        logger.warning(
            f"Dropping {len(oversaturated_users)} users with 100% interaction density (causes Sampler crash)."
        )
        train_df = train_df[~train_df[user_col].isin(oversaturated_users)]
        # Also remove from valid/test to be clean
        valid_df = valid_df[~valid_df[user_col].isin(oversaturated_users)]
        test_df = test_df[~test_df[user_col].isin(oversaturated_users)]

    # Update active sets
    active_df = pd.concat([train_df, valid_df, test_df])
    active_users = set(active_df[user_col].unique())
    active_items = set(active_df[item_col].unique())

    # Write files
    train_path = out_dir / f"{dataset_name}.train.inter"
    valid_path = out_dir / f"{dataset_name}.valid.inter"
    test_path = out_dir / f"{dataset_name}.test.inter"

    logger.info(f"Writing train split: {len(train_df)} rows -> {train_path.name}")
    train_df.to_csv(train_path, sep="\t", index=False)

    logger.info(f"Writing valid split: {len(valid_df)} rows -> {valid_path.name}")
    valid_df.to_csv(valid_path, sep="\t", index=False)

    logger.info(f"Writing test split: {len(test_df)} rows -> {test_path.name}")
    test_df.to_csv(test_path, sep="\t", index=False)

    # Also save MERGED file for RS (Random Split) support if benchmark is disabled
    merged_path = out_dir / f"{dataset_name}.inter"
    logger.info(f"Writing merged file for RS: {len(active_df)} rows -> {merged_path.name}")
    active_df.to_csv(merged_path, sep="\t", index=False)

    return active_users, active_items


def prepare_obd_sample_dataset(sample_dir: str, out_dir: str, dataset_name: str = "obd_sample"):
    """Main execution flow."""
    src = Path(sample_dir)
    dst = Path(out_dir) / dataset_name  # RecBole expects dataset_name/dataset_name.xyz

    # We need to create dst first
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Create splits first to get active users/items
    active_users, active_items = create_splits(src, dst, dataset_name)

    # 2. Setup atomic files with filtering
    setup_recbole_dir(src, dst, dataset_name, active_users, active_items)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RecBole dataset from OBD sample")
    parser.add_argument(
        "--sample-dir", type=str, default="data/sample", help="Path to sample artifacts"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/recbole_dataset", help="Output root directory"
    )
    parser.add_argument("--dataset-name", type=str, default="obd_sample", help="Dataset name")

    args = parser.parse_args()

    prepare_obd_sample_dataset(args.sample_dir, args.out_dir, args.dataset_name)
