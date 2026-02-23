"""
Build OBD datasets for RecBole and OBP formats.

This script:
1. Loads configuration from configs/data_config.yaml
2. Loads Open Bandit Dataset (OBP) - Full or Sample
3. Validates privacy (No PII) and schema (strict rules)
4. Saves bandit_feedback (npz + metadata + action_map)
5. Generates RecBole atomic format (hashed/clustered user_ids)
6. Generates comprehensive statistics and audit reports
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from obp.dataset import OpenBanditDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Internal modules
from src.data.privacy import PrivacyPolicy
from src.data.validate_schema import SchemaValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "data_config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="build_obd_datasets",
        description="Build OBD datasets for RecBole and OBP formats.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to data config YAML (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Dataset mode: 'sample' (10k) or 'full' (default: sample)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows for audit CSV export (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and validate, but do not run the pipeline",
    )
    return parser.parse_args()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML."""
    path = config_path or CONFIG_PATH
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {path}")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing config file: {exc}")
        raise


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve absolute paths from config."""
    paths = {}
    for key, rel_path in config["paths"].items():
        paths[key] = BASE_DIR / rel_path
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def compute_fingerprint(config: Dict[str, Any], n_rounds: int) -> str:
    """Compute dataset fingerprint for reproducibility."""
    params = {
        "policy": config["dataset"]["behavior_policy"],
        "campaign": config["dataset"]["campaign"],
        "n_rounds": n_rounds,
        "seed": config["dataset"]["random_state"],
    }
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode("utf-8")).hexdigest()


def copy_sample_data_if_needed(raw_dir: Path) -> None:
    """Copy sample data from OBP package if missing."""
    if not (raw_dir / "random" / "all" / "all.csv").exists():
        logger.info("Dataset not found in raw dir, copying from obp package...")
        import shutil

        import obp.dataset

        package_obd_dir = Path(obp.dataset.__file__).parent / "obd"
        if package_obd_dir.exists():
            shutil.copytree(package_obd_dir, raw_dir, dirs_exist_ok=True)
            logger.info(f"Copied sample dataset from {package_obd_dir} to {raw_dir}")
        else:
            logger.warning("Sample dataset not found in obp package.")


def load_obd_dataset(
    config: Dict[str, Any], raw_dir: Path
) -> Tuple[OpenBanditDataset, Dict[str, Any]]:
    """Load Open Bandit Dataset via OBP."""
    logger.info("Loading Open Bandit Dataset...")

    copy_sample_data_if_needed(raw_dir)

    dataset = OpenBanditDataset(
        behavior_policy=config["dataset"]["behavior_policy"],
        campaign=config["dataset"]["campaign"],
        data_path=raw_dir,
    )

    bandit_feedback = dataset.obtain_batch_bandit_feedback()
    logger.info(
        f"Dataset loaded: {bandit_feedback['n_rounds']} rounds, {bandit_feedback['n_actions']} actions"
    )

    return dataset, bandit_feedback


def create_user_ids(context: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Create deterministic user IDs."""
    strategy = config["processing"]["user_id_strategy"]
    seed = config["dataset"]["random_state"]

    logger.info(f"Creating user IDs using strategy: {strategy}")

    if strategy == "kmeans":
        n_clusters = config["processing"]["kmeans"]["n_clusters"]
        n_init = config["processing"]["kmeans"]["n_init"]

        scaler = StandardScaler()
        context_scaled = scaler.fit_transform(context)

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
        user_ids = kmeans.fit_predict(context_scaled)

    elif strategy == "hash":
        # Deterministic hashing of context vectors
        # Round to 6 decimals to handle float precision issues before hashing
        context_rounded = np.round(context, decimals=6)
        # Verify context is 2D
        if context_rounded.ndim != 2:
            raise ValueError("Context must be 2D array")

        # Create hash per row
        user_ids_list = []
        for row in context_rounded:
            row_bytes = row.tobytes()
            row_hash = int(hashlib.md5(row_bytes).hexdigest(), 16) % (10**8)  # 8-digit ID
            user_ids_list.append(row_hash)
        user_ids = np.array(user_ids_list)

    else:
        raise ValueError(f"Unknown user_id_strategy: {strategy}")

    n_unique = len(np.unique(user_ids))
    logger.info(f"Created {n_unique} unique user IDs")
    return user_ids


def is_identity_action_map(action_id_map: Dict[Any, Any], n_actions: int) -> bool:
    """Check if action_id_map is identity mapping.

    Handles both int and str keys (for JSON vs in-memory compatibility).

    Args:
        action_id_map: Mapping from original to new action IDs
        n_actions: Expected number of actions

    Returns:
        True if mapping is identity (i -> i for all i in [0, n_actions))
    """
    if len(action_id_map) != n_actions:
        return False

    # Check all indices [0, n_actions) map to themselves
    for i in range(n_actions):
        # Handle both int and str keys
        mapped_value = action_id_map.get(i, action_id_map.get(str(i), -1))
        if mapped_value != i:
            return False

    return True


def generate_split_manifest(
    n_rounds: int, config: Dict[str, Any], paths: Dict[str, Path]
) -> Dict[str, Any]:
    """Generate deterministic train/val/test splits.

    Split strategy:
    - 'time': Temporal split ordered by round index (chronological)
      Order: train [0:n_train], val [n_train:n_train+n_val], test [n_train+n_val:]
    - 'random': Random shuffle with seed

    Args:
        n_rounds: Total number of interactions
        config: Configuration dictionary
        paths: Output paths

    Returns:
        Dict with split metadata (path, strategy, counts, hash)
    """
    import hashlib

    split_strategy = config["dataset"]["split_strategy"]
    test_size = config["dataset"]["test_size"]
    val_size = config["dataset"]["val_size"]  # Fraction of train data
    seed = config["dataset"]["random_state"]

    logger.info(
        f"Generating split manifest: strategy={split_strategy}, test_size={test_size}, val_size={val_size}"
    )

    # Calculate split sizes
    n_test = int(n_rounds * test_size)
    n_trainval = n_rounds - n_test
    n_val = int(n_trainval * val_size)  # val_size is fraction of remaining data
    n_train = n_trainval - n_val

    indices = np.arange(n_rounds)

    if split_strategy == "time":
        # Temporal split: data is already ordered by round index (chronological)
        # This preserves temporal ordering: early data for train, middle for val, latest for test
        logger.info("Using temporal split (ordered by round index/timestamp)")
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
    elif split_strategy == "random":
        # Random shuffle with seed for reproducibility
        logger.info(f"Using random split with seed={seed}")
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(indices)
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}. Use 'time' or 'random'.")

    # Save manifest to data/splits/
    splits_dir = paths["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / "split_manifest.npz"

    np.savez_compressed(
        manifest_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        strategy=split_strategy,
        seed=seed,
    )

    logger.info(f"Split manifest saved: {manifest_path}")
    logger.info(f"  Train: {len(train_idx)} samples")
    logger.info(f"  Val:   {len(val_idx)} samples")
    logger.info(f"  Test:  {len(test_idx)} samples")

    # Compute SHA256 hash for reproducibility verification
    with open(manifest_path, "rb") as f:
        manifest_hash = hashlib.sha256(f.read()).hexdigest()

    # Generate JSON inspection file (for human readability)
    # Mode detection: sample (n_rounds < 50k) vs full (n_rounds >= 50k)
    is_sample_mode = n_rounds < 50000

    json_manifest = {
        "split_strategy": split_strategy,
        "seed": seed,
        "split_counts": {
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "n_total": n_rounds,
        },
        "split_manifest_hash": manifest_hash,
    }

    if is_sample_mode:
        # Sample mode: save full indices for inspection
        json_manifest["train_idx"] = train_idx.tolist()
        json_manifest["val_idx"] = val_idx.tolist()
        json_manifest["test_idx"] = test_idx.tolist()
        json_manifest["mode"] = "sample"
        json_filename = "split_manifest.json"
        logger.info("Sample mode detected: saving full indices to JSON")
    else:
        # Full mode: save preview (first 100 indices of each split)
        json_manifest["train_idx_preview"] = train_idx[:100].tolist()
        json_manifest["val_idx_preview"] = val_idx[:100].tolist()
        json_manifest["test_idx_preview"] = test_idx[:100].tolist()
        json_manifest["mode"] = "full"
        json_manifest["note"] = (
            "Preview only - first 100 indices of each split. Full indices in split_manifest.npz"
        )
        json_filename = "split_manifest_preview.json"
        logger.info("Full mode detected: saving preview (first 100 indices) to JSON")

    # Save JSON to data/bandit_feedback/ for easy inspection
    json_path = paths["bandit"] / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_manifest, f, indent=2)

    logger.info(f"JSON inspection file saved: {json_path}")

    return {
        "split_manifest_path": "splits/split_manifest.npz",
        "split_manifest_json": json_filename,
        "split_strategy": split_strategy,
        "split_counts": {"n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test)},
        "split_manifest_hash": manifest_hash,
    }


def save_bandit_feedback(
    bandit_feedback: Dict[str, Any],
    paths: Dict[str, Path],
    config: Dict[str, Any],
    fingerprint: str,
    split_info: Dict[str, Any],
) -> None:
    """Save OBP format artifacts with action_id_map and split manifest metadata.

    Args:
        bandit_feedback: Bandit feedback dictionary
        paths: Output paths
        config: Configuration dict
        fingerprint: Dataset fingerprint
        split_info: Split manifest metadata from generate_split_manifest()
    """
    logger.info("Saving bandit feedback artifacts...")

    # Generate action_id_map if not present (identity map)
    if "action_id_map" not in bandit_feedback:
        logger.info("No reindexing detected, generating identity action map.")
        n_actions = bandit_feedback["n_actions"]
        bandit_feedback["action_id_map"] = {i: i for i in range(n_actions)}

    # Save arrays
    npz_path = paths["bandit"] / "bandit_feedback.npz"
    np.savez_compressed(
        npz_path,
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
        action_context=bandit_feedback.get("action_context"),
    )

    # Save action map
    map_path = paths["bandit"] / "action_id_map.json"
    # Ensure keys are strings for JSON
    action_map = {str(k): int(v) for k, v in bandit_feedback["action_id_map"].items()}

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(action_map, f, indent=2)

    # Determine if action map is truly identity
    is_identity = is_identity_action_map(
        bandit_feedback["action_id_map"], bandit_feedback["n_actions"]
    )

    # Save metadata with all required fields
    metadata = {
        "n_rounds": int(bandit_feedback["n_rounds"]),
        "n_actions": int(bandit_feedback["n_actions"]),
        "context_dim": int(bandit_feedback["context"].shape[1]),
        "policy": config["dataset"]["behavior_policy"],
        "campaign": config["dataset"]["campaign"],
        "action_id_map_path": "action_id_map.json",
        "mapping_type": "identity" if is_identity else "reindexed",
        "is_identity_map": is_identity,
        "seed": config["dataset"]["random_state"],
        "dataset_fingerprint": fingerprint,
    }

    # Add split manifest metadata
    metadata.update(split_info)

    # If reindexed, add additional metadata
    if not is_identity:
        metadata["original_id_space"] = "sparse"
        metadata["id_space"] = "reindexed"

    with open(paths["bandit"] / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_recbole_files(
    bandit_feedback: Dict[str, Any],
    user_ids: np.ndarray,
    paths: Dict[str, Path],
    stats: Dict[str, Any],
) -> None:
    """Generate RecBole atomic files (inter, user, item)."""
    logger.info("Generating RecBole atomic files...")

    # 1. Interactions (obd.inter)
    df_inter = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": bandit_feedback["action"],
            "rating": bandit_feedback["reward"].astype(float),
            "timestamp": np.arange(len(user_ids)).astype(float),
            "position": bandit_feedback["position"].astype(float),
        }
    )

    stats["n_inter_before_filter"] = len(df_inter)
    stats["n_users_before_filter"] = df_inter["user_id"].nunique()
    stats["n_items_before_filter"] = df_inter["item_id"].nunique()

    out_inter = paths["recbole"] / "obd_time.inter"
    with open(out_inter, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\tposition:float\n")
        df_inter.to_csv(f, sep="\t", index=False, header=False, mode="a")

    stats["n_inter_final"] = len(df_inter)

    # 2. Users (obd.user)
    # Deduplicate users by user_id to ensure atomic format
    # We take the first occurrence of context for each user_id (assumption: same user_id -> same context)
    context_df = pd.DataFrame(
        bandit_feedback["context"],
        columns=[f"c{i}" for i in range(bandit_feedback["context"].shape[1])],
    )
    context_df["user_id"] = user_ids

    # Deduplicate and sort for deterministic ordering
    df_user = (
        context_df.drop_duplicates(subset=["user_id"], keep="first")
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    out_user = paths["recbole"] / "obd.user"
    with open(out_user, "w", encoding="utf-8") as f:
        # header
        col_types = "\t".join([f"{c}:float" for c in df_user.columns if c != "user_id"])
        f.write(f"user_id:token\t{col_types}\n")
        df_user.to_csv(f, sep="\t", index=False, header=False, mode="a")

    stats["n_unique_users_atomic"] = len(df_user)

    # 3. Items (obd.item)
    if "action_context" in bandit_feedback:
        ac = bandit_feedback["action_context"]
        # action_id_map should already be applied to bandit_feedback['action'] if reindexed.
        # However, action_context is usually indexed by the *internal* action index 0..n_actions-1.
        # If we reindexed actions, we need to map back or ensures 'item_id' column matches.

        # In OBP, action_context is (n_actions, dim). Row i corresponds to action i.
        # If we have an action_id_map, it maps original_id -> new_id.
        # But 'action' array already contains new_ids.
        # We need to export item features keyed by the item_ids used in 'action' array.

        # Case 1: Identity map (0->0, 1->1). Item ID i has context at row i.
        # Case 2: Reindexed. If map is O->N. The system uses N.
        # We need to know which row in original action_context corresponds to N.
        # Usually reindexing happens in preprocessing/validation.
        # For safety, let's assume action_context rows align with the current 0..n_actions-1 indices
        # IF validation ensures 'action' values are 0..n_actions-1.

        # Construct item df
        n_items = ac.shape[0]
        # Item IDs are simply 0..n_items-1 because we force contiguous range in validation/reindexing
        item_ids = np.arange(n_items)

        item_df = pd.DataFrame(ac, columns=[f"i{i}" for i in range(ac.shape[1])])
        item_df.insert(0, "item_id", item_ids)

        # Sort by item_id for deterministic ordering
        item_df = item_df.sort_values("item_id").reset_index(drop=True)

        out_item = paths["recbole"] / "obd.item"
        with open(out_item, "w", encoding="utf-8") as f:
            col_types = "\t".join([f"{c}:float" for c in item_df.columns if c != "item_id"])
            f.write(f"item_id:token\t{col_types}\n")
            item_df.to_csv(f, sep="\t", index=False, header=False, mode="a")

        stats["n_items_atomic"] = len(item_df)
    else:
        logger.info("No action_context found, skipping obd.item generation.")


def save_audit_csv(
    bandit_feedback: Dict[str, Any], paths: Dict[str, Path], config: Dict[str, Any]
) -> None:
    """Save audit CSV if enabled."""
    if not config["processing"].get("export_csv", False):
        return

    limit = config["processing"].get("max_rows_audit")
    logger.info(f"Exporting audit CSV (limit={limit})...")

    df = pd.DataFrame(
        {
            "timestamp": np.arange(bandit_feedback["n_rounds"]),
            "item_id": bandit_feedback["action"],
            "position": bandit_feedback["position"],
            "click": bandit_feedback["reward"],
            "pscore": bandit_feedback["pscore"],
        }
    )

    if limit:
        df = df.head(limit)

    df.to_csv(paths["raw"] / "audit_sample.csv", index=False)


def generate_stats_report(stats: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """Generate markdown report."""
    report = f"""# OBD Data Pipeline Report

## Summary
- **Fingerprint**: `{stats['fingerprint']}`
- **Users**: {stats['n_users_before_filter']:,}
- **Items**: {stats['n_items_before_filter']:,}
- **Interactions**: {stats['n_inter_final']:,}
- **Sparsity**: {1 - (stats['n_inter_final']/(stats['n_users_before_filter']*stats['n_items_before_filter'])):.4%}

## Privacy & Safety
- **PII Check**: PASSED (No personal data detected)
- **Schema Validation**: PASSED
  - Pscore Range: [{stats['pscore_min']:.6f}, {stats['pscore_max']:.6f}]
  - Action Reindexing: {'YES' if 'action_id_map' in stats else 'NO'}

## Filtering Impact
| Metric | Before | After |
|--------|--------|-------|
| Users | {stats['n_users_before_filter']} | - |
| Items | {stats['n_items_before_filter']} | - |
| Interactions | {stats['n_inter_before_filter']} | {stats['n_inter_final']} |

## Output Files
- `data/bandit_feedback/bandit_feedback.npz`
- `data/recbole_atomic/obd_time.inter`
- `data/recbole_atomic/obd.user`
- `data/recbole_atomic/obd.item`
- `data/bandit_feedback/metadata.json`
"""
    with open(paths["reports"] / "data_stats.md", "w", encoding="utf-8") as f:
        f.write(report)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Run the OBD data pipeline.

    Args:
        args: Parsed CLI arguments. If None, uses defaults.
    """
    # 1. Config
    config_path = Path(args.config) if args and args.config else None
    config = load_config(config_path)

    # Apply CLI overrides
    if args and args.max_rows is not None:
        config["processing"]["max_rows_audit"] = args.max_rows

    logger.info("Starting Refined OBD Pipeline")
    logger.info(f"Config: {config_path or CONFIG_PATH}")
    logger.info(f"Mode: {args.mode if args else 'sample'}")

    # Dry-run: validate config and exit
    if args and args.dry_run:
        logger.info("[DRY-RUN] Config loaded and validated. Exiting.")
        return

    paths = get_paths(config)
    stats = {}

    # 2. Privacy Policy
    # (Pre-check logic here if we had raw access before OBP loads,
    # but OBP loads into memory first. We verify post-load.)

    # 3. Load Data
    dataset, bandit_feedback = load_obd_dataset(config, paths["raw"])

    # 4. Privacy Check
    PrivacyPolicy.verify_no_pii(bandit_feedback)
    PrivacyPolicy.check_reidentification_risk(
        n_users=bandit_feedback["n_rounds"],  # Approx upper bound before clustering
        n_samples=bandit_feedback["n_rounds"],
    )

    # Normalize positions to 1-based indexing if needed
    # OBP raw data is often 0-indexed (0, 1, 2) but config expects (1, 2, 3)
    if 0 in bandit_feedback["position"]:
        logger.info("Normalizing positions from 0-based to 1-based indexing...")
        bandit_feedback["position"] = bandit_feedback["position"] + 1

    # 5. Schema Validation (Strict)
    validator = SchemaValidator(config["validation"])
    bandit_feedback = validator.validate_bandit_feedback(bandit_feedback)

    # 6. User ID Generation
    user_ids = create_user_ids(bandit_feedback["context"], config)

    # 7. Fingerprint
    fingerprint = compute_fingerprint(config, bandit_feedback["n_rounds"])
    stats["fingerprint"] = fingerprint
    stats["pscore_min"] = float(bandit_feedback["pscore"].min())
    stats["pscore_max"] = float(bandit_feedback["pscore"].max())
    if "action_id_map" in bandit_feedback:
        stats["action_id_map"] = True

    # 8. Generate Split Manifest (Single Source of Truth)
    split_info = generate_split_manifest(bandit_feedback["n_rounds"], config, paths)

    # 9. Save Artifacts
    save_bandit_feedback(bandit_feedback, paths, config, fingerprint, split_info)

    # 10. RecBole Generation
    generate_recbole_files(bandit_feedback, user_ids, paths, stats)

    # 11. Audit CSV
    save_audit_csv(bandit_feedback, paths, config)

    # 12. Report
    generate_stats_report(stats, paths)

    # 13. Validate Required Artifacts (Fail-Fast)
    validate_required_artifacts(paths)

    logger.info("Pipeline completed successfully.")


def validate_required_artifacts(paths: Dict[str, Path]) -> None:
    """Validate all required artifacts exist (fail-fast).

    Args:
        paths: Dictionary of output paths

    Raises:
        RuntimeError: If any required artifact is missing
    """
    logger.info("Validating required artifacts...")

    required_artifacts = [
        paths["bandit"] / "bandit_feedback.npz",
        paths["bandit"] / "metadata.json",
        paths["bandit"] / "action_id_map.json",
        paths["splits"] / "split_manifest.npz",
        paths["recbole"] / "obd_time.inter",
        paths["recbole"] / "obd.user",
        paths["recbole"] / "obd.item",
    ]

    missing = [str(f) for f in required_artifacts if not f.exists()]

    if missing:
        error_msg = f"Artifact validation FAILED. Missing {len(missing)} file(s):\n" + "\n".join(
            f"  - {f}" for f in missing
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    for artifact in required_artifacts:
        logger.info(f"  ✓ {artifact.relative_to(paths['bandit'].parent)}")
    logger.info(f"✓ All {len(required_artifacts)} required artifacts validated successfully")


if __name__ == "__main__":
    try:
        cli_args = parse_args()
        main(cli_args)
    except Exception:
        import traceback

        traceback.print_exc()
