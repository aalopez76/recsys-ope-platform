"""
Train RecBole baselines and export results.

Supports: Pop, BPR, NeuMF, LightGCN.
Requires: prepare_recbole_dataset to be run first.
Output: Markdown report and CSV table in reports/tables.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from recbole.quick_start import run_recbole

# Ensure we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.recsys.prepare_recbole_dataset import prepare_obd_sample_dataset  # noqa: E402

MODELS = ["Pop", "BPR", "NeuMF", "LightGCN"]


def run_baseline(model_name: str, config_file: str) -> dict[str, Any]:
    """Run a single RecBole model and return test metrics."""
    print(f"\n{'='*40}\nRunning {model_name}...\n{'='*40}")

    # RecBole's run_recbole takes kwargs to override config
    # We pass model=model_name
    # Note: run_recbole returns: (test_result, valid_result)
    try:
        result = run_recbole(
            model=model_name,
            dataset="obd_sample",
            config_file_list=[config_file],
            config_dict={
                "filter_inter_by_user_or_item": False,
                "user_inter_num_interval": "[0,inf)",
                "item_inter_num_interval": "[0,inf)",
                "val_interval": 0,
            },
        )
    except Exception:
        traceback.print_exc()
        return {}

    # result is (test_result_dict, valid_result_dict) usually
    if isinstance(result, tuple):
        test_res = result[0]
    else:
        # Maybe it's just one dict?
        print(f"Warning: Unexpected result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else result}")
        if isinstance(result, dict):
            # Check for nested test_result (typical in some RecBole versions)
            if "test_result" in result:
                test_res = result["test_result"]
            else:
                test_res = result
        else:
            return {}

    # metrics are like 'recall@10', etc.
    return test_res


def main():
    parser = argparse.ArgumentParser(description="Train RecBole Baselines")
    parser.add_argument("--models", type=str, default=",".join(MODELS), help="Comma-separated models")
    parser.add_argument("--config", type=str, default="configs/recbole.yaml", help="Path to config file")
    parser.add_argument("--sample-dir", type=str, default="data/sample", help="Input sample dir")
    parser.add_argument("--out-dir", type=str, default="reports/tables", help="Output directory for tables")
    parser.add_argument("--plots-dir", type=str, default="reports/plots", help="Output directory for plots")

    args = parser.parse_args()

    # 1. Prepare Dataset (Standard Split)
    dataset_name = "obd_sample"
    # We implicitly assume data/recbole_dataset is where config points to.
    # Config has data_path: data/recbole_dataset/
    # So we must prepare it there.

    # Parse config to find data_path? Or just hardcode logic since we control both?
    # Let's    # Parse config to find data_path? Or just hardcode logic since we control both?
    # We'll use the prepare function with default out-dir
    print(f"Preparing dataset from {args.sample_dir}...")
    prepare_obd_sample_dataset(args.sample_dir, "data/recbole_dataset", dataset_name)

    # 2. Train Loop
    models_to_run = args.models.split(",")
    results = []

    import time

    for model in models_to_run:
        start_time = time.time()
        res = run_baseline(model, args.config)
        end_time = time.time()

        if res:
            res["model"] = model
            res["train_time_sec"] = round(end_time - start_time, 2)
            results.append(res)

    if not results:
        print("No models finished successfully.")
        return

    # 3. Consolidate & Export
    df = pd.DataFrame(results)

    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    print(f"Available metrics: {df.columns.tolist()}")

    # Select columns
    # Select columns
    cols = ["model", "recall@10", "ndcg@10", "mrr@10", "recall@20", "ndcg@20", "train_time_sec"]
    # RecBole keys might be lowercase
    final_cols = [c for c in cols if c in df.columns]

    df_out = df[final_cols].copy()

    # Rename for pretty output
    df_out.columns = [c.upper() if c != "model" else "Model" for c in df_out.columns]

    # Ensure directories
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = Path(args.out_dir) / "recbole_baselines_sample.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved table to {csv_path}")

    # Markdown
    md_path = Path(args.out_dir) / "recbole_baselines_sample.md"
    with open(md_path, "w") as f:
        f.write("# RecBole Baselines (Sample Dataset)\n\n")
        f.write(df_out.to_markdown(index=False))
    print(f"Saved report to {md_path}")

    # 4. Plot NDCG@10
    if "ndcg@10" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df["model"], df["ndcg@10"], color="skyblue")
        plt.title("NDCG@10 by Model (Sample)")
        plt.xlabel("Model")
        plt.ylabel("NDCG@10")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plot_path = Path(args.plots_dir) / "recbole_ndcg_at10.png"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
