"""
Reward Model for OBD Bandit Environment.

Trains a logistic regression (sklearn) / neural model on:
  X = concat(context[t], action_context[action[t]])  # shape (25,)
  y = reward[t]  # 0/1

Uses ONLY train_idx from split_manifest_sample.npz (SSOT).

Usage:
    python -m src.bandits.reward_model \\
        --bandit data/sample/bandit_feedback_sample.npz \\
        --splits data/sample/split_manifest_sample.npz \\
        --out-dir saved/reward_model \\
        --report reports/tables/reward_model_metrics.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_features(
    context: np.ndarray, action_context: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """Concatenate context and action_context for each round. Returns (n, 25)."""
    action_feats = action_context[actions]  # (n, 5)
    return np.concatenate([context, action_feats], axis=1)  # (n, 25)


def train_reward_model(
    bandit_path: str,
    splits_path: str,
    out_dir: str,
    report_path: str,
    plots_dir: str = "reports/plots",
    seed: int = 42,
) -> Pipeline:
    """Train click prediction model on train_idx, evaluate on val_idx."""
    # Load data
    d = np.load(bandit_path, allow_pickle=True)
    s = np.load(splits_path, allow_pickle=True)

    context = d["context"]
    action_context = d["action_context"]
    rewards = d["reward"].astype(float)
    actions = d["action"].astype(int)

    train_idx = s["train_idx"].astype(int)
    val_idx = s["val_idx"].astype(int)

    # Build features
    X_train = build_features(context[train_idx], action_context, actions[train_idx])
    y_train = rewards[train_idx]
    X_val = build_features(context[val_idx], action_context, actions[val_idx])
    y_val = rewards[val_idx]

    logger.info(f"Train: {X_train.shape}, pos_rate={y_train.mean():.4f}")
    logger.info(f"Val:   {X_val.shape},   pos_rate={y_val.mean():.4f}")

    # Train logistic regression pipeline (fast, interpretable, no GPU needed)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=500, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    # Metrics
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]

    auc_val = roc_auc_score(y_val, y_prob_val)
    auc_train = roc_auc_score(y_train, y_prob_train)
    ll_val = log_loss(y_val, y_prob_val)
    ll_train = log_loss(y_train, y_prob_train)
    # Baseline: constant prediction of mean
    baseline_ll = log_loss(y_val, np.full_like(y_val, y_train.mean()))
    baseline_auc = 0.5

    logger.info(f"Train AUC={auc_train:.4f} LogLoss={ll_train:.4f}")
    logger.info(f"Val   AUC={auc_val:.4f}  LogLoss={ll_val:.4f}")

    # Save model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(out_dir) / "reward_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    # Save metadata for inference
    meta_path = Path(out_dir) / "model_meta.npz"
    np.savez(meta_path, action_context=action_context)

    # Calibration plot
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    _plot_calibration(y_val, y_prob_val, plots_dir)

    # Report
    _write_report(
        report_path,
        auc_train,
        auc_val,
        ll_train,
        ll_val,
        baseline_ll,
        baseline_auc,
        len(train_idx),
        len(val_idx),
        y_train.mean(),
        y_val.mean(),
        str(model_path),
    )

    return model


def _plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label="Model", color="#2196F3", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Reward Model Calibration (Validation Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_p = Path(plots_dir) / "reward_model_calibration.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved calibration plot: {out_p}")


def _write_report(
    report_path: str,
    auc_train: float,
    auc_val: float,
    ll_train: float,
    ll_val: float,
    baseline_ll: float,
    baseline_auc: float,
    n_train: int,
    n_val: int,
    pos_train: float,
    pos_val: float,
    model_path: str,
) -> None:
    report = f"""# Reward Model Metrics

**Date**: 2026-02-22
**Model**: LogisticRegression (sklearn), C=1.0, balanced class_weight
**Features**: concat(context[25d], action_context[5d]) = 25d
**Training data**: `train_idx` only (SSOT: split_manifest_sample.npz)

## Dataset

| Split | Rounds | Click Rate |
|:------|-------:|-----------:|
| Train | {n_train:,} | {pos_train:.4f} |
| Val   | {n_val:,} | {pos_val:.4f} |

## Metrics

| Metric | Train | Val | Baseline (constant) |
|:-------|------:|----:|--------------------:|
| AUC-ROC | {auc_train:.4f} | {auc_val:.4f} | {baseline_auc:.4f} |
| Log-Loss | {ll_train:.4f} | {ll_val:.4f} | {baseline_ll:.4f} |

**AUC Lift over baseline**: {auc_val - baseline_auc:.4f}
**LogLoss reduction**: {baseline_ll - ll_val:.4f}

## Artifacts

- Model: `{model_path}`
- Calibration plot: `reports/plots/reward_model_calibration.png`

## Notes

The model is used to simulate the bandit environment in `obd_sim_bandit_env.py`.
Training is SOLELY on `train_idx` to prevent data leakage into policy evaluation.
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report, encoding="utf-8")
    logger.info(f"Saved report: {report_path}")


def load_reward_model(model_dir: str) -> tuple[Pipeline, np.ndarray]:
    """Load saved reward model and action_context metadata."""
    model = joblib.load(Path(model_dir) / "reward_model.joblib")
    meta = np.load(Path(model_dir) / "model_meta.npz")
    return model, meta["action_context"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train reward model for bandit env")
    p.add_argument("--bandit", default="data/sample/bandit_feedback_sample.npz")
    p.add_argument("--splits", default="data/sample/split_manifest_sample.npz")
    p.add_argument("--out-dir", default="saved/reward_model")
    p.add_argument("--report", default="reports/tables/reward_model_metrics.md")
    p.add_argument("--plots-dir", default="reports/plots")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_reward_model(
        bandit_path=args.bandit,
        splits_path=args.splits,
        out_dir=args.out_dir,
        report_path=args.report,
        plots_dir=args.plots_dir,
        seed=args.seed,
    )
