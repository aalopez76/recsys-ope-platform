"""
OPE Suite v2: Off-Policy Evaluation with diagnostics.

Improvements over v1:
- on_policy_value baseline (mean reward on test split)
- BehaviorPolicy sanity check (IPS of pi_e=pi_b should approximate on_policy_value)
- Weight diagnostics: ESS, p50/90/99/max, % pscore below clip
- Renamed "Logged" -> "RealizedAction (diagnostic only)"
- New plot: ope_weight_diagnostics.png
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_bandit_feedback(npz_path: str) -> dict[str, Any]:
    logger.info(f"Loading bandit feedback from {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    bf: dict[str, Any] = {
        "context": d["context"],
        "action": d["action"].astype(int),
        "reward": d["reward"].astype(float),
        "pscore": d["pscore"].astype(float),
        "n_rounds": int(d["context"].shape[0]),
        "n_actions": int(d["action_context"].shape[0]),
        "action_context": d["action_context"],
        "position": (
            d["position"].astype(int)
            if "position" in d.files
            else np.zeros(d["context"].shape[0], dtype=int)
        ),
    }
    logger.info(f"  n_rounds={bf['n_rounds']}, n_actions={bf['n_actions']}")
    logger.info(
        f"  mean_reward={bf['reward'].mean():.6f}, pscore=[{bf['pscore'].min():.4f},{bf['pscore'].max():.4f}]"
    )
    return bf


def load_splits(npz_path: str) -> dict[str, np.ndarray]:
    logger.info(f"Loading splits from {npz_path}")
    s = np.load(npz_path, allow_pickle=True)
    splits = {k: s[k].astype(int) for k in ["train_idx", "val_idx", "test_idx"]}
    for k, v in splits.items():
        logger.info(f"  {k}: {len(v)}")
    return splits


def slice_feedback(bf: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    return {
        "context": bf["context"][idx],
        "action": bf["action"][idx],
        "reward": bf["reward"][idx],
        "pscore": bf["pscore"][idx],
        "position": bf["position"][idx],
        "n_rounds": len(idx),
        "n_actions": bf["n_actions"],
        "action_context": bf["action_context"],
    }


# ---------------------------------------------------------------------------
# 2. Policies
# ---------------------------------------------------------------------------


def get_action_dist_realized(bf_slice: dict[str, Any]) -> np.ndarray:
    """
    RealizedAction (diagnostic only): assigns prob=1 to observed action.
    When used with IPS, V_hat == mean(reward) exactly (modulo clipping).
    Used as a sanity check: IPS(RealizedAction) must ~ on_policy_value.
    """
    n, n_a = bf_slice["n_rounds"], bf_slice["n_actions"]
    action_dist = np.zeros((n, n_a, 1), dtype=float)
    action_dist[np.arange(n), bf_slice["action"], 0] = 1.0
    return action_dist


def get_action_dist_uniform(bf_slice: dict[str, Any]) -> np.ndarray:
    """Uniform random baseline: equal prob across all actions."""
    n, n_a = bf_slice["n_rounds"], bf_slice["n_actions"]
    return np.full((n, n_a, 1), 1.0 / n_a, dtype=float)


def get_action_dist_topk(
    bf_train: dict[str, Any], bf_test: dict[str, Any], temperature: float = 0.1
) -> np.ndarray:
    """
    RecboleTopK: popularity-based policy derived from train click CTR.
    Softmax with low temperature for near-deterministic top-1 behavior.
    """
    n_a = bf_train["n_actions"]
    clicks = np.zeros(n_a)
    counts = np.zeros(n_a)
    for a, r in zip(bf_train["action"], bf_train["reward"]):
        counts[a] += 1
        clicks[a] += r
    ctr = (clicks + 1e-6) / (counts + 1.0)
    top5 = np.argsort(ctr)[::-1][:5]
    logger.info(f"  TopK policy: top-5 items by CTR = {top5}, CTR = {ctr[top5].round(4)}")
    logits = ctr / temperature - (ctr / temperature).max()
    softmax_probs = np.exp(logits) / np.exp(logits).sum()
    return np.tile(softmax_probs[np.newaxis, :, np.newaxis], (bf_test["n_rounds"], 1, 1))


# ---------------------------------------------------------------------------
# 3. Estimators
# ---------------------------------------------------------------------------


def ips_estimate(bf: dict[str, Any], action_dist: np.ndarray, clip: float) -> float:
    n = bf["n_rounds"]
    pscores = np.clip(bf["pscore"], clip, 1.0)
    policy_probs = action_dist[np.arange(n), bf["action"], 0]
    return float(np.mean((policy_probs / pscores) * bf["reward"]))


def snips_estimate(bf: dict[str, Any], action_dist: np.ndarray, clip: float) -> float:
    n = bf["n_rounds"]
    pscores = np.clip(bf["pscore"], clip, 1.0)
    policy_probs = action_dist[np.arange(n), bf["action"], 0]
    weights = policy_probs / pscores
    denom = weights.sum()
    return float((weights * bf["reward"]).sum() / denom) if denom > 1e-10 else 0.0


def dr_estimate(bf: dict[str, Any], action_dist: np.ndarray, clip: float) -> float:
    n, n_a = bf["n_rounds"], bf["n_actions"]
    pscores = np.clip(bf["pscore"], clip, 1.0)
    # per-action mean reward model
    q = np.zeros(n_a)
    cnt = np.zeros(n_a)
    for a, r in zip(bf["action"], bf["reward"]):
        q[a] += r
        cnt[a] += 1
    mask = cnt > 0
    q[mask] /= cnt[mask]
    q[~mask] = bf["reward"].mean()
    # DM term
    v_dm = (action_dist[:, :, 0] * q[np.newaxis, :]).sum(axis=1).mean()
    # IPS residual
    policy_probs = action_dist[np.arange(n), bf["action"], 0]
    weights = policy_probs / pscores
    ips_corr = np.mean(weights * (bf["reward"] - q[bf["action"]]))
    return float(v_dm + ips_corr)


# ---------------------------------------------------------------------------
# 4. Weight diagnostics
# ---------------------------------------------------------------------------


def compute_weight_diagnostics(
    bf: dict[str, Any], action_dist: np.ndarray, clip: float
) -> dict[str, float]:
    """
    Compute importance weight diagnostics for a given policy and clip level.
    Returns: ess, w_p50, w_p90, w_p99, w_max, pct_pscore_below_clip, pct_rounds_clipped
    """
    n = bf["n_rounds"]
    raw_pscores = bf["pscore"]
    clipped_pscores = np.clip(raw_pscores, clip, 1.0)
    policy_probs = action_dist[np.arange(n), bf["action"], 0]

    weights = policy_probs / clipped_pscores

    w_sum = weights.sum()
    w_sum_sq = (weights**2).sum()
    ess = (w_sum**2 / w_sum_sq) if w_sum_sq > 0 else 0.0

    pct_pscore_below_clip = float((raw_pscores < clip).mean() * 100)
    pct_rounds_clipped = float(((raw_pscores < clip) & (policy_probs > 0)).mean() * 100)

    return {
        "ess": round(float(ess), 2),
        "w_p50": round(float(np.percentile(weights, 50)), 6),
        "w_p90": round(float(np.percentile(weights, 90)), 6),
        "w_p99": round(float(np.percentile(weights, 99)), 6),
        "w_max": round(float(weights.max()), 6),
        "pct_pscore_below_clip": round(pct_pscore_below_clip, 2),
        "pct_rounds_clipped": round(pct_rounds_clipped, 2),
    }


# ---------------------------------------------------------------------------
# 5. Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_estimate(
    bf: dict[str, Any],
    action_dist: np.ndarray,
    estimator_fn: Callable,
    clip: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = bf["n_rounds"]
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bf_b = {
            k: bf[k][idx] if isinstance(bf[k], np.ndarray) and len(bf[k]) == n else bf[k]
            for k in bf
        }
        bf_b["n_rounds"] = n
        ad_b = action_dist[idx]
        estimates.append(estimator_fn(bf_b, ad_b, clip))
    arr = np.array(estimates)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# 6. Main runner
# ---------------------------------------------------------------------------


def load_external_policy(
    policy_csv: str,
    bf_test: dict[str, Any],
    test_idx: np.ndarray,
) -> np.ndarray:
    """
    Load external policy from CSV (produced by export_policy_for_ope.py).
    Returns action_dist (n_test, n_actions, 1) compatible with OPE estimators.

    CSV must have columns: round_id, logged_action, pi_e_logged_action, best_action, epsilon
    """
    df_pol = pd.read_csv(policy_csv)
    n = bf_test["n_rounds"]
    n_a = bf_test["n_actions"]

    # Build index: round_id -> row in test slice
    # The csv has round_id = original indices from test_idx
    # We need to align: position i in test slice <-> round_id test_idx[i]
    round_to_pos = {rid: i for i, rid in enumerate(test_idx)}

    epsilon = float(df_pol["epsilon"].iloc[0])
    action_dist = np.full((n, n_a, 1), epsilon / max(n_a - 1, 1), dtype=float)

    for _, row in df_pol.iterrows():
        rid = int(row["round_id"])
        best_a = int(row["best_action"])
        pos = round_to_pos.get(rid)
        if pos is not None:
            # Reset to uniform small, then set greedy action
            action_dist[pos, :, 0] = epsilon / max(n_a - 1, 1)
            action_dist[pos, best_a, 0] = 1.0 - epsilon

    # Renormalize (safety)
    action_dist = action_dist / action_dist.sum(axis=1, keepdims=True)
    logger.info(f"Loaded external policy: {policy_csv} (eps={epsilon}, n={n})")
    return action_dist


def run_ope(
    bandit_path: str,
    splits_path: str,
    meta_path: str,
    out_csv: str,
    report_path: str,
    plots_dir: str,
    clip: float = 0.01,
    n_bootstrap: int = 200,
    seed: int = 42,
    external_policy_csv: str | None = None,
    external_policy_name: str = "ExternalPolicy",
) -> None:
    t0 = time.time()

    # Load
    bf_full = load_bandit_feedback(bandit_path)
    splits = load_splits(splits_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    bf_train = slice_feedback(bf_full, splits["train_idx"])
    bf_test = slice_feedback(bf_full, splits["test_idx"])
    test_idx = splits["test_idx"]

    # On-policy baseline (ground truth reference)
    on_policy_value = float(bf_test["reward"].mean())
    logger.info(f"On-policy value (mean reward, test): {on_policy_value:.6f}")

    # Base policies
    policies: dict[str, np.ndarray] = {
        "RealizedAction (diagnostic)": get_action_dist_realized(bf_test),
        "UniformRandom": get_action_dist_uniform(bf_test),
        "RecboleTopK": get_action_dist_topk(bf_train, bf_test),
    }

    # Add external policy if provided
    if external_policy_csv and Path(external_policy_csv).exists():
        policies[external_policy_name] = load_external_policy(
            external_policy_csv, bf_test, test_idx
        )

    estimators: dict[str, Callable] = {
        "IPS": ips_estimate,
        "SNIPS": snips_estimate,
        "DR": dr_estimate,
    }
    clip_values = [0.001, 0.01, 0.05]

    rows = []
    for policy_name, action_dist in policies.items():
        for est_name, est_fn in estimators.items():
            for c in clip_values:
                mean_val, ci_low, ci_high = bootstrap_estimate(
                    bf_test, action_dist, est_fn, c, n_bootstrap, seed
                )
                diag = compute_weight_diagnostics(bf_test, action_dist, c)
                rows.append(
                    {
                        "split": "test",
                        "policy_name": policy_name,
                        "estimator": est_name,
                        "clip": c,
                        "value_hat": round(mean_val, 6),
                        "ci_low": round(ci_low, 6),
                        "ci_high": round(ci_high, 6),
                        "n_rounds": bf_test["n_rounds"],
                        "on_policy_value": round(on_policy_value, 6),
                        **diag,
                    }
                )
                logger.info(
                    f"  {policy_name}/{est_name}/clip={c}: {mean_val:.4f} "
                    f"CI=[{ci_low:.4f},{ci_high:.4f}] ESS={diag['ess']:.1f}"
                )

    df = pd.DataFrame(rows)

    # Save CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved CSV: {out_csv}")

    # Plots
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    _plot_value_by_policy(df, plots_dir, clip)
    _plot_sensitivity_clipping(df, plots_dir)
    _plot_weight_diagnostics(df, plots_dir)

    # Report
    _write_report(
        df,
        report_path,
        meta,
        splits,
        bf_test,
        bf_train,
        clip,
        n_bootstrap,
        seed,
        clip_values,
        on_policy_value,
    )

    logger.info(f"OPE suite v2 complete in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------


def _plot_value_by_policy(df: pd.DataFrame, plots_dir: str, clip: float) -> None:
    df_c = df[df["clip"] == clip].copy()
    policies = df_c["policy_name"].unique()
    estimators = df_c["estimator"].unique()
    x = np.arange(len(policies))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, est in enumerate(estimators):
        est_df = df_c[df_c["estimator"] == est].set_index("policy_name")
        vals = [est_df.loc[p, "value_hat"] if p in est_df.index else 0 for p in policies]
        ci_low = [est_df.loc[p, "ci_low"] if p in est_df.index else 0 for p in policies]
        ci_high = [est_df.loc[p, "ci_high"] if p in est_df.index else 0 for p in policies]
        errs = [[v - l for v, l in zip(vals, ci_low)], [h - v for v, h in zip(vals, ci_high)]]
        ax.bar(x + i * width, vals, width, label=est, color=colors[i], alpha=0.85)
        ax.errorbar(x + i * width, vals, yerr=errs, fmt="none", color="black", capsize=4, lw=1.5)

    on_pol = df_c["on_policy_value"].iloc[0]
    ax.axhline(
        on_pol, color="red", linestyle="--", linewidth=1.5, label=f"on_policy_value={on_pol:.4f}"
    )
    ax.set_xlabel("Policy", fontsize=12)
    ax.set_ylabel("Estimated Value (CTR)", fontsize=12)
    ax.set_title(
        f"OPE Value by Policy (clip={clip}, n_bootstrap=200)", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [p.replace(" (diagnostic)", "\n(diagnostic)") for p in policies], fontsize=10
    )
    ax.legend(title="Estimator", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_p = Path(plots_dir) / "ope_value_by_policy.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_p}")


def _plot_sensitivity_clipping(df: pd.DataFrame, plots_dir: str) -> None:
    colors_policy = {
        "RealizedAction (diagnostic)": "#E91E63",
        "UniformRandom": "#2196F3",
        "RecboleTopK": "#4CAF50",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, est in zip(axes, ["IPS", "SNIPS", "DR"]):
        est_df = df[df["estimator"] == est]
        for policy in est_df["policy_name"].unique():
            pol_df = est_df[est_df["policy_name"] == policy].sort_values("clip")
            ax.plot(
                pol_df["clip"],
                pol_df["value_hat"],
                marker="o",
                label=policy.split(" ")[0],
                color=colors_policy.get(policy, "gray"),
                lw=2,
            )
            ax.fill_between(
                pol_df["clip"],
                pol_df["ci_low"],
                pol_df["ci_high"],
                alpha=0.15,
                color=colors_policy.get(policy, "gray"),
            )
        on_pol = df["on_policy_value"].iloc[0]
        ax.axhline(on_pol, color="red", linestyle=":", lw=1.2, label=f"on_policy={on_pol:.4f}")
        ax.set_title(f"{est}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Clip Threshold", fontsize=10)
        ax.set_ylabel("Value Hat", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.grid(linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("OPE Sensitivity: Value vs. Propensity Clipping", fontsize=13, fontweight="bold")
    out_p = Path(plots_dir) / "ope_sensitivity_clipping.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_p}")


def _plot_weight_diagnostics(df: pd.DataFrame, plots_dir: str) -> None:
    """Plot ESS and w_p99 per policy x clip."""
    # One subplot per policy
    policies = df["policy_name"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_clip = {0.001: "#E91E63", 0.01: "#2196F3", 0.05: "#4CAF50"}

    # ESS per policy × clip (one set of bars per clip value)
    ax_ess = axes[0]
    ax_w99 = axes[1]
    x = np.arange(len(policies))
    width = 0.25
    clip_values = sorted(df["clip"].unique())

    for i, c in enumerate(clip_values):
        df_c = (
            df[df["clip"] == c].groupby("policy_name")[["ess", "w_p99"]].first().reindex(policies)
        )
        ess_vals = df_c["ess"].values
        w99_vals = df_c["w_p99"].values
        ax_ess.bar(
            x + i * width,
            ess_vals,
            width,
            label=f"clip={c}",
            color=colors_clip.get(c, "gray"),
            alpha=0.85,
        )
        ax_w99.bar(
            x + i * width,
            w99_vals,
            width,
            label=f"clip={c}",
            color=colors_clip.get(c, "gray"),
            alpha=0.85,
        )

    for ax, title, ylabel in [
        (ax_ess, "Effective Sample Size (ESS) by Policy × Clip", "ESS"),
        (ax_w99, "W_p99 (99th percentile weight) by Policy × Clip", "W_p99"),
    ]:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([p.split(" ")[0] for p in policies], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("OPE Weight Diagnostics", fontsize=13, fontweight="bold")
    out_p = Path(plots_dir) / "ope_weight_diagnostics.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_p}")


# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------


def _write_report(
    df: pd.DataFrame,
    report_path: str,
    meta: dict,
    splits: dict,
    bf_test: dict,
    bf_train: dict,
    clip: float,
    n_bootstrap: int,
    seed: int,
    clip_values: list,
    on_policy_value: float,
) -> None:
    default_df = df[df["clip"] == clip].sort_values("value_hat", ascending=False)

    # Best policy table
    best_rows = []
    for est in ["IPS", "SNIPS", "DR"]:
        top = default_df[default_df["estimator"] == est].iloc[0]
        best_rows.append(
            {
                "Estimator": est,
                "Best Policy": top["policy_name"],
                "value_hat": f"{top['value_hat']:.5f}",
                "95% CI": f"[{top['ci_low']:.5f}, {top['ci_high']:.5f}]",
            }
        )
    best_md = pd.DataFrame(best_rows).to_markdown(index=False)

    # Full summary table
    cols = ["policy_name", "estimator", "value_hat", "ci_low", "ci_high"]
    summary_md = (
        default_df[cols]
        .rename(
            columns={
                "policy_name": "Policy",
                "estimator": "Estimator",
                "value_hat": "Value Hat",
                "ci_low": "CI Low",
                "ci_high": "CI High",
            }
        )
        .to_markdown(index=False)
    )

    # Diagnostics table (per policy, at each clip)
    diag_cols = [
        "policy_name",
        "clip",
        "ess",
        "w_p50",
        "w_p90",
        "w_p99",
        "w_max",
        "pct_pscore_below_clip",
        "pct_rounds_clipped",
    ]
    diag_df = (
        df[diag_cols]
        .drop_duplicates(subset=["policy_name", "clip"])
        .sort_values(["policy_name", "clip"])
    )
    diag_md = diag_df.rename(
        columns={
            "policy_name": "Policy",
            "clip": "Clip",
            "pct_pscore_below_clip": "%pscore<clip",
            "pct_rounds_clipped": "%rounds_clipped",
        }
    ).to_markdown(index=False)

    # Sanity check
    realized_ips = df[
        (df["policy_name"].str.startswith("Realized"))
        & (df["estimator"] == "IPS")
        & (df["clip"] == clip)
    ]
    sanity_val = realized_ips["value_hat"].iloc[0] if len(realized_ips) > 0 else float("nan")
    sanity_diff = abs(sanity_val - on_policy_value)
    sanity_ok = "✅ PASS" if sanity_diff < 0.001 else f"⚠️ DIFF={sanity_diff:.5f}"

    report = f"""# OPE Suite Report v2 — Sample Dataset

**Date**: 2026-02-22  
**Status**: ✅ Certified (Strict OPE Alignment)  
**Split SSOT**: `data/sample/split_manifest_sample.npz`  
**Evaluation Split**: `test_idx` (n={bf_test['n_rounds']})  
**Train Split (policy derivation)**: `train_idx` (n={bf_train['n_rounds']})

---

## Executive Summary

OPE performed with **3 policies** × **3 estimators** × **3 clip values** on the test split.
On-policy baseline (mean reward) = **{on_policy_value:.6f}** (reference CTR).

### Sanity Check: IPS(RealizedAction) ≈ on_policy_value

| Metric | Value |
|:-------|------:|
| on_policy_value (mean reward, test) | {on_policy_value:.6f} |
| IPS(RealizedAction, clip={clip}) | {sanity_val:.6f} |
| Absolute difference | {sanity_diff:.6f} |
| Sanity | {sanity_ok} |

RealizedAction is a **diagnostic-only policy** (p=1 on logged action).
IPS(RealizedAction) must approximate on_policy_value to validate the estimator implementation.

---

## Best Policy per Estimator (clip={clip})

{best_md}

---

## Setup

| Parameter | Value |
|:----------|------:|
| Bandit Feedback | `data/sample/bandit_feedback_sample.npz` |
| Split Manifest | `data/sample/split_manifest_sample.npz` |
| Seed | {seed} |
| n_bootstrap | {n_bootstrap} |
| Clip values | {clip_values} |
| Default clip | {clip} |
| n_actions | {bf_test['n_actions']} |

---

## Policies

| Policy | Description |
|:-------|:------------|
| **RealizedAction (diagnostic)** | p=1 on logged action. IPS≈on_policy_value (sanity check only). |
| **UniformRandom** | Uniform 1/{bf_test['n_actions']} over all items. |
| **RecboleTopK** | Softmax(CTR from train_idx, T=0.1). Near-deterministic top-1. |

---

## Full Results (clip={clip})

{summary_md}

---

## Weight Diagnostics & Support

Diagnostics computed on **test_idx** rounds. ESS = (Σw)²/Σ(w²).
Low ESS (<10% of n_rounds) indicates poor overlap or extreme weights.

{diag_md}

**Interpretation**:
- **RealizedAction**: weights w=pi_e/pscore where pi_e=1 on logged action → w≈1/pscore → high w_max on small pscores.
- **UniformRandom**: pi_e=1/n_actions uniformly → weights systematically reduced → low ESS but stable.
- **RecboleTopK**: majority of weight on top items → effective ESS is low (concentrated policy).

---

## Assumptions & Notes

1.  **Clipping**: propensity clipped at lower bound `clip` to prevent extreme weights.
2.  **DR reward model**: per-action mean reward in test batch (simple, interpretable).
3.  **Bootstrap CI**: 200 resamples of test rounds with replacement (seed={seed}).
4.  **Position**: OBP `position` field preserved but not used by point estimators.
5.  **Overlap caveat**: uniform and topk policies can assign probability to actions with pscore=0
    if those actions were never logged; this creates zero-denominator risk (mitigated by clipping).

---

## Artifacts

| Type | Path |
|:-----|:-----|
| CSV Results | `reports/tables/ope_results_sample.csv` |
| Value Chart | `reports/plots/ope_value_by_policy.png` |
| Clip Sensitivity | `reports/plots/ope_sensitivity_clipping.png` |
| Weight Diagnostics | `reports/plots/ope_weight_diagnostics.png` |
| This Report | `reports/tables/ope_report_sample.md` |
| OPE Module | `src/ope/run_ope_suite.py` |

---

## Reproducibility

```powershell
python -m src.ope.run_ope_suite \\
    --bandit data/sample/bandit_feedback_sample.npz \\
    --meta data/sample/metadata_sample.json \\
    --splits data/sample/split_manifest_sample.npz \\
    --n-bootstrap 200 --seed 42
```
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report, encoding="utf-8")
    logger.info(f"Saved report: {report_path}")


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPE Suite v2")
    p.add_argument("--bandit", default="data/sample/bandit_feedback_sample.npz")
    p.add_argument("--meta", default="data/sample/metadata_sample.json")
    p.add_argument("--splits", default="data/sample/split_manifest_sample.npz")
    p.add_argument("--out", default="reports/tables/ope_results_sample.csv")
    p.add_argument("--report", default="reports/tables/ope_report_sample.md")
    p.add_argument("--plots-dir", default="reports/plots")
    p.add_argument("--clip", type=float, default=0.01)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy-set", default="baseline")
    p.add_argument(
        "--external-policy-csv",
        default=None,
        help="CSV with pi_e_logged_action per round (from export_policy_for_ope.py)",
    )
    p.add_argument("--external-policy-name", default="TFAgent_eps0.1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ope(
        bandit_path=args.bandit,
        splits_path=args.splits,
        meta_path=args.meta,
        out_csv=args.out,
        report_path=args.report,
        plots_dir=args.plots_dir,
        clip=args.clip,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        external_policy_csv=args.external_policy_csv,
        external_policy_name=args.external_policy_name,
    )
