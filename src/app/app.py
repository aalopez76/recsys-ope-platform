"""
RecSys + OPE Certification Dashboard
Streamlit multi-page app consuming reports/tables, reports/plots, data/sample.

Run with:
    streamlit run src/app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
TABLES = ROOT / "reports" / "tables"
PLOTS = ROOT / "reports" / "plots"
SAMPLE = ROOT / "data" / "sample"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecSys OPE Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame | None:
    """Load CSV with graceful fallback."""
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def show_missing(artifact_name: str, hint: str = "") -> None:
    st.warning(
        f"⚠️ **Missing artifact**: `{artifact_name}`\n\n"
        f"{hint or 'Run the corresponding pipeline step to generate it.'}"
    )


def show_plot(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        show_missing(str(path.name), f"Plot not found at `{path}`")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📊 RecSys OPE Dashboard")
    st.markdown("**Off-Policy Evaluation + Bandit Certification Platform**")
    st.divider()
    page = st.radio(
        "Navigate to:",
        ["🏠 Overview", "📈 RecBole Baselines", "🎯 OPE Suite", "🤖 Bandits & Policy"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data source: `data/sample/` only")
    st.caption("Split SSOT: `split_manifest_sample.npz`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏠 Project Overview")
    st.markdown("""
    This platform evaluates recommendation system policies using **Off-Policy Evaluation (OPE)**
    on the Open Bandit Dataset (OBD). It combines **RecBole collaborative filtering baselines**
    (Pop, BPR, NeuMF, LightGCN) with **contextual bandit agents** (LinUCB, EpsilonGreedy)
    evaluated offline via IPS/SNIPS/DR estimators with bootstrap confidence intervals.

    All evaluations use *only* the `data/sample/` package — no raw dataset downloads required.
    Reproducibility is enforced via `split_manifest_sample.npz` as the single source of truth
    for train/val/test splits.
    """)

    st.subheader("📐 Dataset KPIs")
    meta = load_json(SAMPLE / "metadata_sample.json")
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rounds", f"{meta.get('n_rounds', 10000):,}")
        c2.metric("Actions (items)", f"{meta.get('n_actions', 80)}")
        c3.metric("Context Dim", f"{meta.get('context_dim', 20)}")
        c4.metric("Action Context Dim", f"{meta.get('action_context_dim', 5)}")
    else:
        show_missing("metadata_sample.json", "Run `python -m src.data.build_obd_datasets`")

    # Split counts
    splits_path = SAMPLE / "split_manifest_sample.npz"
    if splits_path.exists():
        s = np.load(str(splits_path), allow_pickle=True)
        st.subheader("📊 Split Counts")
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Train", f"{len(s['train_idx']):,}", "63%")
        cs2.metric("Val", f"{len(s['val_idx']):,}", "7%")
        cs3.metric("Test", f"{len(s['test_idx']):,}", "30%")
    else:
        show_missing("split_manifest_sample.npz")

    st.subheader("🗂️ Key Artifacts")
    artifacts = {
        "RecBole Baselines CSV": TABLES / "recbole_baselines_sample.csv",
        "OPE Results CSV": TABLES / "ope_results_sample.csv",
        "OPE + TFAgent CSV": TABLES / "ope_results_with_tf_agents.csv",
        "Reward Model Metrics": TABLES / "reward_model_metrics.md",
        "TF-Agents Training Summary": TABLES / "tf_agents_training_summary.md",
        "Value by Policy Plot": PLOTS / "ope_value_by_policy.png",
        "Weight Diagnostics Plot": PLOTS / "ope_weight_diagnostics.png",
        "Training Curve Plot": PLOTS / "tf_agents_training_curve.png",
    }
    rows = [
        {
            "Artifact": name,
            "Path": str(path.relative_to(ROOT)),
            "Status": "✅" if path.exists() else "❌",
        }
        for name, path in artifacts.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("🏗️ Pipeline Architecture")
    st.code(
        """
data/sample/
  ├── bandit_feedback_sample.npz   (n=10k, context 20d, 80 actions)
  ├── split_manifest_sample.npz    ← SSOT (train/val/test)
  └── metadata_sample.json

         ┌─────────────┐      ┌─────────────────┐
         │  RecBole    │      │  Reward Model   │
         │Baselines    │      │ (sklearn LogReg) │
         │Pop/BPR/NeuMF│      └────────┬────────┘
         │/LightGCN    │               │
         └──────┬──────┘      ┌────────▼────────┐
                │             │ Bandit Agents   │
                │             │ LinUCB/EpsGreedy│
                │             └────────┬────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                   ┌───────────────┐
                   │  OPE Suite v2 │
                   │  IPS/SNIPS/DR │
                   │  + Diagnostics│
                   └───────┬───────┘
                           ▼
                   📊 This Dashboard
    """,
        language="text",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RecBole Baselines
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 RecBole Baselines":
    st.title("📈 RecBole Collaborative Filtering Baselines")

    df = load_csv(TABLES / "recbole_baselines_sample.csv")
    if df is not None:
        st.subheader("📋 Metrics Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Quick bar chart of available metrics
        if "ndcg@10" in df.columns or "NDCG@10" in df.columns:
            col_ndcg = "ndcg@10" if "ndcg@10" in df.columns else "NDCG@10"
            col_model = (
                [c for c in df.columns if "model" in c.lower()][0]
                if any("model" in c.lower() for c in df.columns)
                else df.columns[0]
            )
            st.subheader("📊 NDCG@10 Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                df[col_model],
                df[col_ndcg],
                color=["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"][: len(df)],
            )
            ax.set_xlabel("NDCG@10", fontsize=12)
            ax.set_title("RecBole Baselines — NDCG@10", fontsize=13, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, val in zip(bars, df[col_ndcg]):
                ax.text(
                    bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=10,
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        show_missing(
            "recbole_baselines_sample.csv", "Run: `python -m src.recsys.train_recbole_baselines`"
        )

    # NDCG plot
    st.subheader("📊 NDCG@10 Plot (saved)")
    show_plot(PLOTS / "recbole_ndcg_at10.png", "RecBole NDCG@10 by model")

    # Audit report
    st.subheader("📋 Dataset Contract / Audit Report")
    audit_path = TABLES / "recbole_baselines_audit_report.md"
    if audit_path.exists():
        txt = audit_path.read_text(encoding="utf-8", errors="replace")
        with st.expander("Show audit report", expanded=False):
            st.markdown(txt[:3000] + ("\n\n_[truncated]_" if len(txt) > 3000 else ""))
    else:
        show_missing(
            "recbole_baselines_audit_report.md", "Run: `python scripts/audit_recbole_contract.py`"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: OPE Suite
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 OPE Suite":
    st.title("🎯 Off-Policy Evaluation (OBP)")

    # Load either the TF-Agents extended CSV or the base one
    df_ext = load_csv(TABLES / "ope_results_with_tf_agents.csv")
    df_base = load_csv(TABLES / "ope_results_sample.csv")
    df = df_ext if df_ext is not None else df_base

    if df is None:
        show_missing("ope_results_sample.csv", "Run: `python -m src.ope.run_ope_suite`")
    else:
        st.info(
            f"Loaded: `{'ope_results_with_tf_agents.csv' if df_ext is not None else 'ope_results_sample.csv'}` "
            f"({len(df)} rows, {df['policy_name'].nunique()} policies)"
        )

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        policies = sorted(df["policy_name"].unique())
        estimators = sorted(df["estimator"].unique())
        clips = sorted(df["clip"].unique())

        sel_policies = col_f1.multiselect("Policy", policies, default=policies)
        sel_estimators = col_f2.multiselect("Estimator", estimators, default=estimators)
        sel_clips = col_f3.multiselect("Clip", clips, default=[0.01])

        mask = (
            df["policy_name"].isin(sel_policies)
            & df["estimator"].isin(sel_estimators)
            & df["clip"].isin(sel_clips)
        )
        df_filt = df[mask].copy()

        st.subheader("📋 OPE Results Table")
        show_cols = ["policy_name", "estimator", "clip", "value_hat", "ci_low", "ci_high"]
        if "ess" in df_filt.columns:
            show_cols += ["ess", "w_p99", "pct_pscore_below_clip"]
        st.dataframe(df_filt[show_cols], use_container_width=True, hide_index=True)

        # Value chart with CI
        if len(df_filt) > 0 and len(sel_clips) == 1:
            st.subheader("📊 Value Estimates with 95% CI")
            fig, ax = plt.subplots(figsize=(12, 5))
            palette = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
            for i, (policy, pdf) in enumerate(df_filt.groupby("policy_name")):
                for j, (_est, edf) in enumerate(pdf.groupby("estimator")):
                    x_pos = i * (len(estimators) + 1) + j
                    v = edf["value_hat"].values[0]
                    lo = edf["ci_low"].values[0]
                    hi = edf["ci_high"].values[0]
                    ax.bar(
                        x_pos,
                        v,
                        color=palette[i % len(palette)],
                        alpha=0.8,
                        width=0.7,
                        label=policy if j == 0 else "",
                    )
                    ax.errorbar(
                        x_pos, v, yerr=[[v - lo], [hi - v]], fmt="none", color="black", capsize=4
                    )

            on_pol = (
                df_filt["on_policy_value"].iloc[0] if "on_policy_value" in df_filt.columns else None
            )
            if on_pol is not None:
                ax.axhline(
                    on_pol, color="red", linestyle="--", lw=1.5, label=f"on_policy={on_pol:.4f}"
                )
            ax.set_ylabel("value_hat (CTR)", fontsize=11)
            ax.set_title(f"OPE Values — clip={sel_clips[0]}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Weight diagnostics
        if "ess" in df.columns:
            st.subheader("⚖️ Weight Diagnostics")
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
            avail = [c for c in diag_cols if c in df.columns]
            diag_df = (
                df[avail]
                .drop_duplicates(subset=["policy_name", "clip"])
                .sort_values(["policy_name", "clip"])
            )
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

        # Saved plots
        st.subheader("🖼️ Saved Plots")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            show_plot(PLOTS / "ope_value_by_policy.png", "Value by Policy")
        with pc2:
            show_plot(PLOTS / "ope_sensitivity_clipping.png", "Clip Sensitivity")
        with pc3:
            show_plot(PLOTS / "ope_weight_diagnostics.png", "Weight Diagnostics")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Bandits & Policy Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Bandits & Policy":
    st.title("🤖 Bandits & Policy Evaluation")

    # Policy CSV
    st.subheader("📋 TFAgent Policy: Action Probabilities (test_idx)")
    pol_df = load_csv(TABLES / "tf_agents_policy_test_action_prob.csv")
    if pol_df is not None:
        eps = float(pol_df["epsilon"].iloc[0])
        agreement = (pol_df["logged_action"] == pol_df["best_action"]).mean()
        pct_greedy = (pol_df["pi_e_logged_action"] > 0.5).mean()

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Epsilon", f"{eps:.2f}")
        cm2.metric("Policy/Log Agreement", f"{agreement*100:.1f}%")
        cm3.metric("Rounds where pi_e > 0.5", f"{pct_greedy*100:.1f}%")

        st.markdown("""
        **Interpretation**: The bandit policy (greedy on predicted CTR) selects a *different* item
        than the behavior logging policy in ~99% of rounds. This low **support/coverage**
        causes IPS/SNIPS estimators to produce near-zero estimates because the importance
        weights `pi_e(a|x) / pi_b(a|x)` are extremely small for all non-agreement rounds.
        """)

        # Histogram of pi_e
        st.subheader("📊 Distribution of pi_e (logged action)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            pol_df["pi_e_logged_action"], bins=30, color="#2196F3", edgecolor="white", alpha=0.85
        )
        ax.axvline(
            pol_df["pi_e_logged_action"].mean(),
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"mean={pol_df['pi_e_logged_action'].mean():.4f}",
        )
        ax.set_xlabel("pi_e(logged_action | context)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            "TFAgent: Importance Weight Distribution on Logged Actions",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("Show first 20 rows"):
            st.dataframe(pol_df.head(20), use_container_width=True, hide_index=True)
    else:
        show_missing(
            "tf_agents_policy_test_action_prob.csv",
            "Run: `python -m src.bandits.export_policy_for_ope`",
        )

    # OPE comparison
    st.subheader("⚖️ OPE: Baselines vs. TFAgent Policy")
    df_comparison = load_csv(TABLES / "ope_results_with_tf_agents.csv")
    if df_comparison is not None:
        df_c = df_comparison[df_comparison["clip"] == 0.01]
        pivot = df_c.pivot_table(
            index="policy_name", columns="estimator", values="value_hat"
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

        st.subheader("📊 OPE with TFAgent — Training Curve")
        show_plot(PLOTS / "tf_agents_training_curve.png", "LinUCB & EpsGreedy — Rolling Avg Reward")
        show_plot(PLOTS / "ope_value_by_policy.png", "All Policies: OPE Value Estimates")
    else:
        show_missing(
            "ope_results_with_tf_agents.csv",
            "Run: `python -m src.ope.run_ope_suite --external-policy-csv ...`",
        )

    # Training summary
    st.subheader("🏋️ Training Summary")
    ts_path = TABLES / "tf_agents_training_summary.md"
    if ts_path.exists():
        txt = ts_path.read_text(encoding="utf-8", errors="replace")
        st.markdown(txt[:3000])
    else:
        show_missing(
            "tf_agents_training_summary.md", "Run: `python -m src.bandits.train_tf_agents`"
        )

    # Reward model metrics
    st.subheader("🎯 Reward Model Metrics")
    rm_path = TABLES / "reward_model_metrics.md"
    if rm_path.exists():
        txt = rm_path.read_text(encoding="utf-8", errors="replace")
        with st.expander("Show reward model metrics"):
            st.markdown(txt[:2000])
    else:
        show_missing("reward_model_metrics.md", "Run: `python -m src.bandits.reward_model`")

    st.subheader("📢 Key Finding: Support Problem")
    st.error("""
    **Why IPS/SNIPS(TFAgent) ≈ 0?**

    The TFAgent's greedy policy selects items with *high predicted CTR*, but the logging policy
    selected items *randomly* (or pseudo-randomly). The item the TFAgent recommends almost never
    appears in the historical logs → the importance weight `pi_e(a_logged) / pi_b(a_logged)` is
    near-zero for ~99% of rounds.

    **Solution for production**: Use an exploration policy with explicit overlap with the target
    policy (e.g., mix 30% random + 70% greedy), or use DR with a strong direct model.
    """)
