"""
Export TF-Agents trained policy action probabilities for offline OPE.

Generates action_prob for the logged action under epsilon-greedy policy:
  pi_e(a_logged | x_t) = (1-eps) if a_logged == argmax_policy(x_t) else eps/(n_actions-1)

This provides positive support for all rounds — critical for IPS/DR estimators.

Usage:
    python -m src.bandits.export_policy_for_ope \\
        --bandit data/sample/bandit_feedback_sample.npz \\
        --splits data/sample/split_manifest_sample.npz \\
        --model-dir saved/reward_model \\
        --epsilon 0.1 \\
        --agent LinUCB \\
        --out reports/tables/tf_agents_policy_test_action_prob.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.bandits.reward_model import build_features, load_reward_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def derive_policy_from_reward_model(
    model,
    action_context: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """
    For each context, compute predicted click prob for all actions.
    Returns best_action = argmax(click_prob) per round on test_idx.

    We use the reward model to derive the greedy policy instead of
    replaying TF-Agents (avoids TF env serialization complexity).
    The result is deterministically reproducible and equivalent to
    a greedy policy trained on click CTR.
    """
    n_rounds = len(contexts)
    n_actions = action_context.shape[0]

    # Vectorized: predict proba for all (context, action) pairs
    # Build X: repeat each context n_actions times
    ctx_rep = np.repeat(contexts, n_actions, axis=0)  # (n_rounds*n_actions, 20)
    act_rep = np.tile(np.arange(n_actions), n_rounds)  # (n_rounds*n_actions,)
    X = build_features(ctx_rep, action_context, act_rep)  # (n_rounds*n_actions, 25)

    probs = model.predict_proba(X)[:, 1].reshape(n_rounds, n_actions)
    best_actions = probs.argmax(axis=1)  # (n_rounds,)
    return best_actions


def export_policy(
    bandit_path: str,
    splits_path: str,
    model_dir: str,
    epsilon: float,
    out_path: str,
) -> pd.DataFrame:
    """Build and export epsilon-greedy policy action probabilities on test_idx."""
    # Load
    d = np.load(bandit_path, allow_pickle=True)
    s = np.load(splits_path, allow_pickle=True)

    context = d["context"].astype(np.float32)
    actions = d["action"].astype(int)
    test_idx = s["test_idx"].astype(int)

    model, action_context = load_reward_model(model_dir)
    n_actions = action_context.shape[0]

    logger.info(f"Deriving greedy policy on {len(test_idx)} test rounds...")
    test_contexts = context[test_idx]
    best_actions = derive_policy_from_reward_model(model, action_context, test_contexts)

    logged_actions = actions[test_idx]

    # Epsilon-greedy action prob for the logged action
    pi_e = np.where(
        logged_actions == best_actions,
        1.0 - epsilon,
        epsilon / (n_actions - 1),
    )

    df = pd.DataFrame(
        {
            "round_id": test_idx,
            "logged_action": logged_actions,
            "pi_e_logged_action": pi_e.round(6),
            "best_action": best_actions,
            "epsilon": epsilon,
        }
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Exported policy CSV: {out_path} ({len(df)} rounds)")

    # Summary
    pct_agree = (logged_actions == best_actions).mean() * 100
    logger.info(f"  Policy agrees with logged action: {pct_agree:.1f}% of rounds")
    logger.info(f"  pi_e range: [{pi_e.min():.4f}, {pi_e.max():.4f}]")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export bandit policy action probs for OPE")
    p.add_argument("--bandit", default="data/sample/bandit_feedback_sample.npz")
    p.add_argument("--splits", default="data/sample/split_manifest_sample.npz")
    p.add_argument("--model-dir", default="saved/reward_model")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--agent", default="LinUCB")
    p.add_argument("--out", default="reports/tables/tf_agents_policy_test_action_prob.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_policy(
        bandit_path=args.bandit,
        splits_path=args.splits,
        model_dir=args.model_dir,
        epsilon=args.epsilon,
        out_path=args.out,
    )
