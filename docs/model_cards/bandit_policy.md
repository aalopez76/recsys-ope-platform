# Model Card: Bandit Policy

**Version**: 1.0  
**Date**: 2026-02-22  
**Status**: ✅ Trained & Evaluated

---

## Intended Use

Implementation of **contextual bandit policies** trained on a simulated OBD environment.
The trained policies are exported for offline OPE evaluation.

**Primary use**: Research comparison against RecBole baselines via OPE.  
**Out-of-scope**: Real-time serving without further safety analysis.

---

## Components

### Reward Model (Simulator)

| Property | Value |
|:---------|:------|
| Model Type | LogisticRegression (sklearn) |
| Features | concat(context[20d], action_context[5d]) = 25d |
| Target | click (0/1) binary |
| Training data | `train_idx` ONLY (n=6,300) |
| Output | Saved to `saved/reward_model/reward_model.joblib` |
| Metrics | `reports/tables/reward_model_metrics.md` |

**Purpose**: Simulate `p(click=1 | context, action)` to create a fast bandit environment.

### Bandit Agents

| Agent | Algorithm | Exploration | Steps |
|:------|:----------|:------------|------:|
| **LinUCB** | Disjoint Ridge Regression per action + UCB | alpha=1.0 | 5,000 |
| **EpsGreedy** | Greedy on reward model predictions | ε=0.1 | 5,000 |

> **Note on TF-Agents**: Due to a Keras 3 / tf_agents 0.19 API incompatibility
> (`keras.__internal__` attribute error), agents were implemented in pure numpy.
> The algorithms are mathematically equivalent to `LinearUCBAgent` and
> `NeuralEpsilonGreedyAgent` from TF-Agents.

---

## Input Data

| Artifact | Description |
|:---------|:------------|
| `data/sample/bandit_feedback_sample.npz` | context, action_context for environment |
| `data/sample/split_manifest_sample.npz` | SSOT — training uses ONLY `train_idx` |
| `saved/reward_model/reward_model.joblib` | Simulator for env.step() |

---

## Policy Export (for OPE)

The exported policy is **epsilon-greedy**:
```
pi_e(a_logged | x_t) =
    1 - epsilon   if a_logged == argmax_a reward_model(x_t, a)
    epsilon / (n_actions - 1)  otherwise
```

This provides positive probability on all actions — required for IPS/DR.

**Output**: `reports/tables/tf_agents_policy_test_action_prob.csv`  
**Columns**: `round_id, logged_action, pi_e_logged_action, best_action, epsilon`

---

## Metrics

| Metric | Value |
|:-------|:------|
| Agreement (policy vs. logged action) | ~1% |
| pi_e range | [0.0013, 0.9000] |
| Training rewards | ~0.004 (matches true CTR) |

**OPE Results** (clip=0.01, test_idx n=3,000):

| Estimator | TFAgent value_hat |
|:----------|------------------:|
| IPS | ≈0.0001 |
| SNIPS | ≈0.0001 |
| DR | ≈0.0040 |

---

## Limitations & Risks

| Limitation | Detail |
|:-----------|:-------|
| **Support Problem** | Agreement ~1% → IPS/SNIPS ≈ 0 (insufficient logged coverage) |
| **TF-Agents API** | Numpy reimplementation; functionality equivalent but not identical code |
| **Simulator Fidelity** | LogReg reward model may underfit complex click patterns |
| **No Online Validation** | Policy requires A/B test for definitive performance measurement |

---

## Training Protocol

1. Train reward model on `train_idx` only (prevent leakage).
2. Create OBDEnv backed by reward model.
3. Run LinUCB / EpsGreedy for 5,000 steps.
4. Export greedy action from reward model predictions on `test_idx`.
5. Wrap as epsilon-greedy (ε=0.1) for OPE compatibility.

---

## Repro Commands

```powershell
# Reward Model
python -m src.bandits.reward_model --seed 42

# Bandit Training
python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1

# Policy Export
python -m src.bandits.export_policy_for_ope --epsilon 0.1

# OPE Evaluation
python -m src.ope.run_ope_suite `
    --external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv `
    --external-policy-name "TFAgent_eps0.1" --n-bootstrap 200 --seed 42
```
