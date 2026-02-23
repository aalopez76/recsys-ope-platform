# Model Card: OPE Suite v2

**Version**: 2.0  
**Date**: 2026-02-22  
**Status**: ✅ Certified (IPS sanity check passed)

---

## Intended Use

**Off-Policy Evaluation** of recommendation policies using logged bandit feedback.
Estimates the expected reward (CTR) of target policies without online deployment.

**Primary use**: Offline comparison of candidate policies before A/B testing.  
**Out-of-scope**: Online learning, causal inference beyond IPS/DR framework.

---

## Estimators

| Estimator | Type | Properties |
|:----------|:-----|:-----------|
| **IPS** | Inverse Propensity Score | Consistent; high variance with extreme weights |
| **SNIPS** | Self-normalized IPS | Lower variance; slight bias; sums to 1 |
| **DR** | Doubly Robust | Consistent if either reward model OR propensity is correct |

All estimators include:
- **Bootstrap CI**: 200 resamples, 95% intervals, seed=42.
- **Propensity clipping**: lower bound ∈ [0.001, 0.01, 0.05].

---

## Input Data

| Artifact | Description |
|:---------|:------------|
| `data/sample/bandit_feedback_sample.npz` | context, action, reward, pscore, action_context |
| `data/sample/split_manifest_sample.npz` | SSOT — OPE uses ONLY `test_idx` |
| `data/sample/metadata_sample.json` | n_rounds, n_actions |

**Evaluation split**: `test_idx` only (n=3,000). Train data not seen by estimators.

---

## Diagnostics

| Metric | Description |
|:-------|:------------|
| ESS | Effective Sample Size — detects severe weight concentration |
| w_p50/p90/p99 | Percentiles of importance weights |
| w_max | Maximum weight — risk of single-round dominance |
| %pscore<clip | Fraction of rounds where pscore was clipped |
| %rounds_clipped | Fraction of rounds where both pscore<clip AND pi_e>0 |

---

## Policies Evaluated

| Policy | Role |
|:-------|:-----|
| RealizedAction (diagnostic) | Sanity check: IPS ≈ on_policy_value |
| UniformRandom | Lower bound baseline |
| RecboleTopK | RecBole-derived policy (softmax CTR) |
| TFAgent_eps0.1 | Epsilon-greedy bandit policy |

---

## Training / Eval Protocol

1. Load `test_idx` from `split_manifest_sample.npz`.
2. Slice bandit feedback to test rounds only.
3. Build action_dist (n_test, n_actions, 1) per policy.
4. Run IPS/SNIPS/DR for each (policy, clip) combination.
5. Bootstrap 200 resamples per combination.
6. Compute weight diagnostics per combination.

---

## Limitations & Risks

| Limitation | Detail |
|:-----------|:-------|
| **Support** | If pi_e assigns prob to actions rarely logged, IPS/SNIPS ≈ 0 (not reliable) |
| **DR reward model** | Uses per-action mean reward (simple); richer DM improves DR accuracy |
| **Positivity** | Assumes pscore > 0 for all logged actions — clipping as approximation |
| **Clip sensitivity** | Value estimates vary with clip threshold — report all three |

---

## Repro Commands

```powershell
# Baseline OPE
python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42

# With TFAgent policy
python -m src.ope.run_ope_suite `
    --out reports/tables/ope_results_with_tf_agents.csv `
    --report reports/tables/ope_report_with_tf_agents.md `
    --external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv `
    --external-policy-name "TFAgent_eps0.1" `
    --n-bootstrap 200 --seed 42
```
