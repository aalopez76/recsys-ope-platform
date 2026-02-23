# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] — 2026-02-22

### Added

#### Data Pipeline
- `src/data/build_obd_datasets.py` — reproducible OBD dataset pipeline with schema validation, fingerprinting, and privacy check
- `data/sample/` — versioned sample package (n=10k rounds, 80 actions, 20d context)
- `data/sample/split_manifest_sample.npz` — single source of truth for train/val/test splits (6300/700/3000)
- `data/sample/metadata_sample.json` — dataset dimensions and statistics

#### RecBole Baselines
- `src/recsys/train_recbole_baselines.py` — Pop, BPR, NeuMF, LightGCN training wrapper
- `configs/recbole.yaml` — RecBole configuration
- Outputs: `reports/tables/recbole_baselines_sample.csv` (NDCG@10, Recall@10, HR@10)

#### OPE Suite v2
- `src/ope/run_ope_suite.py` — IPS, SNIPS, DR estimators with bootstrap CI (200 resamples)
- Propensity clipping sensitivity across [0.001, 0.01, 0.05]
- Weight diagnostics: ESS, w_p50/90/99, %pscore<clip
- On-policy sanity check: `IPS(RealizedAction) ≈ on_policy_value`
- Plots: `ope_value_by_policy.png`, `ope_sensitivity_clipping.png`, `ope_weight_diagnostics.png`
- External policy CSV support via `--external-policy-csv`

#### Bandits / RL
- `src/bandits/reward_model.py` — LogisticRegression click model (25d features, train_idx only)
- `src/bandits/obd_sim_bandit_env.py` — TF-Agents `BanditPyEnvironment` backed by reward model
- `src/bandits/train_tf_agents.py` — LinUCB + EpsilonGreedy (numpy implementation, 5000 steps)
- `src/bandits/export_policy_for_ope.py` — epsilon-greedy policy export for OPE evaluation
- Note: TF-Agents API conflict with Keras 3 mitigated via numpy-equivalent implementations

#### Dashboard
- `src/app/app.py` — Streamlit dashboard (4 sections: Overview, RecBole, OPE, Bandits)
- Graceful fallback for missing artifacts

#### Certification Docs
- `docs/PIDA.md` — full CRISP-DM project document (8 sections)
- `docs/model_cards/recsys_baselines.md`
- `docs/model_cards/ope_suite.md`
- `docs/model_cards/bandit_policy.md`
- `docs/data_dictionary.md`

#### Repo & Governance
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `SECURITY.md`
- `.editorconfig`, `.gitattributes`
- `.github/workflows/ci.yml` — pytest CI

[1.0.0]: https://github.com/aalopez/recsys-ope-platform/releases/tag/v1.0.0
