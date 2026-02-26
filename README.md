# RecSys OPE Platform

[![CI](https://github.com/aalopez76/recsys-ope-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aalopez76/recsys-ope-platform/actions/workflows/ci.yml)

> **End-to-end platform comparing RecBole collaborative filtering vs. contextual bandits with Off-Policy Evaluation — Certification Grade**

---

## 🏗️ Architecture

```
data/sample/ (SSOT — versioned)
  ├── bandit_feedback_sample.npz   (n=10k, context 20d, 80 actions)
  ├── split_manifest_sample.npz    ← Train/Val/Test index SSOT
  └── metadata_sample.json

         ┌─────────────────────┐      ┌──────────────────────────┐
         │   RecBole Baselines │      │   Reward Model           │
         │  Pop / BPR / NeuMF  │      │   (LogisticRegression)   │
         │  LightGCN           │      └──────────┬───────────────┘
         └──────────┬──────────┘                 │
                    │                 ┌──────────▼───────────────┐
                    │                 │   Bandit Agents          │
                    │                 │   LinUCB / EpsGreedy     │
                    │                 └──────────┬───────────────┘
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │   OPE Suite v2        │
                       │   IPS / SNIPS / DR    │
                       │   Bootstrap CI        │
                       │   Weight Diagnostics  │
                       └───────────┬───────────┘
                                   ▼
                         📊 Streamlit Dashboard
                           src/app/app.py
```

---

## 📦 Project Structure

```
.
├── configs/                         # YAML configs (versioned)
│   ├── data_config.yaml
│   └── recbole.yaml
│
├── data/
│   └── sample/                      # ✅ Versioned sample package
│       ├── bandit_feedback_sample.npz
│       ├── split_manifest_sample.npz
│       └── metadata_sample.json
│
├── docs/
│   ├── PIDA.md                      # CRISP-DM project document
│   ├── data_dictionary.md
│   └── model_cards/
│       ├── recsys_baselines.md
│       ├── ope_suite.md
│       └── bandit_policy.md
│
├── reports/
│   ├── tables/                      # CSV + MD outputs
│   └── plots/                       # PNG plots
│
├── saved/                           # Model artifacts (not versioned)
│   └── reward_model/
│
├── src/
│   ├── app/                         # Streamlit dashboard
│   │   └── app.py
│   ├── bandits/                     # Bandit agents
│   │   ├── reward_model.py
│   │   ├── obd_sim_bandit_env.py
│   │   ├── train_tf_agents.py
│   │   └── export_policy_for_ope.py
│   ├── data/                        # Data pipeline
│   │   └── build_obd_datasets.py
│   ├── ope/                         # OPE Suite v2
│   │   └── run_ope_suite.py
│   └── recsys/                      # RecBole wrappers
│       └── train_recbole_baselines.py
│
├── tests/
│   └── data/                        # pytest tests
│
├── Makefile
└── pyproject.toml
```

---

## ⚡ Quickstart (Windows, no make required)

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run tests
python -m pytest tests/data -q
# Expected: ..................  18 passed

# 3. Launch dashboard
streamlit run src/app/app.py
# Opens at http://localhost:8501
```

---

## 🔄 Reproducibility — End-to-End (data/sample only)

```powershell
# Step 0: Sanity checks
python -c "
import numpy as np
d = np.load('data/sample/bandit_feedback_sample.npz')
s = np.load('data/sample/split_manifest_sample.npz')
print('context:', d['context'].shape)
print('n_actions:', d['action_context'].shape[0])
print('splits:', {k: len(s[k]) for k in ['train_idx','val_idx','test_idx']})
"
# Expected: context: (10000, 20), n_actions: 80, splits: {train:6300, val:700, test:3000}

# Step 1: RecBole Baselines
python -m src.recsys.train_recbole_baselines --config configs/recbole.yaml

# Step 2: OPE v2 (baselines)
python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42

# Step 3: Reward Model
python -m src.bandits.reward_model --seed 42

# Step 4: Bandit Training
python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1

# Step 5: Policy Export
python -m src.bandits.export_policy_for_ope --epsilon 0.1

# Step 6: OPE with TFAgent
python -m src.ope.run_ope_suite `
    --out reports/tables/ope_results_with_tf_agents.csv `
    --report reports/tables/ope_report_with_tf_agents.md `
    --external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv `
    --external-policy-name "TFAgent_eps0.1" `
    --n-bootstrap 200 --seed 42

# Step 7: Dashboard
streamlit run src/app/app.py
```

### Makefile Targets (if GNU make is available)

```bash
make test           # pytest tests/data -q
make ope            # OPE v2 baselines
make reward_model   # Reward model
make bandits        # Reward model + train agents + export policy
make ope_full       # OPE with TFAgent policy
make app            # Launch Streamlit dashboard
```

---

## ⚠️ TF-Agents / Keras 3 Incompatibility Note

`tf_agents 0.19.*` is incompatible with Keras 3 (`keras.__internal__` attribute error).
The bandit agents (LinUCB, EpsilonGreedy) are implemented in **pure numpy** in
`src/bandits/train_tf_agents.py` with mathematically equivalent algorithms.
See `docs/model_cards/bandit_policy.md` for details.

**Workaround** (if you want native TF-Agents):
```powershell
pip install "tensorflow==2.15.*" "tf-agents==0.19.*" "keras==2.15.*"
set TF_USE_LEGACY_KERAS=1
```

---

## 📊 Key Results (data/sample, test_idx n=3,000)

| Policy | IPS | SNIPS | DR |
|:-------|----:|------:|---:|
| on_policy_value (ground truth) | **0.003967** | — | — |
| RealizedAction (sanity) | ≈0.003967 ✅ | ≈0.003967 | ≈0.003967 |
| UniformRandom | ≈0.0001 | ≈0.0001 | ≈0.004 |
| RecboleTopK | ≈0.0000 | ≈0.0001 | ≈0.004 |
| TFAgent_eps0.1 | ≈0.0001 | ≈0.0001 | ≈0.004 |

**DR estimates are reliable**; IPS/SNIPS are near-zero for non-logging policies
due to support mismatch (~1% agreement between bandit and logging policy).

---

## 🧪 Tests

```powershell
python -m pytest tests/data -q
# ..................  18 passed
```

---

## 📋 Docs

| Document | Description |
|:---------|:------------|
| [PIDA.md](docs/PIDA.md) | CRISP-DM project document |
| [data_dictionary.md](docs/data_dictionary.md) | Variable definitions |
| [recsys_baselines.md](docs/model_cards/recsys_baselines.md) | Model card: RecBole |
| [ope_suite.md](docs/model_cards/ope_suite.md) | Model card: OPE Suite v2 |
| [bandit_policy.md](docs/model_cards/bandit_policy.md) | Model card: Bandit Agents |
