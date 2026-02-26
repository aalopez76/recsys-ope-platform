# Release v1.0.1 — Certification-grade RecSys OPE Platform

> **Date:** 2026-02-26  
> **Tag:** `v1.0.1`  
> **Commit:** `6cf9285` (`style: black format app module`)  
> **Branch:** `main`

---

## 🚀 Highlights

| # | Feature | Details |
|---|---------|---------|
| 1 | **`data/sample` SSOT + `split_manifest`** | Versioned 10K sample (80 actions, 20-dim context). Deterministic train/val/test splits via `split_manifest_sample.npz`. Single source of truth for all pipeline stages. |
| 2 | **RecBole Baselines** | Pop, BPR, NeuMF, LightGCN trained with `src/recsys/train_recbole_baselines.py`. Results exported as CSV + Markdown report. |
| 3 | **OPE Suite v2 (IPS / SNIPS / DR + Bootstrap + Diagnostics)** | `src/ope/run_ope_suite.py` — doubly-robust estimator with 200-sample bootstrap CIs, weight diagnostics (ESS, support overlap), and policy comparison table. |
| 4 | **Bandit Policy + OPE Integration** | LinUCB and ε-Greedy agents (`src/bandits/train_tf_agents.py`) implemented in pure NumPy. Policy export → OPE via `export_policy_for_ope.py`. Full E2E comparison. |
| 5 | **Streamlit Dashboard (4 pages)** | Interactive dashboard (`src/app/app.py`) with four pages via sidebar navigation: Overview (KPIs + architecture), RecBole Baselines (table + plot), OPE Suite (estimators + CI + diagnostics), Bandits & Policy (training summary + support analysis). |
| 6 | **Tests + CI (GitHub Actions)** | 18 pytest tests covering data pipeline integrity. CI runs on `ubuntu-latest` with Python 3.10 + 3.11 matrix: pytest, ruff lint, black format check. |

---

## 📋 CI Status

[![CI](https://github.com/aalopez76/recsys-ope-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aalopez76/recsys-ope-platform/actions/workflows/ci.yml)

- **Workflow:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
- **Runs on:** `ubuntu-latest`, Python 3.10 + 3.11
- **Steps:** `pytest tests/data -q` → `ruff check` → `black --check`
- **Local verification:** `python -m pytest tests/data -q` → **...................  18 passed** ✅

---

## 🔄 Reproducibility Commands

### PowerShell (Windows)

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# 0. Tests
python -m pytest tests/data -q
# Expected: ..................  18 passed

# 1. RecBole Baselines
python -m src.recsys.train_recbole_baselines --config configs/recbole.yaml

# 2. OPE v2 (baselines)
python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42

# 3. Reward Model
python -m src.bandits.reward_model --seed 42

# 4. Bandit Training
python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1

# 5. Policy Export
python -m src.bandits.export_policy_for_ope --epsilon 0.1

# 6. OPE with TFAgent (full E2E)
python -m src.ope.run_ope_suite `
    --out reports/tables/ope_results_with_tf_agents.csv `
    --report reports/tables/ope_report_with_tf_agents.md `
    --external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv `
    --external-policy-name "TFAgent_eps0.1" `
    --n-bootstrap 200 --seed 42

# 7. Streamlit Dashboard (4 pages)
streamlit run src/app/app.py
# Opens at http://localhost:8501
# Pages: Overview | RecBole Baselines | OPE Suite | Bandits & Policy
```

### Bash / GNU Make

```bash
source .venv/bin/activate
make test           # pytest tests/data -q
make ope            # OPE v2 baselines
make reward_model   # Reward model
make bandits        # Reward model + train agents + export policy
make ope_full       # OPE with TFAgent policy
make app            # Launch Streamlit dashboard
```

---

## 📊 Key Results (data/sample, test_idx n=3,000)

| Policy | IPS | SNIPS | DR |
|:-------|----:|------:|---:|
| on_policy_value (ground truth) | **0.003967** | — | — |
| RealizedAction (sanity check) | ≈0.003967 ✅ | ≈0.003967 | ≈0.003967 |
| UniformRandom | ≈0.0001 | ≈0.0001 | ≈0.004 |
| RecboleTopK | ≈0.0000 | ≈0.0001 | ≈0.004 |
| TFAgent_eps0.1 | ≈0.0001 | ≈0.0001 | ≈0.004 |

> **Note:** DR estimates are reliable. IPS/SNIPS approach zero for non-logging policies due to ~1% action support overlap — this is expected and documented in `docs/model_cards/ope_suite.md`.

---

## 📸 Dashboard Screenshots

### Overview — KPIs + Architecture
![Overview Dashboard](https://raw.githubusercontent.com/aalopez76/recsys-ope-platform/v1.0.1/docs/assets/overview.png)

### RecBole Baselines — Table + Plot
![RecBole Results](https://raw.githubusercontent.com/aalopez76/recsys-ope-platform/v1.0.1/docs/assets/recbole.png)

### OPE Suite + Bandits & Policy — Estimators + Diagnostics
![OPE + Bandits](https://raw.githubusercontent.com/aalopez76/recsys-ope-platform/v1.0.1/docs/assets/ope_bandits.png)

> **Manual screenshot instructions** (if regenerating):
> 1. Run `streamlit run src/app/app.py`
> 2. Navigate to `http://localhost:8501`
> 3. Capture each page and save to `docs/assets/` as `overview.png`, `recbole.png`, `ope_bandits.png`

---

## ⚠️ Known Issue: TF-Agents / Keras 3 Incompatibility

`tf_agents 0.19.*` is incompatible with Keras 3 (`keras.__internal__` attribute error).

**Resolution:** Bandit agents (LinUCB, ε-Greedy) are implemented in **pure NumPy** with mathematically equivalent algorithms. No TF-Agents dependency at runtime.

**Optional native TF-Agents workaround:**
```powershell
pip install "tensorflow==2.15.*" "tf-agents==0.19.*" "keras==2.15.*"
$env:TF_USE_LEGACY_KERAS = "1"
```

See [`docs/model_cards/bandit_policy.md`](model_cards/bandit_policy.md) for full details.

---

## 📚 Documentation Links

| Document | Description |
|:---------|:------------|
| [PIDA.md](PIDA.md) | CRISP-DM project document (problem, data, modeling, evaluation) |
| [data_dictionary.md](data_dictionary.md) | Variable definitions and schema |
| [model_cards/recsys_baselines.md](model_cards/recsys_baselines.md) | Model card: RecBole baselines |
| [model_cards/ope_suite.md](model_cards/ope_suite.md) | Model card: OPE Suite v2 |
| [model_cards/bandit_policy.md](model_cards/bandit_policy.md) | Model card: Bandit agents |
| [../README.md](../README.md) | Project README with architecture and quickstart |

---

## 📦 Files Changed in This Release

```
docs/RELEASE_NOTES.md          ← this file (NEW)
docs/assets/overview.png       ← Dashboard overview screenshot (NEW)
docs/assets/recbole.png        ← RecBole baselines screenshot (NEW)
docs/assets/ope_bandits.png    ← OPE + Bandits screenshot (NEW)
README.md                      ← CI badge added
```

---

*Generated: 2026-02-26 | Platform: RecSys OPE Platform | Certification: v1.0.1*
