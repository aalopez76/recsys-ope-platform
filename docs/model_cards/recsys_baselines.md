# Model Card: RecSys Collaborative Filtering Baselines

**Version**: 1.0  
**Date**: 2026-02-22  
**Status**: ✅ Certified

---

## Intended Use

Provide **offline collaborative filtering baselines** for item recommendation.
Used as reference policies in OPE evaluation to establish performance bounds
against which contextual bandit agents are compared.

**Primary use**: Research / certification — not for production deployment without A/B validation.  
**Out-of-scope**: Cold-start users/items, real-time serving, incremental updates.

---

## Models

| Model | Type | Reference |
|:------|:-----|:---------|
| **Pop** | Popularity-based baseline | Non-personalized frequency count |
| **BPR** | Bayesian Personalized Ranking (MF) | Rendle et al., 2009 |
| **NeuMF** | Neural Matrix Factorization | He et al., 2017 |
| **LightGCN** | Light Graph Convolutional Network | He et al., 2020 |

---

## Input Data

| Artifact | Description |
|:---------|:------------|
| `data/sample/recbole_atomic/obd.inter` | User-item click interactions (click=1 only) |
| `data/sample/recbole_atomic/obd.user` | User features (c0..c19, float) |
| `data/sample/recbole_atomic/obd.item` | Item features (i0..i4, float) |
| `data/sample/split_manifest_sample.npz` | Train/val/test SSOT |
| `configs/recbole_config.yaml` | RecBole training configuration |

**Schema**: `user_id`, `item_id`, `click:float`  
**Context dim**: 20 (user), 5 (item)

---

## Metrics

**Evaluation**: top-N recommendation metrics at K=10.

| Metric | Description |
|:-------|:------------|
| NDCG@10 | Normalized Discounted Cumulative Gain |
| Recall@10 | Fraction of relevant items retrieved |
| Precision@10 | Fraction of retrieved items that are relevant |
| HR@10 | Hit Rate at 10 |
| MRR@10 | Mean Reciprocal Rank |

**Results**: See `reports/tables/recbole_baselines_sample.csv`

---

## Training Protocol

1. Data prepared with `src/data/build_obd_datasets.py`.
2. Strict filtering: `click=1` only for RecBole interactions.
3. Train/test split from `split_manifest_sample.npz` (SSOT).
4. Config: `configs/recbole_config.yaml` (epochs, embedding_size, lr).
5. Seed: 42 for all random operations.

---

## Limitations & Risks

| Limitation | Detail |
|:-----------|:-------|
| **Data Sparsity** | CTR ~0.4% — few positive interactions |
| **Cold Start** | Only warm users/items in evaluation |
| **Position Bias** | Click data affected by logging policy's position assignment |
| **OPE Alignment** | Softmax(CTR) policy used for OPE; direct RecBole ranking not OPE-compatible |

---

## Repro Commands

```powershell
python -m src.recsys.train_recbole_baselines --config configs/recbole_config.yaml
Get-Content reports/tables/recbole_baselines_sample.csv
```
