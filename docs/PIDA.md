# PIDA — Proyecto Integrador de Aplicación de Datos

**Título**: Plataforma de Evaluación de Políticas de Recomendación con Off-Policy Evaluation  
**Alumno**: [Nombre]  
**Fecha**: 2026-02-22  
**Framework**: CRISP-DM  
**Status**: ✅ Certificación-Grade

---

## 1. Business Understanding

### 1.1 Contexto y Problema

Los sistemas de recomendación industriales deben tomar decisiones en tiempo real sobre qué ítems mostrar al usuario (acción) basándose en su contexto (features de sesión, historial). El reto principal es que los datos históricos fueron generados por una **política de logging** diferente a la política que se quiere evaluar, lo que impide evaluar candidatos directamente en producción sin un A/B test costoso.

**Problema central**: ¿Cómo comparar políticas de recomendación (RecBole collaboratve filtering vs. bandits contextual) de manera **offline, auditable y sin sesgo**, usando únicamente datos históricos de logging bandit?

### 1.2 Objetivo

Construir una plataforma E2E que:
1. Entrene baselines de filtrado colaborativo (RecBole: Pop, BPR, NeuMF, LightGCN).
2. Entrene agentes bandit contextual (LinUCB, Epsilon-Greedy).
3. Evalúe todas las políticas con **Off-Policy Evaluation** (OBP: IPS, SNIPS, DR) con bootstrap CI y diagnósticos de pesos.
4. Exponga resultados en un dashboard Streamlit auditable.

### 1.3 Criterio de Éxito

| Componente | Criterio |
|:-----------|:---------|
| RecBole Baselines | NDCG@10 > baseline random, entrenamiento reproducible |
| OPE IPS sanity check | IPS(RealizedAction) ≈ on_policy_value (test) |
| Weight diagnostics | ESS, p99, %clipped documentados por política |
| Bandit agents | 5,000 pasos entrenados, rolling avg documentada |
| Dashboard | 4 secciones funcionales, fallback en artefactos faltantes |
| Reproducibilidad | `pytest tests/data -q` → 18 passed, seed=42 determinista |

---

## 2. Data Understanding

### 2.1 Dataset: Open Bandit Dataset (OBD) — Sample Package

| Campo | Descripción | Dimensión |
|:------|:------------|----------:|
| `context` | Vectores de contexto de usuario/sesión | (n_rounds, 20) |
| `action` | Ítem recomendado (índice 0..79) | (n_rounds,) |
| `reward` | Click binario (0/1) | (n_rounds,) |
| `pscore` | Propensity score del logging policy | (n_rounds,) |
| `action_context` | Features de ítem | (80, 5) |
| `position` | Posición de presentación | (n_rounds,) |

**Splits (SSOT: split_manifest_sample.npz)**:

| Split | n_rounds | % |
|:------|--------:|--:|
| train | 6,300 | 63% |
| val | 700 | 7% |
| test | 3,000 | 30% |

### 2.2 Riesgos del Dataset

| Riesgo | Descripción | Mitigación |
|:-------|:------------|:-----------|
| **Sparsity** | CTR real ~0.4% — muy pocas interacciones positivas | Filtrado estricto click=1 en RecBole |
| **Support/Coverage** | Política logging cubre sólo ~80 ítems; bandits targets a ítems distintos | Uso de IPS clipping + diagnósticos ESS |
| **Position Bias** | Items en posición 0 tienen mayor CTR observado | Campo `position` preservado; no modelado en v1 |
| **Spill-over** | Pscore no observado directamente (estimado) | Usar clips [0.001, 0.01, 0.05] y reportar sensibilidad |

### 2.3 Privacidad

- El dataset OBD no contiene PII directa.
- Los vectores de contexto de 20d son embeddings anonimizados.
- Re-identificación risk: bajo (embeddings sintéticos del sample).

---

## 3. Data Preparation

### 3.1 Pipeline de Preparación

```
data/sample/ (input — solo este directorio)
│
├── bandit_feedback_sample.npz    ← raw OBD sample (n=10k)
├── split_manifest_sample.npz     ← SSOT: train/val/test indices
└── metadata_sample.json          ← n_rounds, n_actions, dims
         │
         ▼
src/data/build_obd_datasets.py    ← parseo, validación schema, fingerprint
         │
         ├─ data/sample/recbole_atomic/  ← archivos atómicos RecBole
         │    ├── obd.user (.uid, c0..c19)
         │    ├── obd.item (.iid, i0..i4)
         │    └── obd.inter (.uid, .iid, click)
         └─ data/sample/action_id_map.json
```

### 3.2 Artefactos Clave

| Artefacto | Ruta | Descripción |
|:----------|:-----|:------------|
| Bandit Feedback | `data/sample/bandit_feedback_sample.npz` | NPZ: context, action, reward, pscore |
| Split Manifest | `data/sample/split_manifest_sample.npz` | SSOT de indices train/val/test |
| Metadata | `data/sample/metadata_sample.json` | n_rounds=10k, n_actions=80, dims |
| RecBole Atomic | `data/sample/recbole_atomic/` | Formato InteractionDataset de RecBole |
| Action Map | `data/sample/action_id_map.json` | Mapeo item_id → action_idx |

### 3.3 Política de Versionado

- **Versionado**: sólo `data/sample/` se versiona en git.
- **data/raw/**, **data/recbole_atomic_large/**: excluidos por `.gitignore`.
- Todos los outputs de modelos (`saved/`) y reports son regenerables.

---

## 4. Modeling

### 4.1 RecBole Collaborative Filtering Baselines

**Modelos entrenados**: Pop, BPR, NeuMF, LightGCN  
**Script**: `src/recsys/train_recbole_baselines.py`  
**Config**: `configs/recbole_config.yaml`  
**Dataset**: `data/sample/recbole_atomic/` — split por `split_manifest_sample.npz`.

**Métricas evaluadas**: NDCG@10, Recall@10, Precision@10, MRR@10, HR@10.

```powershell
python -m src.recsys.train_recbole_baselines --config configs/recbole_config.yaml
```

### 4.2 OPE Suite v2

**Estimadores**: IPS, SNIPS, DR  
**Script**: `src/ope/run_ope_suite.py`  
**Clips**: [0.001, 0.01, 0.05]  
**Bootstrap**: 200 resamples, seed=42

**Políticas evaluadas**:
| Política | Tipo | Propósito |
|:---------|:-----|:----------|
| RealizedAction (diagnostic) | Diagnóstico | IPS ≈ on_policy_value (sanity check) |
| UniformRandom | Baseline | Lower bound de performance |
| RecboleTopK | Candidato | Softmax(CTR, T=0.1) sobre train_idx |
| TFAgent_eps0.1 | Candidato | Epsilon-greedy derivada del reward model |

**Diagnósticos de pesos**: ESS, w_p50/90/99/max, %pscore<clip, %rounds_clipped.

```powershell
python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42
```

### 4.3 Bandit Agents

**Reward Model** (simulador de entorno):
- `LogisticRegression` (sklearn, C=1.0, balanced)
- Features: concat(context[20d], action_context[5d]) = 25d
- Entrenar SOLO sobre `train_idx`.

**Agentes entrenados**:
- **LinUCB**: Ridge regression disjoint + UCB exploration (alpha=1.0)
- **EpsilonGreedy**: ε=0.1 greedy sobre predicciones del reward model

> **Nota técnica**: TF-Agents 0.19 / Keras 3 tienen una incompatibilidad
> (`keras.__internal__` attribute error). Los algoritmos se reimplementaron
> en numpy puro con lógica matemáticamente equivalente.

**Policy Export**: Política epsilon-greedy exportada con probabilidades por round en `test_idx`.

```powershell
python -m src.bandits.reward_model
python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1
python -m src.bandits.export_policy_for_ope --epsilon 0.1
```

---

## 5. Evaluation

### 5.1 Resultados Clave

**OPE on_policy_value (test)**: ≈0.003967 (CTR real en test split).

**IPS Sanity Check**: `IPS(RealizedAction, clip=0.01) ≈ on_policy_value` → ✅ PASS.

**Comparación de políticas** (clip=0.01, estimador IPS):

| Política | IPS | SNIPS | DR |
|:---------|----:|------:|---:|
| RealizedAction | ≈0.0040 | ≈0.0040 | ≈0.0040 |
| UniformRandom | ≈0.0001 | ≈0.0001 | ≈0.0040 |
| RecboleTopK | ≈0.0000 | ≈0.0001 | ≈0.0040 |
| TFAgent_eps0.1 | ≈0.0001 | ≈0.0001 | ≈0.0040 |

### 5.2 Interpretación

**Support Problem**: El agente TFAgent selecciona ítems con alto CTR predicho, pero el logging policy seleccionó items casi uniformemente. Acuerdo entre políticas: ~1%. Por esto:
- `IPS/SNIPS(TFAgent)` ≈ 0 — los pesos de importancia son casi cero para la mayoría de rounds.
- `DR(TFAgent)` ≈ on_policy_value — el término de modelo directo (DM) domina, estimando el valor esperado global.

**Conclusión**: Para evaluación offline precisa de una política concentrada, se necesita:
1. Una política de logging con soporte explícito over las acciones del target.
2. Usar DR con un reward model fuerte.
3. Recolectar datos on-policy (A/B test) para validación final.

---

## 6. Deployment (MVP)

### 6.1 Dashboard Streamlit

```powershell
# Activar entorno
.venv\Scripts\activate

# Ejecutar dashboard
streamlit run src/app/app.py
```

El dashboard expone 4 secciones:
1. **Overview** — KPIs, arquitectura, artefactos.
2. **RecBole Baselines** — métricas, gráficos NDCG.
3. **OPE Suite** — filtros interactivos, CI plots, weight diagnostics.
4. **Bandits & Policy** — distribución de pi_e, comparación OPE, conclusión de support.

### 6.2 Pipeline Completo End-to-End

```powershell
# 1. Data (si recbole_atomic no existe)
python -m src.data.build_obd_datasets --mode sample

# 2. RecBole Baselines
python -m src.recsys.train_recbole_baselines --config configs/recbole_config.yaml

# 3. OPE v2 baseline
python -m src.ope.run_ope_suite --n-bootstrap 200 --seed 42

# 4. Reward Model
python -m src.bandits.reward_model --seed 42

# 5. Bandit Agents
python -m src.bandits.train_tf_agents --steps 5000 --seed 42 --epsilon 0.1

# 6. Policy Export
python -m src.bandits.export_policy_for_ope --epsilon 0.1

# 7. OPE con TFAgent
python -m src.ope.run_ope_suite \
    --out reports/tables/ope_results_with_tf_agents.csv \
    --report reports/tables/ope_report_with_tf_agents.md \
    --external-policy-csv reports/tables/tf_agents_policy_test_action_prob.csv \
    --external-policy-name "TFAgent_eps0.1" --n-bootstrap 200 --seed 42

# 8. Dashboard
streamlit run src/app/app.py
```

---

## 7. Risks & Governance

| Riesgo | Severidad | Mitigación |
|:-------|:---------:|:-----------|
| **CTR Sparsity** | Alta | Upsampling para RecBole; OPE sobre sample original |
| **Support/Coverage** | Alta | Diagnósticos ESS + %pscore<clip; documentar cuando IPS es inválido |
| **Position Bias** | Media | Campo `position` preservado; future work: modelar posición |
| **Data Leakage** | Alta | Reward model entrenado SOLO sobre `train_idx`; OPE sobre `test_idx` |
| **Reproducibilidad** | Media | Seeds fijas (42), configs versionadas, split_manifest SSOT |
| **TF-Agents API** | Baja | Reimplementación numpy equivalente; documentada en model card |
| **Fairness** | Media | No se auditan sesgos por grupo demográfico (future work) |

---

## 8. Reproducibility Checklist

```powershell
# Tests
python -m pytest tests/data -q
# Expected: ..................  18 passed

# Verificar artefactos data/sample
python -c "
import numpy as np
d = np.load('data/sample/bandit_feedback_sample.npz')
s = np.load('data/sample/split_manifest_sample.npz')
print('context:', d['context'].shape)
print('n_actions:', d['action_context'].shape[0])
print('splits:', {k: len(s[k]) for k in ['train_idx','val_idx','test_idx']})
"
# Expected:
# context: (10000, 20)
# n_actions: 80
# splits: {'train_idx': 6300, 'val_idx': 700, 'test_idx': 3000}

# OPE sanity check
python -c "
import numpy as np, pandas as pd
df = pd.read_csv('reports/tables/ope_results_with_tf_agents.csv')
ra = df[(df.policy_name.str.startswith('Realized')) & (df.estimator=='IPS') & (df.clip==0.01)]
print('Sanity check IPS(RealizedAction):', ra.value_hat.values[0])
print('on_policy_value:', ra.on_policy_value.values[0])
"
```
