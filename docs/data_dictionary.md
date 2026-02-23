# Data Dictionary

## Open Bandit Dataset (OBD)

### Source Files
Located in `data/raw/obd/`

### Raw Format
The OBD dataset contains logged bandit feedback from ZOZOTOWN fashion e-commerce.

#### Fields

| Field | Type | Description | Range/Values |
|-------|------|-------------|--------------|
| **context** | float array | User context features (hashed for privacy) | 20-dimensional |
| **action** | int | Recommended item ID (arm) | [0, n_actions) |
| **reward** | int | Click indicator (1=clicked, 0=not clicked) | {0, 1} |
| **position** | int | Display position on interface | {1, 2, 3} |
| **pscore** | float | Propensity score (probability of recommendation) | [0, 1] |
| **action_context** | float array | Item feature vectors | 5-dimensional |

---

## RecBole Atomic Format

### Output Files
Located in `data/recbole_atomic/`

#### obd.inter (Interaction File)
Tab-separated file with typed header.
| Field | Type | Description | Transformation |
|-------|------|-------------|----------------|
| **user_id** | token | User identifier | Created via hash/KMeans clustering on context |
| **item_id** | token | Item identifier | Directly from OBD action field (consistent ID) |
| **rating** | float | Interaction label | Directly from OBD reward (0 or 1) |
| **timestamp** | float | Temporal ordering | Sequential index (0 to n_rounds-1) |

**Format**: `user_id:token	item_id:token	rating:float	timestamp:float	position:float`

#### obd.user (User Features)
User features deduplicated by `user_id`.
**Format**: `user_id:token  c0:float  c1:float  c2:float  ...  c19:float`  (20 features)

#### obd.item (Item Features)
Item features indexed by `item_id`.
**Format**: `item_id:token	i0:float	i1:float	i2:float	i3:float	i4:float`

---

## OBP Bandit Feedback Format

### Output Files
Located in `data/bandit_feedback/`

#### bandit_feedback.npz

Compressed NumPy archive with arrays:

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| **context** | (n_rounds, 20) | float32 | User context features (20-dimensional) |
| **action** | (n_rounds,) | int32 | Selected actions (item IDs) |
| **reward** | (n_rounds,) | float32 | Observed rewards (clicks) |
| **position** | (n_rounds,) | int32 | Display positions |
| **pscore** | (n_rounds,) | float32 | Propensity scores |
| **action_context** | (n_actions, 5) | float32 | Item feature matrix |

#### metadata.json

JSON file with dataset metadata:
- `n_rounds`: Total number of logged interactions
- `n_actions`: Total number of items (arms)
- `context_dim`: Dimension of user context features (20)
- `action_context_dim`: Dimension of item features (5)
- `behavior_policy`: Logging policy used ("random")
- `campaign`: Dataset campaign ("all")

---

## Privacy Notes

- **User features** are hashed and anonymized (no PII)
- **User IDs** in RecBole format are synthetic (created via clustering)
- **Original user identities** are not recoverable

---

## Transformation Summary

```
OBD Raw Data
    |
    +--> Hash/KMeans on context (20-dim) --> user_id
    +--> action                          --> item_id
    +--> reward                          --> rating (RecBole) / reward (OBP)
    +--> sequential index                --> timestamp
    +--> preserve pscore                     for OPE
```

---

## References

- [Open Bandit Dataset](https://research.zozo.com/data.html)
- [RecBole Atomic Format](https://recbole.io/docs/user_guide/data/atomic_files.html)
- [OBP Documentation](https://zr-obp.readthedocs.io/)

