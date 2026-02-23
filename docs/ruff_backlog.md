# Ruff Linting Backlog

**Date**: 2026-02-22  
**Tool**: ruff 0.x  
**Remaining after style commits**: 18 errors

---

## Residual Issues

| Rule | Count | Files | Description | Recommendation |
|:-----|:-----:|:------|:------------|:---------------|
| `E501` | 18 | `src/data/build_obd_datasets.py`, `src/ope/run_ope_suite.py`, `src/app/app.py`, `src/recsys/prepare_recbole_dataset.py` | Line too long (> 100 chars) | Wrap manually or add `# noqa: E501` with justification for lines that are docstrings / f-strings where wrapping hurts readability |

---

## Why E501 Left in Backlog

The remaining E501 lines are primarily:

1. **Long f-strings / logger messages** — wrapping mid-string requires introducing
   string concatenation that adds visual noise without improving readability.
2. **Docstrings with long URLs or technical identifiers** — wrapping mid-URL breaks
   copy-paste usability.
3. **Matplotlib / st.info(...) parameter calls** — already wrapped where possible;
   remaining lines would require multiple extra lines with marginal clarity gain.

All 18 lines are **non-logic/non-semantic** — no correctness risk.

---

## Resolution Options

### Option A — noqa per-line (fastest)
```python
logger.warning("Very long message that exceeds 100 chars...")  # noqa: E501
```

### Option B — Increase line-length to 120 in pyproject.toml
```toml
[tool.ruff]
line-length = 120  # up from 100
```
This would resolve all 18 E501 at once with zero code change.  
**Risk**: black must also be updated: `black --line-length 120`

### Option C — Wrap each line manually (most effort, cleanest)
Wrap long strings with implicit concatenation or parenthesized expressions.

---

## Recommended Action

**Option B** (raise to 120) is preferred for a research/certification codebase where
long docstrings and diagnostic messages are common. Requires a single change to
`pyproject.toml` + re-running `black`.
