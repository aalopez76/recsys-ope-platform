# Contributing to RecSys OPE Platform

Thank you for contributing! Please follow these guidelines to keep the project clean and reproducible.

## Running Tests

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/data -q
# Expected: ..................  18 passed
```

```bash
# bash / macOS / Linux
source .venv/bin/activate
python -m pytest tests/data -q
```

## Code Style

- **Formatter**: `black` (line-length=100)
- **Linter**: `ruff` (see `pyproject.toml` for rules)
- **Types**: all public functions must have type hints
- **Docstrings**: minimum one-liner per module and public function

```powershell
ruff check src/ tests/ --fix
black src/ tests/
```

## Adding a New Feature

1. Branch from `main`: `git checkout -b feat/<scope>-<short-description>`
2. Implement with tests in `tests/`
3. Update `docs/` if architecture changes
4. Update `CHANGELOG.md` under `## [Unreleased]`
5. Open a PR against `main`

## PR Checklist

- [ ] `pytest tests/data -q` passes with no failures
- [ ] No large files committed (`data/raw/`, `saved/`, `reports/*.csv`)
- [ ] `ruff` and `black` pass without errors
- [ ] CHANGELOG.md updated
- [ ] Docs updated if needed

## Data Policy

- **Only** `data/sample/` may be committed.
- Never commit `data/raw/`, `data/raw_large/`, `data/splits/`, `saved/`.
- Regenerable reports (`reports/tables/*.csv`, `reports/plots/*.png`) are excluded by `.gitignore`.
