"""
Cross-platform artifact cleaner for the RecSys OPE project.

Removes ONLY regenerable artifacts (data pipeline outputs, report outputs).
Preserves versioned sample, configs, docs, and source code.

Usage:
    python scripts/clean_artifacts.py --dry-run   # Preview what would be deleted
    python scripts/clean_artifacts.py --execute   # Actually delete
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).parent.parent

# Directories whose CONTENTS are regenerable and safe to delete
CLEAN_TARGETS = [
    "data/bandit_feedback",
    "data/recbole_atomic",
    "data/splits",
    "data/interim",
]

# Files in these dirs to delete (only non-gitkeep contents)
CLEAN_OUTPUT_DIRS = [
    "reports/plots",
    "reports/tables",
]

# Regenerable report files (not gitkeeps)
CLEAN_FILES = [
    "reports/data_stats.md",
]

# NEVER touch these
PRESERVED = [
    "data/sample",
    "data/raw",
    "configs",
    "docs",
    "src",
    "tests",
    "scripts",
]


def collect_targets(base: Path) -> List[Path]:
    """Collect all paths that would be cleaned."""
    targets: List[Path] = []

    for rel in CLEAN_TARGETS:
        d = base / rel
        if d.exists():
            for item in sorted(d.iterdir()):
                targets.append(item)

    for rel in CLEAN_OUTPUT_DIRS:
        d = base / rel
        if d.exists():
            for item in sorted(d.iterdir()):
                if item.name != ".gitkeep":
                    targets.append(item)

    for rel in CLEAN_FILES:
        f = base / rel
        if f.exists():
            targets.append(f)

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clean_artifacts",
        description="Remove regenerable artifacts (preserves data/sample, configs, docs, src).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files that would be deleted (no changes made)",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete regenerable artifacts",
    )
    args = parser.parse_args()

    targets = collect_targets(BASE_DIR)

    if not targets:
        print("Nothing to clean — no regenerable artifacts found.")
        return

    if args.dry_run:
        print(f"[DRY-RUN] Would delete {len(targets)} item(s):\n")
        for t in targets:
            kind = "DIR " if t.is_dir() else "FILE"
            print(f"  {kind}  {t.relative_to(BASE_DIR)}")
        print(f"\nPreserved (untouched): {', '.join(PRESERVED)}")
        print("Run with --execute to delete.")
    else:
        deleted = 0
        for t in targets:
            try:
                if t.is_dir():
                    shutil.rmtree(t)
                else:
                    t.unlink()
                print(f"  DELETED  {t.relative_to(BASE_DIR)}")
                deleted += 1
            except OSError as e:
                print(f"  ERROR    {t.relative_to(BASE_DIR)}: {e}", file=sys.stderr)
        print(f"\nCleaned {deleted}/{len(targets)} item(s).")
        print(f"Preserved: {', '.join(PRESERVED)}")


if __name__ == "__main__":
    main()
