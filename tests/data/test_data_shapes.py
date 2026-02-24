"""
Test data shapes, dimensions, and schema consistency.

Validates that data/sample/ artifacts are consistent with
metadata_sample.json and RecBole atomic format expectations.
All tests use only the versioned sample package (data/sample/).
"""

import json
from pathlib import Path

import numpy as np
import pytest

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metadata() -> dict:
    """Load metadata_sample.json."""
    path = SAMPLE_DIR / "metadata_sample.json"
    assert path.exists(), f"Missing versioned artifact: {path}"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bandit_feedback() -> dict:
    """Load bandit_feedback_sample.npz arrays."""
    path = SAMPLE_DIR / "bandit_feedback_sample.npz"
    assert path.exists(), f"Missing versioned artifact: {path}"
    return dict(np.load(path, allow_pickle=True))


@pytest.fixture(scope="module")
def split_manifest() -> dict:
    """Load split_manifest_sample.npz arrays."""
    path = SAMPLE_DIR / "split_manifest_sample.npz"
    assert path.exists(), f"Missing versioned artifact: {path}"
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# P2-1a: bandit_feedback shapes
# ---------------------------------------------------------------------------


class TestBanditFeedbackShapes:
    """Validate bandit_feedback_sample.npz dimensions."""

    def test_context_dim_is_20(self, bandit_feedback: dict) -> None:
        assert (
            bandit_feedback["context"].shape[1] == 20
        ), f"Expected context_dim=20, got {bandit_feedback['context'].shape[1]}"

    def test_action_context_dim_is_5(self, bandit_feedback: dict) -> None:
        assert (
            bandit_feedback["action_context"].shape[1] == 5
        ), f"Expected action_context_dim=5, got {bandit_feedback['action_context'].shape[1]}"

    def test_n_rounds_matches_metadata(self, bandit_feedback: dict, metadata: dict) -> None:
        n_rounds = metadata["n_rounds"]
        assert bandit_feedback["reward"].shape[0] == n_rounds, (
            f"reward has {bandit_feedback['reward'].shape[0]} rows, " f"metadata says n_rounds={n_rounds}"
        )

    def test_context_rows_match_n_rounds(self, bandit_feedback: dict, metadata: dict) -> None:
        n_rounds = metadata["n_rounds"]
        assert bandit_feedback["context"].shape[0] == n_rounds


# ---------------------------------------------------------------------------
# P2-1b: split_manifest consistency
# ---------------------------------------------------------------------------


class TestSplitManifestConsistency:
    """Validate split_manifest_sample.npz vs metadata split_counts."""

    def test_split_keys_exist(self, split_manifest: dict) -> None:
        for key in ("train_idx", "val_idx", "test_idx"):
            assert key in split_manifest, f"Missing split key: {key}"

    def test_total_equals_n_rounds(self, split_manifest: dict, metadata: dict) -> None:
        total = len(split_manifest["train_idx"]) + len(split_manifest["val_idx"]) + len(split_manifest["test_idx"])
        assert total == metadata["n_rounds"], f"Split total {total} != n_rounds {metadata['n_rounds']}"

    def test_split_counts_match_metadata(self, split_manifest: dict, metadata: dict) -> None:
        sc = metadata["split_counts"]
        assert len(split_manifest["train_idx"]) == sc["n_train"]
        assert len(split_manifest["val_idx"]) == sc["n_val"]
        assert len(split_manifest["test_idx"]) == sc["n_test"]


# ---------------------------------------------------------------------------
# P2-1c: RecBole atomic sample headers
# ---------------------------------------------------------------------------


class TestRecBoleAtomicSample:
    """Validate RecBole atomic sample files exist and have correct headers."""

    def test_obd_time_inter_exists_and_has_header(self) -> None:
        path = SAMPLE_DIR / "obd_time_sample.inter"
        assert path.exists(), f"Missing versioned artifact: {path}"
        header = path.read_text(encoding="utf-8").split("\n")[0].strip()
        assert "user_id:token" in header, f"Invalid header: {header}"
        assert "item_id:token" in header, f"Invalid header: {header}"

    def test_obd_user_has_20_feature_columns(self) -> None:
        path = SAMPLE_DIR / "obd_sample.user"
        assert path.exists(), f"Missing versioned artifact: {path}"
        header = path.read_text(encoding="utf-8").split("\n")[0].strip()
        cols = header.split("\t")
        # user_id + c0..c19 = 21 columns
        assert len(cols) == 21, f"Expected 21 columns (user_id + c0..c19), got {len(cols)}: {cols}"
        feature_cols = [c for c in cols if c.startswith("c") and ":float" in c]
        assert len(feature_cols) == 20, f"Expected 20 feature columns c0..c19, got {len(feature_cols)}"

    def test_obd_item_has_5_feature_columns(self) -> None:
        path = SAMPLE_DIR / "obd_sample.item"
        assert path.exists(), f"Missing versioned artifact: {path}"
        header = path.read_text(encoding="utf-8").split("\n")[0].strip()
        cols = header.split("\t")
        # item_id + i0..i4 = 6 columns
        assert len(cols) == 6, f"Expected 6 columns (item_id + i0..i4), got {len(cols)}: {cols}"
        feature_cols = [c for c in cols if c.startswith("i") and ":float" in c]
        assert len(feature_cols) == 5, f"Expected 5 feature columns i0..i4, got {len(feature_cols)}"
