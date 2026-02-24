import numpy as np

from src.data.build_obd_datasets import generate_split_manifest


def test_split_manifest_reproducible_hash(tmp_path):
    """Test that split manifest generates identical hash across runs."""
    config = {"dataset": {"split_strategy": "time", "test_size": 0.3, "val_size": 0.1, "random_state": 42}}

    paths = {"splits": tmp_path / "splits", "bandit": tmp_path / "bandit"}
    paths["splits"].mkdir(parents=True)
    paths["bandit"].mkdir(parents=True)

    # First run
    result1 = generate_split_manifest(1000, config, paths)
    hash1 = result1["split_manifest_hash"]

    # Second run (should be identical)
    result2 = generate_split_manifest(1000, config, paths)
    hash2 = result2["split_manifest_hash"]

    assert hash1 == hash2, "Split manifest hash not reproducible"


def test_split_manifest_deterministic_indices(tmp_path):
    """Test that split indices are deterministic."""
    config = {"dataset": {"split_strategy": "time", "test_size": 0.3, "val_size": 0.1, "random_state": 42}}

    paths = {"splits": tmp_path / "splits", "bandit": tmp_path / "bandit"}
    paths["splits"].mkdir(parents=True)
    paths["bandit"].mkdir(parents=True)

    # Generate twice
    generate_split_manifest(1000, config, paths)

    # Load twice
    manifest_path = paths["splits"] / "split_manifest.npz"
    data1 = np.load(manifest_path)
    data2 = np.load(manifest_path)

    assert np.array_equal(data1["train_idx"], data2["train_idx"])
    assert np.array_equal(data1["val_idx"], data2["val_idx"])
    assert np.array_equal(data1["test_idx"], data2["test_idx"])
