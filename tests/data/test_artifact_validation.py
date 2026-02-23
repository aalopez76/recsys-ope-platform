import pytest

from src.data.build_obd_datasets import validate_required_artifacts


def test_validate_all_artifacts_present(tmp_path):
    """Test validation passes when all 7 artifacts exist."""
    paths = {
        "bandit": tmp_path / "bandit_feedback",
        "recbole": tmp_path / "recbole_atomic",
        "splits": tmp_path / "splits",
    }

    # Create directories
    paths["bandit"].mkdir(parents=True)
    paths["recbole"].mkdir(parents=True)
    paths["splits"].mkdir(parents=True)

    # Create all 7 required files
    (paths["bandit"] / "bandit_feedback.npz").touch()
    (paths["bandit"] / "metadata.json").touch()
    (paths["bandit"] / "action_id_map.json").touch()
    (paths["splits"] / "split_manifest.npz").touch()
    (paths["recbole"] / "obd_time.inter").touch()
    (paths["recbole"] / "obd.user").touch()
    (paths["recbole"] / "obd.item").touch()

    # Should not raise
    validate_required_artifacts(paths)


def test_validate_missing_artifact_raises(tmp_path):
    """Test validation raises RuntimeError when artifact missing."""
    paths = {
        "bandit": tmp_path / "bandit_feedback",
        "recbole": tmp_path / "recbole_atomic",
        "splits": tmp_path / "splits",
    }

    # Create directories but NOT all files
    paths["bandit"].mkdir(parents=True)
    paths["recbole"].mkdir(parents=True)
    paths["splits"].mkdir(parents=True)
    (paths["bandit"] / "metadata.json").touch()
    # Missing: bandit_feedback.npz, action_id_map.json, split_manifest, all recbole files

    with pytest.raises(RuntimeError, match="Artifact validation FAILED"):
        validate_required_artifacts(paths)
