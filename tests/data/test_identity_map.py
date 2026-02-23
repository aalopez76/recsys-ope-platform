from src.data.build_obd_datasets import is_identity_action_map


def test_identity_map_with_int_keys():
    """Test identity map validation with integer keys."""
    action_map = {0: 0, 1: 1, 2: 2, 3: 3}
    assert is_identity_action_map(action_map, 4) == True


def test_identity_map_with_str_keys():
    """Test identity map validation with string keys (JSON format)."""
    action_map = {"0": 0, "1": 1, "2": 2, "3": 3}
    assert is_identity_action_map(action_map, 4) == True


def test_non_identity_map():
    """Test non-identity map detection."""
    action_map = {0: 2, 1: 0, 2: 1, 3: 3}  # Reindexed
    assert is_identity_action_map(action_map, 4) == False


def test_incomplete_map():
    """Test map with missing keys."""
    action_map = {0: 0, 1: 1, 2: 2}  # Missing key 3
    assert is_identity_action_map(action_map, 4) == False
