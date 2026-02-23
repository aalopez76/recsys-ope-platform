"""
Schema validation module for OBD dataset.
Enforces strict validaton rules on raw and processed data.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Schema validator for OBD bandits and RecBole data."""

    def __init__(self, config: dict[str, Any]):
        """Initialize validator with configuration.

        Args:
            config: Validation configuration dictionary
        """
        self.config = config
        self.pscore_min = config.get("pscore", {}).get("min", 1e-6)
        self.pscore_max = config.get("pscore", {}).get("max", 1.0)
        self.pscore_action = config.get("pscore", {}).get("action", "clip")
        self.allowed_positions = set(config.get("position", {}).get("allowed_values", {1, 2, 3}))
        self.reindex_items = config.get("item_id", {}).get("reindex", True)

    def validate_bandit_feedback(self, bandit_feedback: dict[str, Any]) -> dict[str, Any]:
        """Validate and fix bandit feedback data.

        Args:
            bandit_feedback: Dictionary with raw feedback

        Returns:
            Dict: Validated (and potentially fixed) feedback dictionary
        """
        logger.info("Validating bandit feedback schema...")

        # 1. Pscore Validation
        pscores = bandit_feedback["pscore"]
        invalid_pscores = (pscores < self.pscore_min) | (pscores > self.pscore_max)
        n_invalid = np.sum(invalid_pscores)

        if n_invalid > 0:
            logger.warning(
                f"Found {n_invalid} invalid propensity scores (range [{self.pscore_min}, {self.pscore_max}])."
            )
            if self.pscore_action == "clip":
                logger.info("Clipping invalid pscores...")
                bandit_feedback["pscore"] = np.clip(pscores, self.pscore_min, self.pscore_max)
            elif self.pscore_action == "drop":
                logger.info("Dropping rows with invalid pscores...")
                valid_mask = ~invalid_pscores
                for key in ["context", "action", "reward", "position", "pscore"]:
                    bandit_feedback[key] = bandit_feedback[key][valid_mask]
                # Note: 'action_context' stays same as it's indexed by action ID, not round
            else:
                raise ValueError(
                    f"Invalid pscores found and action '{self.pscore_action}' is not supported."
                )

        # 2. Position Validation
        positions = bandit_feedback["position"]
        unique_positions = set(np.unique(positions))
        if not unique_positions.issubset(self.allowed_positions):
            raise ValueError(
                f"Invalid positions found: {unique_positions - self.allowed_positions}. Allowed: {self.allowed_positions}"
            )

        # 3. Item ID (Action) Validation
        actions = bandit_feedback["action"]
        n_actions = bandit_feedback["n_actions"]

        if np.min(actions) < 0 or np.max(actions) >= n_actions:
            logger.warning(
                f"Action IDs out of range [0, {n_actions-1}]. Min: {np.min(actions)}, Max: {np.max(actions)}"
            )
            if self.reindex_items:
                logger.info("Reindexing actions to [0, n_actions-1]...")
                unique_actions = np.unique(actions)
                action_map = {old: new for new, old in enumerate(unique_actions)}
                bandit_feedback["action"] = np.array([action_map[a] for a in actions])
                bandit_feedback["n_actions"] = len(unique_actions)

                # Reindex action context if present
                if "action_context" in bandit_feedback:
                    old_context = bandit_feedback["action_context"]
                    # Create new context matrix sorted by new index
                    new_context = np.zeros((len(unique_actions), old_context.shape[1]))
                    for old, new in action_map.items():
                        if old < len(old_context):
                            new_context[new] = old_context[old]
                    bandit_feedback["action_context"] = new_context

                # Store map in feedback for saving later
                bandit_feedback["action_id_map"] = action_map
            else:
                raise ValueError("Action IDs out of range and reindex=False.")

        logger.info("Bandit feedback schema validation passed.")
        return bandit_feedback

    def validate_recbole_inter(self, df: pd.DataFrame) -> None:
        """Validate RecBole interaction DataFrame.

        Args:
            df: DataFrame with columns [user_id, item_id, rating, timestamp]
        """
        logger.info("Validating RecBole interactions...")
        required_cols = ["user_id", "item_id", "rating", "timestamp"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in RecBole df. Found: {df.columns}")

        # Check rating (click) is binary-like float
        unique_ratings = df["rating"].unique()
        if not set(unique_ratings).issubset({0.0, 1.0}):
            logger.warning(f"Ratings (clicks) contain non-binary values: {unique_ratings}")

        # Check timestamp is float/int
        if not pd.api.types.is_numeric_dtype(df["timestamp"]):
            raise ValueError("Timestamp column must be numeric.")

        logger.info("RecBole interaction validation passed.")
