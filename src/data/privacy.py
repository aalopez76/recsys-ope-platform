"""
Privacy compliance module for OBD dataset.
Enforces privacy checks and policies.
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class PrivacyPolicy:
    """Privacy policy enforcement for OBD data processing."""

    @staticmethod
    def verify_no_pii(bandit_feedback: Dict[str, Any]) -> bool:
        """Verify dataset contains no PII (Personal Identifiable Information).

        Args:
            bandit_feedback: Raw dataset dictionary

        Returns:
            bool: True if compliant, raises ValueError otherwise
        """
        logger.info("Verifying privacy compliance (No PII)...")

        # Check context features are hashed/numerical
        context = bandit_feedback.get("context")
        if context is not None:
            if not np.issubdtype(context.dtype, np.number):
                raise ValueError(
                    "Privacy Violation: Context features must be numerical (hashed). Found object/string types."
                )

        # Check no explicit user identifiers
        forbidden_keys = ["user_id", "email", "username", "phone", "address"]
        found_keys = [k for k in bandit_feedback.keys() if k in forbidden_keys]

        if found_keys:
            raise ValueError(f"Privacy Violation: Found potential PII keys: {found_keys}")

        logger.info("Privacy Check Passed: No PII detected in raw features.")
        return True

    @staticmethod
    def check_reidentification_risk(n_users: int, n_samples: int) -> None:
        """Log re-identification risk metrics.

        Args:
            n_users: Number of unique users (derived)
            n_samples: Total interactions
        """
        avg_inter = n_samples / n_users
        logger.info(
            f"Privacy Metrics: {n_users} derived users, {avg_inter:.2f} avg interactions/user."
        )
        if avg_inter < 2.0:
            logger.warning(
                "Privacy Warning: Low average interactions (<2.0) may increase re-identification risk for specific patterns."
            )
