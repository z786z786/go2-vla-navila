"""Full official-episode NaVILA data path for the 3-D Go2 SmolVLA policy.

This package is deliberately separate from ``navila_n0``/``navila_n1``.  The
latter remain an auditable semantic-short-route experiment and are not an
input to the full-episode data, training, or inference path.
"""

from .contracts import DATASET_SCHEMA_VERSION, POLICY_CONTRACT

__all__ = ("DATASET_SCHEMA_VERSION", "POLICY_CONTRACT")
