"""N0 contracts for NaVILA semantic short-instruction VLN.

This package is deliberately runtime-free: it only plans, validates and audits
the N1 data pipeline.  It neither opens Isaac Sim nor imports a policy.
"""

from .contracts import POLICY_CONTRACT, required_training_steps, validate_training_plan

__all__ = ["POLICY_CONTRACT", "required_training_steps", "validate_training_plan"]
