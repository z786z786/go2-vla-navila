"""CPU-only guardrails for the DT1 non-training SmolVLA resource probe."""

from __future__ import annotations

import copy
import unittest

from src.dual_target.contracts import RED_TASK
from src.dual_target.smolvla_probe_contract import (
    IDENTITY_NORMALIZATION,
    SMOLVLA_PROBE_SCHEMA,
    SmolVlaProbeContractError,
    SmolVlaProbeInput,
    fresh_identity_dataset_stats,
    probe_input_from_dict,
)


def _input() -> dict[str, object]:
    return {
        "schema_version": SMOLVLA_PROBE_SCHEMA,
        "rgb_path": "rgb_pre/000101.png",
        "rgb_sha256": "a" * 64,
        "state": [0.1, 0.0, -0.2],
        "task": RED_TASK,
        "action_dim": 3,
        "action_chunk_size": 50,
        "execute_action_steps": 10,
        "normalization": IDENTITY_NORMALIZATION,
        "dataset_stats": fresh_identity_dataset_stats(),
        "inference_only": True,
        "execute_model_actions": False,
        "training": False,
        "navigation_success_approved": False,
        "dt1_approved": False,
    }


class SmolVlaProbeContractTests(unittest.TestCase):
    def test_exact_3d_identity_nontraining_probe_is_accepted(self) -> None:
        item = probe_input_from_dict(_input())
        self.assertEqual(item.state, (0.1, 0.0, -0.2))
        self.assertEqual(item.as_dict()["action_dim"], 3)

    def test_old_dimensions_normalizer_execution_or_approval_are_rejected(self) -> None:
        for key, value in (
            ("state", [0.0] * 30),
            ("action_dim", 32),
            ("normalization", "old_dataset_normalizer"),
            ("execute_model_actions", True),
            ("training", True),
            ("dt1_approved", True),
        ):
            payload = copy.deepcopy(_input())
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(SmolVlaProbeContractError):
                probe_input_from_dict(payload)

    def test_extra_policy_or_resource_fields_cannot_slip_into_probe_contract(self) -> None:
        payload = _input()
        payload["legacy_normalizer_path"] = "/old/normalizer.pt"
        with self.assertRaises(SmolVlaProbeContractError):
            probe_input_from_dict(payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
