from __future__ import annotations

import math
import unittest

from src.inference.full_protocol import PROTOCOL_VERSION, validate_request_header, validate_response_header
from src.inference.full_state import apply_action_safety, build_policy_state, validate_action_chunk


def request() -> dict:
    return {
        "version": PROTOCOL_VERSION, "request_id": "full:0", "episode_id": "full", "replan_index": 0,
        "instruction": "Walk past the chair and stop.", "state": [0.1, 0.0, -0.2],
        "image_encoding": "jpeg", "image_height": 512, "image_width": 512, "image_channels": 3, "jpeg_size": 4,
    }


class FullProtocolStateTests(unittest.TestCase):
    def test_protocol_accepts_only_3d_state_and_current_rgb_fields(self) -> None:
        validate_request_header(request(), payload_size=4)
        invalid = request()
        invalid["state"] = [0.0] * 30
        with self.assertRaisesRegex(ValueError, "3-D state"):
            validate_request_header(invalid, payload_size=4)
        privileged = request()
        privileged["gt_locations"] = [[0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "unknown request fields"):
            validate_request_header(privileged, payload_size=4)

    def test_response_is_exact_50_by_3(self) -> None:
        response = {
            "version": PROTOCOL_VERSION, "request_id": "full:0", "status": "ok",
            "actions": [[0.1, 0.0, -0.1] for _ in range(50)], "checkpoint_sha256": "b" * 64,
            "timings_ms": {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0, "total": 4.0}, "seed": 7,
        }
        validate_response_header(response)
        response["actions"][0][0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_response_header(response)

    def test_state_builder_discards_all_non_velocity_arguments(self) -> None:
        state = build_policy_state([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0], [9.0] * 12)
        self.assertEqual(state, [1.0, 2.0, 6.0])
        self.assertEqual(len(validate_action_chunk([[0.2, 0.0, 0.1] for _ in range(50)])), 50)
        decision = apply_action_safety([0.7, 0.1, -0.8])
        self.assertEqual(decision.applied, (0.5, 0.0, -0.5))
