from __future__ import annotations

import unittest

from src.inference.check_full_rollout import evaluate
from src.inference.full_protocol import PROTOCOL_VERSION


def request() -> dict:
    return {"version": PROTOCOL_VERSION, "request_id": "x:0", "episode_id": "x", "replan_index": 0, "instruction": "Walk forward.", "state": [0.1, 0.0, 0.0], "image_encoding": "jpeg", "image_height": 512, "image_width": 512, "image_channels": 3, "jpeg_size": 4}


def response() -> dict:
    return {"version": PROTOCOL_VERSION, "request_id": "x:0", "status": "ok", "actions": [[0.1, 0.0, 0.0] for _ in range(50)], "checkpoint_sha256": "a" * 64, "timings_ms": {"preprocess": 1.0, "inference": 1.0, "postprocess": 1.0, "total": 3.0}, "seed": 1}


class FullClosedLoopCheckTests(unittest.TestCase):
    def test_requires_three_dimensional_requests_and_600ms_gate(self) -> None:
        summary = {"protocol_version": PROTOCOL_VERSION, "execute_steps_per_chunk": 10, "temporal_contract": {"execute_frames_per_replan": 10, "wall_clock_control_rate_claim": None}, "latency_gate": {"passed": True}, "p95_inference_roundtrip_ms": 599.0, "passed": True}
        records = [{"policy_state": [0.1, 0.0, 0.0]}]
        report = evaluate(summary, records, [{"request": request(), "response": response()}])
        self.assertTrue(report["passed"])
        summary["p95_inference_roundtrip_ms"] = 601.0
        self.assertFalse(evaluate(summary, records, [{"request": request(), "response": response()}])["passed"])

