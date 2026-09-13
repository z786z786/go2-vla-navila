from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.navila_n0.sampling import cap_terminal_near_zero_samples, terminal_action_statistics
from src.navila_n0.semantics import (
    SemanticFrame,
    build_instruction_swap_set,
    class_pixel_ratios_from_rgb,
    generate_instruction,
    load_navila_mpcat40_colors,
    select_landmark,
    validate_instruction,
)


class TestSemanticInstructions(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = {
            "candidate_id": "candidate-1",
            "category": "left_turn",
            "instruction_anchor_arc_length_m": 1.5,
        }

    def test_unique_visible_landmark_generates_only_template_text(self) -> None:
        frames = [
            SemanticFrame(index, 1.4 + index * 0.1, {"door": 0.03}, f"sha-{index}")
            for index in range(3)
        ]
        selection = select_landmark(self.candidate, frames)
        self.assertTrue(selection["accepted"])
        instruction = generate_instruction(self.candidate, selection["landmark"])
        self.assertEqual(instruction["text"], "Turn left at the door.")
        self.assertEqual(validate_instruction(self.candidate, instruction), [])
        swaps = build_instruction_swap_set([{**self.candidate, "instruction": instruction}])
        self.assertEqual({swap["kind"] for swap in swaps}, {"left_right", "landmark_category"})

    def test_ambiguous_category_is_rejected(self) -> None:
        frames = [
            SemanticFrame(index, 1.4 + index * 0.1, {"door": 0.03, "chair": 0.04}, f"sha-{index}")
            for index in range(3)
        ]
        selection = select_landmark(self.candidate, frames)
        self.assertFalse(selection["accepted"])
        self.assertEqual(selection["reason"], "ambiguous_landmark_categories")

    def test_navila_color_mapping_uses_full_frame_pixel_ratio(self) -> None:
        with TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "mpcat40.tsv"
            mapping.write_text(
                "mpcat40index\tmpcat40\thex\n4\tdoor\t#c5b0d5\n3\tchair\t#98df8a\n",
                encoding="utf-8",
            )
            colors = load_navila_mpcat40_colors(mapping)
        ratios = class_pixel_ratios_from_rgb(
            [[[197, 176, 213], [0, 0, 0]], [[197, 176, 213], [152, 223, 138]]], colors
        )
        self.assertEqual(ratios["door"], 0.5)
        self.assertEqual(ratios["chair"], 0.25)


class TestTerminalSampling(unittest.TestCase):
    def test_terminal_tail_and_sampler_cap(self) -> None:
        actions = [[0.1, 0.0, 0.0] for _ in range(25)] + [[0.0, 0.0, 0.0] for _ in range(50)]
        stats = terminal_action_statistics(actions, 25)
        self.assertTrue(stats["terminal_zero_tail_exactly_50"])
        self.assertGreater(stats["near_zero_chunk_ratio_ge_80pct"], 0.0)
        samples = [
            *[{"sample_id": f"normal-{index}", "terminal": False, "near_zero_chunk": False} for index in range(18)],
            *[{"sample_id": f"terminal-{index}", "terminal": True, "near_zero_chunk": True} for index in range(10)],
        ]
        cap = cap_terminal_near_zero_samples(samples)
        self.assertTrue(cap["passed"])
        self.assertLessEqual(cap["terminal_or_near_zero_selected_ratio"], 0.10)
