"""Dispatch P4-T3 contract tests, organized by its ten required groups."""

from __future__ import annotations

import unittest

from actions.language_parser import LanguageParser
from actions.nav_command import PI_OVER_SIX, NavCommand, duration_to_hold_steps


FORWARD_25 = NavCommand(0.5, 0.0, 0.0, 25, False)
BRAKE_25 = NavCommand(0.0, 0.0, 0.0, 25, False)
STOP = NavCommand(0.0, 0.0, 0.0, 0, True)


class TestLanguageParserContract(unittest.TestCase):
    def assertParse(
        self,
        parser: LanguageParser,
        text: str,
        command: NavCommand,
        classification: str,
    ) -> None:
        result = parser.parse(text)
        self.assertEqual(result.command, command)
        self.assertEqual(result.classification, classification)
        self.assertEqual(result.parse_failure, classification != "exact")

    # 1. The ten frozen §2.1 inputs are asserted field-by-field.
    def test_01_all_ten_frozen_vocabulary_entries(self) -> None:
        cases = [
            ("The next action is move forward 25 cm.", 0.5, 0.0, 25, False),
            ("The next action is move forward 50 cm.", 0.5, 0.0, 50, False),
            ("The next action is move forward 75 cm.", 0.5, 0.0, 75, False),
            ("The next action is turn left 15 degree.", 0.0, PI_OVER_SIX, 25, False),
            ("The next action is turn left 30 degree.", 0.0, PI_OVER_SIX, 50, False),
            ("The next action is turn left 45 degree.", 0.0, PI_OVER_SIX, 75, False),
            ("The next action is turn right 15 degree.", 0.0, -PI_OVER_SIX, 25, False),
            ("The next action is turn right 30 degree.", 0.0, -PI_OVER_SIX, 50, False),
            ("The next action is turn right 45 degree.", 0.0, -PI_OVER_SIX, 75, False),
            (
                "I think I should stop because I have finished the instruction.",
                0.0,
                0.0,
                0,
                True,
            ),
        ]
        for text, vx, wz, hold_steps, stop in cases:
            with self.subTest(text=text):
                result = LanguageParser().parse(text)
                self.assertEqual(result.classification, "exact")
                self.assertEqual(result.command.vx, vx)
                self.assertEqual(result.command.vy, 0.0)
                self.assertEqual(result.command.wz, wz)
                self.assertEqual(result.command.hold_steps, hold_steps)
                self.assertEqual(result.command.stop, stop)

    # 2. Case, surrounding/extra spaces, and a missing final period normalize.
    def test_02_surface_form_variants_are_exact(self) -> None:
        cases = [
            ("  the NEXT action is MOVE forward 25 cm  ", FORWARD_25),
            (
                "The  next   action is  turn   left  30 degree",
                NavCommand(0.0, 0.0, PI_OVER_SIX, 50, False),
            ),
            (" i THINK I should STOP because I have finished the instruction ", STOP),
        ]
        parser = LanguageParser()
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertParse(parser, text, expected, "exact")

    # 3. Native's "45" substring quirk is deliberate and applies to every policy.
    def test_03_native_145_substring_quirk(self) -> None:
        expected = NavCommand(0.0, 0.0, PI_OVER_SIX, 75, False)
        for policy in ("layered", "native", "brake"):
            with self.subTest(policy=policy):
                self.assertParse(
                    LanguageParser(policy), "turn left 145 degree", expected, "partial_match"
                )

    # 4. Branch ordering remains left, then right, then move, then stop.
    def test_04_native_keyword_priority(self) -> None:
        text = "turn left 30 degree; turn right 15 degree; move 75 cm; stop"
        expected = NavCommand(0.0, 0.0, PI_OVER_SIX, 50, False)
        for policy in ("layered", "native", "brake"):
            with self.subTest(policy=policy):
                result = LanguageParser(policy).parse(text)
                self.assertEqual(result.classification, "partial_match")
                self.assertEqual(result.matched_branch, "turn_left")
                self.assertEqual(result.command, expected)

    # 5. A bare "move" uses the native forward branch even without "forward".
    def test_05_move_without_forward(self) -> None:
        expected_by_policy = {
            "layered": FORWARD_25,
            "native": FORWARD_25,
            "brake": BRAKE_25,
        }
        for policy, expected in expected_by_policy.items():
            with self.subTest(policy=policy):
                self.assertParse(LanguageParser(policy), "move carefully", expected, "partial_match")

    # 6. Four total-miss forms exercise every policy and tiered counters.
    def test_06_total_misses_and_tiered_counters(self) -> None:
        misses = ["", "   \t\n", "x" * 10000, "向前走，然后停下来"]
        expected_by_policy = {
            "layered": BRAKE_25,
            "native": FORWARD_25,
            "brake": BRAKE_25,
        }
        for policy, expected in expected_by_policy.items():
            parser = LanguageParser(policy)
            for text in misses:
                with self.subTest(policy=policy, text_preview=text[:12]):
                    self.assertParse(parser, text, expected, "total_miss")
            self.assertEqual(parser.counters.exact, 0)
            self.assertEqual(parser.counters.partial_match, 0)
            self.assertEqual(parser.counters.total_miss, len(misses))
            self.assertEqual(parser.parse_error_rate, 1.0)

        parser = LanguageParser()
        parser.parse("The next action is move forward 25 cm.")
        parser.parse("turn left 90 degree")
        parser.parse("unrelated words")
        self.assertEqual(parser.counters.as_dict(), {
            "exact": 1,
            "partial_match": 1,
            "total_miss": 1,
            "total": 3,
            "parse_error_rate": 2 / 3,
        })

    # 7. Out-of-vocabulary numbers are partial matches and policy fallbacks apply.
    def test_07_out_of_range_numbers(self) -> None:
        cases = [
            (
                "turn left 90 degree",
                {
                    "layered": NavCommand(0.0, 0.0, PI_OVER_SIX, 25, False),
                    "native": NavCommand(0.0, 0.0, PI_OVER_SIX, 25, False),
                    "brake": BRAKE_25,
                },
            ),
            (
                "move forward 200 cm",
                {"layered": FORWARD_25, "native": FORWARD_25, "brake": BRAKE_25},
            ),
        ]
        for text, expected_by_policy in cases:
            for policy, expected in expected_by_policy.items():
                with self.subTest(text=text, policy=policy):
                    self.assertParse(LanguageParser(policy), text, expected, "partial_match")

    # 8. A non-frozen stop phrase is partial, but native's stop branch remains stop.
    def test_08_stop_variants(self) -> None:
        for policy in ("layered", "native", "brake"):
            with self.subTest(policy=policy):
                self.assertParse(LanguageParser(policy), "Please stop now.", STOP, "partial_match")

    # 9. Fixed durations use the exact 0.02 s conversion without truncation.
    def test_09_duration_to_hold_step_boundaries(self) -> None:
        self.assertEqual(duration_to_hold_steps(0.5), 25)
        self.assertEqual(duration_to_hold_steps(1.0), 50)
        self.assertEqual(duration_to_hold_steps(1.5), 75)

    # 10. Every emitted command obeys the frozen NavCommand value domain.
    def test_10_emitted_commands_stay_in_domain(self) -> None:
        texts = [
            "The next action is move forward 25 cm.",
            "The next action is move forward 50 cm.",
            "The next action is move forward 75 cm.",
            "The next action is turn left 15 degree.",
            "The next action is turn left 30 degree.",
            "The next action is turn left 45 degree.",
            "The next action is turn right 15 degree.",
            "The next action is turn right 30 degree.",
            "The next action is turn right 45 degree.",
            "I think I should stop because I have finished the instruction.",
            "turn left 90 degree",
            "unrelated words",
        ]
        for policy in ("layered", "native", "brake"):
            parser = LanguageParser(policy)
            for text in texts:
                with self.subTest(policy=policy, text=text):
                    command = parser.parse(text).command
                    self.assertGreaterEqual(command.vx, 0.0)
                    self.assertLessEqual(command.vx, 0.5)
                    self.assertEqual(command.vy, 0.0)
                    self.assertLessEqual(abs(command.wz), PI_OVER_SIX + 1e-12)
                    self.assertIn(command.hold_steps, {0, 25, 50, 75})


if __name__ == "__main__":
    unittest.main()
