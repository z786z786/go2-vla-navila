"""Arithmetic and real-source verification; no simulator/model/GPU imports."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "p3_navila_action_map", ROOT / "scripts/p3_navila_action_map.py"
)
MAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAPPER)
SOURCE = Path(
    "/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/"
    "omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py"
)


class ActionMapTest(unittest.TestCase):
    def test_forward_integrals(self):
        for cm, ticks, residual in [(25, 2, 0.05), (50, 5, 0.0), (75, 7, 0.05)]:
            with self.subTest(cm=cm):
                out = MAPPER.expand_action(f"move forward {cm} cm")
                self.assertEqual(out["tick_count"], ticks)
                integral = sum(c["vx"] * 0.2 for c in out["commands"])
                self.assertAlmostEqual(cm / 100 - integral, residual)
                self.assertAlmostEqual(out["truncation_residual"], residual)
                self.assertEqual(out["low_level_steps"], ticks * 10)

    def test_turn_integrals_and_signs(self):
        for direction, sign in [("left", 1), ("right", -1)]:
            for degrees, ticks in [(15, 2), (30, 5), (45, 7)]:
                with self.subTest(direction=direction, degrees=degrees):
                    out = MAPPER.expand_action(f"turn {direction} {degrees} degrees")
                    self.assertEqual(out["tick_count"], ticks)
                    integral = sum(c["wz"] * 0.2 for c in out["commands"])
                    residual = math.radians(degrees) - sign * integral
                    self.assertAlmostEqual(out["truncation_residual"], residual)
                    self.assertGreaterEqual(residual, 0)
                    self.assertLess(residual, 0.1)
                    self.assertTrue(all(c["vx"] == 0 and c["wz"] == sign * 0.5
                                        for c in out["commands"]))

    def test_stop_and_normalization(self):
        out = MAPPER.expand_action("  STOP. \n")
        self.assertEqual(out["commands"], [{"vx": 0.0, "vy": 0.0, "wz": 0.0, "stop": True}])
        self.assertEqual(out["motion_tick_count"], 0)
        self.assertEqual(out["truncation_residual"], 0)
        self.assertEqual(out["tick_count"], 1)
        self.assertEqual(MAPPER.expand_action(" TURN  Left 30 degrees. ")["tick_count"], 5)

    def test_reject_unverified_values_and_malformed_input(self):
        for text in [None, 25, [], "", "move forward 100 cm", "move forward -25 cm",
                     "move forward 25.0 cm", "turn left 60 degrees", "turn right 0 degrees",
                     "turn left 450 degrees", "remove 75", "move backwards 50 cm",
                     "turn left nan degrees", "turn right inf degrees", "stop now",
                     "turn left 30 degrees and stop", "move forward 25 m", "gibberish"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    MAPPER.expand_action(text)

    def test_dispatch_rounding_distinction(self):
        self.assertAlmostEqual(0.5 * 0.2, 0.1)
        self.assertAlmostEqual(math.radians(30) / 0.5, math.pi / 3)
        self.assertAlmostEqual(math.radians(45) / 0.5, math.pi / 2)
        self.assertEqual(math.floor((math.pi / 2) / 0.2), 7)
        self.assertEqual(round((math.pi / 2) / 0.2), 8)
        self.assertGreater(math.degrees(8 * 0.2 * 0.5), 45)

    def test_real_local_parser_outputs(self):
        # Compile just this pure function, never import third_party dependencies.
        self.assertTrue(SOURCE.is_file(), "required local evidence file missing")
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "get_vel_command")
        namespace = {"np": SimpleNamespace(pi=math.pi)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        parse = namespace["get_vel_command"]
        for cm, duration in [(25, 0.5), (50, 1.0), (75, 1.5)]:
            velocity, actual_duration = parse(f"move forward {cm} cm")
            self.assertEqual(velocity, [0.5, 0.0, 0.0])
            self.assertEqual(actual_duration, duration)
            self.assertAlmostEqual(velocity[0] * actual_duration, cm / 100)
        for direction, sign in [("left", 1), ("right", -1)]:
            for degrees, duration in [(15, 0.5), (30, 1.0), (45, 1.5)]:
                velocity, actual_duration = parse(f"turn {direction} {degrees} degrees")
                self.assertEqual(velocity, [0.0, 0.0, sign * math.pi / 6])
                self.assertEqual(actual_duration, duration)
                self.assertAlmostEqual(math.degrees(velocity[2] * duration), sign * degrees)
        self.assertEqual(parse("stop"), ([0.0, 0.0, 0.0], 0.0))
        # Upstream is permissive; these are evidence of fallback, not allowed labels.
        self.assertEqual(parse("gibberish"), parse("move forward 25 cm"))
        self.assertEqual(parse("turn left 450 degrees"), parse("turn left 45 degrees"))
        self.assertEqual(parse("remove 75"), parse("move forward 75 cm"))


if __name__ == "__main__":
    unittest.main()
