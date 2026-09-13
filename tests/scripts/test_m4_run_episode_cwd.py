"""Regression coverage for the M4 collector's package import context."""

from pathlib import Path
import stat
import unittest


class M4RunEpisodeCwdTests(unittest.TestCase):
    def test_enters_project_before_running_collector_as_a_module(self) -> None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "m4_run_episode.sh"
        script = path.read_text(encoding="utf-8")
        collector = '"$PREFIX/bin/python" -m src.collector.collect_expert'
        self.assertIn(collector, script)
        self.assertLess(script.index('cd "$PROJECT"'), script.index(collector))

    def test_is_executable_for_the_batch_collector(self) -> None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "m4_run_episode.sh"
        self.assertTrue(path.stat().st_mode & stat.S_IXUSR)

    def test_passes_opt_in_r7_terminal_hold_to_the_collector(self) -> None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "m4_run_episode.sh"
        script = path.read_text(encoding="utf-8")
        self.assertIn('M6_2_R7_TERMINAL_HOLD_FRAMES', script)
        self.assertIn('--terminal-hold-frames "$TERMINAL_HOLD_FRAMES"', script)

    def test_runs_the_postprocessor_as_a_module_in_the_project_context(self) -> None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "m4_run_episode.sh"
        script = path.read_text(encoding="utf-8")
        checker = '"$PREFIX/bin/python" -m src.collector.check_expert'
        self.assertIn(checker, script)
        self.assertLess(script.index('cd "$PROJECT"'), script.index(checker))

    def test_r7_canary_stops_on_the_first_failed_sanity_gate(self) -> None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "r7_run_canary.sh"
        script = path.read_text(encoding="utf-8")
        self.assertIn("R7 canary fail-fast sanity failure", script)
        self.assertLess(script.index("R7 canary fail-fast sanity failure"), script.index("episode_dirs+=("))


if __name__ == "__main__":
    unittest.main()
