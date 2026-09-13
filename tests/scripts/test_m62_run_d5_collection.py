from pathlib import Path
import unittest


class D5RunnerCollectionIntegrationTests(unittest.TestCase):
    def test_uses_resilient_collector_and_only_accepted_root(self):
        script = (
            Path(__file__).resolve().parents[2] / "scripts" / "m62_run_d5.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("-m src.collector.d5_collection", script)
        self.assertIn('--accepted-root "$EXPERT_ROOT"', script)
        self.assertNotIn('m4_collect_gate.sh" "$EXPERT_ROOT"', script)

    def test_has_explicit_same_root_resume_mode(self):
        script = (
            Path(__file__).resolve().parents[2] / "scripts" / "m62_run_d5.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"--resume"', script)
        self.assertIn('COLLECTION_ARGS+=(--resume)', script)


if __name__ == "__main__":
    unittest.main()
