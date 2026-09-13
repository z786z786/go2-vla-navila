import unittest
from pathlib import Path


class D5ClosedLoopScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path("scripts/m62_run_d5_closed_loop.sh").read_text(encoding="utf-8")

    def test_resume_validates_before_skip(self):
        self.assertIn("src.inference.d5_resume", self.script)
        self.assertIn("--resume", self.script)
        self.assertIn("refuses incomplete or invalid episode", self.script)

    def test_comparison_only_runs_two_missing_models(self):
        self.assertIn("--comparison-only", self.script)
        self.assertIn("run_model_seed expanded_step_002000", self.script)
        self.assertIn("run_model_seed expanded_equal_exposure", self.script)
        self.assertIn("src.inference.d5_checkpoint_compare", self.script)
        self.assertIn("continuing through the action safety gate", self.script)

    def test_server_cleanup_runs_on_function_return_and_script_exit(self):
        self.assertIn("trap cleanup_server RETURN EXIT", self.script)
        self.assertIn("trap - RETURN EXIT", self.script)

    def test_post_training_entrypoint_requests_closed_loop_resume(self):
        resume_script = Path("scripts/m62_resume_d5_post_training.sh").read_text(encoding="utf-8")
        self.assertIn('"$FINAL_STEPS" false --resume', resume_script)

    def test_enters_project_before_any_resume_audit(self):
        cd_index = self.script.index('cd "$PROJECT"')
        function_index = self.script.index("run_model_seed()")
        self.assertLess(cd_index, function_index)


if __name__ == "__main__":
    unittest.main()
