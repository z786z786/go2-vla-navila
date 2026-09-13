import json
from pathlib import Path
import tempfile
import unittest

from src.smolvla.revalidate_extended_training import acceptance_checks
from src.smolvla.training_acceptance import smoke_acceptance_checks


class ExtendedTrainingAcceptanceTests(unittest.TestCase):
    def test_smoke_accepts_finite_equal_exposure_run(self):
        history = [{"loss": 0.1}] * 43000
        checks = smoke_acceptance_checks(43000, history, {"finite": True, "output_shape": [1, 50, 3]})
        self.assertTrue(all(checks.values()))

    def test_smoke_rejects_short_or_incomplete_run(self):
        checks = smoke_acceptance_checks(43000, [{"loss": 0.1}], {"finite": True, "output_shape": [1, 50, 3]})
        self.assertFalse(checks["history_length_matches_steps"])

    def test_offline_revalidation_checks_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "step_002000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "m6_checkpoint.json").write_text(json.dumps({"step": 2000}))
            for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
                (checkpoint / name).write_bytes(b"ok")
            row = {"loss": 0.1, "gradient_norm": 1.0, "lr": 1e-4, "step_time_s": 0.2, "samples_per_s": 5.0, "val_loss": None}
            history = [{"step": step, **row} for step in range(1, 2001)]
            report = {"stage": "smoke", "training": {"steps": 2000}, "reloaded_checkpoint_probe": {"finite": True, "output_shape": [1, 50, 3]}}
            self.assertTrue(all(acceptance_checks(root, report, history).values()))


if __name__ == "__main__":
    unittest.main()
