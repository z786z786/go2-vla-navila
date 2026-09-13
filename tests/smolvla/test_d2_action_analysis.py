import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "NumPy is available in the server SmolVLA environment, not this local host")
class D2ActionAnalysisTests(unittest.TestCase):
    @staticmethod
    def _archive() -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", list[str], "np.ndarray"]:
        physical = np.zeros((3, 4, 50, 3), dtype=np.float32)
        latent = np.zeros_like(physical)
        expert = np.zeros((4, 50, 3), dtype=np.float32)
        valid = np.zeros((4, 50), dtype=bool)
        first_actions = np.array(
            [[0.1, 0.0, -0.2], [0.2, 0.0, 0.2], [0.3, 0.0, -0.1], [0.4, 0.0, 0.1]],
            dtype=np.float32,
        )
        physical[:, :, 0, :] = first_actions
        expert[:, 0, :] = first_actions
        valid[:, 0] = True
        return physical, latent, expert, valid, ["short_vln_v1_0000", "short_vln_v1_0000", "short_vln_v1_0004", "short_vln_v1_0004"], np.array([0, 1, 0, 1])

    def test_metrics_preserve_first_step_ratios_and_integrate_per_episode_commands(self):
        from src.smolvla.analyze_train_actions import compute_d2_metrics

        physical, _latent, expert, valid, episode_ids, frame_indices = self._archive()
        result = compute_d2_metrics(physical, expert, valid, episode_ids, frame_indices)
        global_vx = result["global"]["per_action"]["vx"]
        self.assertAlmostEqual(global_vx["mean_abs_magnitude_ratio"], 1.0)
        self.assertAlmostEqual(global_vx["std_ratio"], 1.0)
        self.assertAlmostEqual(result["per_episode"]["short_vln_v1_0000"]["cumulative"]["model_sum_vx_dt"], 0.006)
        self.assertAlmostEqual(result["per_episode"]["short_vln_v1_0004"]["cumulative"]["expert_sum_abs_wz_dt"], 0.004)
        self.assertIsNone(result["global"]["per_action"]["vy"]["mean_abs_magnitude_ratio"])

    def test_classification_marks_material_vx_magnitude_shrinkage(self):
        from src.smolvla.analyze_train_actions import classify_d2, compute_d2_metrics

        physical, _latent, expert, valid, episode_ids, frame_indices = self._archive()
        physical[:, :, 0, 0] *= 0.5
        metrics = compute_d2_metrics(physical, expert, valid, episode_ids, frame_indices)
        classification = classify_d2(metrics)
        self.assertTrue(classification["vx_magnitude_shrinkage"])
        self.assertFalse(classification["wz_magnitude_shrinkage"])

    def test_acceptance_passes_exact_reconstruction_with_zero_lateral_action(self):
        from src.smolvla.analyze_train_actions import acceptance, compute_d2_metrics

        physical, _latent, expert, valid, episode_ids, frame_indices = self._archive()
        verdict = acceptance(compute_d2_metrics(physical, expert, valid, episode_ids, frame_indices))
        self.assertTrue(verdict["passed"])

    def test_archive_validation_rejects_out_of_range_physical_actions(self):
        from src.smolvla.analyze_train_actions import validate_prediction_arrays

        physical, latent, expert, valid, episode_ids, frame_indices = self._archive()
        physical[0, 0, 0, 0] = 0.5001
        with self.assertRaisesRegex(ValueError, "physical_predictions"):
            validate_prediction_arrays(
                physical,
                latent,
                expert,
                valid,
                episode_ids,
                frame_indices,
                np.array([1, 2, 3]),
                expected_episode_counts={"short_vln_v1_0000": 2, "short_vln_v1_0004": 2},
            )

    def test_archive_validation_rejects_unknown_training_episode(self):
        from src.smolvla.analyze_train_actions import validate_prediction_arrays

        physical, latent, expert, valid, _episode_ids, frame_indices = self._archive()
        with self.assertRaisesRegex(ValueError, "episode IDs"):
            validate_prediction_arrays(
                physical,
                latent,
                expert,
                valid,
                ["short_vln_v1_0000", "short_vln_v1_0000", "short_vln_v1_0004", "other"],
                frame_indices,
                np.array([1, 2, 3]),
                expected_episode_counts={"short_vln_v1_0000": 2, "short_vln_v1_0004": 2},
            )


if __name__ == "__main__":
    unittest.main()
