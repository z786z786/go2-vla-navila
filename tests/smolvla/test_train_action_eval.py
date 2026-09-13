import unittest

try:
    import numpy as np

    from src.smolvla.evaluate_train_actions import (
        ACTION_NAMES,
        action_metrics,
        alignment_metrics,
        derive_inference_seed,
        flatten_valid,
        pearson_correlation,
        saturation_ratios,
        wz_sign_metrics,
    )
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "NumPy is available in the server SmolVLA environment, not this local host")
class OfflineTrainActionEvalTests(unittest.TestCase):
    def test_seed_is_stable_and_order_independent(self):
        seed = derive_inference_seed(20260831, "short_vln_v1_0000", 17)
        self.assertEqual(seed, derive_inference_seed(20260831, "short_vln_v1_0000", 17))
        self.assertNotEqual(seed, derive_inference_seed(20260831, "short_vln_v1_0000", 18))
        self.assertNotEqual(seed, derive_inference_seed(20260832, "short_vln_v1_0000", 17))

    def test_flatten_valid_excludes_padding_and_preserves_action_pairs(self):
        expected = np.array([[[1.0, 0.0, -0.1], [2.0, 0.0, 0.2]]])
        predicted = expected + 0.5
        valid = np.array([[True, False]])
        pred_flat, expected_flat = flatten_valid(predicted, expected, valid)
        self.assertEqual(pred_flat.shape, (1, 3))
        np.testing.assert_allclose(pred_flat, [[1.5, 0.5, 0.4]])
        np.testing.assert_allclose(expected_flat, [[1.0, 0.0, -0.1]])

    def test_metrics_report_constant_vy_correlation_as_na(self):
        expected = np.array([[0.1, 0.0, -0.2], [0.2, 0.0, 0.2]])
        predicted = np.array([[0.1, 0.0, -0.2], [0.3, 0.0, 0.1]])
        metrics = action_metrics(predicted, expected)
        self.assertEqual(set(metrics["per_action"]), set(ACTION_NAMES))
        self.assertIsNone(metrics["per_action"]["vy"]["correlation_pearson"])
        self.assertAlmostEqual(metrics["per_action"]["vx"]["mse"], 0.005)

    def test_wz_sign_deadband_counts_small_prediction_as_wrong(self):
        expected = np.array([[0.2, 0.0, -0.2], [0.2, 0.0, 0.2], [0.2, 0.0, 0.02]])
        predicted = np.array([[0.2, 0.0, -0.3], [0.2, 0.0, 0.01], [0.2, 0.0, -0.2]])
        result = wz_sign_metrics(predicted, expected, deadband=0.05)
        self.assertEqual(result["eligible_count"], 2)
        self.assertAlmostEqual(result["eligible_ratio"], 2 / 3)
        self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_saturation_thresholds_are_explicit(self):
        actions = np.array([[0.0, 0.0, -0.5], [0.5, 0.0, 0.49], [0.2, 1e-4, 0.0]])
        result = saturation_ratios(actions)
        self.assertAlmostEqual(result["vx_low_le_0_01"], 1 / 3)
        self.assertAlmostEqual(result["vx_high_ge_0_49"], 1 / 3)
        self.assertAlmostEqual(result["wz_abs_ge_0_49"], 2 / 3)
        self.assertAlmostEqual(result["vy_nonzero_gt_1e_6"], 1 / 3)

    def test_alignment_reports_seedwise_and_ensemble_with_both_masks(self):
        expected = np.array([[[0.1, 0.0, -0.2], [0.2, 0.0, 0.2]]])
        predictions = np.stack((expected, expected + 0.1, expected - 0.1))
        valid = np.array([[True, False]])
        result = alignment_metrics(predictions, expected, valid, np.array([0.15, 0.0, 0.0]))
        self.assertEqual(result["valid_pair_count_per_seed"], 1)
        self.assertEqual(len(result["per_seed"]), 3)
        self.assertAlmostEqual(result["ensemble_mean"]["per_action"]["vx"]["mse"], 0.0)

    def test_pearson_constant_is_na(self):
        self.assertIsNone(pearson_correlation(np.array([1.0, 1.0]), np.array([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
