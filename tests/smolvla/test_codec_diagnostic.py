import unittest

try:
    import numpy as np
    import torch

    from src.smolvla.bounded_actions import decode_latent_actions, encode_physical_actions
    from src.smolvla.evaluate_action_codec import acceptance, boundary_limits, per_dimension_metrics
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "NumPy/PyTorch are available in the server SmolVLA environment")
class CodecDiagnosticTests(unittest.TestCase):
    def test_epsilon_contract_accepts_exact_bounds_and_interior_values(self):
        raw = np.array(
            [[0.0, 0.0, -0.5], [0.125, 0.0, -0.25], [0.375, 0.0, 0.25], [0.5, 0.0, 0.5]],
            dtype=np.float32,
        )
        decoded = decode_latent_actions(encode_physical_actions(torch.from_numpy(raw))).numpy()
        metrics = per_dimension_metrics(raw, decoded, epsilon=1e-4)
        verdict = acceptance(metrics)
        self.assertTrue(verdict["passed"])
        self.assertAlmostEqual(boundary_limits(1e-4)["vx"], 2.6e-5)
        self.assertAlmostEqual(boundary_limits(1e-4)["wz"], 5.1e-5)

    def test_sign_inversion_fails_acceptance(self):
        raw = np.array([[0.1, 0.0, -0.2], [0.2, 0.0, 0.2]], dtype=np.float32)
        decoded = raw.copy()
        decoded[:, 2] *= -1.0
        metrics = per_dimension_metrics(raw, decoded, epsilon=1e-4)
        self.assertEqual(metrics["wz"]["sign"]["inversion_count"], 2)
        self.assertFalse(acceptance(metrics)["passed"])

    def test_acceptance_skips_boundary_check_when_no_boundary_samples_exist(self):
        raw = np.array([[0.1, 0.0, -0.2], [0.2, 0.0, 0.2]], dtype=np.float32)
        metrics = per_dimension_metrics(raw, raw.copy(), epsilon=1e-4)
        verdict = acceptance(metrics)
        self.assertTrue(verdict["passed"])
        self.assertNotIn("vx_boundary_error_within_epsilon_contract", verdict["checks"])
        self.assertNotIn("wz_boundary_error_within_epsilon_contract", verdict["checks"])

    def test_distribution_drift_fails_acceptance(self):
        raw = np.array([[0.1, 0.0, -0.2], [0.2, 0.0, 0.2]], dtype=np.float32)
        decoded = raw.copy()
        decoded[:, 0] += 0.01
        metrics = per_dimension_metrics(raw, decoded, epsilon=1e-4)
        self.assertFalse(acceptance(metrics)["checks"]["vx_distribution_preserved"])


if __name__ == "__main__":
    unittest.main()
