import unittest

try:
    import numpy as np

    from src.smolvla.analyze_temporal_alignment import (
        clear_nonzero_peak,
        lag_curve,
        signal_difference,
        wrap_angle,
        yaw_from_wxyz,
        zero_lag_image_pass,
    )
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "NumPy is available in the server SmolVLA environment, not this local host")
class D4TemporalAlignmentTests(unittest.TestCase):
    def test_positive_lag_means_source_leads_target(self):
        source = np.zeros(30)
        source[[4, 12, 21]] = [1.0, -1.0, 0.5]
        target = np.zeros(30)
        target[[6, 14, 23]] = [1.0, -1.0, 0.5]
        peak = clear_nonzero_peak(lag_curve(source, target, range(-5, 6)), threshold=0.01)
        self.assertEqual(peak["best_lag"], 2)
        self.assertTrue(peak["clear_nonzero_peak"])

    def test_zero_lag_is_not_mislabeled_as_shift(self):
        source = np.linspace(-1.0, 1.0, 20) ** 3
        peak = clear_nonzero_peak(lag_curve(source, source, range(-5, 6)))
        self.assertEqual(peak["best_lag"], 0)
        self.assertFalse(peak["clear_nonzero_peak"])

    def test_first_difference_preserves_expected_one_frame_shift(self):
        source = np.asarray([0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 0.5, 0.0, 0.0])
        target = np.concatenate(([0.0], source[:-1]))
        peak = clear_nonzero_peak(lag_curve(signal_difference(source), signal_difference(target), range(-3, 4)), threshold=0.01)
        self.assertEqual(peak["best_lag"], 1)

    def test_angle_wrap_and_yaw_are_wxyz_correct(self):
        self.assertAlmostEqual(wrap_angle(3.0 * np.pi), -np.pi)
        yaw = yaw_from_wxyz([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        self.assertAlmostEqual(yaw, np.pi / 2.0)

    def test_image_gate_accepts_zero_lag_near_best_and_rejects_material_shift(self):
        accepted = zero_lag_image_pass({-1: 0.2, 0: 0.101, 1: 0.1})
        rejected = zero_lag_image_pass({-1: 0.2, 0: 0.2, 1: 0.1})
        self.assertTrue(accepted["zero_within_one_percent_of_best"])
        self.assertFalse(rejected["zero_within_one_percent_of_best"])


if __name__ == "__main__":
    unittest.main()
