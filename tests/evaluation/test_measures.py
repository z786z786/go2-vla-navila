import unittest
import warnings

import numpy as np

from evaluation.measures import METRIC_NAMES, evaluate_trajectory


def episode(waypoints=None):
    return {'gt_locations': waypoints if waypoints is not None else [[0., 0., 0.]],
            'goals': [{'radius': 3.0}], 'info': {'geodesic_distance': 999.0}}


class MeasureTests(unittest.TestCase):
    def test_path_length_is_3d_step_sum(self):
        result = evaluate_trajectory([[1., 0., 0.], [1., 0., 3.], [1., 4., 3.]], None, episode())
        self.assertEqual([s['path_length'] for s in result['steps']], [0., 3., 7.])

    def test_kdtree_uses_z_and_remaining_waypoints(self):
        e = episode([[0., 0., 0.], [0., 0., 10.], [0., 0., 20.]])
        result = evaluate_trajectory([[0., 0., 9.]], None, e)
        self.assertEqual(result['final']['distance_to_goal'], 11.)

    def test_allclose_cache_keeps_last_recomputed_position(self):
        result = evaluate_trajectory([[1., 0., 0.], [1.00005, 0., 0.],
                                      [1.0001, 0., 0.], [1.00015, 0., 0.]], None, episode())
        self.assertEqual([s['distance_to_goal'] for s in result['steps']], [1., 1., 1., 1.00015])
        self.assertGreater(result['steps'][1]['path_length'], 0.)

    def test_allclose_preserves_default_relative_tolerance(self):
        result = evaluate_trajectory([[100., 0., 0.], [100.0005, 0., 0.]], None, episode())
        self.assertEqual(result['final']['distance_to_goal'], 100.)

    def test_exact_radius_boundaries_are_strict(self):
        for radius, suffix in ((3., ''), (2., '_2m'), (1., '_1m')):
            with self.subTest(radius=radius):
                result = evaluate_trajectory([[radius, 0., 0.], [radius, 0., 0.]], 1, episode())
                self.assertEqual(result['final']['success' + suffix], 0.)
                self.assertEqual(result['final']['oracle_success' + suffix], 0.)
                inside = evaluate_trajectory([[radius - .01, 0., 0.]] * 2, 1, episode())
                self.assertEqual(inside['final']['success' + suffix], 1.)
                self.assertEqual(inside['final']['oracle_success' + suffix], 1.)

    def test_stop_first_middle_last_never(self):
        positions = [[2., 0., 0.]] * 5
        for stop in (1, 2, 4, None):
            result = evaluate_trajectory(positions, stop, episode())
            self.assertEqual([s['success'] for s in result['steps']],
                             [float(stop is not None and i >= stop) for i in range(5)])

    def test_spl_uses_reset_distance_and_walked_length(self):
        result = evaluate_trajectory([[4., 0., 0.], [8., 0., 0.], [2., 0., 0.]], 2, episode())
        self.assertEqual(result['final']['spl'], 4. / 10.)

    def test_measure_update_order_and_oracle_history(self):
        result = evaluate_trajectory([[4., 0., 0.], [.5, 0., 0.], [5., 0., 0.]], 1, episode())
        self.assertEqual(tuple(result['final'])[:6], METRIC_NAMES)
        self.assertEqual(result['steps'][1]['spl'], 1.)
        self.assertEqual(result['final']['success'], 0.)
        self.assertEqual(result['final']['oracle_navigation_error'], .5)
        self.assertEqual(result['final']['oracle_success'], 1.)
        self.assertEqual(result['final']['oracle_success_1m'], 1.)

    def test_final_matches_last_snapshot_without_aliasing(self):
        result = evaluate_trajectory([[4., 0., 0.]], None, episode())
        self.assertEqual(result['final'], result['steps'][-1])
        self.assertIsNot(result['final'], result['steps'][-1])

    def test_preserves_float32_step_arithmetic(self):
        positions = np.array([[4., .2, .1], [2.3, .6, .9], [1., .1, .2]], dtype=np.float32)
        expected = 0.0
        for a, b in zip(positions, positions[1:]):
            expected += np.linalg.norm(a - b, ord=2)
        self.assertEqual(evaluate_trajectory(positions, 2, episode())['final']['path_length'], expected)

    def test_zero_start_distance_preserves_official_undefined_spl(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = evaluate_trajectory([[0., 0., 0.]], None, episode())
        self.assertTrue(np.isnan(result['final']['spl']))
        self.assertTrue(any('invalid value' in str(w.message) for w in caught))

    def test_invalid_inputs_rejected(self):
        for positions, stop in (([], None), ([[1., 2.]], None), ([[np.nan, 0., 0.]], None),
                                ([[1., 0., 0.]], 0), ([[1., 0., 0.]], 1)):
            with self.subTest(positions=positions, stop=stop), self.assertRaises(ValueError):
                evaluate_trajectory(positions, stop, episode())
