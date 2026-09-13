import unittest
from evaluation.measures import evaluate_trajectory
class Smoke(unittest.TestCase):
 def test_multiple_waypoints(self): self.assertEqual(evaluate_trajectory([[0,0,0],[1,0,0]],None,{'gt_locations':[[0,0,0],[1,0,0]],'goals':[{'radius':3.0}]})['final']['distance_to_goal'],0.)
