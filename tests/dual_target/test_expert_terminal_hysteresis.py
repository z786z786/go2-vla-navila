"""Regression for the real r3 limit cycle around the terminal yaw boundary."""
import unittest

from src.dual_target.expert import ParkingExpert
from src.dual_target.layouts import Vec2


class TerminalHoldTest(unittest.TestCase):
    def test_small_yaw_rebound_does_not_restart_turn_after_entering_hold(self):
        expert = ParkingExpert()
        def command(yaw, xy=Vec2(1.03, 1.18)):
            return expert.command(robot_xy=xy, robot_yaw_rad=yaw,
                                  parking_center=Vec2(1.0, 1.2), terminal_heading_rad=0.0)
        self.assertLess(command(.13).wz, 0.)
        self.assertEqual(command(.119).as_list(), [0., 0., 0.])
        # Real r3 repeatedly rebounds from .119 to .127 radians after zero;
        # a stateless switch restarts turning and never allows physical settling.
        for yaw in [.127, .119, .13, .125]:
            self.assertEqual(command(yaw).as_list(), [0., 0., 0.])
        self.assertLess(command(.30).wz, 0.)  # large drift must reacquire
        self.assertEqual(command(.05).as_list(), [0., 0., 0.])
        self.assertNotEqual(command(.05, Vec2(.5, 1.2)).as_list(), [0., 0., 0.])
        self.assertLess(command(.13).wz, 0.)  # leaving position cleared hold


if __name__ == '__main__':
    unittest.main()
