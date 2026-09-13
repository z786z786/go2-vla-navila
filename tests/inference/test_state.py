import math
import unittest

from src.inference.state import (
    apply_action_safety,
    build_policy_state,
    build_policy_state_from_m4_record,
    quaternion_wxyz_to_rpy,
    validate_action_chunk,
)


class StateTests(unittest.TestCase):
    def test_quaternion_conversion_matches_wrapped_isaac_xyz_convention(self):
        half = math.sqrt(0.5)

        roll, pitch, yaw = quaternion_wxyz_to_rpy([half, 0.0, 0.0, half])

        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=6)

    def test_policy_state_uses_m5_order_and_has_30_values(self):
        joint_pos = [float(index) for index in range(12)]
        default_joint_pos = [0.5] * 12
        joint_vel = [float(index) / 10.0 for index in range(12)]

        state = build_policy_state(
            linear_body=[1.0, 2.0, 3.0],
            angular_body=[4.0, 5.0, 6.0],
            quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            joint_position=joint_pos,
            default_joint_position=default_joint_pos,
            joint_velocity=joint_vel,
        )

        self.assertEqual(len(state), 30)
        self.assertEqual(state[:6], [1.0, 2.0, 6.0, 0.0, 0.0, 0.0])
        self.assertEqual(state[6:18], [value - 0.5 for value in joint_pos])
        self.assertEqual(state[18:], joint_vel)

    def test_policy_state_rejects_nonunit_quaternion(self):
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            build_policy_state(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0] * 12,
                [0.0] * 12,
                [0.0] * 12,
            )

    def test_action_safety_preserves_raw_action_and_clips_application(self):
        result = apply_action_safety([0.7, 0.2, -0.8])

        self.assertEqual(result.raw, (0.7, 0.2, -0.8))
        self.assertEqual(result.applied, (0.5, 0.0, -0.5))
        self.assertFalse(result.in_range)
        self.assertEqual(result.clipped_dimensions, ("vx", "vy", "wz"))

    def test_finite_in_range_action_is_unchanged(self):
        result = apply_action_safety([0.2, 0.0, 0.3])

        self.assertEqual(result.raw, result.applied)
        self.assertTrue(result.in_range)
        self.assertEqual(result.clipped_dimensions, ())

    def test_action_chunk_rejects_wrong_shape_and_nonfinite_value(self):
        with self.assertRaisesRegex(ValueError, "50 actions"):
            validate_action_chunk([[0.0, 0.0, 0.0] for _ in range(49)])

        bad = [[0.0, 0.0, 0.0] for _ in range(50)]
        bad[3][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_action_chunk(bad)

    def test_m4_record_conversion_matches_m5_field_selection(self):
        raw_state = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0]
        raw_state += [float(index) for index in range(12)]
        raw_state += [100.0 + index for index in range(12)]
        record = {
            "robot_state": raw_state,
            "current_velocity": {
                "linear_body": [1.0, 2.0, 3.0],
                "angular_body": [4.0, 5.0, 6.0],
            },
            "robot_pose": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "joint_velocity": [200.0 + index for index in range(12)],
        }

        state = build_policy_state_from_m4_record(record)

        self.assertEqual(state[:6], [1.0, 2.0, 6.0, 0.0, 0.0, 0.0])
        self.assertEqual(state[6:18], [float(index) for index in range(12)])
        self.assertEqual(state[18:], [200.0 + index for index in range(12)])

    def test_full_33d_m4_record_has_same_policy_state_as_legacy_31d_record(self):
        legacy = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0]
        legacy += [float(index) for index in range(12)]
        legacy += [100.0 + index for index in range(12)]
        complete = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0]
        complete += [float(index) for index in range(12)]
        complete += [100.0 + index for index in range(12)]
        common = {
            "current_velocity": {
                "linear_body": [1.0, 2.0, 3.0],
                "angular_body": [4.0, 5.0, 6.0],
            },
            "robot_pose": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "joint_velocity": [200.0 + index for index in range(12)],
        }

        self.assertEqual(
            build_policy_state_from_m4_record({**common, "robot_state": legacy}),
            build_policy_state_from_m4_record({**common, "robot_state": complete}),
        )


if __name__ == "__main__":
    unittest.main()
