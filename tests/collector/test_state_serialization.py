import unittest

from src.collector.state_serialization import serialize_batched_euler_xyz


class StateSerializationTests(unittest.TestCase):
    def test_serializes_all_three_batched_euler_components(self):
        self.assertEqual(
            serialize_batched_euler_xyz(([0.1], [0.2], [0.3])),
            [0.1, 0.2, 0.3],
        )

    def test_rejects_non_singleton_or_nonfinite_components(self):
        with self.assertRaisesRegex(ValueError, "three"):
            serialize_batched_euler_xyz(([0.1], [0.2]))
        with self.assertRaisesRegex(ValueError, "single"):
            serialize_batched_euler_xyz(([0.1, 0.2], [0.2], [0.3]))
        with self.assertRaisesRegex(ValueError, "finite"):
            serialize_batched_euler_xyz(([0.1], [float("nan")], [0.3]))


if __name__ == "__main__":
    unittest.main()
