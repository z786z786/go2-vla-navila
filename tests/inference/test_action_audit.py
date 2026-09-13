import unittest

from src.inference.action_audit import summarize_action_range


def safe_chunk() -> list[list[float]]:
    return [[0.1, 0.0, 0.0] for _ in range(50)]


class ActionAuditTests(unittest.TestCase):
    def test_separates_vector_chunk_and_execute_window_counts(self):
        first = safe_chunk()
        first[0] = [-0.01, 0.0, -0.6]  # One vector, two violating dimensions, executed.
        first[10] = [0.1, 0.0, -0.51]  # Discarded tail action.
        second = safe_chunk()
        second[25] = [0.6, 0.0, 0.0]
        second[49] = [0.1, 0.1, 0.0]

        result = summarize_action_range([first, second], execute_steps=10)

        self.assertEqual(result.total_vectors, 100)
        self.assertEqual(result.violating_vectors, 4)
        self.assertEqual(result.chunks_with_violation, 2)
        self.assertEqual(result.executed_vectors, 20)
        self.assertEqual(result.executed_violating_vectors, 1)
        self.assertEqual(result.tail_vectors, 80)
        self.assertEqual(result.tail_violating_vectors, 3)
        self.assertEqual(
            result.per_dimension,
            {
                "vx_below_min": 1,
                "vx_above_max": 1,
                "vy_not_zero": 1,
                "wz_below_min": 2,
                "wz_above_max": 0,
            },
        )

    def test_safe_chunks_have_zero_violations(self):
        result = summarize_action_range([safe_chunk()], execute_steps=10)

        self.assertEqual(result.violating_vectors, 0)
        self.assertEqual(result.chunks_with_violation, 0)
        self.assertEqual(result.executed_violating_vectors, 0)
        self.assertEqual(result.tail_violating_vectors, 0)


if __name__ == "__main__":
    unittest.main()
