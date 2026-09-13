import unittest

from src.inference.d5_checkpoint_compare import response_checkpoint_sha256


class D5CheckpointCompareTests(unittest.TestCase):
    def test_extracts_single_response_checkpoint_hash(self):
        requests = [
            {"status": "ok", "response": {"checkpoint_sha256": "abc"}},
            {"status": "ok", "response": {"checkpoint_sha256": "abc"}},
        ]
        self.assertEqual(response_checkpoint_sha256(requests), "abc")

    def test_rejects_mixed_checkpoint_hashes(self):
        requests = [
            {"status": "ok", "response": {"checkpoint_sha256": "abc"}},
            {"status": "ok", "response": {"checkpoint_sha256": "def"}},
        ]
        with self.assertRaisesRegex(ValueError, "expected one response checkpoint hash"):
            response_checkpoint_sha256(requests)


if __name__ == "__main__":
    unittest.main()
