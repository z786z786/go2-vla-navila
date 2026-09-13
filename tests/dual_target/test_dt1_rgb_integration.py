"""Run on the Isaac host with real OpenCV; catches missing evidence directories."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.dual_target.records import EpisodeEvidenceWriter
from src.dual_target.runner import _write_rgb


class RgbEvidenceIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
                         "requires real OpenCV/numpy on the Isaac host; must pass there before GPU")
    def test_writer_supports_real_reset_pre_post_png_roundtrip(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="dt1-rgb-integration-") as scratch:
            root = Path(scratch) / "episode"
            EpisodeEvidenceWriter(root, {"test": "real-rgb-roundtrip"})
            # Distinct channels reveal an accidental RGB/BGR swap. No manual
            # mkdir: creation is the writer's responsibility in the live path.
            rgb = np.zeros((1, 16, 24, 4), dtype=np.uint8)
            rgb[..., :] = [231, 37, 89, 255]
            for phase in ("reset", "pre", "post"):
                path = root / f"rgb_{phase}" / "000001.png"
                _write_rgb(path, rgb)
                decoded = cv2.imread(str(path))
                self.assertIsNotNone(decoded)
                self.assertEqual(decoded.shape, (16, 24, 3))
                self.assertTrue(np.all(decoded == [89, 37, 231]))


if __name__ == "__main__":
    unittest.main()
