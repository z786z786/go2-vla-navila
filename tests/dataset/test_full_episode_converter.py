from __future__ import annotations

import unittest

from src.collector.r7_supervision import PD_ACTION_SOURCE
from src.dataset.convert_full_episode_to_lerobot import POLICY_FIELDS, feature_spec


class FullEpisodeConverterTests(unittest.TestCase):
    def test_policy_schema_is_only_rgb_state3_action3_task(self) -> None:
        features = feature_spec()
        self.assertEqual(features["observation.state"]["shape"], (3,))
        self.assertEqual(features["action"]["shape"], (3,))
        self.assertEqual(POLICY_FIELDS, ["observation.images.front", "observation.state", "action", "task"])
        self.assertNotIn("reference_path", features)
        self.assertNotIn("gt_locations", features)

