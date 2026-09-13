from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.navila_r2r import NavilaR2RDataset


class NavilaHistoryRuleTests(unittest.TestCase):
    def test_matches_navila_linspace_for_every_length(self):
        import numpy as np
        from src.smolvla.phase1_prompt import navila_history_layout
        for n in range(1, 600):
            padded = [None] * max(0, 8 - n) + list(range(n))
            ref = [padded[i] for i in np.linspace(0, len(padded) - 1, num=7, endpoint=False, dtype=int)]
            self.assertEqual(navila_history_layout(n), ref + [padded[-1]], n)


class NavilaR2RTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "train/v1").mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (4, 5), (i * 80, 20, 10)).save(self.root / f"train/v1/frame_{i}.jpg")
        self.record = {
            "video_id": "v1-1", "video": "v1", "step": 1,
            "instruction_raw": "Go LEFT!  ", "instruction_normalized": "go left!",
            "action_text": "The next action is move forward 25 cm.", "n_frames": 3,
            "frames_first": "v1/frame_0.jpg", "frames_last": "v1/frame_2.jpg",
        }
        self.index = self.root / "records.jsonl"
        self.index.write_text(json.dumps(self.record) + "\n\n" + json.dumps(self.record) + "\n")
        self.ds = NavilaR2RDataset(self.index, self.root / "train")
        self.addCleanup(self.ds.close)

    def test_cpu_pixels_raw_instruction_and_batch(self):
        item = self.ds[0]
        self.assertEqual(item["frames"].shape, (8, 3, 5, 4))
        self.assertEqual(item["frames"].dtype, torch.uint8)
        self.assertEqual(item["frames"].device.type, "cpu")
        self.assertEqual(item["instruction_raw"], "Go LEFT!  ")
        self.assertEqual(item["instruction"], self.record["instruction_raw"])
        # 3 real frames -> 5 black frames prepended, then frames 0,1,2 (NaVILA rule)
        self.assertEqual(item["history_padding_mask"], [True] * 5 + [False] * 3)
        self.assertTrue(torch.equal(item["frames"][0], torch.zeros(3, 5, 4, dtype=torch.uint8)))
        for k in range(3):
            self.assertTrue(torch.equal(item["frames"][5 + k], self.ds.read_frame(item["frame_paths"][5 + k])))
        batch = next(iter(DataLoader(self.ds, batch_size=2, num_workers=0)))
        self.assertEqual(batch["frames"].shape, (2, 8, 3, 5, 4))

    def test_duplicates_and_boundaries(self):
        self.assertEqual(len(self.ds), 2)
        self.assertEqual(self.ds[0]["record"], self.ds[1]["record"])
        self.assertEqual(self.ds[-1]["record"], self.record)
        for index in (2, -3):
            with self.assertRaises(IndexError):
                self.ds[index]

    def test_history_and_path_reconstruction(self):
        # Expected values computed with NaVILA's own np.linspace(..., endpoint=False, dtype=int)
        self.assertEqual(self.ds._history_indices(1), [None] * 7 + [0])
        self.assertEqual(self.ds._history_indices(3), [None] * 5 + [0, 1, 2])
        self.assertEqual(self.ds._history_indices(8), list(range(8)))
        self.assertEqual(self.ds._history_indices(9), [0, 1, 2, 3, 4, 5, 6, 8])
        self.assertEqual(self.ds._history_indices(49), [0, 6, 13, 20, 27, 34, 41, 48])
        paths = self.ds.frame_paths(self.record)
        self.assertEqual(paths[5], self.root / "train" / self.record["frames_first"])
        self.assertEqual(paths[-1], self.root / "train" / self.record["frames_last"])
        with self.assertRaises(ValueError):
            self.ds._history_indices(0)

    def test_bad_metadata_and_missing_image_fail(self):
        for changed in ({"frames_last": "v1/frame_99.jpg"}, {"video": "../escape"}):
            with self.assertRaises(ValueError):
                self.ds.frame_paths({**self.record, **changed})
        (self.root / "train/v1/frame_2.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            self.ds[0]

    def test_metadata_only_and_holdout_filter(self):
        ds = NavilaR2RDataset(self.index, self.root / "absent", load_frames=False)
        self.addCleanup(ds.close)
        self.assertNotIn("frames", ds[0])
        filtered = NavilaR2RDataset(self.index, self.root, exclude_videos={"v1"})
        self.addCleanup(filtered.close)
        self.assertEqual(len(filtered), 0)

    def test_pickling_after_open(self):
        self.ds[0]
        restored = pickle.loads(pickle.dumps(self.ds))
        self.addCleanup(restored.close)
        self.assertEqual(restored[1]["record"], self.record)

    def test_frozen_history(self):
        with self.assertRaises(ValueError):
            NavilaR2RDataset(self.index, self.root, history_frames=4)


if __name__ == "__main__":
    unittest.main()
