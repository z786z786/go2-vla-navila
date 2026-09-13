"""Phase 1 prompt contract and training-loop helpers (stdlib unittest).

The tokenizer tests need transformers and the SmolVLM2 snapshot; they skip where
either is absent (the local box has no transformers, the remote GPU host does).
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import torch

from src.smolvla.phase1_prompt import (EXPECTED_IDS, HISTORY_FRAMES, MAX_INSTRUCTION_TOKENS,
                                       TOKENS_PER_FRAME, Phase1PromptBuilder)

DATA_ROOT = Path(os.environ.get("GO2_DATA_ROOT", "/mnt/wxh/go2_short_vln"))
MODEL = DATA_ROOT / ("cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
                     "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467")
HAVE_TOKENIZER = importlib.util.find_spec("transformers") is not None and MODEL.is_dir()

ACTIONS = ["The next action is move forward 25 cm.", "The next action is move forward 50 cm.",
           "The next action is move forward 75 cm.", "The next action is turn left 15 degree.",
           "The next action is turn left 30 degree.", "The next action is turn left 45 degree.",
           "The next action is turn right 15 degree.", "The next action is turn right 30 degree.",
           "The next action is turn right 45 degree.",
           "I think I should stop because I have finished the instruction."]


def load_train_module():
    path = Path(__file__).resolve().parents[2] / "scripts/p5_phase1_train.py"
    spec = importlib.util.spec_from_file_location("p5_phase1_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_TOKENIZER, "transformers or SmolVLM2 snapshot unavailable")
class PromptBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cls.b = Phase1PromptBuilder(cls.tok)

    def test_image_slots_and_blocks(self):
        ids = self.b.prompt("Walk to the kitchen.").ids
        self.assertEqual(ids.count(EXPECTED_IDS["<image>"]), HISTORY_FRAMES * TOKENS_PER_FRAME)
        self.assertEqual(ids.count(EXPECTED_IDS["<global-img>"]), HISTORY_FRAMES)
        self.assertEqual(ids.count(EXPECTED_IDS["<fake_token_around_image>"]), 2 * HISTORY_FRAMES)
        # every image run is exactly one 64-token block
        runs, cur = [], 0
        for t in ids + [-1]:
            if t == EXPECTED_IDS["<image>"]:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        self.assertEqual(runs, [TOKENS_PER_FRAME] * HISTORY_FRAMES)

    def test_prompt_ends_at_assistant_and_eos_in_prompt_once(self):
        ids = self.b.prompt("Walk to the kitchen.").ids
        self.assertTrue(self.tok.decode(ids[-3:]).endswith("Assistant:"))
        self.assertEqual(ids[0], self.tok.convert_tokens_to_ids("<|im_start|>"))
        self.assertEqual(ids.count(EXPECTED_IDS["<end_of_utterance>"]), 1)   # closes the user turn

    def test_render_matches_navila_question(self):
        text = self.b.render(" Walk to the kitchen. ", ACTIONS[2])
        self.assertIn('Your assigned task is: "Walk to the kitchen." Analyze this series', text)
        self.assertIn("Imagine you are a robot programmed for navigation tasks.", text)
        self.assertTrue(text.endswith("Assistant: The next action is move forward 75 cm.<end_of_utterance>"))

    def test_whitespace_strip_makes_train_and_eval_identical(self):
        self.assertEqual(self.b.prompt("Go left. ").ids, self.b.prompt("Go left.").ids)

    def test_every_action_round_trips_and_ends_with_eos(self):
        from actions.language_parser import parse_nav_command
        for a in ACTIONS:
            t = self.b.target(a)
            self.assertEqual(t[-1], EXPECTED_IDS["<end_of_utterance>"])
            self.assertEqual(t.count(EXPECTED_IDS["<end_of_utterance>"]), 1)
            decoded = self.b.decode_answer(t)
            self.assertEqual(decoded, a)
            self.assertEqual(parse_nav_command(decoded).classification, "exact")

    def test_instruction_truncated_never_template_or_target(self):
        long = "walk forward past the table and " * 80
        p = self.b.prompt(long)
        self.assertTrue(p.instruction_truncated)
        self.assertEqual(len(p.ids), self.b.template_tokens + MAX_INSTRUCTION_TOKENS)
        self.assertTrue(self.tok.decode(p.ids[-3:]).endswith("Assistant:"))

    def test_collate_supervises_exactly_the_answer(self):
        mod = load_train_module()
        fn = mod.collate_fn(self.b, pad_id=self.tok.pad_token_id)
        frames = torch.zeros(HISTORY_FRAMES, 3, 4, 4, dtype=torch.uint8)
        items = [{"instruction_raw": "Go left. ", "action_text": ACTIONS[3], "frames": frames,
                  "record": {"action_id": 3}},
                 {"instruction_raw": "Walk out of the bedroom and stop at the top of the stairs. ",
                  "action_text": ACTIONS[9], "frames": frames, "record": {"action_id": 9}}]
        batch = fn(items)
        for i, it in enumerate(items):
            sup_ids = batch["ids"][i][batch["sup"][i]].tolist()
            self.assertEqual(sup_ids, self.b.target(it["action_text"]))
            self.assertEqual(int(batch["attn"][i].sum()),
                             len(self.b.prompt(it["instruction_raw"]).ids) + len(sup_ids))
        shuffled = mod.collate_fn(self.b, pad_id=self.tok.pad_token_id, shuffle_instructions=True)(items)
        self.assertEqual(shuffled["instruction"], [items[1]["instruction_raw"], items[0]["instruction_raw"]])
        self.assertEqual(shuffled["differs"].tolist(), [True, True])


class TrainingHelperTests(unittest.TestCase):
    def test_sampler_resume_continues_the_same_permutation(self):
        mod = load_train_module()
        s = mod.ResumableEpochSampler(100, seed=7)
        s.set_position(1, 0); full = list(s)
        s.set_position(1, 48); tail = list(s)
        self.assertEqual(tail, full[48:])
        s.set_position(2, 0)
        self.assertNotEqual(list(s), full)
        self.assertEqual(sorted(full), list(range(100)))

    def test_lr_schedule_warmup_then_cosine_to_zero(self):
        mod = load_train_module()
        f = mod.lr_lambda(warmup=10, total=110)
        self.assertAlmostEqual(f(0), 0.1)
        self.assertAlmostEqual(f(9), 1.0)
        self.assertAlmostEqual(f(10), 1.0)
        self.assertAlmostEqual(f(60), 0.5)
        self.assertAlmostEqual(f(110), 0.0)


if __name__ == "__main__":
    unittest.main()
