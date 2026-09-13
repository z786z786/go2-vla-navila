"""Explicit CPU preflight on the remote SmolVLA environment, no model/CUDA."""
import importlib.util
import unittest


class ProcessorIntegration(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('lerobot') and importlib.util.find_spec('torch'),
                         'requires remote LeRobot/SmolVLA environment')
    def test_fresh_three_dimensional_processor_roundtrip(self):
        import torch
        from src.dual_target.smolvla_probe_client import _build_new_3d_config, DEFAULT_BASE_MODEL
        from src.dual_target.contracts import IMAGE_KEY, STATE_KEY, RED_TASK
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        config = _build_new_3d_config(DEFAULT_BASE_MODEL)
        config.device = 'cpu'
        pre, post = make_smolvla_pre_post_processors(config)
        result = pre({IMAGE_KEY: torch.zeros(3, 512, 512),
                      STATE_KEY: torch.tensor([.1, .0, -.2]), 'task': RED_TASK})
        self.assertEqual(tuple(result[STATE_KEY].shape), (1, 3))
        self.assertEqual(tuple(result[IMAGE_KEY].shape), (1, 3, 512, 512))
        self.assertTrue(torch.equal(result[STATE_KEY], torch.tensor([[.1, 0., -.2]])))
        self.assertEqual(tuple(post(torch.zeros(1, 50, 3)).shape), (1, 50, 3))
        self.assertFalse(torch.cuda.is_initialized())
