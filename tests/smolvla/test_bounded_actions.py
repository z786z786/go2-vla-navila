import unittest

try:
    import torch

    from src.smolvla.bounded_actions import decode_latent_actions, encode_physical_actions
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is available in the server SmolVLA environment, not this local host")
class BoundedActionCodecTests(unittest.TestCase):
    def test_exact_physical_boundaries_encode_and_decode_inside_immutable_bounds(self):
        physical = torch.tensor(
            [[0.0, 0.0, -0.5], [0.5, 0.0, 0.5]], dtype=torch.float64
        )

        decoded = decode_latent_actions(encode_physical_actions(physical))

        self.assertTrue(torch.all(decoded[..., 0] >= 0.0))
        self.assertTrue(torch.all(decoded[..., 0] <= 0.5))
        self.assertTrue(torch.equal(decoded[..., 1], torch.zeros_like(decoded[..., 1])))
        self.assertTrue(torch.all(decoded[..., 2] >= -0.5))
        self.assertTrue(torch.all(decoded[..., 2] <= 0.5))

    def test_interior_values_round_trip(self):
        physical = torch.tensor(
            [[[0.125, 0.0, -0.25], [0.375, 0.0, 0.25]]], dtype=torch.float64
        )

        decoded = decode_latent_actions(encode_physical_actions(physical))

        self.assertTrue(torch.allclose(decoded, physical, atol=1e-9, rtol=0.0))

    def test_large_finite_latents_are_always_decoded_to_safe_physical_actions(self):
        latent = torch.tensor(
            [[[-1e6, 3.0, 1e6], [1e6, -2.0, -1e6]]], dtype=torch.float32
        )

        decoded = decode_latent_actions(latent)

        self.assertTrue(torch.all(torch.isfinite(decoded)))
        self.assertTrue(torch.all((decoded[..., 0] >= 0.0) & (decoded[..., 0] <= 0.5)))
        self.assertTrue(torch.equal(decoded[..., 1], torch.zeros_like(decoded[..., 1])))
        self.assertTrue(torch.all((decoded[..., 2] >= -0.5) & (decoded[..., 2] <= 0.5)))

    def test_codec_preserves_dtype_and_device(self):
        physical = torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32)

        latent = encode_physical_actions(physical)
        decoded = decode_latent_actions(latent)

        self.assertEqual(latent.dtype, physical.dtype)
        self.assertEqual(latent.device, physical.device)
        self.assertEqual(decoded.dtype, physical.dtype)
        self.assertEqual(decoded.device, physical.device)

    def test_nonfinite_and_invalid_physical_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_physical_actions(torch.tensor([[float("nan"), 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            decode_latent_actions(torch.tensor([[float("inf"), 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            encode_physical_actions(torch.tensor([[0.1, 0.01, 0.0]]))


if __name__ == "__main__":
    unittest.main()
