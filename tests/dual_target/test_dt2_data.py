import unittest
from src.dual_target.tiny_data import normalized_stats, chunk_indices


class TinyDataTests(unittest.TestCase):
    def test_train_normalizer_constant_vy_is_safe(self):
        stats = normalized_stats([[0.,0.,-.5], [.5,0.,.5]])
        self.assertEqual(stats['mean'], [.25,0.,0.])
        self.assertEqual(stats['raw_std'], [.25,0.,.5])
        self.assertEqual(stats['std'], [.25,1.,.5])
        self.assertEqual(stats['constant_channels'], [1])
        for bad in ([], [[0.,0.]], [[0.,0.,float('nan')]]):
            with self.assertRaises(ValueError):
                normalized_stats(bad)

    def test_chunk_padding_never_crosses_episode(self):
        idx, pad = chunk_indices(98,100)
        self.assertEqual(idx, [98]+[99]*49)
        self.assertEqual(pad, [False,False]+[True]*48)
        self.assertEqual(chunk_indices(50,100)[1], [False]*50)
        self.assertEqual(chunk_indices(99,100)[1], [False]+[True]*49)
        with self.assertRaises(ValueError):
            chunk_indices(100,100)


if __name__ == '__main__':
    unittest.main()
