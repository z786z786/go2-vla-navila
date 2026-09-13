import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'requires torch')
class ResumeSamplerTests(unittest.TestCase):
    def test_prefix_and_continuation_exact(self):
        from src.dual_target.zoh_train_resume5k import extend_sampler
        weights=torch.tensor([.2,.3,.5],dtype=torch.float64)
        gen=torch.Generator().manual_seed(20260906)
        old=torch.multinomial(weights,16,replacement=True,generator=gen)
        new=torch.multinomial(weights,64,replacement=True,generator=gen)
        result=extend_sampler(weights,old,16,80)
        self.assertTrue(torch.equal(result,torch.cat([old,new])))
        self.assertTrue(torch.equal(result[:16],old))

    def test_wrong_prefix_rejected(self):
        from src.dual_target.zoh_train_resume5k import extend_sampler
        with self.assertRaises(ValueError):
            extend_sampler(torch.tensor([.5,.5]),torch.full((16,),9),16,80)

if __name__=='__main__':unittest.main()
