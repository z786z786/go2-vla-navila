import copy
import math
import unittest
try:
    import torch
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'requires actual torch/lerobot environment')
class SchedulerTests(unittest.TestCase):
    def setup_run(self):
        from src.dual_target.zoh_scheduler_core import make_scheduler
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=1e-4,betas=(.9,.95),weight_decay=1e-10,foreach=False)
        return p,opt,make_scheduler(opt)

    def test_actual_5000_lr_curve(self):
        p,opt,s=self.setup_run();rates=[]
        for step in range(5000):
            rates.append(opt.param_groups[0]['lr'])
            p.grad=torch.ones_like(p);opt.step();s.step()
        self.assertAlmostEqual(rates[0],1e-4/167,places=15)
        self.assertLess(rates[0],rates[100])
        self.assertLess(rates[100],rates[165])
        self.assertGreater(rates[167],rates[2500])
        self.assertGreater(rates[2500],rates[-1])
        self.assertAlmostEqual(opt.param_groups[0]['lr'],2.5e-6,places=14)
        self.assertEqual(s.last_epoch,5000)
        self.assertTrue(all(math.isfinite(x) and 0<x<=1e-4 for x in rates))

    def test_optimizer_scheduler_resume_exact(self):
        from src.dual_target.zoh_scheduler_core import make_scheduler
        p,opt,s=self.setup_run()
        for i in range(17):
            p.grad=torch.tensor([.3]);opt.step();s.step()
        before=p.detach().clone();os=copy.deepcopy(opt.state_dict());ss=copy.deepcopy(s.state_dict())
        expected=[]
        for i in range(10):
            expected.append(opt.param_groups[0]['lr']);p.grad=torch.tensor([.7]);opt.step();s.step()
        p2,opt2,s2=self.setup_run();p2.data.copy_(before)
        opt2.load_state_dict(os);s2.load_state_dict(ss)
        actual=[]
        for i in range(10):
            actual.append(opt2.param_groups[0]['lr']);p2.grad=torch.tensor([.7]);opt2.step();s2.step()
        self.assertEqual(expected,actual)
        self.assertTrue(torch.equal(p,p2))
        self.assertEqual(s.state_dict(),s2.state_dict())

class TensorBoardTests(unittest.TestCase):
    def test_actual_lr_survives_parser(self):
        import json
        from scripts.zoh_scheduled_training_to_tensorboard import training_row
        row=training_row(json.dumps(dict(event='update',optimizer_updates=1,loss=.5,lr=1e-4/167)))
        self.assertEqual(row['lr'],1e-4/167)

if __name__=='__main__':unittest.main()
