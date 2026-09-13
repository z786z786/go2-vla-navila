import json
from pathlib import Path
import tempfile
import unittest
from src.dual_target.zoh_train_core import approved_dataset


class BindingTests(unittest.TestCase):
    def test_legacy_approval_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text(json.dumps({'stage':'DT2','status':'APPROVED'}))
            with self.assertRaises(ValueError):approved_dataset(Path(d)/'dataset',p)

    def test_heldout_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text(json.dumps({'stage':'ZOH_V2_DATASET',
                'status':'APPROVED','formal_dataset_approved':True,'remote_dataset_root':d}))
            for name in ['validation','test','train']:
                with self.assertRaises(ValueError):approved_dataset(Path(d)/name,p)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'requires training torch environment')
class ChunkTests(unittest.TestCase):
    def test_five_step_chunks_never_cross_episode(self):
        from src.dual_target.zoh_train import FrameCache
        from src.dual_target.contracts import IMAGE_KEY
        c=FrameCache.__new__(FrameCache)
        c.rgb=torch.zeros((7,3,2,2),dtype=torch.uint8)
        c.state=torch.zeros((7,3));c.action=torch.arange(21).reshape(7,3).float()
        c.ends=torch.tensor([3,3,3,7,7,7,7]);c.tasks=['red']*3+['blue']*4
        b=c.batch([2,3,6])
        self.assertEqual(tuple(b['action'].shape),(3,5,3))
        self.assertEqual(b['action_is_pad'].tolist(),[[False,True,True,True,True],
            [False,False,False,False,True],[False,True,True,True,True]])
        self.assertTrue(torch.equal(b['action'][0],c.action[2].expand(5,3)))
        self.assertEqual(b['task'],['red','blue','blue'])

    def test_five_step_mask_accumulation(self):
        from src.dual_target.tiny_train_core import masked_anchor_losses
        x=torch.randn(16,5,32,requires_grad=True)
        pad=torch.arange(5)[None,:]>=(torch.arange(16)%5+1)[:,None]
        whole=masked_anchor_losses(x,pad).mean()
        accumulated=sum(masked_anchor_losses(x[i:i+4],pad[i:i+4]).mean()/4 for i in range(0,16,4))
        self.assertTrue(torch.allclose(whole,accumulated))
        whole.backward();self.assertEqual(x.grad[:,:,1].abs().sum().item(),0)
        self.assertEqual(x.grad.masked_select(pad[:,:,None].expand_as(x)).abs().sum().item(),0)


if __name__=='__main__':unittest.main()
