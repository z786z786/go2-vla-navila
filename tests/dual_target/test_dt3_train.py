import unittest
import json
from pathlib import Path
import tempfile

from src.dual_target.tiny_train_core import resource_candidates, choose_microbatch, approved_dataset, file_sha


class ResourceTests(unittest.TestCase):
    def test_fixed_effective_batch(self):
        self.assertEqual(resource_candidates(), (1, 2, 4, 8, 16))
        with self.assertRaises(ValueError):
            resource_candidates(32)

    def test_throughput_not_blind_memory_fill(self):
        self.assertEqual(choose_microbatch([
            {'micro_batch': 2, 'passed': True, 'samples_per_second': 10},
            {'micro_batch': 4, 'passed': True, 'samples_per_second': 10.3},
            {'micro_batch': 8, 'passed': False}]), 2)
        with self.assertRaises(RuntimeError):
            choose_microbatch([])

    def test_no_training_without_full_bound_dt2_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            norm = {'source_split':'train','action_loss_active_channels':[True,False,True],
                    'action':{'mean':[0,0,0],'std':[1,1,1]}}
            (root/'normalizer.json').write_text(json.dumps(norm))
            manifest = {'episode_count':16,'split':'train','status':'CONVERTED_NOT_DT2_APPROVED',
                        'files_sha256':{'normalizer.json':file_sha(root/'normalizer.json')}}
            (root/'dataset_manifest.json').write_text(json.dumps(manifest))
            approval = {'stage':'DT2','status':'APPROVED','dataset_root':str(root),
                        'dataset_manifest_sha256':file_sha(root/'dataset_manifest.json')}
            ap = root/'approval.json'
            ap.write_text(json.dumps(approval))
            self.assertEqual(approved_dataset(root,ap)[0]['episode_count'],16)
            approval['status'] = 'NOT_APPROVED'
            ap.write_text(json.dumps(approval))
            with self.assertRaises(ValueError):
                approved_dataset(root,ap)
            approval['status'] = 'APPROVED'
            ap.write_text(json.dumps(approval))
            (root/'normalizer.json').write_text('{}')
            with self.assertRaises(ValueError):
                approved_dataset(root,ap)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires remote torch environment')
class MaskTests(unittest.TestCase):
    def test_mask_and_ignored_channels_have_zero_gradient(self):
        from src.dual_target.tiny_train_core import masked_anchor_losses
        x = torch.arange(2*3*32, dtype=torch.float32).reshape(2,3,32).requires_grad_()
        pad = torch.tensor([[False,False,False], [False,True,True]])
        loss = masked_anchor_losses(x, pad)
        self.assertTrue(torch.equal(loss, torch.stack([x[0,:,[0,2]].mean(), x[1,0,[0,2]].mean()])))
        loss.mean().backward()
        self.assertEqual(x.grad[:,:,1].abs().sum().item(), 0)
        self.assertEqual(x.grad[:,:,3:].abs().sum().item(), 0)
        self.assertEqual(x.grad[1,1:].abs().sum().item(), 0)
        self.assertAlmostEqual(x.grad.sum().item(), 1.0)

    def test_micro_accumulation_matches_whole_batch_and_allpad_rejected(self):
        from src.dual_target.tiny_train_core import masked_anchor_losses
        x = torch.randn(16,50,32, requires_grad=True)
        pad = torch.arange(50)[None,:] >= torch.arange(1,17)[:,None]
        whole = masked_anchor_losses(x,pad).mean()
        micro = sum(masked_anchor_losses(x[i:i+4],pad[i:i+4]).mean()/4 for i in range(0,16,4))
        self.assertTrue(torch.allclose(whole,micro))
        with self.assertRaises(ValueError):
            masked_anchor_losses(x, torch.ones((16,50),dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
