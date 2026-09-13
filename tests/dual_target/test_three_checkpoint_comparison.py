import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import zoh_compare_common as common


class ComparisonContractTests(unittest.TestCase):
    def test_full_paired_matrix(self):
        checkpoints={name:dict(checkpoint=name,checkpoint_manifest_sha256='digest') for name in common.RUNS}
        tasks=common.tasks_for(Path('evaluation'),checkpoints)
        self.assertEqual(len(tasks),72)
        self.assertEqual(len({t['run_id'] for t in tasks}),72)
        for slot in range(24):
            rows=[t for t in tasks if t['slot']==slot]
            self.assertEqual([t['condition'] for t in rows],list(common.RUNS))
            self.assertTrue(all(t['policy_seed']==common.SEED and t['step']==5000 for t in rows))
            self.assertTrue(all(t['evaluation_split']==('train' if slot<16 else 'validation') for t in rows))

    def test_pending_not_counted_as_failures(self):
        row=dict(condition='A_constant_expert',evaluation_split='train',status='failed_wrong_target_stop',
            review=dict(collision=False,reached_correct_region=False,reached_other_region=True,final_distance_m=2.))
        metrics=common.metric_rows([row])
        self.assertEqual(metrics[0]['completed'],1)
        self.assertEqual(metrics[0]['wrong_target_stop'],1)
        self.assertEqual(metrics[0]['timeout'],0)
        self.assertEqual(metrics[1]['completed'],0)
        self.assertIsNone(metrics[1]['mean_final_parking_distance_m'])

    def test_initial_report_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(common,'ROOT',Path(directory)):
            common.publish([])
            result=json.loads((Path(directory)/'delivery/results.json').read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['metrics']),6)
            self.assertFalse(list(Path(directory).rglob('*.tmp')))


class FullScopeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.model=torch.nn.Module()
        for name in ('vision_model','text_model','lm_expert','state_proj','action_out_proj'):
            setattr(self.model,name,torch.nn.Linear(2,2))

    def test_full_scope_and_gradients(self):
        from scripts.zoh_full_support import checked_parameters,full_gradient_audit,weight_updates
        self.assertTrue(all(r['trainable'] for r in checked_parameters(self.model)))
        before={'vision':('vision_model.weight',self.model.vision_model.weight.detach().clone())}
        optimizer=self.torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        sum(p.square().sum() for p in self.model.parameters()).backward()
        self.assertEqual(len(full_gradient_audit(self.model)['groups']),3)
        optimizer.step()
        self.assertGreater(weight_updates(self.model,before)['vision']['max_abs_update'],0)

    def test_reject_frozen_or_missing_gradients(self):
        from scripts.zoh_full_support import checked_parameters,full_gradient_audit
        self.model.vision_model.weight.requires_grad_(False)
        with self.assertRaises(ValueError):checked_parameters(self.model)
        with self.assertRaises(ValueError):full_gradient_audit(self.model)

    def test_inference_dispatch(self):
        from scripts.zoh_full_support import load_for_inference
        from types import SimpleNamespace
        with patch('lerobot.configs.PreTrainedConfig.from_pretrained',return_value=SimpleNamespace(train_expert_only=False)), \
                patch('scripts.zoh_full_support.load_training_policy',return_value='full') as full:
            self.assertEqual(load_for_inference(Path('/checkpoint')),'full')
            full.assert_called_once()


if __name__=='__main__':unittest.main()
