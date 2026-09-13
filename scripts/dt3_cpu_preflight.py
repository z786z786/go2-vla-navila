"""CPU-only test of undeployed DT3 candidates against the installed LeRobot."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

candidate = Path(sys.argv[1])
sys.path.insert(0,'/home/wxh/go2_short_vln')
for short in ('tiny_train_core','tiny_train','tiny_train_launch'):
    name = 'src.dual_target.'+short
    spec = importlib.util.spec_from_file_location(name,candidate/f'src/dual_target/{short}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
spec = importlib.util.spec_from_file_location('dt3_candidate_tests',candidate/'tests/dual_target/test_dt3_train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
assert result.wasSuccessful() and not result.skipped

import torch
from lerobot.configs import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.dual_target.tiny_train import processors,checked_parameters,load_training_policy,reload_processors
from src.dual_target.smolvla_probe_client import _build_new_3d_config,DEFAULT_BASE_MODEL
from src.dual_target.contracts import IMAGE_KEY,STATE_KEY

assert not torch.cuda.is_available(), 'CPU-only preflight required'
torch.set_num_threads(8)
config = _build_new_3d_config(DEFAULT_BASE_MODEL)
config.device = 'cpu'
config.normalization_mapping = {'VISUAL':NormalizationMode.IDENTITY,
    'STATE':NormalizationMode.MEAN_STD,'ACTION':NormalizationMode.MEAN_STD}
probe = Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt2_converter_probe_0906_v1')
normalizer = json.loads((probe/'normalizer.json').read_text())
pre,post = processors(config,normalizer)
dataset = LeRobotDataset('local/dual_target_v1_tiny',root=probe/'lerobot',
                        delta_timestamps={'action':[i/50 for i in range(50)]})
items = [dataset[0],dataset[486]]
raw = {key:torch.stack([item[key] for item in items])
       for key in (IMAGE_KEY,STATE_KEY,'action','action_is_pad')}
raw['task'] = [i['task'] for i in items]
batch = pre(raw)
assert batch[STATE_KEY].shape == (2,3) and batch['action'].shape == (2,50,3)
assert torch.equal(batch['action_is_pad'],raw['action_is_pad'])
assert batch['action_is_pad'][1].sum().item() == 49
assert batch['action'][:,:,1].abs().sum().item() == 0
assert torch.equal(batch[IMAGE_KEY],raw[IMAGE_KEY])
expected = (raw['action']-torch.tensor(normalizer['action']['mean']))/torch.tensor(normalizer['action']['std'])
assert torch.allclose(expected,batch['action'],atol=1e-6,rtol=1e-6)
assert torch.allclose(post(batch['action']),raw['action'],atol=1e-6,rtol=1e-6)
import tempfile
saved_processors = Path(tempfile.mkdtemp(prefix='processors_',dir=candidate))
pre.save_pretrained(saved_processors)
post.save_pretrained(saved_processors)
pre2,post2 = reload_processors(saved_processors)
batch2 = pre2(raw)
assert all(torch.equal(v,batch2[k]) for k,v in batch.items() if isinstance(v,torch.Tensor))
assert torch.equal(post(batch['action']),post2(batch['action']))
policy = load_training_policy(DEFAULT_BASE_MODEL,config)
scope = checked_parameters(policy)
assert not policy.model.vlm_with_expert.vlm.training
assert {r['dtype'] for r in scope if r['trainable']} == {'torch.float32'}
evidence = {'passed':True,'cpu_only':True,'training':False,
    'unit_tests':result.testsRun,'real_processor_padding_and_normalizer':True,
    'parameters':sum(r['count'] for r in scope),
    'trainable_parameters':sum(r['count'] for r in scope if r['trainable']),
    'trainable_dtypes':sorted({r['dtype'] for r in scope if r['trainable']}),
    'saved_processors_real_reload_equal':True}
(candidate/'cpu_preflight_result.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence))
