"""Single-variable scheduler experiment: pinned baseline samples and official LR schedule."""
import json
from pathlib import Path
from .tiny_train_core import file_sha

BASELINE=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_5000_0906_v2')
INITIAL=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_1000_0906_v1')
TOTAL=5000


def scheduler_config():
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    return CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=1000,num_decay_steps=30000,
        peak_lr=1e-4,decay_lr=2.5e-6)


def make_scheduler(optimizer):
    return scheduler_config().build(optimizer,num_training_steps=TOTAL)


def baseline_samples(root):
    import numpy as np
    import torch
    cfg=json.loads((BASELINE/'training_config.json').read_text())
    first=json.loads((INITIAL/'training_config.json').read_text())
    result=json.loads((BASELINE/'result.json').read_text())
    if result['optimizer_updates']!=5000 or result['status']!='ZOH_5000_COMPLETE_AWAITING_REVIEW':
        raise ValueError('completed constant 5000 baseline required')
    expected=dict(effective_batch=16,micro_batch=4,gradient_accumulation=4,learning_rate=1e-4,
        scheduler='constant',fps=5,chunk_size=5,execute_steps=1,seed=20260906)
    for c in (first,cfg):
        if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('baseline config differs')
        for key,name in [('dataset_manifest_sha256','dataset_manifest.json'),('normalizer_sha256','normalizer.json')]:
            if c[key]!=file_sha(root/name):raise ValueError('baseline dataset mismatch')
    for path,c in ((BASELINE,cfg),(INITIAL,first)):
        if file_sha(path/'sampled_indices.npy')!=c['sampled_indices_sha256']:
            raise ValueError('baseline sampler hash mismatch')
    sampled=np.load(BASELINE/'sampled_indices.npy')
    if sampled.shape!=(80000,) or sampled.dtype.kind not in 'iu' or sampled.min()<0 or sampled.max()>=770:
        raise ValueError('wrong baseline sampling domain')
    if not np.array_equal(sampled[:16000],np.load(INITIAL/'sampled_indices.npy')):
        raise ValueError('baseline prefix changed')
    return torch.from_numpy(sampled.copy())


def assert_scope_matches(scope):
    old=json.loads((INITIAL/'parameter_scope.json').read_text())
    if scope!=old:raise ValueError('trainable scope/count/dtype changed from constant baseline')


def baseline_first_loss():
    for line in (INITIAL/'events.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('event')=='update' and row['optimizer_updates']==1:return row['loss']
    raise ValueError('baseline first loss missing')
