"""Pinned 10k scheduler, deterministic sampler extension and checkpoint gates."""
import json
from pathlib import Path

from src.dual_target.tiny_train_core import file_sha

BASE=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2')
BASELINE_5K=BASE/'zoh_scheduler_5000_0907_v1'
TOTAL=10000
EFFECTIVE_BATCH=16
SAMPLES=TOTAL*EFFECTIVE_BATCH
CHECKPOINT_STEPS=(2000,4000,6000,8000,10000)


def scheduler_config():
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    return CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=1000,num_decay_steps=30000,
        peak_lr=1e-4,decay_lr=2.5e-6)


def make_scheduler(optimizer):
    return scheduler_config().build(optimizer,num_training_steps=TOTAL)


def training_samples(root):
    import numpy as np
    import torch
    root=Path(root)
    weights=torch.from_numpy(np.load(root/'episode_balanced_sampler_weights.npy'))
    if weights.shape!=(770,) or not bool(torch.isfinite(weights).all()) or abs(float(weights.sum())-1)>1e-9:
        raise ValueError('reviewed train sampler weights changed')
    generator=torch.Generator().manual_seed(20260906)
    sampled=torch.multinomial(weights,SAMPLES,replacement=True,generator=generator)
    prior=np.load(BASELINE_5K/'sampled_indices.npy')
    if prior.shape!=(80000,) or not np.array_equal(sampled[:80000].numpy(),prior):
        raise ValueError('10k sampler does not preserve the reviewed 5k prefix')
    if sampled.min()<0 or sampled.max()>=770:raise ValueError('sample index outside train frames')
    return sampled


def verify_checkpoint(path,step,root,sampler_sha256):
    import torch
    path=Path(path);root=Path(root)
    manifest=json.loads((path/'checkpoint_manifest.json').read_text())
    if manifest['optimizer_updates']!=step:raise ValueError('checkpoint manifest step mismatch')
    for name,digest in manifest['files_sha256'].items():
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()) or file_sha(target)!=digest:
            raise ValueError('checkpoint file hash mismatch '+name)
    saved=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
    if (saved['optimizer_updates']!=step or saved['consumed_samples']!=step*EFFECTIVE_BATCH
            or saved['sampled_indices_sha256']!=sampler_sha256
            or saved['dataset_manifest_sha256']!=file_sha(root/'dataset_manifest.json')
            or saved['normalizer_sha256']!=file_sha(root/'normalizer.json')):
        raise ValueError('checkpoint training state binding mismatch')
    if saved['scheduler']['last_epoch']!=step:raise ValueError('checkpoint scheduler cursor mismatch')
    if saved['optimizer']['param_groups'][0]['lr']!=saved['lr_next']:
        raise ValueError('checkpoint optimizer/scheduler LR mismatch')
    adam_steps=[int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
    if not adam_steps or min(adam_steps)!=step or max(adam_steps)!=step:
        raise ValueError('checkpoint Adam step mismatch')
    if not {'python','numpy','torch','cuda'}<=set(saved['rng']):raise ValueError('checkpoint RNG state incomplete')
    result=dict(step=step,manifest_sha256=file_sha(path/'checkpoint_manifest.json'),
        files_verified=len(manifest['files_sha256']),optimizer_state_entries=len(adam_steps),
        scheduler_last_epoch=saved['scheduler']['last_epoch'],lr_next=saved['lr_next'],
        consumed_samples=saved['consumed_samples'],passed=True)
    del saved
    return result


def verify_run(path,scope,root,deep=True):
    """Verify one completed train-only run without loading validation or Isaac."""
    import numpy as np
    path,root=Path(path).resolve(),Path(root).resolve()
    expected_status=('ZOH_EXPERT_10000_COMPLETE_AWAITING_REVIEW' if scope=='expert'
        else 'ZOH_FULL_10000_COMPLETE_AWAITING_REVIEW')
    result=json.loads((path/'result.json').read_text())
    config=json.loads((path/'training_config.json').read_text())
    if (result.get('status')!=expected_status or result.get('optimizer_updates')!=TOTAL
            or result.get('scope')!=scope):
        raise ValueError('completed 10k result contract failed')
    for key in ('validation_read','evaluation_loss','rollout'):
        if result.get(key) is not False or config.get(key) is not False:
            raise ValueError(f'train-only contract failed: {key}')
    expected=dict(scope=scope,effective_batch=16,micro_batch=4,gradient_accumulation=4,
        scheduler='cosine_decay_with_warmup',target_optimizer_updates=10000,
        formal_checkpoint_interval=2000,fps=5,chunk_size=5,execute_steps=1,
        fresh_from_base=True)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('10k training configuration drift')
    actual=sorted(int(p.name.split('_')[1]) for p in path.glob('checkpoint_*') if p.is_dir())
    if actual!=list(CHECKPOINT_STEPS):raise ValueError('formal checkpoint set changed')
    sampled=np.load(path/'sampled_indices.npy')
    prior=np.load(BASELINE_5K/'sampled_indices.npy')
    if (sampled.shape!=(SAMPLES,) or not np.array_equal(sampled[:80000],prior)
            or file_sha(path/'sampled_indices.npy')!=config['sampled_indices_sha256']):
        raise ValueError('10k sampled-index binding failed')
    for gate_name in ('single_update_gate.json','final_reload_gate.json'):
        gate=json.loads((path/gate_name).read_text())
        if gate.get('passed') is not True:raise ValueError(f'{gate_name} failed')
    if scope=='full':
        for gate_name in ('full_gradient_gate.json','full_weight_update_gate.json'):
            gate=json.loads((path/gate_name).read_text())
            if gate.get('passed') is not True:raise ValueError(f'{gate_name} failed')
    recorded=json.loads((path/'checkpoint_verification.json').read_text())
    if sorted(map(int,recorded))!=list(CHECKPOINT_STEPS) or not all(x.get('passed') for x in recorded.values()):
        raise ValueError('checkpoint verification receipts incomplete')
    verified={}
    if deep:
        for step in CHECKPOINT_STEPS:
            verified[str(step)]=verify_checkpoint(path/f'checkpoint_{step:06d}',step,root,
                config['sampled_indices_sha256'])
    return dict(scope=scope,status=expected_status,optimizer_updates=TOTAL,
        sampled_indices_sha256=config['sampled_indices_sha256'],checkpoints=list(CHECKPOINT_STEPS),
        checkpoint_receipts_sha256=file_sha(path/'checkpoint_verification.json'),
        result_sha256=file_sha(path/'result.json'),deep_verified=deep,
        deep_verification=verified,validation_read=False,evaluation_loss=False,rollout=False,passed=True)
