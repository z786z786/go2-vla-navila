"""Bounded fresh 5 Hz ZOH first training stage, invoked only by the CPU resource supervisor.

Uses the approved lossless LeRobot dataset, fresh train statistics, frozen
VLM and trainable expert/projections. No Isaac/GT/planner is imported.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

from .contracts import IMAGE_KEY, STATE_KEY
from .zoh_train_core import approved_dataset
from .tiny_train_core import file_sha, flow_loss, resource_candidates, choose_microbatch
from .smolvla_probe_client import (
    DEFAULT_BASE_MODEL, DEFAULT_BASE_CONFIG_SHA256, DEFAULT_BASE_MODEL_SHA256, _build_new_3d_config)


def emit(root, event, **values):
    row = {'stage': 'DT3', 'event': event, 'wall_time_s': time.time(), **values}
    with (root/'events.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class LosslessFrameView:
    """CPU-only conversion to bounded uint8 cache, retaining real loader values."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        import torch
        item = self.dataset[i]
        return {'rgb': (item[IMAGE_KEY]*255).round().to(torch.uint8),
                'state': item[STATE_KEY], 'action': item['action'],
                'episode': item['episode_index'], 'task': item['task']}


class FrameCache:
    def __init__(self, root, manifest, out):
        import torch
        from torch.utils.data import DataLoader
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        n = manifest['frames']
        if n*3*512*512 > 16*1024**3:
            raise ValueError('tiny uint8 RGB cache exceeds explicit 16 GiB CPU RAM cap')
        dataset = LeRobotDataset('local/go2_zoh_v2_train', root=root/'train')
        if len(dataset) != n:
            raise ValueError('training loader length changed')
        self.rgb = torch.empty((n,3,512,512),dtype=torch.uint8)
        self.state, self.action = torch.empty((n,3)), torch.empty((n,3))
        self.ends = torch.empty(n,dtype=torch.long)
        self.tasks, episodes = [], []
        offset = 0
        for batch in DataLoader(LosslessFrameView(dataset), batch_size=32, num_workers=4,
                                shuffle=False, prefetch_factor=2, persistent_workers=False):
            k = len(batch['task'])
            self.rgb[offset:offset+k] = batch['rgb']
            self.state[offset:offset+k] = batch['state']
            self.action[offset:offset+k] = batch['action']
            episodes.extend(batch['episode'].tolist())
            self.tasks.extend(batch['task'])
            offset += k
        for ep in range(16):
            indices = [i for i, e in enumerate(episodes) if e == ep]
            if not indices or indices != list(range(indices[0],indices[-1]+1)):
                raise ValueError('cache episode coverage/contiguity changed')
            self.ends[indices[0]:indices[-1]+1] = indices[-1]+1
        if offset != n or self.action[:,1].abs().sum().item() != 0:
            raise ValueError('cache coverage or fixed vy changed')
        emit(out, 'lossless_cpu_cache_ready', frames=n, rgb_bytes=self.rgb.numel(),
             workers=4, cpu_threads=torch.get_num_threads())

    def batch(self, indices):
        import torch
        idx = torch.as_tensor(indices, dtype=torch.long)
        future = idx[:,None] + torch.arange(5)[None,:]
        padding = future >= self.ends[idx,None]
        future = torch.minimum(future, self.ends[idx,None]-1)
        return {IMAGE_KEY: self.rgb[idx].float()/255., STATE_KEY: self.state[idx],
                'action': self.action[future], 'action_is_pad': padding,
                'task': [self.tasks[i] for i in idx.tolist()]}


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def processors(config, normalizer):
    import torch
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    stats = {key: {name: torch.tensor(normalizer[kind][name], dtype=torch.float32)
                   for name in ('mean','std','min','max')}
             for key,kind in ((STATE_KEY,'state'),('action','action'))}
    return make_smolvla_pre_post_processors(config, dataset_stats=stats)


def reload_processors(path):
    from lerobot.processor import PolicyProcessorPipeline,policy_action_to_transition,transition_to_policy_action
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME,POLICY_POSTPROCESSOR_DEFAULT_NAME
    pre = PolicyProcessorPipeline.from_pretrained(path,
        config_filename=POLICY_PREPROCESSOR_DEFAULT_NAME+'.json',local_files_only=True)
    post = PolicyProcessorPipeline.from_pretrained(path,
        config_filename=POLICY_POSTPROCESSOR_DEFAULT_NAME+'.json',local_files_only=True,
        to_transition=policy_action_to_transition,to_output=transition_to_policy_action)
    return pre,post


def optimizer_for(policy):
    import torch
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
        lr=1e-4, betas=(.9,.95), eps=1e-8, weight_decay=1e-10, foreach=False)


def load_training_policy(path, config=None):
    """Keep trainable master weights/moments FP32, frozen VLM in base dtype.

    Cast BEFORE strict safetensors loading, including checkpoint reloads, so
    FP32 updates cannot be rounded through a constructor's BF16 parameters.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_model
    if config is None:
        config = PreTrainedConfig.from_pretrained(path,local_files_only=True)
    policy = SmolVLAPolicy(config)
    for param in policy.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    load_model(policy,str(path/'model.safetensors'),strict=True,device='cpu')
    return policy.to(config.device).train()


def checked_parameters(policy):
    allowed = ('model.vlm_with_expert.lm_expert.', 'model.state_proj.', 'model.action_in_proj.',
               'model.action_out_proj.', 'model.action_time_mlp_in.', 'model.action_time_mlp_out.')
    rows = [{'name': n, 'count': p.numel(), 'dtype': str(p.dtype), 'trainable': p.requires_grad}
            for n,p in policy.named_parameters()]
    bad = [r['name'] for r in rows if r['trainable'] and not r['name'].startswith(allowed)]
    if any(r['trainable'] and r['dtype'] != 'torch.float32' for r in rows):
        raise ValueError('trainable master parameters must be FP32')
    if bad or not any(r['trainable'] and r['name'].startswith(allowed[0]) for r in rows):
        raise ValueError(f'unexpected trainable parameter scope: {bad}')
    for prefix in allowed[1:]:
        if not any(r['trainable'] and r['name'].startswith(prefix) for r in rows):
            raise ValueError(f'missing trainable projection: {prefix}')
    return rows


def profile(policy, pre, cache, out):
    import torch
    from .gpu_wait import probe_gpu
    results = []
    params = [p for p in policy.parameters() if p.requires_grad]
    # Reserve Adam's two moment buffers before forward/backward; no optimizer
    # update during profiling and no accidental warmup changes to base weights.
    moments = [torch.empty((2,p.numel()),device=p.device,dtype=p.dtype) for p in params]
    for micro in (1, 2, 4):
        try:
            policy.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Real RGB, task, states, boundary-aware targets; three repeated passes.
            batch = pre(cache.batch(list(range(micro))))
            if batch['action_is_pad'].shape != (micro,5):
                raise ValueError('processor changed padding mask')
            durations = []
            for trial in range(3):
                policy.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = flow_loss(policy,batch)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('profile loss is nonfinite')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-started)
            free = probe_gpu().free_mib
            result = {'micro_batch': micro, 'passed': free >= 2048,
                      'samples_per_second': 2*micro/sum(durations[1:]),
                      'seconds': durations, 'free_mib_after': free,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
                      'gradient_norm': float(norm), 'adam_moment_buffers_reserved': True,
                      'optimizer_updates': 0}
            del loss, batch
        except torch.cuda.OutOfMemoryError as exc:
            result = {'micro_batch': micro, 'passed': False, 'reason': 'allocator_OOM', 'detail': str(exc)}
        results.append(result)
        emit(out,'micro_batch_profile',**result)
        save_json(out/'resource_profiles.json',results)
        policy.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if not result['passed']:
            break
    del moments
    gc.collect()
    torch.cuda.empty_cache()
    return choose_microbatch(results)


def checkpoint(out, step, policy, pre, post, optimizer, state):
    import torch
    final = out/f'checkpoint_{step:06d}'
    temp = out/f'.checkpoint_{step:06d}.incomplete'
    temp.mkdir(exist_ok=False)
    if final.exists():
        raise FileExistsError(final)
    policy.save_pretrained(temp/'pretrained_model')
    pre.save_pretrained(temp/'pretrained_model')
    post.save_pretrained(temp/'pretrained_model')
    torch.save({'optimizer': optimizer.state_dict(), 'rng': rng_state(), **state}, temp/'training_state.pt')
    save_json(temp/'checkpoint_manifest.json', {
        'optimizer_updates': step, 'training_state': state,
        'files_sha256': {str(p.relative_to(temp)):file_sha(p) for p in temp.rglob('*') if p.is_file()}})
    temp.rename(final)
    emit(out,'checkpoint_saved',optimizer_updates=step,path=str(final))
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    root,out = args.dataset.resolve(),args.output.resolve()
    # Supervisor owns output creation and the lock; guard a direct invocation.
    if not (out/'supervisor_receipt.json').is_file() or (out/'events.jsonl').exists():
        raise ValueError('new supervised DT3 run required')
    manifest, normalizer = approved_dataset(root,args.approval)
    if (file_sha(DEFAULT_BASE_MODEL/'config.json') != DEFAULT_BASE_CONFIG_SHA256
            or file_sha(DEFAULT_BASE_MODEL/'model.safetensors') != DEFAULT_BASE_MODEL_SHA256):
        raise ValueError('reviewed base weights changed')
    import numpy as np
    import torch
    from lerobot.configs import NormalizationMode
    from .gpu_wait import probe_gpu
    torch.set_num_threads(8)
    cache = FrameCache(root,manifest,out)
    free = probe_gpu().free_mib
    if free < 7168:
        raise RuntimeError('less than 7 GiB free before model allocation; requeue required')
    # Cap our allocator to admission free minus 2.25 GiB; outside-driver allocations
    # remain protected by the separate 2 GiB nvidia-smi watchdog.
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(.90,(free-2304)*1024**2/total)
    torch.cuda.set_per_process_memory_fraction(fraction)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable; explicit precision decision required')
    config = _build_new_3d_config(DEFAULT_BASE_MODEL)
    config.chunk_size = 5
    config.n_action_steps = 1
    config.normalization_mapping = {'VISUAL':NormalizationMode.IDENTITY,
        'STATE':NormalizationMode.MEAN_STD,'ACTION':NormalizationMode.MEAN_STD}
    config.use_amp = True
    seed_all(20260906)
    policy = load_training_policy(DEFAULT_BASE_MODEL,config)
    scope = checked_parameters(policy)
    save_json(out/'parameter_scope.json',scope)
    pre,post = processors(config,normalizer)
    micro = profile(policy,pre,cache,out)
    seed_all(20260906)  # resource profiles never consume the training RNG stream
    optimizer = optimizer_for(policy)
    sampler_gen = torch.Generator().manual_seed(20260906)
    weights = torch.from_numpy(np.load(root/'episode_balanced_sampler_weights.npy'))
    sampled = torch.multinomial(weights,1000*16,replacement=True,generator=sampler_gen)
    np.save(out/'sampled_indices.npy',sampled.numpy())
    save_json(out/'training_config.json', {
        'stage':'DT3','dataset_root':str(root),'effective_batch':16,'micro_batch':micro,
        'gradient_accumulation':16//micro,'learning_rate':1e-4,'scheduler':'constant',
        'scheduler_reason':'1000-update v2 first stage: preserve LR=1e-4 constant; checkpoint closed-loop review before further updates',
        'precision':'BF16 autocast; FP32 trainable master weights and Adam moments; frozen VLM base dtype',
        'optimizer':'AdamW foreach=False','betas':[.9,.95],'eps':1e-8,'weight_decay':1e-10,
        'gradient_clip':10.,'max_updates_this_run':1000,'total_DT3_budget':5000,'old_updates_consumed':250,'remaining_after_this_run':3750,
        'fps':5,'chunk_size':5,'execute_steps':1,'fresh_from_base':True,'validation_used_for_updates':False,
        'loss':'per-anchor valid vx/wz mean, then equal anchor mean; excludes vy/padding',
        'seed':20260906,'allocator_fraction':fraction,'admission_free_mib':free,
        'normalizer_sha256':file_sha(root/'normalizer.json'),
        'dataset_manifest_sha256':file_sha(root/'dataset_manifest.json'),
        'approval_sha256':file_sha(args.approval),'sampled_indices_sha256':file_sha(out/'sampled_indices.npy')})
    params = [p for p in policy.parameters() if p.requires_grad]
    projection_before = policy.model.action_out_proj.weight.detach().cpu().clone()
    fixed = pre(cache.batch([0]))
    fixed_noise = torch.randn((1,5,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(91))
    def prediction(model):
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result = model.predict_action_chunk(fixed,noise=fixed_noise).float().cpu()
        model.train()
        if result.shape != (1,5,3) or not bool(torch.isfinite(result).all()):
            raise ValueError('fixed observation/noise prediction failed')
        return result
    before_prediction = prediction(policy)
    emit(out,'training_started',micro_batch=micro,effective_batch=16,max_updates=1000)
    for step in range(1,1001):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        ids = sampled[(step-1)*16:step*16]
        total_loss = 0.
        for start in range(0,16,micro):
            batch = pre(cache.batch(ids[start:start+micro]))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = flow_loss(policy,batch)/(16//micro)
            if not bool(torch.isfinite(loss)):
                raise ValueError('training loss nonfinite')
            loss.backward()
            total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(v).all()) for v in params):
            raise ValueError('updated parameter nonfinite')
        torch.cuda.synchronize()
        if step == 1 or step % 10 == 0:
            emit(out,'update',optimizer_updates=step,loss=total_loss,gradient_norm=float(norm),
                 seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if step in (1,250,500,750,1000):
            state = {'optimizer_updates':step,'consumed_samples':step*16,'micro_batch':micro,
                'effective_batch':16,'scheduler':None,'amp_scaler':None,
                'dataset_manifest_sha256':file_sha(root/'dataset_manifest.json'),
                'normalizer_sha256':file_sha(root/'normalizer.json'),
                'sampled_indices_sha256':file_sha(out/'sampled_indices.npy'),
                'sampler':'precomputed weighted replacement indices; cursor=consumed_samples',
                'resume_claim':'state-complete artifact; exact resumed trajectory not yet tested'}
            if step == 1:
                after = prediction(policy)
                delta = float((policy.model.action_out_proj.weight.detach().cpu()-projection_before).abs().max())
                if delta <= 0 or not bool(torch.isfinite(after).all()) or torch.equal(after,before_prediction):
                    raise ValueError('single-update gate: projection not updated')
            path = checkpoint(out,step,policy,pre,post,optimizer,state)
            if step == 1:
                # Free the old model and optimizer before the real checkpoint load.
                del params, optimizer, policy, batch, loss
                gc.collect()
                torch.cuda.empty_cache()
                policy = load_training_policy(path/'pretrained_model')
                checked_parameters(policy)
                pre,post = reload_processors(path/'pretrained_model')
                reprocessed = pre(cache.batch([0]))
                if set(fixed) != set(reprocessed) or any(
                        not torch.equal(value,reprocessed[key]) for key,value in fixed.items()
                        if isinstance(value,torch.Tensor)):
                    raise ValueError('saved preprocessor reload changed fixed model inputs')
                fixed = reprocessed
                reloaded = prediction(policy)
                difference = float((reloaded-after).abs().max())
                if not torch.allclose(after,reloaded,atol=1e-5,rtol=1e-5):
                    raise ValueError(f'single-update reload prediction drift: {difference}')
                optimizer = optimizer_for(policy)
                saved = torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
                optimizer.load_state_dict(saved['optimizer'])
                restore_rng(saved['rng'])
                del saved
                params = [p for p in policy.parameters() if p.requires_grad]
                save_json(out/'single_update_gate.json',{'passed':True,'optimizer_updates':1,
                    'projection_max_update':delta,'reload_prediction_max_abs':difference,
                    'prediction_change_after_update':float((after-before_prediction).abs().max()),
                    'finite_gradients':True,'scope_checked':True,'fixed_observation_noise':True,
                    'checkpoint':str(path),'not_navigation_success':True})
                emit(out,'single_update_reload_gate_passed',max_abs_difference=difference)
    save_json(out/'result.json',{'stage':'DT3','status':'ZOH_1000_COMPLETE_AWAITING_REVIEW',
        'optimizer_updates':1000,'checkpoint':str(out/'checkpoint_001000'),
        'next':'validation closed-loop checkpoint comparison; no test evaluation or automatic budget expansion'})
    emit(out,'pilot_complete',optimizer_updates=1000,dt3_approved=False)


if __name__ == '__main__':
    main()
