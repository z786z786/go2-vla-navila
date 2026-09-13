"""Resume approved ZOH 1000 checkpoint to total 5000 updates, never reinitialize."""
import argparse
import gc
import json
from pathlib import Path
import time

from .zoh_train import (FrameCache, checked_parameters, checkpoint, emit,
    load_training_policy, optimizer_for, profile, reload_processors, restore_rng, save_json)
from .zoh_train_core import approved_dataset
from .tiny_train_core import file_sha, flow_loss


def extend_sampler(weights, old_indices, old_count=16000, total_count=80000):
    import torch
    generator=torch.Generator().manual_seed(20260906)
    prefix=torch.multinomial(weights,old_count,replacement=True,generator=generator)
    if not torch.equal(prefix,old_indices) or total_count<=old_count:
        raise ValueError('resume sampler prefix mismatch or no new samples')
    return torch.cat([prefix,torch.multinomial(weights,total_count-old_count,
        replacement=True,generator=generator)])


def validate_parent(parent, root):
    manifest=json.loads((parent/'checkpoint_manifest.json').read_text())
    if manifest['optimizer_updates']!=1000:
        raise ValueError('only reviewed 1000-step checkpoint can start this run')
    for name,digest in manifest['files_sha256'].items():
        path=(parent/name).resolve()
        if not path.is_relative_to(parent.resolve()) or file_sha(path)!=digest:
            raise ValueError('parent checkpoint drift: '+name)
    state=manifest['training_state']
    if state['consumed_samples']!=16000 or state['effective_batch']!=16:
        raise ValueError('wrong parent sample cursor')
    for key,name in [('dataset_manifest_sha256','dataset_manifest.json'),('normalizer_sha256','normalizer.json')]:
        if state[key]!=file_sha(root/name):raise ValueError('parent data/normalizer mismatch')
    if state['sampled_indices_sha256']!=file_sha(parent.parent/'sampled_indices.npy'):
        raise ValueError('parent sampler drift')
    return manifest


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','approval','output','resume'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.dataset.resolve();out=args.output.resolve();parent=args.resume.resolve()
    if not (out/'supervisor_receipt.json').is_file() or (out/'events.jsonl').exists():
        raise ValueError('fresh supervised output required')
    manifest,normalizer=approved_dataset(root,args.approval)
    validate_parent(parent,root)
    import numpy as np
    import torch
    from .gpu_wait import probe_gpu
    torch.set_num_threads(8)
    cache=FrameCache(root,manifest,out)
    free=probe_gpu().free_mib
    while free<7168:
        emit(out,'waiting_gpu_before_model_allocation',free_mib=free,required_free_mib=7168,optimizer_updates=1000)
        time.sleep(30)
        free=probe_gpu().free_mib
    fraction=min(.9,(free-2304)*1024**2/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required')
    policy=load_training_policy(parent/'pretrained_model')
    if policy.config.chunk_size!=5 or policy.config.n_action_steps!=1:
        raise ValueError('wrong checkpoint action interface')
    save_json(out/'parameter_scope.json',checked_parameters(policy))
    pre,post=reload_processors(parent/'pretrained_model')
    micro=profile(policy,pre,cache,out)
    optimizer=optimizer_for(policy)
    saved=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
    if saved['optimizer_updates']!=1000:raise ValueError('wrong optimizer cursor')
    optimizer.load_state_dict(saved['optimizer'])
    if any(g['lr']!=1e-4 for g in optimizer.param_groups):raise ValueError('unexpected parent LR')
    for state in optimizer.state.values():
        if int(state['step'])!=1000:raise ValueError('Adam update counter mismatch')
    weights=torch.from_numpy(np.load(root/'episode_balanced_sampler_weights.npy'))
    sampled=extend_sampler(weights,torch.from_numpy(np.load(parent.parent/'sampled_indices.npy')))
    np.save(out/'sampled_indices.npy',sampled.numpy())
    config=json.loads((parent.parent/'training_config.json').read_text())
    config.update(micro_batch=micro,gradient_accumulation=16//micro,max_updates_this_run=4000,
        target_optimizer_updates=5000,start_optimizer_updates=1000,fresh_from_base=False,
        resume_checkpoint=str(parent),resume_manifest_sha256=file_sha(parent/'checkpoint_manifest.json'),
        scheduler_reason='User authorized total 5000 ZOH updates; retain parent constant LR for controlled continuation',
        total_DT3_budget=5250,old_updates_consumed=1250,remaining_after_this_run=0,
        budget_note='User latest 5k request supersedes old cumulative cap: ZOH 5000 plus historical legacy 250',
        allocator_fraction=fraction,admission_free_mib=free,
        sampled_indices_sha256=file_sha(out/'sampled_indices.npy'),
        approval_sha256=file_sha(args.approval))
    save_json(out/'training_config.json',config)
    restore_rng(saved['rng']);del saved
    params=[v for v in policy.parameters() if v.requires_grad]
    emit(out,'training_started',micro_batch=micro,effective_batch=16,start_updates=1000,max_updates=5000)
    for step in range(1001,5001):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True);total_loss=0.
        ids=sampled[(step-1)*16:step*16]
        for start in range(0,16,micro):
            batch=pre(cache.batch(ids[start:start+micro]))
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=flow_loss(policy,batch)/(16//micro)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
            loss.backward();total_loss+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(v).all()) for v in params):raise ValueError('nonfinite parameter')
        torch.cuda.synchronize()
        if step==1001 or step%10==0:
            emit(out,'update',optimizer_updates=step,loss=total_loss,gradient_norm=float(norm),
                seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if step in (1001,2000,3000,4000,5000):
            state=dict(optimizer_updates=step,consumed_samples=step*16,micro_batch=micro,
                effective_batch=16,scheduler=None,amp_scaler=None,
                dataset_manifest_sha256=file_sha(root/'dataset_manifest.json'),
                normalizer_sha256=file_sha(root/'normalizer.json'),
                sampled_indices_sha256=file_sha(out/'sampled_indices.npy'),
                sampler='verified parent prefix plus continuation of seeded weighted replacement generator',
                resume_parent=str(parent),resume_claim='optimizer/RNG/cursor restored; first new checkpoint reload verified')
            path=checkpoint(out,step,policy,pre,post,optimizer,state)
            if step==1001:
                fixed=pre(cache.batch([0]))
                noise=torch.randn((1,5,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(91))
                policy.eval()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    before=policy.predict_action_chunk(fixed,noise=noise).float().cpu()
                del params,optimizer,policy,batch,loss
                gc.collect();torch.cuda.empty_cache()
                policy=load_training_policy(path/'pretrained_model').eval()
                checked_parameters(policy);pre,post=reload_processors(path/'pretrained_model')
                fixed2=pre(cache.batch([0]))
                for key,value in fixed.items():
                    if isinstance(value,torch.Tensor) and not torch.equal(value,fixed2[key]):
                        raise ValueError('resume checkpoint processor reload mismatch')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    after=policy.predict_action_chunk(fixed2,noise=noise).float().cpu()
                if not torch.allclose(before,after,atol=1e-5,rtol=1e-5):raise ValueError('resume checkpoint model reload mismatch')
                optimizer=optimizer_for(policy)
                saved=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
                optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);del saved
                policy.train();params=[v for v in policy.parameters() if v.requires_grad]
                save_json(out/'resume_gate.json',dict(passed=True,optimizer_updates=1001,
                    parent_optimizer_updates=1000,reload_prediction_max_abs=float((before-after).abs().max()),
                    parent_sampler_prefix_exact=True,parent_rng_restored=True,adam_step_restored=True))
                emit(out,'resume_reload_gate_passed',optimizer_updates=1001)
    save_json(out/'result.json',dict(status='ZOH_5000_COMPLETE_AWAITING_REVIEW',optimizer_updates=5000,
        new_updates=4000,checkpoint=str(out/'checkpoint_005000'),next='Stop; await user. No automatic rollout or further training.'))
    emit(out,'pilot_complete',optimizer_updates=5000,dt3_approved=False)


if __name__=='__main__':main()
