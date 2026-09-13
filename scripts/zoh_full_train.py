"""Fresh 5000-step full-policy comparison with the same official scheduler."""
import argparse
import gc
import json
from pathlib import Path
import time
from src.dual_target.contracts import IMAGE_KEY,STATE_KEY
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.tiny_train_core import file_sha,flow_loss
from src.dual_target.smolvla_probe_client import DEFAULT_BASE_MODEL,DEFAULT_BASE_CONFIG_SHA256,DEFAULT_BASE_MODEL_SHA256,_build_new_3d_config
from src.dual_target.zoh_train import (FrameCache,emit,save_json,load_training_policy,checked_parameters,processors,profile,
    seed_all,optimizer_for,checkpoint,reload_processors,restore_rng)
from src.dual_target.zoh_scheduler_core import BASELINE,baseline_samples,assert_scope_matches,baseline_first_loss,make_scheduler,scheduler_config

from scripts.zoh_full_support import load_training_policy,checked_parameters,selected_weights,full_gradient_audit,weight_updates

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
    from src.dual_target.gpu_wait import probe_gpu
    torch.set_num_threads(8)
    cache = FrameCache(root,manifest,out)
    free = probe_gpu().free_mib
    while free < 14336:
        emit(out,'waiting_gpu_before_model_allocation',free_mib=free,required_free_mib=14336)
        time.sleep(30)
        free=probe_gpu().free_mib
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
    config.train_expert_only=False
    config.freeze_vision_encoder=False
    seed_all(20260906)
    policy = load_training_policy(DEFAULT_BASE_MODEL,config)
    scope = checked_parameters(policy)
    full_before=selected_weights(policy)
    save_json(out/'parameter_scope.json',scope)
    pre,post = processors(config,normalizer)
    profile(policy,pre,cache,out)
    profiles=json.loads((out/'resource_profiles.json').read_text())
    if not any(x['micro_batch']==4 and x['passed'] for x in profiles):
        raise RuntimeError('fixed micro 4 did not pass; do not change ablation batch')
    micro=4
    seed_all(20260906)  # resource profiles never consume the training RNG stream
    optimizer = optimizer_for(policy)
    scheduler=make_scheduler(optimizer)
    sampled=baseline_samples(root)
    np.save(out/'sampled_indices.npy',sampled.numpy())
    save_json(out/'training_config.json', {
        'stage':'DT3','dataset_root':str(root),'effective_batch':16,'micro_batch':micro,
        'gradient_accumulation':16//micro,'learning_rate':1e-4,'scheduler':'cosine_decay_with_warmup',
        'scheduler_reason':'User-authorized full-policy vs scheduler-expert comparison; same base/data/sample order/micro4/accum4/loss/scheduler; expand trainable scope',
        'scheduler_warmup_steps_configured':1000,'scheduler_decay_steps_configured':30000,
        'scheduler_warmup_steps_actual':166,'scheduler_decay_steps_actual':5000,'scheduler_decay_lr':2.5e-6,
        'scheduler_source_sha256':file_sha(Path(__import__('inspect').getfile(type(scheduler_config())))),
        'baseline_root':str(BASELINE),'baseline_sampler_sha256':file_sha(BASELINE/'sampled_indices.npy'),
        'lr_logging':'lr used for optimizer update; lr_next after scheduler.step',
        'scheduler_call_order':'record lr -> optimizer.step -> scheduler.step',
        'precision':'BF16 autocast; FP32 masters and Adam for all enabled policy parameters',
        'full_finetuning':True,'freeze_vision_encoder':False,'train_expert_only':False,
        'activation_checkpointing':'vision only; non-reentrant, preserves RNG; memory optimization',
        'optimizer':'AdamW foreach=False','betas':[.9,.95],'eps':1e-8,'weight_decay':1e-10,
        'gradient_clip':10.,'max_updates_this_run':5000,'total_DT3_budget':15250,'old_updates_consumed':10250,'remaining_after_this_run':0,
        'budget_note':'User-authorized fresh 5000-update full-policy run; pipeline subsequently compares all three 5k checkpoints',
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
    emit(out,'training_started',micro_batch=micro,effective_batch=16,max_updates=5000)
    for step in range(1,5001):
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
        if step==1:save_json(out/'full_gradient_gate.json',full_gradient_audit(policy))
        norm = torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        lr_used=float(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        if scheduler.last_epoch!=step:raise ValueError('scheduler cursor drift')
        if not all(bool(torch.isfinite(v).all()) for v in params):
            raise ValueError('updated parameter nonfinite')
        torch.cuda.synchronize()
        if step == 1 or step % 10 == 0 or step in (166,167):
            emit(out,'update',optimizer_updates=step,loss=total_loss,gradient_norm=float(norm),lr=lr_used,lr_next=float(optimizer.param_groups[0]['lr']),
                 seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if step in (1,250,500,750,1000,2000,3000,4000,5000):
            state = {'optimizer_updates':step,'consumed_samples':step*16,'micro_batch':micro,
                'effective_batch':16,'scheduler':scheduler.state_dict(),'amp_scaler':None,
                'lr_next':float(optimizer.param_groups[0]['lr']),
                'dataset_manifest_sha256':file_sha(root/'dataset_manifest.json'),
                'normalizer_sha256':file_sha(root/'normalizer.json'),
                'sampled_indices_sha256':file_sha(out/'sampled_indices.npy'),
                'sampler':'precomputed weighted replacement indices; cursor=consumed_samples',
                'resume_claim':'state-complete artifact; exact resumed trajectory not yet tested'}
            if step == 1:
                after = prediction(policy)
                delta = float((policy.model.action_out_proj.weight.detach().cpu()-projection_before).abs().max())
                save_json(out/'full_weight_update_gate.json',weight_updates(policy,full_before))
                del full_before
                if delta <= 0 or not bool(torch.isfinite(after).all()):
                    raise ValueError('single-update gate: projection not updated')
            path = checkpoint(out,step,policy,pre,post,optimizer,state)
            if step == 1:
                # Free the old model and optimizer before the real checkpoint load.
                del params, scheduler, optimizer, policy, batch, loss
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
                scheduler=make_scheduler(optimizer)
                saved = torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
                optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler'])
                if scheduler.last_epoch!=step or optimizer.param_groups[0]['lr']!=saved['lr_next']:
                    raise ValueError('scheduler/optimizer reload mismatch')
                restore_rng(saved['rng'])
                del saved
                params = [p for p in policy.parameters() if p.requires_grad]
                save_json(out/'single_update_gate.json',{'passed':True,'optimizer_updates':1,
                    'projection_max_update':delta,'reload_prediction_max_abs':difference,
                    'prediction_change_after_update':float((after-before_prediction).abs().max()),
                    'finite_gradients':True,'scope_checked':True,'scheduler_reload_checked':True,
                    'first_loss_abs_diff_from_constant_baseline':abs(total_loss-baseline_first_loss()),
                    'prediction_change_not_required_at_tiny_warmup_lr':True,'fixed_observation_noise':True,
                    'checkpoint':str(path),'not_navigation_success':True})
                emit(out,'single_update_reload_gate_passed',max_abs_difference=difference)
    save_json(out/'result.json',{'stage':'DT3','status':'ZOH_FULL_5000_COMPLETE_AWAITING_REVIEW',
        'optimizer_updates':5000,'checkpoint':str(out/'checkpoint_005000'),
        'next':'Trainer exits; authorized parent pipeline starts train/validation comparison. No test use or further training'})
    emit(out,'pilot_complete',optimizer_updates=5000,dt3_approved=False)


if __name__ == '__main__':
    main()

