"""Fresh, train-only 10k SmolVLA run for expert-only or full-policy scope."""
import argparse
import gc
import json
from pathlib import Path
import time

from src.dual_target.contracts import IMAGE_KEY,STATE_KEY
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.tiny_train_core import file_sha,flow_loss
from src.dual_target.smolvla_probe_client import (DEFAULT_BASE_MODEL,DEFAULT_BASE_CONFIG_SHA256,
    DEFAULT_BASE_MODEL_SHA256,_build_new_3d_config)
from src.dual_target.zoh_train import (FrameCache,emit,save_json,processors,profile,seed_all,
    optimizer_for,checkpoint,reload_processors)
from src.dual_target.zoh_scheduler_core import assert_scope_matches,baseline_first_loss
from scripts.zoh_10k_core import (BASELINE_5K,TOTAL,EFFECTIVE_BATCH,CHECKPOINT_STEPS,
    scheduler_config,make_scheduler,training_samples,verify_checkpoint)
from scripts.zoh_full_support import (load_training_policy as load_full_policy,
    checked_parameters as checked_full_parameters,selected_weights,full_gradient_audit,weight_updates,
    baseline_first_loss as full_baseline_first_loss)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--approval',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('expert','full'),required=True)
    p.add_argument('--required-free-mib',type=int,default=0);args=p.parse_args()
    root,out=args.dataset.resolve(),args.output.resolve()
    if not (out/'supervisor_receipt.json').is_file() or (out/'events.jsonl').exists():
        raise ValueError('fresh supervised 10k run required')
    manifest,normalizer=approved_dataset(root,args.approval)
    if (file_sha(DEFAULT_BASE_MODEL/'config.json')!=DEFAULT_BASE_CONFIG_SHA256
            or file_sha(DEFAULT_BASE_MODEL/'model.safetensors')!=DEFAULT_BASE_MODEL_SHA256):
        raise ValueError('reviewed base weights changed')
    import numpy as np
    import torch
    from lerobot.configs import NormalizationMode
    from src.dual_target.gpu_wait import probe_gpu
    from src.dual_target.zoh_train import load_training_policy as load_expert_policy,checked_parameters as checked_expert_parameters
    torch.set_num_threads(8);cache=FrameCache(root,manifest,out)
    required=int(getattr(args,'required_free_mib',0) or (14336 if args.scope=='full' else 7168))
    free=probe_gpu().free_mib
    while free<required:
        emit(out,'waiting_gpu_before_model_allocation',free_mib=free,required_free_mib=required);time.sleep(30);free=probe_gpu().free_mib
    total=torch.cuda.get_device_properties(0).total_memory
    fraction=min(.90,(free-2304)*1024**2/total);torch.cuda.set_per_process_memory_fraction(fraction)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required')
    config=_build_new_3d_config(DEFAULT_BASE_MODEL);config.chunk_size=5;config.n_action_steps=1
    config.normalization_mapping={'VISUAL':NormalizationMode.IDENTITY,'STATE':NormalizationMode.MEAN_STD,
        'ACTION':NormalizationMode.MEAN_STD};config.use_amp=True
    if args.scope=='full':config.train_expert_only=False;config.freeze_vision_encoder=False
    seed_all(20260906)
    if args.scope=='full':policy=load_full_policy(DEFAULT_BASE_MODEL,config);scope=checked_full_parameters(policy)
    else:policy=load_expert_policy(DEFAULT_BASE_MODEL,config);scope=checked_expert_parameters(policy);assert_scope_matches(scope)
    save_json(out/'parameter_scope.json',scope)
    full_before=selected_weights(policy) if args.scope=='full' else None
    pre,post=processors(config,normalizer);profile(policy,pre,cache,out)
    profiles=json.loads((out/'resource_profiles.json').read_text())
    if not any(x['micro_batch']==4 and x['passed'] for x in profiles):
        raise RuntimeError('fixed micro4 failed; scope/batch must not change')
    micro=4;seed_all(20260906)
    optimizer=optimizer_for(policy);scheduler=make_scheduler(optimizer);sampled=training_samples(root)
    np.save(out/'sampled_indices.npy',sampled.numpy());sampler_sha=file_sha(out/'sampled_indices.npy')
    precision=('BF16 autocast; FP32 masters/Adam for all policy parameters; vision non-reentrant activation recomputation'
        if args.scope=='full' else 'BF16 autocast; FP32 expert/projection masters and Adam; frozen VLM base dtype')
    save_json(out/'training_config.json',dict(stage='DT3',experiment='fresh_10k_train_only',scope=args.scope,
        dataset_root=str(root),effective_batch=16,micro_batch=4,gradient_accumulation=4,
        learning_rate=1e-4,scheduler='cosine_decay_with_warmup',scheduler_warmup_steps_configured=1000,
        scheduler_decay_steps_configured=30000,scheduler_warmup_steps_actual=333,
        scheduler_decay_steps_actual=10000,scheduler_decay_lr=2.5e-6,
        scheduler_max_lr_used_step=334,scheduler_max_lr_used=9.973347576037485e-5,
        scheduler_source_sha256=file_sha(Path(__import__('inspect').getfile(type(scheduler_config())))),
        scheduler_call_order='record lr -> optimizer.step -> scheduler.step',
        lr_logging='lr used for optimizer update; lr_next after scheduler.step',precision=precision,
        full_finetuning=args.scope=='full',freeze_vision_encoder=config.freeze_vision_encoder,
        train_expert_only=config.train_expert_only,activation_checkpointing=('vision non-reentrant' if args.scope=='full' else None),
        optimizer='AdamW foreach=False',betas=[.9,.95],eps=1e-8,weight_decay=1e-10,gradient_clip=10.,
        target_optimizer_updates=TOTAL,checkpoint_steps=list(CHECKPOINT_STEPS),formal_checkpoint_interval=2000,
        fps=5,chunk_size=5,execute_steps=1,fresh_from_base=True,validation_read=False,
        validation_used_for_updates=False,evaluation_loss=False,rollout=False,
        loss='per-anchor valid vx/wz mean, then equal anchor mean; excludes vy/padding',seed=20260906,
        allocator_fraction=fraction,admission_free_mib=free,admission_free_mib_required=required,
        resource_exception=('user-authorized 13.5 GiB full-finetune admission' if required==13824 else None),
        normalizer_sha256=file_sha(root/'normalizer.json'),
        dataset_manifest_sha256=file_sha(root/'dataset_manifest.json'),approval_sha256=file_sha(args.approval),
        sampled_indices_sha256=sampler_sha,reviewed_5k_sampler_sha256=file_sha(BASELINE_5K/'sampled_indices.npy'),
        reviewed_5k_prefix_samples=80000,total_samples=TOTAL*EFFECTIVE_BATCH))
    params=[p for p in policy.parameters() if p.requires_grad]
    projection_before=policy.model.action_out_proj.weight.detach().cpu().clone()
    fixed=pre(cache.batch([0]));fixed_noise=torch.randn((1,5,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(91))
    def prediction(model):
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            value=model.predict_action_chunk(fixed,noise=fixed_noise).float().cpu()
        model.train()
        if value.shape!=(1,5,3) or not bool(torch.isfinite(value).all()):raise ValueError('fixed prediction failed')
        return value
    before_prediction=prediction(policy);gates={}
    emit(out,'training_started',scope=args.scope,micro_batch=4,effective_batch=16,max_updates=TOTAL,
        validation_read=False,rollout=False)
    for step in range(1,TOTAL+1):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True);ids=sampled[(step-1)*16:step*16];total_loss=0.
        for start in range(0,16,micro):
            batch=pre(cache.batch(ids[start:start+micro]))
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=flow_loss(policy,batch)/(16//micro)
            if not bool(torch.isfinite(loss)):raise ValueError('training loss nonfinite')
            loss.backward();total_loss+=float(loss.detach())
        if step==1 and args.scope=='full':
            gradient_gate=full_gradient_audit(policy);gradient_gate['passed']=True
            save_json(out/'full_gradient_gate.json',gradient_gate)
        norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        lr_used=float(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
        if scheduler.last_epoch!=step:raise ValueError('scheduler cursor drift')
        if not all(bool(torch.isfinite(v).all()) for v in params):raise ValueError('updated parameter nonfinite')
        torch.cuda.synchronize()
        if step==1 or step%10==0 or step in (333,334):
            emit(out,'update',optimizer_updates=step,loss=total_loss,gradient_norm=float(norm),lr=lr_used,
                lr_next=float(optimizer.param_groups[0]['lr']),seconds=time.perf_counter()-started,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if step==1:
            after=prediction(policy);projection_delta=float((policy.model.action_out_proj.weight.detach().cpu()-projection_before).abs().max())
            if projection_delta<=0:raise ValueError('single update did not change action projection')
            expected_first_loss=(full_baseline_first_loss(root) if args.scope=='full' else baseline_first_loss())
            first_loss_difference=abs(total_loss-expected_first_loss)
            if first_loss_difference>1e-4:raise ValueError('first pre-update loss differs from reviewed run')
            if any(out.glob('checkpoint_*')):raise ValueError('formal checkpoint created before step 2000')
            if args.scope=='full':
                update_gate={'passed':True,'updates':weight_updates(policy,full_before)}
                save_json(out/'full_weight_update_gate.json',update_gate);del full_before
            save_json(out/'single_update_gate.json',dict(passed=True,optimizer_updates=1,scope=args.scope,
                projection_max_update=projection_delta,prediction_change_after_update=float((after-before_prediction).abs().max()),
                finite_gradients=True,scheduler_cursor_checked=True,first_loss=total_loss,
                expected_first_loss=expected_first_loss,
                first_loss_abs_diff_from_existing_baseline=first_loss_difference,
                no_checkpoint_created=True,validation_read=False,rollout=False))
        if step in CHECKPOINT_STEPS:
            state=dict(optimizer_updates=step,consumed_samples=step*16,micro_batch=4,effective_batch=16,
                scheduler=scheduler.state_dict(),amp_scaler=None,lr_next=float(optimizer.param_groups[0]['lr']),
                dataset_manifest_sha256=file_sha(root/'dataset_manifest.json'),normalizer_sha256=file_sha(root/'normalizer.json'),
                sampled_indices_sha256=sampler_sha,sampler='seeded weighted replacement; reviewed 80k prefix; cursor=consumed_samples',
                resume_claim='state-complete artifact; manifest and optimizer/scheduler/RNG cursor verified')
            path=checkpoint(out,step,policy,pre,post,optimizer,state)
            gates[str(step)]=verify_checkpoint(path,step,root,sampler_sha);save_json(out/'checkpoint_verification.json',gates)
    before_reload=prediction(policy);final=out/'checkpoint_010000'
    del params,scheduler,optimizer,policy,batch,loss;gc.collect();torch.cuda.empty_cache()
    if args.scope=='full':policy=load_full_policy(final/'pretrained_model');checked_full_parameters(policy)
    else:policy=load_expert_policy(final/'pretrained_model');checked_expert_parameters(policy)
    pre2,post2=reload_processors(final/'pretrained_model');fixed2=pre2(cache.batch([0]))
    if set(fixed)!=set(fixed2) or any(not torch.equal(v,fixed2[k]) for k,v in fixed.items() if isinstance(v,torch.Tensor)):
        raise ValueError('final checkpoint processor reload changed inputs')
    fixed=fixed2;after_reload=prediction(policy);difference=float((before_reload-after_reload).abs().max())
    if not torch.allclose(before_reload,after_reload,atol=1e-5,rtol=1e-5):raise ValueError('final checkpoint prediction reload drift')
    save_json(out/'final_reload_gate.json',dict(passed=True,optimizer_updates=10000,
        prediction_max_abs_difference=difference,processor_exact=True,scope_checked=True))
    status='ZOH_EXPERT_10000_COMPLETE_AWAITING_REVIEW' if args.scope=='expert' else 'ZOH_FULL_10000_COMPLETE_AWAITING_REVIEW'
    save_json(out/'result.json',dict(stage='DT3',status=status,scope=args.scope,optimizer_updates=10000,
        checkpoint=str(final),checkpoints=[str(out/f'checkpoint_{s:06d}') for s in CHECKPOINT_STEPS],
        validation_read=False,evaluation_loss=False,rollout=False,next='Stop; no evaluation authorized'))
    emit(out,'training_complete',scope=args.scope,optimizer_updates=10000,validation_read=False,rollout=False)


if __name__=='__main__':main()
