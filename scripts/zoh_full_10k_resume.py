"""Resume the interrupted full-policy 10k run from its verified step-2000 checkpoint."""
import argparse,gc,json,time
from pathlib import Path

from src.dual_target.tiny_train_core import file_sha,flow_loss
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.zoh_train import (FrameCache,checkpoint,emit,optimizer_for,reload_processors,
    restore_rng,save_json)
from scripts.zoh_10k_core import TOTAL,CHECKPOINT_STEPS,make_scheduler,training_samples,verify_checkpoint
from scripts.zoh_full_support import load_training_policy,checked_parameters,selected_weights,weight_updates

PARENT_STEP=2000


def validate_parent(parent,root):
    parent,parent_run=Path(parent).resolve(),Path(parent).resolve().parent
    config=json.loads((parent_run/'training_config.json').read_text())
    if (config.get('scope')!='full' or config.get('target_optimizer_updates')!=10000
            or config.get('sampled_indices_sha256')!=file_sha(parent_run/'sampled_indices.npy')):
        raise ValueError('resume parent is not the reviewed full-policy 10k run')
    receipt=verify_checkpoint(parent,PARENT_STEP,root,config['sampled_indices_sha256'])
    for name in ('single_update_gate.json','full_gradient_gate.json','full_weight_update_gate.json'):
        if json.loads((parent_run/name).read_text()).get('passed') is not True:
            raise ValueError('parent gate failed: '+name)
    status=json.loads((parent_run/'supervisor_status.json').read_text())
    if status.get('status')!='FAILED' or 'below 2 GiB' not in status.get('detail',''):
        raise ValueError('parent did not stop through the reviewed GPU floor')
    return config,receipt


def prior_loss(parent_run,step):
    rows=[json.loads(x) for x in (parent_run/'events.jsonl').read_text().splitlines()]
    hits=[x for x in rows if x.get('event')=='update' and x.get('optimizer_updates')==step]
    if len(hits)!=1:raise ValueError('missing interrupted-run continuity reference')
    return float(hits[0]['loss'])


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','approval','output','resume'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.dataset.resolve();out=args.output.resolve();parent=args.resume.resolve()
    if not (out/'supervisor_receipt.json').is_file() or (out/'events.jsonl').exists():
        raise ValueError('fresh supervised resume output required')
    manifest,_=approved_dataset(root,args.approval);parent_config,parent_receipt=validate_parent(parent,root)
    import numpy as np
    import torch
    from src.dual_target.gpu_wait import probe_gpu
    torch.set_num_threads(8);cache=FrameCache(root,manifest,out);free=probe_gpu().free_mib
    while free<10240:
        emit(out,'waiting_gpu_before_resume_allocation',free_mib=free,required_free_mib=10240);time.sleep(30);free=probe_gpu().free_mib
    total=torch.cuda.get_device_properties(0).total_memory
    fraction=min(.90,(free-2304)*1024**2/total);torch.cuda.set_per_process_memory_fraction(fraction)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required')
    policy=load_training_policy(parent/'pretrained_model');save_json(out/'parameter_scope.json',checked_parameters(policy))
    pre,post=reload_processors(parent/'pretrained_model');optimizer=optimizer_for(policy);scheduler=make_scheduler(optimizer)
    saved=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
    if saved['optimizer_updates']!=PARENT_STEP or saved['consumed_samples']!=PARENT_STEP*16:
        raise ValueError('parent optimizer/sample cursor mismatch')
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    adam_steps=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
    if not adam_steps or min(adam_steps)!=PARENT_STEP or max(adam_steps)!=PARENT_STEP or scheduler.last_epoch!=PARENT_STEP:
        raise ValueError('parent Adam/scheduler cursor mismatch')
    sampled=training_samples(root);np.save(out/'sampled_indices.npy',sampled.numpy())
    if file_sha(out/'sampled_indices.npy')!=parent_config['sampled_indices_sha256']:
        raise ValueError('resume sampled order differs from parent')
    config=dict(parent_config);config.update(experiment='resume_full_10k_train_only',fresh_from_base=False,
        start_optimizer_updates=PARENT_STEP,new_optimizer_updates=TOTAL-PARENT_STEP,
        resume_checkpoint=str(parent),resume_manifest_sha256=file_sha(parent/'checkpoint_manifest.json'),
        parent_checkpoint_verification=parent_receipt,allocator_fraction=fraction,admission_free_mib=free,
        admission_free_mib_required=13824,resource_exception='user-authorized 13.5 GiB resume admission',
        checkpoint_steps=[4000,6000,8000,10000],validation_read=False,evaluation_loss=False,rollout=False)
    save_json(out/'training_config.json',config);restore_rng(saved['rng']);del saved
    params=[v for v in policy.parameters() if v.requires_grad];before=selected_weights(policy);gates={}
    emit(out,'training_resumed',scope='full',start_updates=PARENT_STEP,max_updates=TOTAL,
        micro_batch=4,effective_batch=16,validation_read=False,rollout=False)
    for step in range(PARENT_STEP+1,TOTAL+1):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True);ids=sampled[(step-1)*16:step*16];total_loss=0.
        for start in range(0,16,4):
            batch=pre(cache.batch(ids[start:start+4]))
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=flow_loss(policy,batch)/4
            if not bool(torch.isfinite(loss)):raise ValueError('training loss nonfinite')
            loss.backward();total_loss+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        lr=float(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
        if scheduler.last_epoch!=step or not all(bool(torch.isfinite(v).all()) for v in params):
            raise ValueError('resume update/scheduler failed')
        torch.cuda.synchronize()
        if step==PARENT_STEP+1 or step%10==0:
            emit(out,'update',optimizer_updates=step,loss=total_loss,gradient_norm=float(norm),lr=lr,
                lr_next=float(optimizer.param_groups[0]['lr']),seconds=time.perf_counter()-started,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if step==PARENT_STEP+1:
            save_json(out/'resume_update_gate.json',dict(passed=True,parent_optimizer_updates=PARENT_STEP,
                optimizer_updates=step,scheduler_last_epoch=scheduler.last_epoch,
                full_weight_updates=weight_updates(policy,before),validation_read=False,rollout=False));del before
        if step==2010:
            reference=prior_loss(parent.parent,step);difference=abs(total_loss-reference)
            if difference>1e-4:raise ValueError('resume trajectory differs from interrupted run')
            save_json(out/'resume_continuity_gate.json',dict(passed=True,optimizer_updates=step,
                loss=total_loss,interrupted_run_loss=reference,absolute_difference=difference))
        if step in CHECKPOINT_STEPS and step>PARENT_STEP:
            state=dict(optimizer_updates=step,consumed_samples=step*16,micro_batch=4,effective_batch=16,
                scheduler=scheduler.state_dict(),amp_scaler=None,lr_next=float(optimizer.param_groups[0]['lr']),
                dataset_manifest_sha256=file_sha(root/'dataset_manifest.json'),normalizer_sha256=file_sha(root/'normalizer.json'),
                sampled_indices_sha256=file_sha(out/'sampled_indices.npy'),
                sampler='same seeded 160k order; restored cursor from verified step 2000',
                resume_parent=str(parent),resume_claim='Adam/scheduler/RNG/sample cursor restored')
            path=checkpoint(out,step,policy,pre,post,optimizer,state)
            gates[str(step)]=verify_checkpoint(path,step,root,state['sampled_indices_sha256'])
            save_json(out/'checkpoint_verification.json',gates)
    fixed=pre(cache.batch([0]));noise=torch.randn((1,5,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(91))
    policy.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):before_reload=policy.predict_action_chunk(fixed,noise=noise).float().cpu()
    del params,scheduler,optimizer,policy,batch,loss;gc.collect();torch.cuda.empty_cache()
    final=out/'checkpoint_010000';policy=load_training_policy(final/'pretrained_model');checked_parameters(policy)
    pre2,_=reload_processors(final/'pretrained_model');fixed2=pre2(cache.batch([0]))
    if any(not torch.equal(v,fixed2[k]) for k,v in fixed.items() if isinstance(v,torch.Tensor)):
        raise ValueError('final processor reload changed inputs')
    policy.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):after_reload=policy.predict_action_chunk(fixed2,noise=noise).float().cpu()
    difference=float((before_reload-after_reload).abs().max())
    if not torch.allclose(before_reload,after_reload,atol=1e-5,rtol=1e-5):raise ValueError('final reload drift')
    save_json(out/'final_reload_gate.json',dict(passed=True,optimizer_updates=TOTAL,prediction_max_abs_difference=difference))
    save_json(out/'result.json',dict(stage='DT3',status='ZOH_FULL_10000_RESUME_COMPLETE_AWAITING_REVIEW',scope='full',
        optimizer_updates=TOTAL,new_updates=TOTAL-PARENT_STEP,checkpoint=str(final),parent_checkpoint=str(parent),
        validation_read=False,evaluation_loss=False,rollout=False,next='Stop; await user'))
    emit(out,'training_complete',scope='full',optimizer_updates=TOTAL,validation_read=False,rollout=False)


if __name__=='__main__':main()
