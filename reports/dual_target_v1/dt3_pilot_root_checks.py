"""Root CPU review of a completed pilot, not a DT3 navigation approval."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from src.dual_target.tiny_train_core import file_sha
from src.dual_target.smolvla_probe_client import DEFAULT_BASE_MODEL


def read(p):
    return json.loads(p.read_text())


def review(root):
    torch.set_num_threads(8)
    result=read(root/'result.json')
    assert result['status']=='PILOT_250_COMPLETE_NOT_DT3_APPROVED' and result['optimizer_updates']==250
    config=read(root/'training_config.json')
    assert config['effective_batch']==16 and config['micro_batch']==16 and config['gradient_accumulation']==1
    gate=read(root/'single_update_gate.json')
    assert gate['passed'] and gate['reload_prediction_max_abs']==0 and gate['projection_max_update']>0
    scope=read(root/'parameter_scope.json')
    trainable={r['name'] for r in scope if r['trainable']}
    assert sum(r['count'] for r in scope if r['trainable'])==99880992
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert [e['optimizer_updates'] for e in updates]==[1]+list(range(10,251,10))
    assert all(np.isfinite(e['loss']) and np.isfinite(e['gradient_norm']) for e in updates)
    paths=[root/f'checkpoint_{i:06d}' for i in (1,50,100,150,200,250)]
    for p in paths:
        for name,digest in read(p/'checkpoint_manifest.json')['files_sha256'].items():
            assert file_sha(p/name)==digest,(p,name)
    state=torch.load(paths[-1]/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['optimizer_updates']==250 and state['consumed_samples']==4000
    assert set(state['rng'])=={'python','numpy','torch','cuda'}
    assert file_sha(root/'sampled_indices.npy')==state['sampled_indices_sha256']
    sampled=np.load(root/'sampled_indices.npy')
    assert sampled.shape==(4000,) and sampled.min()>=0 and sampled.max()<7392
    assert state['dataset_manifest_sha256']==config['dataset_manifest_sha256']
    steps={float(s['step']) for s in state['optimizer']['state'].values()}
    assert steps=={250.},steps
    frozen_count=changed_count=0
    with safe_open(str(DEFAULT_BASE_MODEL/'model.safetensors'),framework='pt',device='cpu') as base, \
         safe_open(str(paths[-1]/'pretrained_model/model.safetensors'),framework='pt',device='cpu') as final:
        assert set(base.keys())==set(final.keys()),'unexpected checkpoint tensor keys'
        for key in final.keys():
            before,after=base.get_tensor(key),final.get_tensor(key)
            assert torch.isfinite(after).all(),key
            if key in trainable:
                assert after.dtype==torch.float32,key
                changed_count+=int(not torch.equal(before.float(),after))
            else:
                assert torch.equal(before,after),f'frozen tensor changed: {key}'
                frozen_count+=1
        name='model.action_out_proj.weight'
        delta=final.get_tensor(name)-base.get_tensor(name).float()
        ignored=[i for i in range(32) if i not in (0,2)]
        assert delta[ignored].abs().max().item()==0,'ignored output channel updated'
        assert delta[[0,2]].abs().max().item()>0
        head_update=float(delta[[0,2]].abs().max())
    samples=[json.loads(s) for s in (root/'gpu_samples.jsonl').read_text().splitlines()]
    assert samples and min(s['free_mib'] for s in samples)>=2048
    assert read(root/'supervisor_status.json')['status']=='PILOT_250_COMPLETE_NOT_DT3_APPROVED'
    return {'stage':'DT3','status':'PILOT_ARTIFACT_CHECKS_PASS_NOT_DT3_APPROVAL',
        'optimizer_updates':250,'effective_batch':16,'micro_batch':16,'gradient_accumulation':1,
        'finite_logged_updates':len(updates),'first_logged_loss':updates[0]['loss'],
        'last_logged_loss':updates[-1]['loss'],'last_5_logged_losses':[e['loss'] for e in updates[-5:]],
        'full_checkpoint_hashes_checked':len(paths),'optimizer_state_steps':sorted(steps),
        'frozen_tensors_exact_to_base':frozen_count,'changed_trainable_tensors':changed_count,
        'ignored_output_weight_rows_exact_to_base':True,'vx_wz_head_max_update':head_update,
        'real_single_update_reload_prediction_max_abs':gate['reload_prediction_max_abs'],
        'sampled_indices_count':len(sampled),'min_actual_free_mib':min(s['free_mib'] for s in samples),
        'owned_peak_mib':max(s['owned_used_mib'] for s in samples),
        'navigation_success_approved':False,'closed_loop_evaluated':False,
        'training_config_sha256':file_sha(root/'training_config.json'),
        'checkpoint_250_manifest_sha256':file_sha(paths[-1]/'checkpoint_manifest.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    result=review(args.run)
    with (args.run/'root_pilot_review.json').open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
