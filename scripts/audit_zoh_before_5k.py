"""Read-only data/inference audit; writes only a new evidence directory."""
import json
import os
from pathlib import Path
import math

def main():
    import numpy as np
    import torch
    from PIL import Image
    from src.dual_target.zoh_train import FrameCache, load_training_policy, reload_processors, processors
    from src.dual_target.zoh_train_core import approved_dataset
    from src.dual_target.tiny_train_core import file_sha
    from src.dual_target.gpu_wait import probe_gpu
    from src.dual_target.contracts import IMAGE_KEY, STATE_KEY
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    base=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2')
    out=base/'zoh_pre5k_audit_0906_v1';out.mkdir(exist_ok=False)
    root=base/'zoh_dataset_0906_v2/dataset'
    approval=Path('/home/wxh/go2_short_vln/reports/dual_target_v1/zoh_v2_dataset_approval.json')
    manifest,norm=approved_dataset(root,approval)
    torch.set_num_threads(4)
    cache=FrameCache(root,manifest,out)
    weights=np.load(root/'episode_balanced_sampler_weights.npy')
    starts=[0]+[i for i in range(1,len(cache.tasks)) if cache.ends[i]!=cache.ends[i-1]]
    episodes=[]
    for start in starts:
        end=int(cache.ends[start]);actions=cache.action[start:end]
        assert len(set(cache.tasks[start:end]))==1
        episodes.append(dict(start=start,frames=end-start,task=cache.tasks[start],weight=float(weights[start:end].sum()),
            first_actions=actions[:5].tolist(),mean_action=actions.mean(0).tolist(),
            near_zero_fraction=float(((actions[:,0].abs()<.03)&(actions[:,2].abs()<.05)).float().mean())))
    report=dict(episodes=episodes,sampler=json.loads((root/'sampler.json').read_text()),dataset_hashes_verified=True)
    (out/'data.json').write_text(json.dumps(report,indent=2))
    ckpt=base/'zoh_train_1000_0906_v1/checkpoint_001000'
    cm=json.loads((ckpt/'checkpoint_manifest.json').read_text())
    for name,digest in cm['files_sha256'].items():assert file_sha(ckpt/name)==digest
    assert probe_gpu().free_mib>=4096
    torch.cuda.set_per_process_memory_fraction(.14)
    model=load_training_policy(ckpt/'pretrained_model').eval()
    pre,post=reload_processors(ckpt/'pretrained_model');fresh,_=processors(model.config,norm)
    original=cache.batch([0]);a=pre(original);b=fresh(cache.batch([0]))
    for k,v in a.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,b[k]),k
    def predict(batch,seed=20260906):
        assert probe_gpu().free_mib>=2048
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            noise=torch.randn((1,5,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            return post(model.predict_action_chunk(batch,noise=noise)).float().cpu()
    checks=[]
    for start in starts:
        item=cache.batch([start]);normal=pre(item)
        swap=dict(item);swap['task']=[item['task'][0].replace('red','TEMP').replace('blue','red').replace('TEMP','blue')]
        swapped=pre(swap)
        assert not torch.equal(normal[OBS_LANGUAGE_TOKENS],swapped[OBS_LANGUAGE_TOKENS])
        pa,pb=predict(normal),predict(swapped)
        checks.append(dict(start=start,task=item['task'][0],expert_first=cache.action[start].tolist(),
            predicted_first=pa[0,0].tolist(),swapped_first=pb[0,0].tolist(),
            task_swap_chunk_mean_abs=float((pa-pb).abs().mean())))
    runtime=[]
    runpaths=[base/'zoh_val_scored_0906_v1/runs/zoh_val_scored_0906_v1_c1000_s16']
    runpaths += [base/f'zoh_val_resume_0906_v1/runs/zoh_val_resume_0906_v1_c1000_s{s}' for s in (17,18)]
    for run in runpaths:
        record=json.loads((run/'model_chunks.jsonl').read_text().splitlines()[0]);req=record['input']
        rgb=np.asarray(Image.open(req[IMAGE_KEY]).convert('RGB')).copy()
        batch=pre({IMAGE_KEY:torch.from_numpy(rgb).permute(2,0,1).float()/255,STATE_KEY:torch.tensor(req[STATE_KEY]),'task':req['task']})
        prediction=predict(batch)[0]
        error=float((prediction-torch.tensor(record['raw_chunk'])).abs().max())
        assert error<1e-5,error
        runtime.append(dict(run=run.name,task=req['task'],replay_max_abs=error))
    report.update(checkpoint_hashes_verified=True,processor_train_deploy_exact=True,initial_task_swap=checks,runtime_first_chunk_replay=runtime)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
