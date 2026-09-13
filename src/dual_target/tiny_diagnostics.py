"""Fixed-observation language/colour diagnostics. No simulator or training."""
import argparse
import json
from pathlib import Path
import time

from .tiny_train import load_training_policy, reload_processors
from .tiny_train_core import file_sha, approved_dataset
from .contracts import IMAGE_KEY, STATE_KEY, RED_TASK, BLUE_TASK


def main():
    import numpy as np
    from PIL import Image
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest,_=approved_dataset(a.dataset,a.approval)
    ck= json.loads((a.checkpoint/'checkpoint_manifest.json').read_text())
    for name,digest in ck['files_sha256'].items():
        if file_sha(a.checkpoint/name)!=digest:
            raise ValueError('checkpoint artifact changed: '+name)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    config=PreTrainedConfig.from_pretrained(a.checkpoint/'pretrained_model',local_files_only=True)
    config.device='cpu'
    model=load_training_policy(a.checkpoint/'pretrained_model',config).eval()
    # Fresh CPU processors use the exact approved statistics; saved processor
    # configs target CUDA and must not implicitly allocate the busy device.
    from .tiny_train import processors
    pre,post=processors(config,json.loads((a.dataset/'normalizer.json').read_text()))
    plan=json.loads(Path(manifest['plan_path']).read_text())
    results=[]
    with (a.output/'predictions.jsonl').open('x',buffering=1) as log:
        for slot in plan['slots']:
            ep=Path(slot['episode_path'])
            rows=[json.loads(s) for s in (ep/'pre_action.jsonl').read_text().splitlines()]
            # One actual initial image per configuration, both instructions
            # with identical state and noise; no synthetic pixel recolouring.
            cases=[]
            if slot['slot_index']%2==0:
                cases.extend([('initial',0,RED_TASK),('initial',0,BLUE_TASK)])
            cases.append(('terminal',len(rows)-1,slot['instruction']))
            for phase,index,task in cases:
                row=rows[index]
                rgb=np.asarray(Image.open(ep/row['rgb_path']).convert('RGB')).copy()
                batch=pre({IMAGE_KEY:torch.from_numpy(rgb).permute(2,0,1).float()/255,
                    STATE_KEY:torch.tensor(row['body_velocity_body'],dtype=torch.float32),'task':task})
                noise=torch.randn((1,50,32),generator=torch.Generator().manual_seed(20260906))
                start=time.monotonic()
                with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                    pred=post(model.predict_action_chunk(batch,noise=noise)).float()[0]
                if pred.shape!=(50,3) or not torch.isfinite(pred).all():
                    raise ValueError('nonfinite or wrong-shaped diagnostic prediction')
                record={'slot':slot['slot_index'],'phase':phase,'anchor':index,'task':task,
                    'source_rgb_sha256':file_sha(ep/row['rgb_path']),
                    'state':row['body_velocity_body'],'raw_chunk':pred.tolist(),
                    'mean_first10_vx':float(pred[:10,0].mean()),
                    'mean_first10_wz':float(pred[:10,2].mean()),
                    'first10_raw_stop_fraction':float(((pred[:10,0].abs()<.03)&(pred[:10,2].abs()<.05)).float().mean()),
                    'wall_s':time.monotonic()-start}
                if phase=='initial':
                    target='red' if task==RED_TASK else 'blue'
                    target_slot='a' if slot['scene']['slot_a_color']==target else 'b'
                    record['expected_initial_turn_sign']=1 if slot['scene'][f'slot_{target_slot}_center_xy'][1]>0 else -1
                    record['turn_sign_matches']=record['mean_first10_wz']*record['expected_initial_turn_sign']>0
                results.append(record)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k not in ('raw_chunk','state')}),flush=True)
    initial=[r for r in results if r['phase']=='initial']
    terminal=[r for r in results if r['phase']=='terminal']
    summary={'stage':'DT3','status':'OFFLINE_DIAGNOSTICS_COMPLETE_NOT_CLOSED_LOOP',
        'checkpoint':str(a.checkpoint),'checkpoint_manifest_sha256':file_sha(a.checkpoint/'checkpoint_manifest.json'),
        'device':'cpu','precision':'BF16 autocast with FP32 trainable weights',
        'initial_turn_sign_matches':sum(r['turn_sign_matches'] for r in initial),'initial_cases':len(initial),
        'terminal_cases':len(terminal),'terminal_first10_all_raw_stop':sum(r['first10_raw_stop_fraction']==1 for r in terminal),
        'navigation_success_approved':False,'training':False,
        'limitations':['CPU diagnostic numerics may differ from CUDA; final claims require CUDA closed-loop',
                      'Initial turn sign is a heuristic diagnostic, not a task success criterion']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
