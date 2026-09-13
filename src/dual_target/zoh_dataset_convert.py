"""CPU-only conversion: train/validation separated; test never exported."""
import argparse
import json
import os
from pathlib import Path

from .tiny_plan import sha
from .tiny_data import normalized_stats, chunk_indices
from .zoh_dataset import make_plan, audit_dataset_episode
from .reset_audit import validate_paired_resets
from .contracts import IMAGE_KEY, STATE_KEY


def require_train_split(manifest):
    if manifest.get('split')!='train':raise ValueError('only train may supply updates or normalizer')


def convert(plan_path,output):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    plan=json.loads(plan_path.read_text())
    if plan!=make_plan(plan_path.parent):raise ValueError('noncanonical dataset plan')
    state=json.loads((plan_path.parent/'queue_status.json').read_text())
    if len(state['completed'])!=32:raise ValueError('all 32 expert trajectories required before export')
    output.mkdir(parents=True,exist_ok=False)
    data={};audits=[];pairs={}
    for slot in plan['slots']:
        audit,rows=audit_dataset_episode(Path(slot['episode_path']),slot)
        data[slot['index']]=rows;audits.append(audit)
        key=slot['geometry_group_id'],slot['color_configuration']
        pairs.setdefault(key,[]).append(json.loads((Path(slot['episode_path'])/'paired_reset_audit.json').read_text()))
    pair_reviews=[validate_paired_resets(*pair) for pair in pairs.values()]
    features={IMAGE_KEY:{'dtype':'image','shape':(512,512,3),'names':['height','width','channels']},
              STATE_KEY:{'dtype':'float32','shape':(3,),'names':['body_vx','body_vy','body_yaw_rate']},
              'action':{'dtype':'float32','shape':(3,),'names':['vx','vy','wz']}}
    summaries=[]
    for split in ('train','validation'):
        slots=[s for s in plan['slots'] if s['split']==split]
        directory=output/split
        dataset=LeRobotDataset.create(repo_id=f'local/go2_zoh_v2_{split}',root=directory,fps=5,
            features=features,robot_type='unitree_go2_high_level_velocity',use_videos=False,
            image_writer_processes=0,image_writer_threads=0)
        try:
            for slot in slots:
                for row in data[slot['index']]:
                    with Image.open(Path(slot['episode_path'])/row['rgb_path']) as im:
                        rgb=np.asarray(im.convert('RGB')).copy()
                    dataset.add_frame({IMAGE_KEY:rgb,STATE_KEY:np.asarray(row['body_velocity_body'],dtype=np.float32),
                        'action':np.asarray(row['applied_action'],dtype=np.float32),'task':row['task']})
                dataset.save_episode(parallel_encoding=False)
        finally:dataset.finalize()
        loaded=LeRobotDataset(f'local/go2_zoh_v2_{split}',root=directory,
            delta_timestamps={'action':[i*.2 for i in range(5)]})
        allowed={IMAGE_KEY,STATE_KEY,'action','timestamp','frame_index','episode_index','index','task_index'}
        if set(loaded.features)!=allowed:raise ValueError('GT or unexpected feature in learner dataset')
        checked=[];offset=0
        for episode,slot in enumerate(slots):
            rows=data[slot['index']];n=len(rows)
            for anchor in sorted({0,1,n//4,n//2,3*n//4,n-6,n-5,n-4,n-2,n-1}):
                item=loaded[offset+anchor]
                assert int(item['episode_index'])==episode and int(item['frame_index'])==anchor
                assert item['task']==rows[anchor]['task']
                assert np.allclose(item[STATE_KEY].numpy(),rows[anchor]['body_velocity_body'],atol=1e-7,rtol=1e-6)
                future,pad=chunk_indices(anchor,n,chunk_size=5)
                expected=np.asarray([rows[i]['applied_action'] for i in future],dtype=np.float32)
                assert item['action'].shape==(5,3) and np.array_equal(item['action'].numpy(),expected)
                assert item['action_is_pad'].tolist()==pad
                with Image.open(Path(slot['episode_path'])/rows[anchor]['rgb_path']) as im:
                    rgb=np.asarray(im.convert('RGB'))
                decoded=np.rint(item[IMAGE_KEY].permute(1,2,0).numpy()*255).astype(np.uint8)
                assert np.array_equal(decoded,rgb)
                checked.append(dict(episode=episode,anchor=anchor,padding=sum(pad),rgb_exact=True))
            offset+=n
        assert len(loaded)==offset
        summary=dict(split=split,episodes=len(slots),frames=offset,checked_anchors=checked,
            trajectory_ids=[s['trajectory_id'] for s in slots],fps=5,chunk_size=5,feature_allowlist=sorted(allowed))
        if split=='train':
            require_train_split(summary)
            flat=[row for s in slots for row in data[s['index']]]
            normalizer=dict(source_split='train',plan_sha256=sha(plan_path),
                state=normalized_stats([r['body_velocity_body'] for r in flat]),
                action=normalized_stats([r['applied_action'] for r in flat]),
                visual_normalization='IDENTITY',action_loss_active_channels=[True,False,True])
            (output/'normalizer.json').write_text(json.dumps(normalizer,indent=2)+'\n')
            weights=np.concatenate([np.full(len(data[s['index']]),1/(len(slots)*len(data[s['index']]))) for s in slots])
            np.save(output/'episode_balanced_sampler_weights.npy',weights)
            near=lambda r:abs(r['applied_action'][0])<.03 and abs(r['applied_action'][2])<.05
            exposure={}
            for key,fn in [('initial_5',lambda rows,i:i<5),('near_zero',lambda rows,i:near(rows[i])),
                           ('padded',lambda rows,i:i+5>len(rows)),
                           ('any_chunk_near_zero',lambda rows,i:any(near(r) for r in rows[i:i+5]))]:
                exposure[key]=sum(sum(fn(data[s['index']],i) for i in range(len(data[s['index']])))/len(data[s['index']]) for s in slots)/len(slots)
            (output/'sampler.json').write_text(json.dumps(dict(policy='trajectory_uniform_then_anchor_uniform',replacement=True,
                weights_sum=float(weights.sum()),fractions=exposure,no_terminal_cap=True),indent=2)+'\n')
        (output/(split+'_loader_review.json')).write_text(json.dumps(summary,indent=2)+'\n')
        summaries.append(summary)
    (output/'raw_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (output/'paired_review.json').write_text(json.dumps(pair_reviews,indent=2)+'\n')
    manifest=dict(stage='ZOH_V2_DATASET',status='CONVERTED_AWAITING_ROOT_REVIEW',plan_sha256=sha(plan_path),
        plan_path=str(plan_path),splits=summaries,test_exported=False,training=False,
        files_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(status=manifest['status'],train_episodes=16,validation_episodes=8,test_exported=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(convert(a.plan.resolve(),a.output.resolve())))
