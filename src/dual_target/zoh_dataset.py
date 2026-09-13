"""V2 4/2/2 dataset contract, complete holds and independent episode audit."""
import json
import math
from pathlib import Path

from .tiny_plan import groups as train_groups, SOURCE_ROOT, sha, verify_binding
from .layouts import GeometryGroup, TargetSlot, Vec2, build_four_tasks, SplitTask, validate_grouped_splits
from .scene import DualTargetSceneSpec
from .reset_audit import derive_pair_seed
from .zoh_control import ZohSpec
from .zoh_review import review_episode, read_lines


def grouped_tasks():
    groups=[('train',g) for g in train_groups()]
    for split,name,a,b in (
        ('validation','v2_val_000',(2.90,1.15),(2.60,-1.35)),
        ('validation','v2_val_001',(2.65,1.55),(2.95,-.95)),
        ('test','v2_test_000',(3.20,1.50),(2.85,-1.25)),
        ('test','v2_test_001',(2.70,.75),(3.15,-1.65))):
        groups.append((split,GeometryGroup(name,name,TargetSlot('A',Vec2(*a),Vec2(-1.,0.)),
            TargetSlot('B',Vec2(*b),Vec2(-1.,0.)),Vec2(0.,0.))))
    tasks=[(split,DualTargetSceneSpec(g,t.color_configuration),t) for split,g in groups for t in build_four_tasks(g)]
    errors=validate_grouped_splits([SplitTask(task=t,split=s) for s,_,t in tasks])
    if errors: raise ValueError(errors)
    return tasks


def verify_gate(source=SOURCE_ROOT):
    verify_binding(source)
    approval=json.loads((source/'reports/dual_target_v1/zoh_v2_gate_approval.json').read_text())
    if approval['status']!='APPROVED' or sha(source/approval['report'])!=approval['report_sha256']:
        raise ValueError('missing root physical gate approval')
    manifest_path=Path(approval['remote_run_root'])/'gate_manifest.json'
    if sha(manifest_path)!=approval['gate_manifest_sha256']: raise ValueError('gate manifest drift')
    for path,digest in json.loads(manifest_path.read_text())['source_sha256'].items():
        if sha(source/path)!=digest: raise ValueError('frozen gate code drift: '+path)
    return approval


def make_plan(root):
    approval=verify_gate()
    slots=[]
    for i,(split,scene,task) in enumerate(grouped_tasks()):
        run_id=f'{root.name}_s{i:02d}'
        episode_id=f'{task.geometry_group_id}_{task.color_configuration}_{task.target_color}_r00'
        slots.append(dict(index=i,split=split,trajectory_id=episode_id,run_id=run_id,
            geometry_group_id=task.geometry_group_id,lineage_root_id=task.lineage_root_id,
            instruction=task.instruction,color_configuration=task.color_configuration,target_color=task.target_color,
            target_slot=task.target_slot,seed=derive_pair_seed(geometry_group_id=task.geometry_group_id,
                color_configuration=task.color_configuration,repeat=0),scene=scene.audit_metadata(),
            episode_path=str(root/('test_audit' if split=='test' else 'runs')/run_id/'episodes'/episode_id)))
    names=['zoh_dataset','zoh_dataset_loop','zoh_dataset_runtime','zoh_dataset_queue','zoh_dataset_convert']
    result=dict(stage='ZOH_V2_DATASET',status='PLANNED_NOT_APPROVED',slots=slots,
        physics_gate_report_sha256=approval['report_sha256'],zoh_spec=ZohSpec().metadata(),fps=5,
        chunk_size=5,execute_steps=1,terminal_complete_zero_holds=6,
        source_sha256={f'src/dual_target/{n}.py':sha(SOURCE_ROOT/f'src/dual_target/{n}.py') for n in names},
        visibility=dict(minimum_target_color_pixels=100,edge_margin_pixels=5,warning_target_fraction=.01),
        image_shape=[512,512,3],reuse_old_episodes=False,test_policy_evaluation=False,training=False)
    return json.loads(json.dumps(result))


def low_stop(raw,applied,body,correct,thresholds):
    return (correct and all(abs(v)<=1e-12 for v in list(raw)+list(applied))
        and math.hypot(*body[:2])<thresholds['body_linear_speed_mps']
        and abs(body[2])<thresholds['body_yaw_rate_radps'])


class TerminalHolds:
    def __init__(self):
        self.count=0
        self.current=True

    def observe(self, step, valid):
        if step%10==0: self.current=True
        self.current=self.current and valid
        if not valid:self.count=0
        if (step+1)%10==0:
            self.count=self.count+1 if self.current else 0
        return self.count>=6 and (step+1)%10==0


def complete_rows(high,pre,post):
    """Never turn partial executed commands into full 0.2 s training labels."""
    if len(pre)!=len(post): raise ValueError('pre/post length mismatch')
    kept=[];durations=[]
    for i,h in enumerate(high):
        start=h['low_level_start_index']
        if start!=i*10 or start>=len(pre):raise ValueError('wrong high-level anchor')
        count=min(10,len(pre)-start)
        duration=post[start+count-1]['sim_time_after_s']-pre[start]['sim_time_s']
        full=count==10 and abs(duration-.2)<1e-5
        durations.append(dict(high_level_index=i,low_level_steps=count,actual_duration_s=duration,complete=full))
        if full:kept.append(h)
        elif i!=len(high)-1:raise ValueError('partial hold before end of episode')
    return kept,durations


def audit_dataset_episode(ep,slot):
    ep=Path(ep)
    review=review_episode(ep,'expert')
    meta=json.loads((ep/'episode_manifest.json').read_text())
    if any(meta[k]!=slot[k] for k in ('geometry_group_id','lineage_root_id','instruction','target_color','target_slot','color_configuration')):
        raise ValueError('trajectory does not match frozen split slot')
    if meta['seed']!=slot['seed'] or meta['episode_id']!=slot['trajectory_id']:
        raise ValueError('trajectory seed or ID changed')
    pre=read_lines(ep/'pre_action.jsonl');post=read_lines(ep/'post_step_events.jsonl');high=read_lines(ep/'high_level_pre_action.jsonl')
    rows,durations=complete_rows(high,pre,post)
    written=read_lines(ep/'high_level_segments.jsonl')
    if written!=durations:raise ValueError('collector actual-duration evidence differs')
    tail=TerminalHolds();finished=False
    for i,(a,b) in enumerate(zip(pre,post)):
        valid=low_stop(a['raw_action'],a['applied_action'],a['body_velocity_body'],b['in_correct_parking_region'],meta['thresholds']) and low_stop(a['raw_action'],a['applied_action'],b['body_velocity_body'],b['in_correct_parking_region'],meta['thresholds'])
        finished=tail.observe(i,valid)
    if not finished or len(rows)!=len(high):raise ValueError('fresh data must end with six complete stopped holds')
    from PIL import Image
    import numpy as np
    visibility=[]
    hashes={}
    for h in rows:
        path=ep/h['rgb_path']
        hashes[h['rgb_path']]=sha(path)
        with Image.open(path) as im:
            if im.size!=(512,512):raise ValueError('camera shape changed')
            a=np.asarray(im.convert('RGB')).astype('float32');r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
            masks=dict(red=(r>80)&(r>g*1.15)&(r>b*1.15),blue=(b>80)&(b>g*1.15)&(b>r*1.15))
            counts={k:int(v.sum()) for k,v in masks.items()}
            y,x=np.nonzero(masks[slot['target_color']]);count=len(x)
            box=[int(x.min()),int(y.min()),int(x.max()),int(y.max())] if count else None
            edge=box is None or box[0]<=5 or box[1]<=5 or box[2]>=506 or box[3]>=506
            visibility.append(dict(index=h['high_level_index'],rgb_path=h['rgb_path'],counts=counts,
                target_bbox=box,target_fraction=count/(512*512),edge_warning=edge or count<.01*512*512))
    result=dict(review,slot=slot,complete_frames=len(rows),durations=durations,visibility=visibility,
        image_sha256=hashes,minimum_target_pixels=min(v['counts'][slot['target_color']] for v in visibility),
        edge_warning_frames=sum(v['edge_warning'] for v in visibility),terminal_complete_zero_holds=tail.count,
        evidence_sha256={n:sha(ep/n) for n in ('pre_action.jsonl','post_step_events.jsonl','high_level_pre_action.jsonl','high_level_segments.jsonl','episode_manifest.json')})
    (ep/'dataset_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if any(v['counts'][slot['target_color']]<100 for v in visibility):raise ValueError('target visibility below minimum; preserve warning audit and stop')
    if any(visibility[0]['counts'][c]<100 for c in ('red','blue')):raise ValueError('both targets must be visible initially')
    return result,rows
