#!/usr/bin/env python3
"""Audit causal raw recording and export aligned ablation arrays, without training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_r2r_ablation import history8


def read(path):
    with path.open() as f:return [json.loads(s) for s in f if s.strip()]


def main():
    import numpy as np
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    meta=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
    low=read(root/'low_level.jsonl');actions=read(root/'actions.jsonl');frames=read(root/'frames.jsonl')
    errors=[]
    def check(ok,reason):
        if not ok:errors.append(reason)
    check(bool(low) and bool(actions) and bool(frames),'empty_recording')
    check(len(low)==summary.get('low_level_records'),'low_count')
    check(len(actions)==summary.get('action_records'),'action_count')
    check(len(frames)==summary.get('rgb_frames'),'rgb_count')
    def finite(value):
        if isinstance(value,float):return bool(np.isfinite(value))
        if isinstance(value,list):return all(finite(v) for v in value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        return True
    check(finite(low) and finite(actions) and finite(frames),'nonfinite_raw_data')
    for i,row in enumerate(low):
        check(row['low_index']==i,f'low_index:{i}')
        check(row['post']['physics_step']-row['pre']['physics_step']==4,f'physics_dt:{i}')
        check(np.isclose(row['post']['sim_time_s']-row['pre']['sim_time_s'],.02),f'time_dt:{i}')
        if i:check(row['pre']==low[i-1]['post'],f'state_discontinuity:{i}')
        check(row['action_index']==i//10,f'action_index:{i}')
        check(not row['pre_reset_state_used'] or i==len(low)-1,f'continued_after_reset:{i}')
        expected_hold=(low[i-1]['hold_steps'] if i else 0)+1 if (
            row['stop_intent'] and row['command_zero'] and row['actual_stationary']
            and row['inside_stop_region'] and not row['pre_reset_state_used']) else 0
        check(row['hold_steps']==expected_hold,f'terminal_hold_counter:{i}')
    frame_map={f['frame_id']:f for f in frames}
    hashes=[];camera_counts=[]
    for i,f in enumerate(frames):
        path=root/f['path'];check(f['frame_id']==i,f'frame_index:{i}')
        with Image.open(path) as im:
            im.load();check(im.size==(512,512) and im.mode=='RGB',f'rgb_shape:{i}')
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest());camera_counts.append(f['camera_frame_counter'])
        if i:
            check(f['physics_step']>frames[i-1]['physics_step'],f'frame_clock:{i}')
            check(f['camera_frame_counter']>frames[i-1]['camera_frame_counter'],f'stale_camera:{i}')
    complete=[]
    for i,a in enumerate(actions):
        check(a['action_index']==i,f'action_contiguity:{i}')
        segment=low[a['low_start_index']:a['low_start_index']+a['executed_low_steps']]
        check(a['pre_state']==segment[0]['pre'] and a['post_state']==segment[-1]['post'],f'interval_alignment:{i}')
        check(all(s['applied_command']==a['applied_command'] and s['raw_command']==a['raw_command'] for s in segment),f'zoh_hold:{i}')
        f=frame_map[a['rgb_frame_id']]
        check(f['physics_step']==a['pre_state']['physics_step'],f'image_action_alignment:{i}')
        eligible=[f['frame_id'] for f in frames if f['navila_keyframe'] and f['physics_step']<=a['pre_state']['physics_step']]
        check(a['history']==history8(eligible),f'history_selection:{i}')
        check(all(h is None or frame_map[h]['physics_step']<=a['pre_state']['physics_step'] for h in a['history']['frame_ids']),f'history_future_leak:{i}')
        check(all(j<i for j in a['history_action_indices']),f'action_history_future_leak:{i}')
        values=[[s['post']['linear_body'][0],s['post']['linear_body'][1],s['post']['angular_body'][2]] for s in segment]
        check(np.allclose(np.mean(values,axis=0),a['mean_actual_velocity_body']),f'actual_mean:{i}')
        check(a['stop_intent']==a['expert']['stop_intent'],f'stop_intent:{i}')
        check(a['complete_command_interval']==(len(segment)==10),f'partial_interval:{i}')
        if a['complete_command_interval'] and not a['environment_done']:complete.append(a)
    check(summary.get('success') is True and summary.get('final_hold_steps',0)>=100,'not_successful_terminal_hold')
    check(summary.get('reset_count')==0,'unexpected_reset')
    check(len(set(hashes))>1,'all_rgb_identical')
    contacts_available=any(
        np.linalg.norm(np.asarray(r['post']['contact_force_world']),axis=-1).max()>1 for r in low)
    audit={'schema':'r2r-ablation-audit-v1','passed':not errors,'training_eligible':not errors,
        'errors':errors,'low_records':len(low),'action_records':len(actions),'rgb_frames':len(frames),
        'unique_rgb_sha256':len(set(hashes)),'complete_action_intervals':len(complete),
        'source_split':'train','positive_stop_actions':sum(a['stop_intent'] for a in actions),
        'contact_force_usable':contacts_available,
        'rich_recording_complete':not errors and contacts_available,
        'warnings':[] if contacts_available else ['Contact forces are all zero; availability is unverified. Do not interpret as collision-free.'],
        'checks':['consecutive clocks and pre/post state continuity','RGB/action exact physics-step alignment',
            'camera frame freshness and JPEG decode','ZOH raw/applied labels','measured interval velocity mean',
            'NaVILA history padding and no future frames','terminal hold and no reset','counts and complete intervals']}
    if not errors:
        # Raw image files remain single-copy; all visual variants reference them.
        def vel(s):return [s['linear_body'][0],s['linear_body'][1],s['angular_body'][2]]
        np.savez_compressed(root/'ablation_arrays.npz',
            action_index=np.asarray([a['action_index'] for a in complete],dtype=np.int64),
            rgb_frame_id=np.asarray([a['rgb_frame_id'] for a in complete],dtype=np.int64),
            history_frame_ids=np.asarray([[-1 if x is None else x for x in a['history']['frame_ids']] for a in complete],dtype=np.int64),
            history_padding_mask=np.asarray([a['history']['padding_mask'] for a in complete],dtype=bool),
            state_velocity_pre=np.asarray([vel(a['pre_state']) for a in complete],dtype=np.float32),
            velocity_actual_post=np.asarray([vel(a['post_state']) for a in complete],dtype=np.float32),
            velocity_actual_interval_mean=np.asarray([a['mean_actual_velocity_body'] for a in complete],dtype=np.float32),
            command_raw=np.asarray([a['raw_command'] for a in complete],dtype=np.float32),
            command_applied=np.asarray([a['applied_command'] for a in complete],dtype=np.float32),
            stop_intent=np.asarray([a['stop_intent'] for a in complete],dtype=np.float32),
            action_command_stop=np.asarray([[*a['applied_command'],float(a['stop_intent'])] for a in complete],dtype=np.float32))
        with np.load(root/'ablation_arrays.npz',allow_pickle=False) as exported:
            check(exported['action_command_stop'].shape==(len(complete),4),'export_shape')
            check(np.isfinite(exported['action_command_stop']).all(),'export_nonfinite')
        audit['passed']=audit['training_eligible']=not errors
        # Preserve raw JSONL and benchmark an additional lossless copy.
        with (root/'low_level.jsonl').open('rb') as src,gzip.open(root/'low_level.jsonl.gz','wb',compresslevel=3) as dst:
            for chunk in iter(lambda:src.read(1024*1024),b''):dst.write(chunk)
        audit['numeric_gzip_bytes']=(root/'low_level.jsonl.gz').stat().st_size
    (root/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    byte_manifest=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name!='file_manifest.json':
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            byte_manifest.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    (root/'file_manifest.json').write_text(json.dumps(byte_manifest,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
