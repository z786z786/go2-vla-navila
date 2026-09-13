"""Structural policy trace audit; model failure is a result, not a retry."""
import json
import math
from pathlib import Path
from .zoh_control import CommandBoundary
from .zoh_review import read_lines
from .zoh_parking_review import review_trace
from .contracts import IMAGE_KEY, STATE_KEY
from .zoh_result_contract import normalized_result


def audit_policy_episode(ep, result, *, allow_state_placeholder=False):
    ep=Path(ep);pre=read_lines(ep/'pre_action.jsonl');post=read_lines(ep/'post_step_events.jsonl')
    high=read_lines(ep/'high_level_pre_action.jsonl');chunks=read_lines(ep.parents[1]/'model_chunks.jsonl')
    meta=json.loads((ep/'episode_manifest.json').read_text())
    result=normalized_result(result,len(pre),len(post))
    if not pre or len(pre)!=len(post) or len(pre)!=result['steps'] or len(high)!=(len(pre)+9)//10 or len(chunks)!=len(high):
        raise ValueError('incomplete policy execution evidence')
    if any(p['scorer_status']=='invalid_sample' for p in post):
        raise ValueError('invalid scoring evidence cannot be accepted as model failure')
    boundary=CommandBoundary()
    for j,(h,c) in enumerate(zip(high,chunks)):
        if h['low_level_start_index']!=j*10 or c['execute_steps']!=1 or c['action_dt_s']!=.2:
            raise ValueError('old execution semantics')
        state_matches = h['body_velocity_body'] == c['input'][STATE_KEY]
        placeholder_matches = (
            allow_state_placeholder
            and c['input'][STATE_KEY] == [0.0, 0.0, 0.0]
            and c.get('source_state_read') is False
            and c.get('state_placeholder') == 'constant_zero_vector_dim3'
        )
        if h['raw_action']!=c['raw_chunk'][0] or h['task']!=c['input']['task'] or not (state_matches or placeholder_matches):
            raise ValueError('raw policy/observation evidence differs')
        if Path(c['input'][IMAGE_KEY])!=ep/h['rgb_path']:
            raise ValueError('policy image does not match high-level input')
        expected=boundary.update(h['raw_action'])
        if any(abs(x-y)>1e-12 for x,y in zip(expected,h['applied_action'])):
            raise ValueError('deployment boundary changed')
        for before in pre[j*10:(j+1)*10]:
            if before['raw_action']!=h['raw_action'] or before['applied_action']!=h['applied_action']:
                raise ValueError('not a zero-order hold')
    for i,(a,b) in enumerate(zip(pre,post)):
        if b['physics_step_before']!=a['physics_step'] or b['physics_step_after']-a['physics_step']!=4 or abs(b['sim_time_after_s']-a['sim_time_s']-.02)>1e-5:
            raise ValueError('physical time alignment changed')
        if i and (a['physics_step']!=post[i-1]['physics_step_after'] or a['body_velocity_body']!=post[i-1]['body_velocity_body']):
            raise ValueError('state gap/reset in policy execution')
    target=meta['target_slot'];other='B' if target=='A' else 'A'
    parking={'trace_passed':False,'reason':'runtime task failure; no success claim'}
    if result['status']=='success':
        parking=review_trace(pre,post,dict(correct_parking_center_xy=meta['parking_centers_xy'][target],
            other_parking_center_xy=meta['parking_centers_xy'][other],thresholds=meta['thresholds'],
            physics_dt_s=.005,decimation=4,contact_threshold_n=1.))
    if result['status']=='success' and (not parking['trace_passed'] or post[-1]['scorer_status']!='success'):
        raise ValueError('success claim failed independent parking audit')
    return dict(status=result['status'],steps=len(pre),high_level_commands=len(high),
        reached_correct_region=any(p['in_correct_parking_region'] for p in post),
        reached_other_region=any(p['in_other_parking_region'] for p in post),
        collision=any(any(p['collision_latched_substeps']) for p in post),
        final_distance_m=math.dist(post[-1]['robot_pose_w'][:2],meta['parking_centers_xy'][target]),
        independent_parking_review=parking,terminal_partial_hold_steps=len(pre)%10,
        scope='validation-only; no dataset export or training')
