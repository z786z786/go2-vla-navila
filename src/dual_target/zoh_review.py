"""Independent trace gates, including raw stop requests and actual body settling."""
import json
from pathlib import Path
import math
from .zoh_control import ZohSpec


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def review_episode(root, mode):
    root = Path(root)
    pre = read_lines(root/'pre_action.jsonl')
    post = read_lines(root/'post_step_events.jsonl')
    high = read_lines(root/'high_level_pre_action.jsonl')
    meta = json.loads((root/'episode_manifest.json').read_text())
    spec = ZohSpec()
    if len(pre) != len(post) or len(high) != (len(pre)+9)//10 or not pre:
        raise ValueError('missing low/high level evidence')
    previous = [0.,0.,0.]
    for j,h in enumerate(high):
        if h['low_level_start_index'] != j*10 or h['high_level_index'] != j:
            raise ValueError('not a new 5 Hz source trajectory')
        a = pre[j*10]
        for k in ('rgb_path','body_velocity_body','task','physics_step','sim_time_s','raw_action','applied_action'):
            if h[k] != a[k]: raise ValueError('high/low evidence differs: '+k)
        if j and (h['physics_step']-high[j-1]['physics_step'] != 40 or abs(h['sim_time_s']-high[j-1]['sim_time_s']-.2)>1e-5):
            raise ValueError('high-level time-grid mismatch')
        expected=[]
        for raw,old,(lo,hi),rate in zip(h['raw_action'],previous,spec.bounds,spec.rate_limits):
            clipped=max(lo,min(hi,raw))
            expected.append(max(old-rate*.2,min(old+rate*.2,clipped)))
        if any(abs(a-b)>1e-12 for a,b in zip(expected,h['applied_action'])):
            raise ValueError('shared rate/absolute boundary violation')
        previous=h['applied_action']
        for a in pre[j*10:(j+1)*10]:
            if a['raw_action'] != h['raw_action'] or a['applied_action'] != h['applied_action']:
                raise ValueError('not zero-order held')
    for a,b in zip(pre,post):
        if b['physics_step_after']-b['physics_step_before'] != 4:
            raise ValueError('changed low-level decimation')
        if abs(b['sim_time_after_s']-b['sim_time_before_s']-.02)>1e-5:
            raise ValueError('changed low-level time grid')
        if b['fallen'] or any(b['collision_latched_substeps']) or b['terminated'] or b['truncated']:
            raise ValueError('contact, fall or reset in candidate trace')
    limits=meta['thresholds']
    windows=[]
    if mode == 'probe':
        if len(pre) != 700: raise ValueError('incomplete fixed probe')
        for start,end in ((250,300),(450,500),(650,700)):
            window=post[start:end]
            linear=max(math.hypot(*b['body_velocity_body'][:2]) for b in window)
            yaw=max(abs(b['body_velocity_body'][2]) for b in window)
            if linear >= limits['body_linear_speed_mps'] or yaw >= limits['body_yaw_rate_radps']:
                raise ValueError(f'probe did not settle in final second: {start}, {linear}, {yaw}')
            if any(any(abs(x)>1e-12 for x in a['raw_action']+a['applied_action']) for a in pre[start:end]):
                raise ValueError('nonzero held command in probe stop window')
            windows.append(dict(start=start,end=end,max_linear=linear,max_yaw=yaw))
        if sum(b['body_velocity_body'][0] for b in post[120:150])/30 < .05:
            raise ValueError('no forward response')
        if sum(b['body_velocity_body'][2] for b in post[320:350])/30 < .05:
            raise ValueError('no positive yaw response')
        if sum(b['body_velocity_body'][2] for b in post[520:550])/30 > -.05:
            raise ValueError('no negative yaw response')
    else:
        from .zoh_parking_review import review_trace
        target=meta['target_slot']; other='B' if target=='A' else 'A'
        result=review_trace(pre,post,dict(correct_parking_center_xy=meta['parking_centers_xy'][target],
            other_parking_center_xy=meta['parking_centers_xy'][other],thresholds=limits,
            physics_dt_s=.005,decimation=4,contact_threshold_n=1.))
        if not result['trace_passed'] or post[-1]['scorer_status'] != 'success':
            raise ValueError('ZOH expert did not complete unchanged parking scorer')
    return dict(passed=True, mode=mode, low_level_steps=len(pre), high_level_commands=len(high),
                complete_holds=len(pre)//10, stop_windows=windows, dataset_approved=False)
