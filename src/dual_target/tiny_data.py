"""CPU-only DT2 raw evidence checks and train-only sampling/normalizer stats."""
import json
import math
from pathlib import Path

from .contracts import validate_policy_input, IMAGE_KEY, STATE_KEY, TASK_KEY
from .records import validate_pre_action, validate_post_step
from .reset_audit import reset_audit_from_dict, validate_paired_resets
from .tiny_plan import read, sha, LOCK_HASH
from reports.dual_target_v1.dt1_independent_checks import review_trace


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalized_stats(rows):
    if not rows or any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in rows):
        raise ValueError('normalizer requires finite 3D training rows')
    n = len(rows)
    mean = [sum(v[i] for v in rows)/n for i in range(3)]
    raw_std = [math.sqrt(sum((v[i]-mean[i])**2 for v in rows)/n) for i in range(3)]
    return {'mean': mean, 'std': [s if s >= 1e-6 else 1.0 for s in raw_std],
            'raw_std': raw_std, 'count': n,
            'constant_channels': [i for i, s in enumerate(raw_std) if s < 1e-6],
            'min': [min(v[i] for v in rows) for i in range(3)],
            'max': [max(v[i] for v in rows) for i in range(3)]}


def chunk_indices(anchor, length, chunk_size=50):
    if type(anchor) is not int or not 0 <= anchor < length:
        raise ValueError('anchor outside episode')
    return [min(anchor+i, length-1) for i in range(chunk_size)], [anchor+i >= length for i in range(chunk_size)]


def audit_episode(slot, contract):
    p = Path(slot['episode_path']).resolve()
    m, pre, post = read(p/'episode_manifest.json'), lines(p/'pre_action.jsonl'), lines(p/'post_step_events.jsonl')
    for field in ('geometry_group_id', 'color_configuration', 'target_color', 'repeat', 'instruction', 'target_slot'):
        if m[field] != slot[field]:
            raise ValueError(f'episode differs from fixed slot: {field}')
    if (m['thresholds'] != contract['parking_thresholds'] or m['physics_dt_s'] != .005
            or m['decimation'] != 4 or m['seed'] != slot['pair_seed']
            or m['checkpoint_sha256'] != contract['low_level']['checkpoint_sha256']):
        raise ValueError('episode drift from locked task/low-level contract')
    for a, b in zip(pre, post):
        validate_pre_action(a)
        validate_post_step(b)
        validate_policy_input({IMAGE_KEY: a['rgb_path'], STATE_KEY: a['body_velocity_body'], TASK_KEY: a['task']})
    meta = {'correct_parking_center_xy': m['parking_centers_xy'][m['target_slot']],
            'other_parking_center_xy': m['parking_centers_xy']['B' if m['target_slot']=='A' else 'A'],
            'thresholds': m['thresholds'], 'physics_dt_s': .005, 'decimation': 4, 'contact_threshold_n': 1.}
    trace = review_trace(pre, post, meta)
    if post[-1]['scorer_status'] != 'success':
        raise ValueError('episode did not terminate successfully')
    images = []
    for a in pre:
        image = (p/a['rgb_path']).resolve()
        if not image.is_relative_to(p) or not image.is_file():
            raise ValueError('missing or escaping real pre-action RGB')
        images.append(sha(image))
    audit = reset_audit_from_dict(read(p/'paired_reset_audit.json'))
    for path, digest in ((audit.first_rgb_path, audit.first_rgb_sha256),
                         (audit.learner_first_rgb_path, audit.learner_first_rgb_sha256)):
        image = (p/path).resolve()
        if not image.is_relative_to(p) or sha(image) != digest:
            raise ValueError('reset image binding failed')
    if images[0] != audit.learner_first_rgb_sha256:
        raise ValueError('first learner input differs from paired audit')
    # Require one second of real stopped pre-action observations with zero
    # expert commands, not only post-action scorer flags or padded targets.
    tail = []
    for a, b in reversed(list(zip(pre, post))):
        vx, vy, wz = a['body_velocity_body']
        if (any(abs(v) > 1e-12 for v in a['raw_action'])
                or math.hypot(vx, vy) >= .03 or abs(wz) >= .05
                or not b['in_correct_parking_region']):
            break
        tail.append(a)
    span = tail[0]['sim_time_s'] - tail[-1]['sim_time_s'] if tail else 0.
    # Float32 Isaac dt drift is ~2e-8 seconds for a one-second interval.
    if len(tail) < 51 or span < 1.0 - 1e-6:
        raise ValueError('missing >=1s real terminal pre-action zero-command supervision')
    n = len(pre)
    near = [abs(a['raw_action'][0]) < .03 and abs(a['raw_action'][2]) < .05 for a in pre]
    exposure = {'anchors': n, 'terminal_frame_anchors': len(tail),
                'near_zero_frame_anchors': sum(near),
                'all_valid_chunk_near_zero_anchors': sum(all(near[i:min(n,i+50)]) for i in range(n)),
                'any_valid_chunk_near_zero_anchors': sum(any(near[i:min(n,i+50)]) for i in range(n)),
                'padded_anchors': min(n, 49),
                'padded_target_entries': sum(max(0, i+50-n) for i in range(n)),
                'episode_balanced_anchor_weight': 1.0/n}
    record = {'episode_path': str(p), 'slot': slot, 'frames': n, 'trace': trace,
              'terminal_real_pre_action_frames': len(tail), 'terminal_real_pre_action_span_s': span,
              'camera_sensor_frame_changes': sum(a['camera_sensor_frame'] != b['camera_sensor_frame'] for a,b in zip(pre,pre[1:])),
              'adjacent_rgb_byte_duplicates': sum(a == b for a,b in zip(images, images[1:])),
              'camera_timestamps_s': [a['camera_timestamp_s'] for a in pre],
              'rgb_sha256': images, 'exposure': exposure,
              'source_evidence_sha256': {name: sha(p/name) for name in (
                  'episode_manifest.json', 'pre_action.jsonl', 'post_step_events.jsonl', 'paired_reset_audit.json')},
              'source_manifest_sha256': sha(p.parent.parent/'source_manifest.json')}
    return record, pre, audit


def audit_plan(plan, contract, indices=None):
    if plan['task_contract_sha256'] != LOCK_HASH or plan['stage'] != 'DT2':
        raise ValueError('wrong DT2 plan binding')
    slots = plan['slots'] if indices is None else [plan['slots'][i] for i in indices]
    if indices is None:
        expected = {(g, cfg, color) for g in ('dt1_dev_000', 'dt1_dev_001', 'dt2_train_002', 'dt2_train_003')
                    for cfg in ('A_red_B_blue', 'A_blue_B_red') for color in ('red', 'blue')}
        if len(slots) != 16 or {(s['geometry_group_id'],s['color_configuration'],s['target_color']) for s in slots} != expected:
            raise ValueError('tiny coverage incomplete or duplicated')
    results, raw, resets = [], [], {}
    for slot in slots:
        if slot['split'] != 'train':
            raise ValueError('normalizer/data may only consume tiny train')
        result, pre, reset = audit_episode(slot, contract)
        results.append(result)
        raw.append(pre)
        key = slot['geometry_group_id'], slot['color_configuration']
        resets.setdefault(key, []).append(reset)
    for pair in resets.values():
        if len(pair) != 2:
            raise ValueError('missing instruction pair')
        validate_paired_resets(*pair)
    return results, raw
