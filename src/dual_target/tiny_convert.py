"""DT2 lossless LeRobot conversion, real-loader audit and train-only statistics."""
import argparse
import json
from pathlib import Path

from .contracts import IMAGE_KEY, STATE_KEY
from .tiny_plan import read, sha, SOURCE_ROOT, verify_binding
from .tiny_data import audit_plan, normalized_stats, chunk_indices


def verify_loader(dataset_root, raw, records):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('local/dual_target_v1_tiny', root=dataset_root,
                            delta_timestamps={'action': [i/50. for i in range(50)]})
    expected_features = {IMAGE_KEY, STATE_KEY, 'action', 'timestamp', 'frame_index',
                         'episode_index', 'index', 'task_index'}
    if set(dataset.features) != expected_features:
        raise ValueError(f'unexpected dataset fields: {set(dataset.features)}')
    if len(dataset) != sum(len(rows) for rows in raw):
        raise ValueError('loader length mismatch')
    checked, offset = [], 0
    for episode, (rows, record) in enumerate(zip(raw, records)):
        n = len(rows)
        anchors = sorted({0, 1, n//4, n//2, 3*n//4, n-51, n-50, n-2, n-1})
        for anchor in anchors:
            item = dataset[offset+anchor]
            if item['episode_index'].item() != episode or item['frame_index'].item() != anchor:
                raise ValueError('loader episode boundary or index mismatch')
            if item['task'] != rows[anchor]['task']:
                raise ValueError('loader instruction mismatch')
            if not np.allclose(item[STATE_KEY].numpy(), rows[anchor]['body_velocity_body'], atol=1e-7, rtol=1e-6):
                raise ValueError('loader state differs from pre-action source')
            future, padded = chunk_indices(anchor, n)
            expected = np.asarray([rows[i]['applied_action'] for i in future], dtype=np.float32)
            if item['action'].shape != (50, 3) or not np.array_equal(item['action'].numpy(), expected):
                raise ValueError('loader action chunk is shifted or crosses episode boundary')
            if item['action_is_pad'].tolist() != padded:
                raise ValueError('loader padding mask differs from valid episode interval')
            path = Path(record['episode_path'])/rows[anchor]['rgb_path']
            rgb = np.asarray(Image.open(path).convert('RGB'))
            decoded = np.rint(item[IMAGE_KEY].permute(1,2,0).numpy()*255).astype(np.uint8)
            if decoded.shape != (512,512,3) or not np.array_equal(decoded, rgb):
                raise ValueError('lossless loader RGB differs from true pre-action observation')
            checked.append({'episode_index': episode, 'frame_index': anchor,
                            'padded_targets': sum(padded), 'rgb_exact': True})
        offset += n
    return {'real_loader_passed': True, 'checked_anchors': checked, 'total_frames': len(dataset),
            'allowed_features': sorted(expected_features), 'cross_episode_chunks': False,
            'image_transform': 'lossless RGB uint8 HWC -> float32 CHW /255, no resize/crop/augmentation'}


def convert(plan_path, output, *, probe=False):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.datasets.lerobot_dataset as installed_dataset
    contract = verify_binding(SOURCE_ROOT)
    plan = read(plan_path)
    records, raw = audit_plan(plan, contract, indices=(0,1) if probe else None)
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = output/'lerobot'
    features = {
        IMAGE_KEY: {'dtype': 'image', 'shape': (512,512,3), 'names': ['height','width','channels']},
        STATE_KEY: {'dtype': 'float32', 'shape': (3,), 'names': ['body_vx','body_vy','body_yaw_rate']},
        'action': {'dtype': 'float32', 'shape': (3,), 'names': ['vx','vy','wz']},
    }
    dataset = LeRobotDataset.create(repo_id='local/dual_target_v1_tiny', root=dataset_root,
        fps=50, features=features, robot_type='unitree_go2_high_level_velocity',
        use_videos=False, image_writer_processes=0, image_writer_threads=0)
    try:
        for record, rows in zip(records, raw):
            for row in rows:
                rgb = np.asarray(Image.open(Path(record['episode_path'])/row['rgb_path']).convert('RGB'))
                if rgb.shape != (512,512,3):
                    raise ValueError('source front camera image shape mismatch')
                dataset.add_frame({IMAGE_KEY: rgb, STATE_KEY: np.asarray(row['body_velocity_body'], dtype=np.float32),
                    'action': np.asarray(row['applied_action'], dtype=np.float32), 'task': row['task']})
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    state_rows = [r['body_velocity_body'] for ep in raw for r in ep]
    action_rows = [r['applied_action'] for ep in raw for r in ep]
    if any(r[1] != 0 for r in action_rows):
        raise ValueError('DT2 constant vy channel changed')
    normalizer = {'format': 'dual-target-v1-tiny-train-normalizer-v1',
        'source_split': 'train', 'plan_sha256': sha(plan_path),
        'state': normalized_stats(state_rows), 'action': normalized_stats(action_rows),
        'visual_normalization': 'IDENTITY', 'action_loss_active_channels': [True, False, True],
        'constant_std_policy': 'raw std<1e-6 -> safe std1; preserve raw statistics separately',
        'note': 'DT3 must use this safe normalizer, not legacy or unadjusted meta/stats.json'}
    weights = np.concatenate([np.full(len(ep), 1.0/(len(raw)*len(ep)), dtype=np.float64) for ep in raw])
    np.save(output/'episode_balanced_sampler_weights.npy', weights)
    exposure = {key: sum(r['exposure'][key]/len(ep) for r,ep in zip(records,raw))/len(raw)
                for key in ('terminal_frame_anchors','near_zero_frame_anchors',
                            'all_valid_chunk_near_zero_anchors','any_valid_chunk_near_zero_anchors','padded_anchors')}
    sampler = {'policy': 'episode_uniform_then_frame_anchor_uniform', 'replacement': True,
               'no_terminal_cap': True, 'no_frame_removal_or_duplication': True,
               'probability_sum': float(weights.sum()), 'episode_balanced_anchor_fractions': exposure,
               'padding_policy': 'repeat final action only inside query; action_is_pad masks every padded target',
               'action_loss_mask': 'not action_is_pad, intersect active channels [vx,wz]; fixed vy ignored and deployed as zero'}
    loader_review = verify_loader(dataset_root, raw, records)
    results = {'normalizer.json': normalizer, 'sampler.json': sampler,
               'raw_audit.json': records, 'loader_review.json': loader_review}
    for name, value in results.items():
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    manifest = {'stage': 'DT2', 'status': 'PROBE_ONLY_NOT_DT2_DATASET' if probe else 'CONVERTED_NOT_DT2_APPROVED',
        'episode_count': len(raw), 'frames': len(state_rows), 'split': 'train',
        'plan_path': str(plan_path), 'plan_sha256': sha(plan_path),
        'task_contract_sha256': plan['task_contract_sha256'],
        'lerobot_source_path': installed_dataset.__file__, 'lerobot_source_sha256': sha(installed_dataset.__file__),
        'feature_allowlist': list(features)+['task'], 'source_episode_paths': [r['episode_path'] for r in records],
        'files_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'training': False, 'dt3_started': False}
    (output/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': manifest['status'], 'episodes': len(raw), 'frames': len(state_rows),
                      'loader_checked_anchors': len(loader_review['checked_anchors']), 'output': str(output)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tiny-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-existing', action='store_true')
    args = p.parse_args()
    convert(args.tiny_plan.resolve(), args.output.resolve(), probe=args.probe_existing)
