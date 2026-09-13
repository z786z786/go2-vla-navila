#!/usr/bin/env python3
"""Build the frozen P1-DEV100 manifest offline, using only the standard library.

Run with zero arguments. Existing identical output is left untouched; any
different existing manifest causes an error, never an overwrite.
"""

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz')
INPUT_SHA256 = 'ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec'
SEED = 20260913
# Freeze date, deliberately a fixed ISO date rather than the rerun clock.
FROZEN_AT = '2026-09-13'
EXPECTED = {
    'zsNo4HB9uLZ': (270, 90), 'QUCTc6BB5sX': (240, 80),
    '2azQ1b91cZZ': (204, 68), 'TbHJrupSAjP': (108, 36),
    'X7HyMhZNoso': (69, 23), 'Z6MFQCViBuw': (66, 22),
    'EU6Fwq7SyZv': (54, 18), 'x8F5xyUWy9e': (30, 10),
    'oLBMNvg9in8': (24, 8), '8194nk5LbLH': (9, 3),
    'pLe4wQe7qrG': (3, 1),
}
SELECTION_RULE = {
    'allocation': 'Reserve 1 per scene. For the remaining 89 seats use quotas '
                  '89 * original_scene_route_count / 359. Add floor quotas, '
                  'then award residual seats by descending fractional remainder; '
                  'ties use ascending scene name; skip scenes at route capacity.',
    'route_selection': 'Within each scene rank trajectories by ascending SHA-256 '
                       'hex digest of UTF-8 f"{seed}:{scene}:{trajectory_id}" '
                       '(integer IDs in decimal, no whitespace/newline). '
                       'Break hash ties by numeric trajectory_id; take allocated count.',
    'episode_selection': 'Choose the numerically smallest original episode_id '
                         'among the three episodes of each selected trajectory.',
    'output_order': 'Ascending (scene name, numeric episode_id); episode_ids and '
                    'trajectory_ids are parallel arrays of original integer IDs.',
    'physical_route': 'Original trajectory_id; verify a bijection with '
                      '(scene_id, start_position, goals, gt_locations).',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scene(episode):
    return episode['scene_id'].split('/')[1]


def build():
    raw = INPUT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUT_SHA256, 'Input SHA-256 mismatch')
    episodes = json.loads(gzip.decompress(raw))['episodes']
    require(len(episodes) == 1077, 'Expected 1077 episodes')
    require(all(type(e['episode_id']) is int and type(e['trajectory_id']) is int
                for e in episodes), 'Expected integer original IDs')
    require(len({e['episode_id'] for e in episodes}) == 1077, 'Duplicate episode IDs')
    routes = defaultdict(list)
    for e in episodes:
        routes[e['trajectory_id']].append(e)
        require(e['goals'][0]['radius'] == 3.0, 'Unexpected goal radius')
    require(len(routes) == 359, 'Expected 359 trajectories')
    physical = set()
    by_scene = defaultdict(list)
    for tid, members in routes.items():
        require(len(members) == 3, 'Expected exactly three episodes per trajectory')
        signatures = {json.dumps([e[k] for k in
                      ('scene_id', 'start_position', 'goals', 'gt_locations')],
                      sort_keys=True) for e in members}
        require(len(signatures) == 1, 'Inconsistent physical route')
        physical.update(signatures)
        by_scene[scene(members[0])].append(tid)
    require(len(physical) == 359, 'Different IDs share a physical route')
    counts = Counter(map(scene, episodes))
    require({s: (counts[s], len(ts)) for s, ts in by_scene.items()} == EXPECTED,
            'Counts differ from dispatch section 3')

    # Scene metadata only: no training episode, instruction, or route is read.
    source = ROOT / 'reports/p0/all61_audit.json'
    audit_bytes = source.read_bytes()
    train_scenes = json.loads(audit_bytes)['findings']['scene_coverage'][
        'expected_scenes_from_plan_plus_unavailable']
    require(len(train_scenes) == len(set(train_scenes)) == 61, 'Expected 61 training scenes')
    require(not set(by_scene).intersection(train_scenes), 'Evaluation/training overlap')

    sizes = {s: len(ts) for s, ts in by_scene.items()}
    remaining = 100 - len(sizes)
    total = sum(sizes.values())
    quotients = {s: divmod(remaining * n, total) for s, n in sizes.items()}
    allocated = {s: min(n, 1 + quotients[s][0]) for s, n in sizes.items()}
    residual = 100 - sum(allocated.values())
    for s in sorted(sizes, key=lambda s: (-quotients[s][1], s)):
        if residual and allocated[s] < sizes[s]:
            allocated[s] += 1
            residual -= 1
    require(residual == 0, 'Could not allocate exactly 100')
    require(all(1 <= allocated[s] <= sizes[s] for s in sizes), 'Allocation bounds')
    selected = []
    for s in sorted(sizes):
        def rank(tid):
            return hashlib.sha256(f'{SEED}:{s}:{tid}'.encode('utf-8')).hexdigest(), tid
        for tid in sorted(by_scene[s], key=rank)[:allocated[s]]:
            selected.append(min(routes[tid], key=lambda e: e['episode_id']))
    selected.sort(key=lambda e: (scene(e), e['episode_id']))
    require(len(selected) == len({e['trajectory_id'] for e in selected}) == 100,
            'Expected 100 unique selected trajectories')
    require(Counter(map(scene, selected)) == allocated, 'Selected count mismatch')
    return {
        'task_id': 'P1-DEV100', 'frozen_at': FROZEN_AT, 'seed': SEED,
        'input_path': str(INPUT), 'input_sha256': INPUT_SHA256,
        'selection_rule': SELECTION_RULE,
        'episode_ids': [e['episode_id'] for e in selected],
        'trajectory_ids': [e['trajectory_id'] for e in selected],
        'per_scene_allocation': [
            {'scene': s, 'available_episodes': counts[s], 'available_routes': sizes[s],
             'allocated': allocated[s], 'base': 1, 'extra_floor': quotients[s][0],
             'remainder_numerator': quotients[s][1], 'remainder_denominator': total}
            for s in sorted(sizes)],
        'training_scene_check': {
            'source': str(source.relative_to(ROOT)),
            'source_sha256': hashlib.sha256(audit_bytes).hexdigest(),
            'json_key': 'findings.scene_coverage.expected_scenes_from_plan_plus_unavailable',
            'training_scene_ids': sorted(train_scenes), 'intersection': [],
        },
    }


def main():
    manifest = build()
    output = ROOT / 'configs/benchmark_dev100.json'
    content = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if output.exists():
        require(output.read_bytes() == content,
                'Frozen manifest differs; refusing to overwrite. Investigate without refreezing.')
        print('Existing frozen manifest is byte-identical; no write performed.')
    else:
        with output.open('xb') as f:
            f.write(content)
        print(f'Created {output}')
    print(f'Manifest SHA-256: {hashlib.sha256(content).hexdigest()}')


if __name__ == '__main__':
    main()
