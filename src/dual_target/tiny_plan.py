"""DT2-only fixed train geometries, provenance and immutable collection plan."""
import argparse
import hashlib
import json
from pathlib import Path

from .layouts import GeometryGroup, TargetSlot, Vec2, build_four_tasks
from .scene import DualTargetSceneSpec, dt1_development_groups
from .reset_audit import derive_pair_seed

SOURCE_ROOT = Path('/home/wxh/go2_short_vln')
DATA_ROOT = Path('/mnt/wxh/go2_short_vln')
DT1_MATRIX = 'dt1_m_0906_v2'
LOCK_HASH = 'ae040c5b66c7405197874d9d24408f03736606704c5d3cd278ad5827af5e0e14'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_binding(source_root):
    approval = read(source_root/'reports/dual_target_v1/dt1_approval.json')
    assert approval['stage'] == 'DT1' and approval['status'] == 'APPROVED'
    for name in ('gate', 'task_contract', 'report'):
        assert sha(source_root/approval[name]) == approval[name+'_sha256']
    assert approval['task_contract_sha256'] == LOCK_HASH
    contract = read(source_root/approval['task_contract'])
    compatibility = read(source_root/'config/dual_target_v1/dt2_source_compatibility.json')
    # Only a two-line, explicit scene/task parameter extension differs from
    # the DT1 runtime. All environment/expert/scoring/other locked code stays exact.
    assert compatibility['dt1_runner_sha256'] == contract['source_sha256']['src/dual_target/runner.py']
    for name, expected in contract['source_sha256'].items():
        if name == 'src/dual_target/runner.py':
            expected = compatibility['dt2_runner_sha256']
        assert sha(source_root/name) == expected, f'locked source drift: {name}'
    return contract


def groups():
    fixed = (*dt1_development_groups(),
        GeometryGroup('dt2_train_002', 'dt2_train_002',
                      TargetSlot('A', Vec2(2.55, 1.35), Vec2(-1., 0.)),
                      TargetSlot('B', Vec2(2.25, -1.05), Vec2(-1., 0.)), Vec2(0., 0.)),
        GeometryGroup('dt2_train_003', 'dt2_train_003',
                      TargetSlot('A', Vec2(2.10, 1.00), Vec2(-1., 0.)),
                      TargetSlot('B', Vec2(2.65, -1.60), Vec2(-1., 0.)), Vec2(0., 0.)))
    for group in fixed:
        group.validate_geometry()
    if len({group.geometry_signature for group in fixed}) != 4:
        raise ValueError('tiny requires four distinct geometry signatures')
    return fixed


def task_at(index):
    tasks = [(DualTargetSceneSpec(g, task.color_configuration), task)
             for g in groups() for task in build_four_tasks(g)]
    if type(index) is not int or not 0 <= index < 16:
        raise ValueError('tiny task index must be in 0..15')
    return tasks[index]


def build_plan(batch_root, source_root=SOURCE_ROOT, data_root=DATA_ROOT):
    verify_binding(source_root)
    slots = []
    for index in range(16):
        scene, task = task_at(index)
        group, config, color = task.geometry_group_id, task.color_configuration, task.target_color
        episode_id = f'{group}_{config}_{color}_r00'
        reused = index < 8
        run_id = (f'{DT1_MATRIX}__{group}__{config}__r0__{color}' if reused
                  else f'{batch_root.name}_s{index:02d}')
        parent = data_root/'outputs/dual_target_v1' if reused else batch_root/'runs'
        slots.append({'slot_index': index, 'geometry_group_id': group, 'split': 'train',
                      'lineage_root_id': scene.group.lineage_root_id,
                      'color_configuration': config, 'target_color': color,
                      'instruction': task.instruction, 'target_slot': task.target_slot,
                      'repeat': 0, 'pair_seed': derive_pair_seed(geometry_group_id=group,
                          color_configuration=config, repeat=0),
                      'run_id': run_id, 'episode_id': episode_id,
                      'episode_path': str(parent/run_id/'episodes'/episode_id),
                      'reuse_approved_dt1': reused,
                      'geometry_signature': list(scene.group.geometry_signature),
                      'scene': scene.audit_metadata()})
    return {'format': 'go2-dual-target-dt2-tiny-plan-v1', 'stage': 'DT2',
            'status': 'PLANNED_NOT_APPROVED', 'run_id': batch_root.name,
            'task_contract_sha256': LOCK_HASH,
            'dt1_approval_sha256': sha(source_root/'reports/dual_target_v1/dt1_approval.json'),
            'task_count': 16, 'geometry_group_count': 4, 'slots': slots,
            'reuse_policy': 'DT1 fixed repeat=0, all four combinations; no best-of selection',
            'new_attempt_policy': 'one predeclared attempt per new slot; stop on failure',
            'training': False, 'dt3_started': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch-root', type=Path, required=True)
    args = p.parse_args()
    plan = build_plan(args.batch_root.resolve())
    args.batch_root.mkdir(parents=True, exist_ok=False)
    (args.batch_root/'tiny_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'tasks': 16, 'reuse': 8, 'new': 8, 'root': str(args.batch_root)}))
