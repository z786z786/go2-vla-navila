"""DT2 entry: reuses DT1 physical expert loop with a predeclared scene/task."""
import json
import sys
from pathlib import Path

from .tiny_plan import SOURCE_ROOT, build_plan, read, sha, task_at, verify_binding
from .gpu_wait import acquire_live_admission, GpuAdmissionPolicy, EXCLUSIVE_DT1_POLICY

SHARED_DT2_POLICY = GpuAdmissionPolicy('dt2_expert_shared_8g_v2', 8192, allow_existing_compute=True)


def main():
    from . import runner as r
    from omni.isaac.lab.app import AppLauncher
    parser = r._live_parser()
    parser.add_argument('--tiny-plan', type=Path, required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--shared-dt2', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    contract = verify_binding(SOURCE_ROOT)
    plan = read(args.tiny_plan)
    if plan != build_plan(args.tiny_plan.resolve().parent):
        parser.error('tiny plan differs from predeclared canonical task list')
    if plan['stage'] != 'DT2' or not 8 <= args.task_index < 16:
        parser.error('DT2 collects only the eight predeclared new slots')
    slot = plan['slots'][args.task_index]
    scene, task = task_at(args.task_index)
    if (slot['reuse_approved_dt1'] or slot['geometry_group_id'] != task.geometry_group_id
            or slot['color_configuration'] != task.color_configuration
            or slot['target_color'] != task.target_color or args.shared_smoke
            or args.contact_precheck or args.motion_precheck or not args.live):
        parser.error('slot identity or live mode does not match fixed DT2 plan')
    args.group_id, args.color_configuration, args.target_color = (
        task.geometry_group_id, task.color_configuration, task.target_color)
    args.run_id, args.seed, args.repeat, args.paired_reset = slot['run_id'], slot['pair_seed'], 0, True
    args.output_dir = args.tiny_plan.resolve().parent/'runs'
    args.max_env_steps = contract['max_episode_env_steps']
    args.contact_threshold_n = contract['scene']['target_contact_threshold_n']
    if not args.go2_usd or args.motion_calibration is None:
        parser.error('verified local Go2 asset and frozen motion calibration required')
    if sha(args.motion_calibration) != contract['evidence_sha256']['motion_calibration']:
        parser.error('motion calibration hash changed')
    root = args.output_dir/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = r._current_source_manifest(SOURCE_ROOT, run_id=args.run_id, actual_argv=[
        sys.executable, '-m', 'src.dual_target.tiny_runtime', *sys.argv[1:]])
    manifest.update(stage='DT2', tiny_plan_sha256=sha(args.tiny_plan), resolved_slot=slot,
                    task_contract_sha256=plan['task_contract_sha256'])
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    def status(value, **extra):
        (root/'stage_status.json').write_text(json.dumps({
            'stage': 'DT2', 'status': value, 'run_id': args.run_id, 'actor': 'parent',
            'training': False, 'dt3_started': False, **extra}, indent=2, allow_nan=False))
    status('RUNNING')
    lease = app = None
    try:
        lease = acquire_live_admission(args.gpu_project_root, run_id=args.run_id,
            actual_argv=sys.argv, policy=SHARED_DT2_POLICY if args.shared_dt2 else EXCLUSIVE_DT1_POLICY)
        asset_root = r._asset_root_from_go2_usd(args.go2_usd)
        r._configure_dt1_kit_args_before_app(asset_root)
        app = AppLauncher(args)
        kit = r._configure_dt1_runtime_after_app(asset_root)
        (root/'kit_runtime.json').write_text(json.dumps(kit, indent=2))
        result = r.run_live_expert(args, app.app, scene_task=(scene, task))
        if result['status'] != 'success':
            status('FAILED', episode_result=result)
            raise RuntimeError('DT2 expert did not succeed; preserve attempt and stop batch')
        status('EPISODE_SUCCEEDED_NOT_DT2_APPROVED', episode_result=result)
    except BaseException as exc:
        evidence = r.write_dt1_failure_traceback(root, exc=exc)
        status('FAILED', detail=str(exc), **evidence)
        raise
    finally:
        try:
            if app is not None:
                app.app.close()
        finally:
            if lease is not None:
                lease.close()


if __name__ == '__main__':
    main()
