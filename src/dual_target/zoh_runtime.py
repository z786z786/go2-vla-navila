"""One bounded v2 probe/expert episode; no dataset approval or training."""
import json
from pathlib import Path
import sys

from .tiny_plan import SOURCE_ROOT, verify_binding, task_at, sha
from .zoh_control import ZohSpec
from .gpu_wait import acquire_live_admission, GpuAdmissionPolicy

POLICY = GpuAdmissionPolicy('zoh_v2_shared_8g_probe', 8192, allow_existing_compute=True)


def main():
    from . import runner as r
    from .zoh_loop import run_live_zoh
    from omni.isaac.lab.app import AppLauncher
    p = r._live_parser()
    p.add_argument('--zoh-mode', choices=['probe','expert'], required=True)
    p.add_argument('--slot', type=int, required=True)
    p.add_argument('--zoh-manifest', type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    contract = verify_binding(SOURCE_ROOT)
    declared = json.loads(args.zoh_manifest.read_text())
    spec = ZohSpec()
    if declared['zoh_spec'] != json.loads(json.dumps(spec.metadata())):
        raise ValueError('ZOH candidate changed')
    for name, digest in declared['source_sha256'].items():
        if sha(SOURCE_ROOT/name) != digest:
            raise ValueError('probe source drift: '+name)
    scene, task = task_at(args.slot)
    from .reset_audit import derive_pair_seed
    args.group_id, args.color_configuration, args.target_color = task.geometry_group_id, task.color_configuration, task.target_color
    args.seed = derive_pair_seed(geometry_group_id=task.geometry_group_id, color_configuration=task.color_configuration, repeat=0)
    args.repeat, args.paired_reset = 0, True
    args.max_env_steps = 700 if args.zoh_mode == 'probe' else contract['max_episode_env_steps']
    if sha(args.motion_calibration) != contract['evidence_sha256']['motion_calibration']:
        raise ValueError('frozen calibration changed')
    root = args.output_dir/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    (root/'source_manifest.json').write_text(json.dumps(dict(declared, actual_argv=sys.argv),indent=2)+'\n')
    def status(value, **extra):
        (root/'stage_status.json').write_text(json.dumps(dict(stage='ZOH_V2_GATE',status=value,
            training=False, dataset_approved=False, **extra),indent=2)+'\n')
    app = lease = None
    try:
        status('RUNNING')
        lease = acquire_live_admission(args.gpu_project_root, run_id=args.run_id, actual_argv=sys.argv, policy=POLICY)
        asset_root = r._asset_root_from_go2_usd(args.go2_usd)
        r._configure_dt1_kit_args_before_app(asset_root)
        app = AppLauncher(args)
        r._configure_dt1_runtime_after_app(asset_root)
        result = run_live_zoh(args, app.app, scene_task=(scene,task), mode=args.zoh_mode, spec=spec)
        status('EPISODE_COMPLETE_NOT_APPROVED', episode_result=result)
    except BaseException as exc:
        evidence = r.write_dt1_failure_traceback(root, exc=exc)
        status('FAILED', detail=str(exc), **evidence)
        raise
    finally:
        try:
            if app is not None: app.app.close()
        finally:
            if lease is not None: lease.close()


if __name__ == '__main__': main()
