"""One predeclared fresh v2 dataset trajectory, never training."""
import json
from pathlib import Path
import sys

from .tiny_plan import SOURCE_ROOT, verify_binding, sha
from .zoh_dataset import make_plan, grouped_tasks
from .zoh_control import ZohSpec
from .gpu_wait import acquire_live_admission, GpuAdmissionPolicy

POLICY = GpuAdmissionPolicy('zoh_v2_dataset_shared_7g', 7168, allow_existing_compute=True)


def main():
    from . import runner as r
    from .zoh_dataset_loop import run_live_dataset
    from omni.isaac.lab.app import AppLauncher
    p = r._live_parser()
    p.add_argument('--zoh-mode', choices=['expert'], default='expert')
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
    if declared != make_plan(args.zoh_manifest.resolve().parent):
        raise ValueError('dataset plan differs from fixed 4/2/2 collection')
    if not 0 <= args.slot < 32:
        raise ValueError('slot outside dataset')
    slot = declared['slots'][args.slot]
    _, scene, task = grouped_tasks()[args.slot]
    args.run_id = slot['run_id']
    args.output_dir = Path(slot['episode_path']).parents[2]
    from .reset_audit import derive_pair_seed
    args.group_id, args.color_configuration, args.target_color = task.geometry_group_id, task.color_configuration, task.target_color
    args.seed = derive_pair_seed(geometry_group_id=task.geometry_group_id, color_configuration=task.color_configuration, repeat=0)
    args.repeat, args.paired_reset = 0, True
    args.max_env_steps = contract['max_episode_env_steps']
    if sha(args.motion_calibration) != contract['evidence_sha256']['motion_calibration']:
        raise ValueError('frozen calibration changed')
    root = args.output_dir/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    (root/'source_manifest.json').write_text(json.dumps(dict(declared, actual_argv=sys.argv),indent=2)+'\n')
    def status(value, **extra):
        (root/'stage_status.json').write_text(json.dumps(dict(stage='ZOH_V2_DATASET',status=value,
            training=False, dataset_approved=False, **extra),indent=2)+'\n')
    app = lease = None
    try:
        status('RUNNING')
        lease = acquire_live_admission(args.gpu_project_root, run_id=args.run_id, actual_argv=sys.argv, policy=POLICY)
        asset_root = r._asset_root_from_go2_usd(args.go2_usd)
        r._configure_dt1_kit_args_before_app(asset_root)
        app = AppLauncher(args)
        r._configure_dt1_runtime_after_app(asset_root)
        result = run_live_dataset(args, app.app, scene_task=(scene,task), mode=args.zoh_mode, spec=spec)
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
