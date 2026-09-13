"""One DT3 physical model episode, preserving raw commands and frozen scoring."""
import json
from pathlib import Path
import sys

from .tiny_plan import SOURCE_ROOT,verify_binding,sha
from .zoh_dataset import grouped_tasks
from .gpu_wait import acquire_live_admission,GpuAdmissionPolicy

POLICY=GpuAdmissionPolicy('zoh_validation_probe7_floor1_conditional_v1',7168,allow_existing_compute=True)


def main():
    from . import runner as r
    from omni.isaac.lab.app import AppLauncher
    from .zoh_scored_loop import run_live_zoh
    from .zoh_policy_rpc import PolicyClient
    p=r._live_parser()
    p.add_argument('--slot',type=int,required=True)
    p.add_argument('--eval-plan',type=Path,required=True)
    p.add_argument('--policy-seed',type=int,required=True)
    p.add_argument('--policy-checkpoint',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    contract=verify_binding(SOURCE_ROOT)
    if not 16 <= args.slot < 24:
        raise ValueError('only validation slots 16..23 are authorized')
    split,scene,task=grouped_tasks()[args.slot]
    assert split=='validation'
    plan=json.loads(args.eval_plan.read_text())
    if plan['policy']!=POLICY.policy_id:raise ValueError('wrong resource probe policy')
    for name,digest in plan['source_sha256'].items():
        if sha(SOURCE_ROOT/name)!=digest:raise ValueError('evaluation source drift: '+name)
    match=[t for t in plan['tasks'] if t['run_id']==args.run_id]
    if len(match)!=1 or match[0]['slot']!=args.slot or match[0]['policy_seed']!=args.policy_seed or match[0]['checkpoint']!=str(args.policy_checkpoint):
        raise ValueError('task differs from frozen validation plan')
    if sha(args.policy_checkpoint/'checkpoint_manifest.json')!=match[0]['checkpoint_manifest_sha256']:
        raise ValueError('checkpoint binding drift')
    from .reset_audit import derive_pair_seed
    args.group_id=task.geometry_group_id
    args.color_configuration=task.color_configuration
    args.target_color=task.target_color
    args.seed=derive_pair_seed(geometry_group_id=task.geometry_group_id,
        color_configuration=task.color_configuration,repeat=0)
    args.repeat=0
    args.paired_reset=True
    args.max_env_steps=contract['max_episode_env_steps']
    if r.sha256(args.motion_calibration)!=contract['evidence_sha256']['motion_calibration']:
        raise ValueError('calibration differs from locked task')
    root=args.output_dir/args.run_id
    root.mkdir(parents=True,exist_ok=False)
    manifest=r._current_source_manifest(SOURCE_ROOT,run_id=args.run_id,actual_argv=sys.argv)
    manifest.update(stage='DT3',slot=args.slot,policy_seed=args.policy_seed,
                    policy_checkpoint=str(args.policy_checkpoint))
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    def status(value,**extra):
        (root/'stage_status.json').write_text(json.dumps({'stage':'DT3','status':value,
            'training':False,'navigation_success_approved':False,**extra},indent=2)+'\n')
    app=lease=client=None
    try:
        status('RUNNING')
        lease=acquire_live_admission(args.gpu_project_root,run_id=args.run_id,
                                    actual_argv=sys.argv,policy=POLICY)
        asset_root=r._asset_root_from_go2_usd(args.go2_usd)
        r._configure_dt1_kit_args_before_app(asset_root)
        app=AppLauncher(args)
        r._configure_dt1_runtime_after_app(asset_root)
        client=PolicyClient(args.policy_checkpoint,args.policy_seed,root)
        result=run_live_zoh(args,app.app,scene_task=(scene,task),policy_client=client,mode="policy")
        status('EPISODE_COMPLETE_NOT_DT3_APPROVED',episode_result=result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        evidence=r.write_dt1_failure_traceback(root,exc=exc)
        status('FAILED',detail=str(exc),**evidence)
        raise
    finally:
        if client is not None:
            client.close()
        try:
            if app is not None:
                app.app.close()
        finally:
            if lease is not None:
                lease.close()


if __name__=='__main__':
    main()
