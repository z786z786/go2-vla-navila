"""One B-10k validation rollout; independent test slots are forbidden."""
import json
from pathlib import Path
import sys

from src.dual_target.tiny_plan import SOURCE_ROOT,verify_binding,sha
from src.dual_target.zoh_dataset import grouped_tasks
from src.dual_target.gpu_wait import acquire_live_admission,GpuAdmissionPolicy

POLICY=GpuAdmissionPolicy('zoh_b10k_validation_shared7g_floor1_v1',7168,allow_existing_compute=True)
SCOPE='b-10000-validation-rollout-floor1-exception'


def main():
    from src.dual_target import runner as r
    from omni.isaac.lab.app import AppLauncher
    from src.dual_target.zoh_result_loop import run_live_zoh
    from scripts.zoh_compare_policy_rpc import PolicyClient
    p=r._live_parser();p.add_argument('--slot',type=int,required=True)
    p.add_argument('--rollout-plan',type=Path,required=True);p.add_argument('--policy-seed',type=int,required=True)
    p.add_argument('--policy-checkpoint',type=Path,required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    contract=verify_binding(SOURCE_ROOT)
    if not 16<=args.slot<24:raise ValueError('B-10k validation permits only slots 16..23')
    split,scene,task=grouped_tasks()[args.slot]
    if split!='validation':raise ValueError('non-validation task reached B-10k validation runtime')
    plan=json.loads(args.rollout_plan.read_text())
    if (plan.get('scope')!=SCOPE or plan.get('policy')!=POLICY.policy_id
            or plan.get('validation_used') is not True or plan.get('test_used') is not False):
        raise ValueError('explicit B-10k validation-only plan required')
    for name,digest in plan['source_sha256'].items():
        if sha(SOURCE_ROOT/name)!=digest:raise ValueError('rollout source drift: '+name)
    match=[x for x in plan['tasks'] if x['run_id']==args.run_id]
    if (len(match)!=1 or match[0]['slot']!=args.slot or match[0]['evaluation_split']!='validation'
            or match[0]['policy_seed']!=args.policy_seed or match[0]['checkpoint']!=str(args.policy_checkpoint)):
        raise ValueError('runtime task differs from frozen B-10k validation plan')
    if sha(args.policy_checkpoint/'checkpoint_manifest.json')!=match[0]['checkpoint_manifest_sha256']:
        raise ValueError('B-10k checkpoint binding drift')
    from src.dual_target.reset_audit import derive_pair_seed
    args.group_id=task.geometry_group_id;args.color_configuration=task.color_configuration
    args.target_color=task.target_color;args.seed=derive_pair_seed(geometry_group_id=task.geometry_group_id,
        color_configuration=task.color_configuration,repeat=0);args.repeat=0;args.paired_reset=True
    args.max_env_steps=contract['max_episode_env_steps']
    if r.sha256(args.motion_calibration)!=contract['evidence_sha256']['motion_calibration']:
        raise ValueError('motion calibration differs from locked task')
    root=args.output_dir/args.run_id;root.mkdir(parents=True,exist_ok=False)
    manifest=r._current_source_manifest(SOURCE_ROOT,run_id=args.run_id,actual_argv=sys.argv)
    manifest.update(stage='DT3',slot=args.slot,policy_seed=args.policy_seed,
        policy_checkpoint=str(args.policy_checkpoint),comparison_scope=SCOPE,evaluation_split='validation')
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    def status(value,**extra):
        (root/'stage_status.json').write_text(json.dumps({'stage':'DT3','status':value,'training':False,
            'evaluation_split':'validation','validation_used':True,'test_used':False,'scope':SCOPE,
            'navigation_success_approved':False,**extra},indent=2)+'\n')
    app=lease=client=None
    try:
        status('RUNNING');lease=acquire_live_admission(args.gpu_project_root,run_id=args.run_id,
            actual_argv=sys.argv,policy=POLICY)
        asset_root=r._asset_root_from_go2_usd(args.go2_usd);r._configure_dt1_kit_args_before_app(asset_root)
        app=AppLauncher(args);r._configure_dt1_runtime_after_app(asset_root)
        client=PolicyClient(args.policy_checkpoint,args.policy_seed,root)
        result=run_live_zoh(args,app.app,scene_task=(scene,task),policy_client=client,mode='policy')
        status('EPISODE_COMPLETE_NOT_DT3_APPROVED',episode_result=result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        evidence=r.write_dt1_failure_traceback(root,exc=exc);status('FAILED',detail=str(exc),**evidence);raise
    finally:
        if client is not None:client.close()
        try:
            if app is not None:app.app.close()
        finally:
            if lease is not None:lease.close()


if __name__=='__main__':main()
