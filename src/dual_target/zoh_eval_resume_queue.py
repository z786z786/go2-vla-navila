"""Fixed DT3 16-task evaluation queue; no retries or automatic training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .tiny_plan import SOURCE_ROOT,DATA_ROOT,verify_binding
from .zoh_closed_loop_result import POLICY
from .tiny_train_core import file_sha
from .zoh_train_core import approved_dataset
from .zoh_eval_review_v2 import audit_policy_episode
from .zoh_result_contract import normalized_result
from .gpu_wait import Dt1GpuWaiter
from .coexistence_watch import sample_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-seed',type=int,default=20260906)
    args=p.parse_args()
    verify_binding(SOURCE_ROOT)
    prior=args.resume_from.resolve()
    old=json.loads((prior/'queue_status.json').read_text())
    if old['status']!='FAILED' or old['detail']!="'steps'" or old['completed']:
        raise ValueError('resume only the known post-timeout summary failure')
    if Path(f"/proc/{old['pid']}").exists():raise ValueError('previous queue still alive')
    prior_plan=json.loads((prior/'eval_plan.json').read_text())
    for name,digest in prior_plan['source_sha256'].items():
        if file_sha(SOURCE_ROOT/name)!=digest:raise ValueError('prior execution source drift')
    previous_task=prior_plan['tasks'][0]
    previous_run=prior/'runs'/previous_task['run_id']
    evidence=json.loads((previous_run/'stage_status.json').read_text())
    if evidence['status']!='EPISODE_COMPLETE_NOT_DT3_APPROVED':raise ValueError('previous runtime did not finish')
    previous_ep=previous_run/'episodes'/evidence['episode_result']['episode_id']
    reviewed=audit_policy_episode(previous_ep,evidence['episode_result'])
    recovered=normalized_result(evidence['episode_result'],reviewed['steps'],reviewed['steps'])
    samples=[json.loads(line) for line in (prior/(previous_task['run_id']+'_gpu.jsonl')).read_text().splitlines()]
    if min(x['free_mib'] for x in samples)<1024 or Path(f"/proc/{samples[-1]['owned_pgid']}").exists():
        raise ValueError('prior resource bound failed or child alive')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    approved_dataset(DATA_ROOT/'outputs/dual_target_v2/zoh_dataset_0906_v2/dataset',SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json')
    training=json.loads((args.training_root/'result.json').read_text())
    if training['status']!='ZOH_1000_COMPLETE_AWAITING_REVIEW':raise ValueError('training not complete')
    tasks=[]
    for step in (1000,750,500,250):
        checkpoint=args.training_root.resolve()/f'checkpoint_{step:06d}'
        manifest=checkpoint/'checkpoint_manifest.json'
        for name,digest in json.loads(manifest.read_text())['files_sha256'].items():
            if file_sha(checkpoint/name)!=digest:raise ValueError('checkpoint artifact drift')
        tasks.extend(dict(slot=i,step=step,policy_seed=args.policy_seed,
            run_id=f'{root.name}_c{step:04d}_s{i:02d}',checkpoint=str(checkpoint),
            checkpoint_manifest_sha256=file_sha(manifest)) for i in range(16,24))
    if len(prior_plan['tasks'])!=32 or any(any(a[k]!=b[k] for k in ('slot','step','policy_seed','checkpoint','checkpoint_manifest_sha256')) for a,b in zip(tasks,prior_plan['tasks'])):
        raise ValueError('resume changed validation matrix')
    tasks[0]=dict(previous_task,retained_run_root=str(previous_run))
    (root/'recovered_first_result.json').write_text(json.dumps(dict(task=tasks[0],result=recovered,review=reviewed,
        original_stage_sha256=file_sha(previous_run/'stage_status.json'),
        original_plan_sha256=file_sha(prior/'eval_plan.json'),
        evidence_sha256={name:file_sha(previous_ep/name) for name in ('pre_action.jsonl','post_step_events.jsonl','high_level_pre_action.jsonl')},
        minimum_sampled_free_mib=min(x['free_mib'] for x in samples),
        resource_continuation='user authorized continuation after complete runtime; task failure remains failure'),indent=2)+'\n')
    (root/'eval_plan.json').write_text(json.dumps({'stage':'DT3','training_root':str(args.training_root),
        'source_sha256':{str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in (SOURCE_ROOT/'src/dual_target').glob('*.py')},
        'test_used':False,'training':False,'resource_probe_required':False,'resume_from':str(prior),
        'promotion_condition':'user-authorized continuation; completed timeout retained, no navigation success required','selection':'validation closed-loop success, then wrong-stop/collision and arrival; no offline loss ranking',
        'tasks':tasks,'pid':os.getpid(),'policy':POLICY.policy_id},indent=2)+'\n')
    child=None
    results=[{**tasks[0],**recovered,'review':reviewed}]
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def status(value,**extra):
        (root/'queue_status.json').write_text(json.dumps({'stage':'DT3','status':value,
            'pid':os.getpid(),'completed':results,**extra},indent=2)+'\n')
    try:
        for task in tasks[1:]:
            status('WAITING_GPU',current_slot=task['slot'],checkpoint_step=task['step'])
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=task['run_id'],argv=[],
                policy=POLICY,required_free_mib=7168)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':
                raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','src.dual_target.zoh_closed_loop_result',
                '--live','--headless','--enable_cameras','--slot',str(task['slot']),
                '--policy-seed',str(args.policy_seed),'--policy-checkpoint',task['checkpoint'],
                '--eval-plan',str(root/'eval_plan.json'),
                '--run-id',task['run_id'],'--output-dir',str(root/'runs'),
                '--go2-usd',str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration',str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None)
            env.pop('PYTHONHOME',None)
            with (root/f'{task["run_id"]}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current_slot=task['slot'],checkpoint_step=task['step'],child_pgid=child.pid)
                start=time.monotonic()
                with (root/f'{task["run_id"]}_gpu.jsonl').open('x') as samples:
                    while child.poll() is None:
                        sample=sample_group(child.pid)
                        samples.write(json.dumps(sample)+'\n')
                        samples.flush()
                        if sample['free_mib']<1024:
                            raise RuntimeError('runtime free GPU memory below 1024 MiB; low admission not approved')
                        if time.monotonic()-start>1800:
                            raise RuntimeError('owned episode exceeded 1800 s time bound')
                        time.sleep(2)
                if child.returncode!=0:
                    raise RuntimeError(f'episode runtime failed: {child.returncode}')
            evidence=json.loads((root/'runs'/task['run_id']/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_DT3_APPROVED':
                raise RuntimeError('Isaac exit masked runtime error')
            ep=root/'runs'/task['run_id']/'episodes'/evidence['episode_result']['episode_id']
            review=audit_policy_episode(ep,evidence['episode_result'])
            (ep/'policy_trace_review.json').write_text(json.dumps(review,indent=2)+'\n')
            results.append({**task,**evidence['episode_result'],'review':review})
            child=None
        status('MATRIX_COMPLETE_NOT_DT3_APPROVED')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',detail=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)


if __name__=='__main__':
    main()
