"""Fixed 5000-checkpoint, 72-task paired train/validation queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from src.dual_target.tiny_plan import SOURCE_ROOT,DATA_ROOT,verify_binding
from scripts.zoh_compare_runtime import POLICY
from scripts.zoh_compare_common import ROOT,RUNS,SEED,write_json,verify_training,tasks_for,publish
from scripts.zoh_compare_video import export_video
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.zoh_eval_review_v2 import audit_policy_episode
from src.dual_target.gpu_wait import Dt1GpuWaiter
from src.dual_target.coexistence_watch import sample_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-seed',type=int,default=20260906)
    args=p.parse_args()
    verify_binding(SOURCE_ROOT)
    root=args.output.resolve()
    if root != ROOT/'evaluation_v2' or args.policy_seed != SEED:
        raise ValueError('comparison output and paired seed are pinned')
    root.mkdir(parents=True,exist_ok=False)
    approved_dataset(DATA_ROOT/'outputs/dual_target_v2/zoh_dataset_0906_v2/dataset',SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json')

    checkpoints={name:verify_training(name) for name in RUNS}
    tasks=tasks_for(root,checkpoints)
    (root/'eval_plan.json').write_text(json.dumps({'stage':'DT3','training_roots':{k:str(v) for k,v in RUNS.items()},
        'source_sha256':{str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in [*(SOURCE_ROOT/'src/dual_target').glob('*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_compare*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_full*.py')]},
        'test_used':False,'training':False,'evaluation_split':'train+validation','scope':'three-5000-checkpoint-comparison','resource_probe_required':False,
        'promotion_condition':'user-authorized next-stage paired validation; ordinary task failures count and continue','selection':'fixed checkpoint 5000; same full train and validation matrix for all three 5000 checkpoints',
        'tasks':tasks,'pid':os.getpid(),'policy':POLICY.policy_id},indent=2)+'\n')
    child=None
    results=[]
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def status(value,**extra):
        write_json(root/'queue_status.json',{'stage':'DT3','status':value,
            'pid':os.getpid(),'completed':results,**extra})
    try:
        for task in tasks:
            status('WAITING_GPU',current_slot=task['slot'],current_condition=task['condition'],checkpoint_step=task['step'])
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=task['run_id'],argv=[],
                policy=POLICY,required_free_mib=9216)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':
                raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','scripts.zoh_compare_runtime',
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
                status('RUNNING',current_slot=task['slot'],current_condition=task['condition'],checkpoint_step=task['step'],child_pgid=child.pid)
                start=time.monotonic()
                with (root/f'{task["run_id"]}_gpu.jsonl').open('x') as samples:
                    while child.poll() is None:
                        sample=sample_group(child.pid)
                        samples.write(json.dumps(sample)+'\n')
                        samples.flush()
                        if sample['free_mib']<2048:
                            raise RuntimeError('runtime free GPU memory below 2048 MiB; low admission not approved')
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
            review['scope']=task['evaluation_split']+' comparison; train results are seen-layout only; no optimization'
            (ep/'policy_trace_review.json').write_text(json.dumps(review,indent=2)+'\n')
            video=export_video(ep,task,evidence['episode_result'],ROOT/'delivery/videos')
            results.append({**task,**evidence['episode_result'],'review':review,**video})
            publish(results)
            child=None
        if len(results)!=72:raise ValueError('incomplete comparison matrix')
        publish(results,complete=True)
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

