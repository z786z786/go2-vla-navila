"""Durable B/C checkpoint queue: all 16 train tasks, then stop before eval."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from src.dual_target.tiny_plan import SOURCE_ROOT,DATA_ROOT,verify_binding
from scripts.zoh_compare_runtime import POLICY
from scripts.zoh_compare_video import export_video
from scripts.zoh_train_bc_common import ROOT,SEED,sha,write_json,checkpoints,tasks_for,publish,publish_training_evidence
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.zoh_eval_review_v2 import audit_policy_episode
from src.dual_target.gpu_wait import Dt1GpuWaiter
from src.dual_target.coexistence_watch import sample_group


def main():
    root=ROOT/'evaluation';root.mkdir(parents=True,exist_ok=False)
    verify_binding(SOURCE_ROOT)
    approved_dataset(DATA_ROOT/'outputs/dual_target_v2/zoh_dataset_0906_v2/dataset',
        SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json')
    values=checkpoints();tasks=tasks_for(root,values)
    if len(tasks)!=32 or any(t['slot']>=16 or t['evaluation_split']!='train' for t in tasks):
        raise ValueError('train-only B/C matrix contract failed')
    sources=[*(SOURCE_ROOT/'src/dual_target').glob('*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_compare*.py'),
        *(SOURCE_ROOT/'scripts').glob('zoh_full*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_train_bc*.py')]
    write_json(root/'eval_plan.json',{'stage':'DT3','training_roots':values,
        'source_sha256':{str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in sources},
        'test_used':False,'training':False,'evaluation_split':'train-only',
        'scope':'two-5000-checkpoint-train-comparison','resource_probe_required':False,
        'selection':'fixed checkpoint 5000; B/C only; all 16 train tasks; stop before eval',
        'tasks':tasks,'pid':os.getpid(),'policy':POLICY.policy_id})
    publish_training_evidence();publish([])
    child=None;results=[]
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(value,**extra):
        row={'stage':'DT3','status':value,'pid':os.getpid(),'completed':results,**extra}
        write_json(root/'queue_status.json',row)
        write_json(ROOT/'delivery/progress.json',{'status':value,'pid':os.getpid(),
            'completed':len(results),'planned':32,**{k:v for k,v in extra.items() if k!='detail'},
            **({'detail':extra['detail']} if 'detail' in extra else {})})
    try:
        for task in tasks:
            status('WAITING_GPU',current_slot=task['slot'],current_condition=task['condition'])
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=task['run_id'],argv=[],policy=POLICY,required_free_mib=9216)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','scripts.zoh_compare_runtime',
                '--live','--headless','--enable_cameras','--slot',str(task['slot']),
                '--policy-seed',str(SEED),'--policy-checkpoint',task['checkpoint'],
                '--eval-plan',str(root/'eval_plan.json'),'--run-id',task['run_id'],'--output-dir',str(root/'runs'),
                '--go2-usd',str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration',str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            with (root/f'{task["run_id"]}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current_slot=task['slot'],current_condition=task['condition'],child_pgid=child.pid)
                started=time.monotonic()
                with (root/f'{task["run_id"]}_gpu.jsonl').open('x') as samples:
                    while child.poll() is None:
                        sample=sample_group(child.pid);samples.write(json.dumps(sample)+'\n');samples.flush()
                        if sample['free_mib']<2048:raise RuntimeError('runtime free GPU memory below 2048 MiB')
                        if time.monotonic()-started>1800:raise RuntimeError('owned episode exceeded 1800 s')
                        time.sleep(2)
                if child.returncode!=0:raise RuntimeError(f'episode runtime failed: {child.returncode}')
            evidence=json.loads((root/'runs'/task['run_id']/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_DT3_APPROVED':raise RuntimeError('Isaac exit masked runtime error')
            ep=root/'runs'/task['run_id']/'episodes'/evidence['episode_result']['episode_id']
            review=audit_policy_episode(ep,evidence['episode_result'])
            review['scope']='train-layout comparison only; no optimization or generalization claim'
            write_json(ep/'policy_trace_review.json',review)
            video=export_video(ep,task,evidence['episode_result'],ROOT/'delivery/videos')
            results.append({**task,**evidence['episode_result'],'review':review,**video});publish(results);child=None
        if len(results)!=32:raise ValueError('incomplete B/C train matrix')
        publish(results,complete=True)
        write_json(ROOT/'delivery/artifacts_sha256.json',{str(p.relative_to(ROOT/'delivery')):sha(p)
            for p in (ROOT/'delivery').rglob('*') if p.is_file() and p.name not in ('progress.json','artifacts_sha256.json')})
        status('TRAIN_COMPLETE_AWAITING_USER_EVAL_DECISION',report=str(ROOT/'delivery/README.md'))
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_NEEDS_REVIEW',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


if __name__=='__main__':main()
