"""Durable serial controller: existing scheduler run -> full run -> 72 rollouts -> report."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from scripts.zoh_compare_common import ROOT,BASE,SOURCE,RUNS,write_json,sha,verify_training,publish,publish_training_evidence

PY=str(BASE.parents[1]/'envs/conda/navila-isaac/bin/python')
SMOL=str(BASE.parents[1]/'envs/conda/smolvla/bin/python')

def alive(pid):
    try:
        os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'delivery').mkdir()
    env=dict(os.environ,PYTHONPATH=str(SOURCE),PYTHONUNBUFFERED='1')
    write_json(ROOT/'pipeline_plan.json',dict(pid=os.getpid(),conditions={k:str(v) for k,v in RUNS.items()},
        steps=5000,train_slots=list(range(16)),eval_slots=list(range(16,24)),test_used=False,
        policy_seed=20260906,episodes=72,prior_scheduler_pid=2113786,
        authorization='user: queue full fine-tuning after current scheduler training; then automatically compare three 5k checkpoints on train and eval',
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'scripts').glob('zoh_*.py')}))
    child=None;bridge=None
    def status(stage,**extra):
        value=dict(status=stage,pid=os.getpid(),wall_time_s=time.time(),**extra)
        write_json(ROOT/'pipeline_status.json',value);write_json(ROOT/'delivery/progress.json',value)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        publish([])
        status('WAITING_SCHEDULER_TRAINING',prior_pid=2113786)
        while True:
            state=json.loads((RUNS['B_scheduler_expert']/'supervisor_status.json').read_text())
            if state['status']=='ZOH_SCHEDULER_5000_COMPLETE_AWAITING_REVIEW' and not alive(2113786):break
            if state['status'] in ('FAILED','INTERRUPTED_RESUMABLE'):
                raise RuntimeError('scheduler run failed; preserve evidence and do not train C: '+str(state))
            if not alive(2113786):raise RuntimeError('scheduler supervisor exited without completion')
            time.sleep(20)
        verify_training('A_constant_expert');verify_training('B_scheduler_expert')
        command=[PY,'-m','scripts.zoh_full_train_launch','--dataset',str(BASE/'zoh_dataset_0906_v2/dataset'),
            '--approval',str(SOURCE/'reports/dual_target_v1/zoh_v2_dataset_approval.json'),
            '--output',str(RUNS['C_scheduler_full'])]
        with (ROOT/'full_supervisor.log').open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('FULL_TRAINING_QUEUED',child_pid=child.pid)
        while child.poll() is None:
            state_path=RUNS['C_scheduler_full']/'supervisor_status.json'
            state=json.loads(state_path.read_text()) if state_path.is_file() else {}
            status('FULL_TRAINING',child_pid=child.pid,supervisor=state)
            input_log=RUNS['C_scheduler_full']/'training.log'
            if bridge is None and input_log.is_file():
                command=[SMOL,str(SOURCE/'scripts/zoh_scheduled_training_to_tensorboard.py'),
                    '--input-log',str(input_log),'--log-dir',str(RUNS['B_scheduler_expert'].parent/'zoh_train_1000_0906_v1/tensorboard/zoh_full_5000'),'--follow']
                with (ROOT/'full_tensorboard_bridge.log').open('x') as log:
                    bridge=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(ROOT/'full_tensorboard_receipt.json',dict(pid=bridge.pid,command=command))
            time.sleep(20)
        if child.returncode!=0:raise RuntimeError('full training supervisor failed; see full_supervisor.log')
        child=None
        verify_training('C_scheduler_full')
        publish_training_evidence()
        status('CHECKPOINTS_VERIFIED_STARTING_COMPARISON')
        with (ROOT/'comparison_queue.log').open('x') as log:
            child=subprocess.Popen([PY,'-m','scripts.zoh_compare_queue','--output',str(ROOT/'evaluation')],
                cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            path=ROOT/'evaluation/queue_status.json'
            state=json.loads(path.read_text()) if path.is_file() else {}
            status('COMPARISON_RUNNING',child_pid=child.pid,completed=len(state.get('completed',[])),
                current_slot=state.get('current_slot'),current_condition=state.get('current_condition'),queue_status=state.get('status'))
            time.sleep(20)
        if child.returncode!=0:raise RuntimeError('comparison failed; see evaluation queue evidence')
        child=None
        result=json.loads((ROOT/'delivery/results.json').read_text())
        if not result['complete'] or len(result['results'])!=72:raise RuntimeError('comparison final result incomplete')
        # Hash every deliverable except mutable progress/checksum files.
        write_json(ROOT/'delivery/artifacts_sha256.json',{str(p.relative_to(ROOT/'delivery')):sha(p)
            for p in (ROOT/'delivery').rglob('*') if p.is_file() and p.name not in ('progress.json','artifacts_sha256.json')})
        status('COMPLETE',completed=72,report=str(ROOT/'delivery/README.md'),
            note='Automatic trace audits passed; report and videos ready. Not a claim of navigation success.')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_NEEDS_REVIEW',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            # Children are supervisors with their own owned-group cleanup handlers.
            child.terminate()
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                status('CLEANUP_NEEDS_REVIEW',child_pid=child.pid)
        if bridge is not None and bridge.poll() is None:
            time.sleep(3);bridge.terminate()
            try:bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:bridge.kill();bridge.wait(timeout=10)

if __name__=='__main__':main()
