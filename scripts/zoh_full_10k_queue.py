"""Durable C-only fresh 10k full fine-tuning queue with train-only TensorBoard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from src.dual_target.tiny_plan import DATA_ROOT,SOURCE_ROOT,verify_binding
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.zoh_train_core import approved_dataset
from scripts.zoh_10k_core import training_samples,verify_run

BASE=DATA_ROOT/'outputs/dual_target_v2'
ROOT=BASE/'zoh_full_10000_queue_0907_v4'
RUN=BASE/'zoh_full_10000_0907_v4'
OLD_RUN=BASE/'zoh_full_10000_0907_v3'
DATASET=BASE/'zoh_dataset_0906_v2/dataset'
APPROVAL=SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json'
TB_DIR=BASE/'zoh_train_1000_0906_v1/tensorboard/zoh_full_10000_v4'


def write_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sources():
    paths=[*(SOURCE_ROOT/'src/dual_target').glob('*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_*10k*.py'),
        SOURCE_ROOT/'scripts/zoh_full_support.py',SOURCE_ROOT/'scripts/zoh_scheduled_training_to_tensorboard.py']
    return {str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in paths}


def stop_process(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def main():
    verify_binding(SOURCE_ROOT);approved_dataset(DATASET,APPROVAL)
    sampled=training_samples(DATASET).numpy()
    b=np.load(BASE/'zoh_scheduler_10000_0907_v1/sampled_indices.npy')
    if not np.array_equal(sampled,b):raise ValueError('C sample order differs from completed B-10k')
    old=json.loads((OLD_RUN/'supervisor_status.json').read_text())
    events=[json.loads(x) for x in (OLD_RUN/'events.jsonl').read_text().splitlines()]
    updates=[x for x in events if x.get('event')=='update']
    if (old.get('status')!='FAILED' or len(updates)!=1 or updates[0].get('optimizer_updates')!=1
            or 'first pre-update loss differs' not in (OLD_RUN/'training.log').read_text()):
        raise ValueError('old C attempt is not the preserved wrong-baseline step-1 failure')
    for path in (ROOT,RUN,TB_DIR):
        if path.exists():raise FileExistsError(path)
    ROOT.mkdir(parents=True,exist_ok=False);frozen=sources()
    write_json(ROOT/'queue_plan.json',{'stage':'DT3','status':'PLAN_FROZEN','pid':os.getpid(),
        'scope':'full','fresh_from_base':True,'target_optimizer_updates':10000,
        'dataset':str(DATASET),'dataset_manifest_sha256':file_sha(DATASET/'dataset_manifest.json'),
        'approval_sha256':file_sha(APPROVAL),'same_sample_order_as_B':True,
        'B_sampled_indices_sha256':file_sha(BASE/'zoh_scheduler_10000_0907_v1/sampled_indices.npy'),
        'output':str(RUN),'tensorboard_dir':str(TB_DIR),'required_free_mib':13824,
        'resource_exception':'user-authorized 13.5 GiB full-finetune admission',
        'runtime_free_floor_mib':2048,'validation_read':False,'evaluation_loss':False,'rollout':False,
        'checkpoint_steps':[2000,4000,6000,8000,10000],
        'tensorboard_tags':['Loss/train','Optimization/learning_rate','Optimization/gradient_norm_before_clip',
            'Performance/step_time_s','Memory/peak_allocated_MiB'],
        'old_wrong_baseline_attempt':{'path':str(OLD_RUN),'status':old['status'],'updates':1,
            'first_loss':updates[0]['loss'],'supervisor_status_sha256':file_sha(OLD_RUN/'supervisor_status.json'),
            'training_log_sha256':file_sha(OLD_RUN/'training.log')},'source_sha256':frozen})
    supervisor=bridge=None;bridge_log=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def status(value,**extra):
        write_json(ROOT/'queue_status.json',{'stage':'DT3','status':value,'pid':os.getpid(),
            'scope':'full','output':str(RUN),'validation_read':False,'evaluation_loss':False,
            'rollout':False,**extra})
    try:
        if sources()!=frozen:raise ValueError('source changed after C-only plan freeze')
        command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','scripts.zoh_train_10k_launch',
            '--dataset',str(DATASET),'--approval',str(APPROVAL),'--output',str(RUN),'--scope','full',
            '--required-free-mib','13824']
        env={**os.environ,'PYTHONUNBUFFERED':'1',
            'PYTHONPATH':f'{SOURCE_ROOT}:{DATA_ROOT}/third_party/lerobot/src'}
        status('STARTING_SUPERVISOR',command=command)
        with (ROOT/'supervisor.log').open('x') as log:
            supervisor=subprocess.Popen(command,cwd=SOURCE_ROOT,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=env)
            while not (RUN/'training.log').is_file():
                if supervisor.poll() is not None:raise RuntimeError(f'C supervisor exited {supervisor.returncode}')
                state={}
                if (RUN/'supervisor_status.json').is_file():state=json.loads((RUN/'supervisor_status.json').read_text())
                status(state.get('status','STARTING_SUPERVISOR'),supervisor_pgid=supervisor.pid,
                    supervisor_detail=state.get('detail'))
                time.sleep(5)
            bridge_command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),
                str(SOURCE_ROOT/'scripts/zoh_scheduled_training_to_tensorboard.py'),
                '--input-log',str(RUN/'training.log'),'--log-dir',str(TB_DIR),'--poll-seconds','2',
                '--follow','--training-only-five-tags']
            bridge_log=(ROOT/'tensorboard.log').open('x')
            bridge=subprocess.Popen(bridge_command,cwd=SOURCE_ROOT,stdout=bridge_log,stderr=subprocess.STDOUT,
                start_new_session=True)
            status('TRAINING',supervisor_pgid=supervisor.pid,tensorboard_bridge_pgid=bridge.pid)
            supervisor.wait()
        if supervisor.returncode!=0:raise RuntimeError(f'C supervisor exited {supervisor.returncode}')
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if bridge.poll() is not None:raise RuntimeError(f'TensorBoard bridge exited {bridge.returncode}')
            state=TB_DIR/'bridge_state.json'
            if state.is_file() and json.loads(state.read_text()).get('last_step')==10000:break
            time.sleep(2)
        else:raise TimeoutError('TensorBoard bridge did not import step 10000')
        stop_process(bridge);bridge=None
        verification=verify_run(RUN,'full',DATASET,deep=True)
        write_json(ROOT/'verification.json',verification)
        result={'stage':'DT3','status':'C_FULL_10000_COMPLETE_AWAITING_USER','output':str(RUN),
            'verification_sha256':file_sha(ROOT/'verification.json'),'tensorboard_dir':str(TB_DIR),
            'validation_read':False,'evaluation_loss':False,'rollout':False,'next':'Stop and await user'}
        write_json(ROOT/'result.json',result);status(result['status'],result=result)
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_NEEDS_REVIEW',
            detail=f'{type(exc).__name__}: {exc}');raise
    finally:
        if bridge_log is not None:bridge_log.close()
        stop_process(bridge);stop_process(supervisor)


if __name__=='__main__':main()
