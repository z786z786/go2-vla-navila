"""Durable B then C 10k train-only pipeline; explicitly no evaluation or rollout."""
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
from scripts.zoh_10k_core import verify_run,CHECKPOINT_STEPS

BASE=DATA_ROOT/'outputs/dual_target_v2'
ROOT=BASE/'zoh_bc_10000_0907_v1'
DATASET=BASE/'zoh_dataset_0906_v2/dataset'
APPROVAL=SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json'
RUNS=(('B_scheduler_expert','expert',BASE/'zoh_scheduler_10000_0907_v1','zoh_scheduler_10000'),
      ('C_scheduler_full','full',BASE/'zoh_full_10000_0907_v1','zoh_full_10000'))
TB_ROOT=BASE/'zoh_train_1000_0906_v1/tensorboard'
OLD_QUEUE=BASE/'two_ckpt_train_0907_v2/evaluation/queue_status.json'


def write_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_hashes():
    paths=[*(SOURCE_ROOT/'src/dual_target').glob('*.py'),*(SOURCE_ROOT/'scripts').glob('zoh_*10k*.py'),
        SOURCE_ROOT/'scripts/zoh_full_support.py',SOURCE_ROOT/'scripts/zoh_scheduled_training_to_tensorboard.py']
    return {str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in paths}


def old_queue_stopped():
    value=json.loads(OLD_QUEUE.read_text())
    if value.get('status')!='INTERRUPTED':raise ValueError('old rollout queue is not preserved as INTERRUPTED')
    pid=value.get('pid')
    if isinstance(pid,int) and Path(f'/proc/{pid}/cmdline').is_file():
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'zoh_train_bc_queue' in command:raise ValueError('old rollout queue is still alive')
    return {'status':value['status'],'pid':pid,'sha256':file_sha(OLD_QUEUE)}


def stop_process(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def wait_bridge(bridge,log_dir):
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        if bridge.poll() is not None:raise RuntimeError(f'TensorBoard bridge exited {bridge.returncode}')
        state=log_dir/'bridge_state.json'
        if state.is_file() and json.loads(state.read_text()).get('last_step')==10000:return
        time.sleep(2)
    raise TimeoutError('TensorBoard bridge did not import step 10000')


def main():
    verify_binding(SOURCE_ROOT);approved_dataset(DATASET,APPROVAL);old=old_queue_stopped()
    for _,_,run,tb_name in RUNS:
        if run.exists():raise FileExistsError(run)
        if (TB_ROOT/tb_name).exists():raise FileExistsError(TB_ROOT/tb_name)
    ROOT.mkdir(parents=True,exist_ok=False)
    frozen=source_hashes()
    plan={'stage':'DT3','status':'PLAN_FROZEN','pipeline_pid':os.getpid(),'sequence':['B','C'],
        'runs':[{ 'condition':n,'scope':s,'output':str(r),'tensorboard_run':t} for n,s,r,t in RUNS],
        'dataset':str(DATASET),'dataset_manifest_sha256':file_sha(DATASET/'dataset_manifest.json'),
        'approval_sha256':file_sha(APPROVAL),'checkpoint_steps':list(CHECKPOINT_STEPS),
        'validation_read':False,'evaluation_loss':False,'intermediate_rollout':False,
        'final_train_rollout':False,'validation_rollout':False,'test_rollout':False,
        'tensorboard_tags':['Loss/train','Optimization/learning_rate',
            'Optimization/gradient_norm_before_clip','Performance/step_time_s','Memory/peak_allocated_MiB'],
        'old_rollout_queue':old,'source_sha256':frozen}
    write_json(ROOT/'frozen_plan.json',plan)
    completed=[];supervisor=bridge=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def status(value,**extra):
        write_json(ROOT/'pipeline_status.json',{'stage':'DT3','status':value,'pid':os.getpid(),
            'completed':completed,'validation_read':False,'evaluation_loss':False,'rollout':False,**extra})
    try:
        for condition,scope,run,tb_name in RUNS:
            if source_hashes()!=frozen:raise ValueError('source changed after pipeline plan freeze')
            command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','scripts.zoh_train_10k_launch',
                '--dataset',str(DATASET),'--approval',str(APPROVAL),'--output',str(run),'--scope',scope]
            status('STARTING_'+condition,current=condition,command=command)
            with (ROOT/f'{condition}.supervisor.log').open('x') as log:
                supervisor=subprocess.Popen(command,cwd=SOURCE_ROOT,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1',
                    'PYTHONPATH':f'{SOURCE_ROOT}:{DATA_ROOT}/third_party/lerobot/src'})
                while not (run/'training.log').is_file():
                    if supervisor.poll() is not None:raise RuntimeError(f'{condition} supervisor exited {supervisor.returncode}')
                    time.sleep(2)
                tb_dir=TB_ROOT/tb_name
                bridge_command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),
                    str(SOURCE_ROOT/'scripts/zoh_scheduled_training_to_tensorboard.py'),
                    '--input-log',str(run/'training.log'),'--log-dir',str(tb_dir),'--poll-seconds','2',
                    '--follow','--training-only-five-tags']
                bridge_log=(ROOT/f'{condition}.tensorboard.log').open('x')
                bridge=subprocess.Popen(bridge_command,cwd=SOURCE_ROOT,stdout=bridge_log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                status('TRAINING_'+condition,current=condition,supervisor_pgid=supervisor.pid,
                    tensorboard_bridge_pgid=bridge.pid,tensorboard_dir=str(tb_dir))
                supervisor.wait();bridge_log.close()
            if supervisor.returncode!=0:raise RuntimeError(f'{condition} supervisor exited {supervisor.returncode}')
            wait_bridge(bridge,tb_dir);stop_process(bridge);bridge=None;supervisor=None
            verification=verify_run(run,scope,DATASET,deep=True)
            write_json(ROOT/f'{condition}.verification.json',verification)
            completed.append({'condition':condition,'scope':scope,'output':str(run),
                'verification_sha256':file_sha(ROOT/f'{condition}.verification.json'),
                'tensorboard_dir':str(tb_dir)})
            status('COMPLETED_'+condition,current=None)
        b=np.load(RUNS[0][2]/'sampled_indices.npy');c=np.load(RUNS[1][2]/'sampled_indices.npy')
        if not np.array_equal(b,c):raise ValueError('B/C sampled index order differs')
        comparison={'stage':'DT3','status':'BC_10K_TRAINING_COMPLETE_AWAITING_USER',
            'completed':completed,'same_sample_order':True,'sampled_indices_sha256':file_sha(RUNS[0][2]/'sampled_indices.npy'),
            'checkpoint_steps':list(CHECKPOINT_STEPS),'validation_read':False,'evaluation_loss':False,
            'rollout':False,'next':'Stop and await user instruction'}
        write_json(ROOT/'comparison.json',comparison)
        status('BC_10K_TRAINING_COMPLETE_AWAITING_USER',comparison_sha256=file_sha(ROOT/'comparison.json'))
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_NEEDS_REVIEW',
            detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop_process(bridge);stop_process(supervisor)


if __name__=='__main__':main()
