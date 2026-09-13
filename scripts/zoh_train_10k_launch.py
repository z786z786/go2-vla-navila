"""CPU supervisor for one fresh train-only 10k SmolVLA run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.gpu_wait import Dt1GpuWaiter,GpuAdmissionPolicy,acquire_live_admission
from src.dual_target.coexistence_watch import sample_group
from src.dual_target.tiny_plan import DATA_ROOT,SOURCE_ROOT,verify_binding
from scripts.zoh_10k_core import training_samples


def atomic_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--approval',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('expert','full'),required=True)
    p.add_argument('--required-free-mib',type=int,default=0);args=p.parse_args()
    out=args.output.resolve();dataset=args.dataset.resolve();approval=args.approval.resolve()
    approved_dataset(dataset,approval);training_samples(dataset);verify_binding(SOURCE_ROOT)
    out.mkdir(parents=True,exist_ok=False)
    required=int(args.required_free_mib or (14336 if args.scope=='full' else 7168))
    if args.scope=='full' and required<13824:
        raise ValueError('full 10k admission may not be lowered below the reviewed 13.5 GiB exception')
    max_runtime=43200 if args.scope=='full' else 21600
    policy=GpuAdmissionPolicy(f'zoh_{args.scope}_fresh10k_{required//1024}g_v1',required,
        allow_existing_compute=True)
    child=lease=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def status(value,**extra):
        atomic_json(out/'supervisor_status.json',{'stage':'DT3','status':value,
            'scope':args.scope,'wall_time_s':time.time(),**extra})
    try:
        status('WAITING_GPU',validation_read=False,evaluation_loss=False,rollout=False)
        waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=out.name,argv=sys.argv,
            policy=policy,required_free_mib=policy.required_free_mib)
        if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':raise KeyboardInterrupt
        lease=acquire_live_admission(DATA_ROOT,run_id=out.name,actual_argv=sys.argv,policy=policy)
        env=dict(os.environ,HF_HOME=str(DATA_ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
            MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',
            PYTHONPATH=f'{SOURCE_ROOT}:{DATA_ROOT}/third_party/lerobot/src')
        command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','scripts.zoh_train_10k',
            '--dataset',str(dataset),'--approval',str(approval),'--output',str(out),'--scope',args.scope,
            '--required-free-mib',str(required)]
        sources=[*(SOURCE_ROOT/'src/dual_target').glob('*.py'),
            *(SOURCE_ROOT/'scripts').glob('zoh_*10k*.py'),SOURCE_ROOT/'scripts/zoh_full_support.py']
        receipt={'stage':'DT3','supervisor_pid':os.getpid(),'run_id':out.name,'scope':args.scope,
            'command':command,'admission':lease.state,'watchdog_free_floor_mib':2048,
            'max_runtime_s':max_runtime,'dt2_approval_sha256':file_sha(approval),
            'validation_read':False,'evaluation_loss':False,'rollout':False,
            'source_sha256':{str(x.relative_to(SOURCE_ROOT)):file_sha(x) for x in sources}}
        atomic_json(out/'supervisor_receipt.json',receipt)
        with (out/'training.log').open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            if os.getpgid(child.pid)!=child.pid or child.pid==os.getpgrp():
                raise RuntimeError('child process group not isolated')
            status('RUNNING',child_pid=child.pid,child_pgid=child.pid,
                validation_read=False,evaluation_loss=False,rollout=False)
            started=time.monotonic()
            with (out/'gpu_samples.jsonl').open('x') as samples:
                while child.poll() is None:
                    row=sample_group(child.pid);samples.write(json.dumps(row)+'\n');samples.flush()
                    if row['free_mib']<2048:raise RuntimeError('actual free GPU memory below 2 GiB')
                    if time.monotonic()-started>max_runtime:raise TimeoutError('owned 10k runtime exceeded bound')
                    time.sleep(2)
            if child.returncode!=0:raise RuntimeError(f'trainer exited {child.returncode}')
        result=json.loads((out/'result.json').read_text())
        expected=('ZOH_EXPERT_10000_COMPLETE_AWAITING_REVIEW' if args.scope=='expert'
            else 'ZOH_FULL_10000_COMPLETE_AWAITING_REVIEW')
        if result.get('status')!=expected:raise RuntimeError('unexpected training result')
        status(expected,result=result,validation_read=False,evaluation_loss=False,rollout=False)
    except BaseException as exc:
        status('INTERRUPTED_RESUMABLE' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
            detail=f'{type(exc).__name__}: {exc}',validation_read=False,evaluation_loss=False,rollout=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        if lease is not None:lease.close()


if __name__=='__main__':main()
