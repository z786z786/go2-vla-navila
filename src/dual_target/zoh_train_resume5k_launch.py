"""CPU supervisor: fresh DT2 approval, shared GPU admission and owned-group guard."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .zoh_train_core import approved_dataset
from .zoh_train_resume5k import validate_parent
from .tiny_train_core import file_sha
from .gpu_wait import Dt1GpuWaiter, GpuAdmissionPolicy, acquire_live_admission
from .coexistence_watch import sample_group
from .tiny_plan import DATA_ROOT, SOURCE_ROOT, verify_binding

POLICY = GpuAdmissionPolicy('zoh_train_shared_profile_7g_v1',7168,allow_existing_compute=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path,required=True)
    args = p.parse_args()
    out = args.output.resolve()
    approved_dataset(args.dataset,args.approval)
    validate_parent(args.resume.resolve(),args.dataset.resolve())
    verify_binding(SOURCE_ROOT)
    out.mkdir(parents=True,exist_ok=False)
    child = lease = None
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    def status(value, **extra):
        temporary = out/'.supervisor_status.tmp'
        temporary.write_text(json.dumps({'stage':'DT3','status':value,'wall_time_s':time.time(),**extra},indent=2)+'\n')
        temporary.replace(out/'supervisor_status.json')
    try:
        status('WAITING_GPU')
        waiter = Dt1GpuWaiter(project_root=DATA_ROOT,run_id=out.name,argv=sys.argv,
            policy=POLICY,required_free_mib=POLICY.required_free_mib)
        result = waiter.wait()
        if result['status'] != 'GPU_READY_LOCKED_RECHECKED':
            raise KeyboardInterrupt
        lease = acquire_live_admission(DATA_ROOT,run_id=out.name,actual_argv=sys.argv,policy=POLICY)
        env = dict(os.environ, HF_HOME=str(DATA_ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
                   MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',
                   PYTHONPATH=f'{SOURCE_ROOT}:{DATA_ROOT}/third_party/lerobot/src')
        command = [str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','src.dual_target.zoh_train_resume5k',
                   '--dataset',str(args.dataset.resolve()),'--approval',str(args.approval.resolve()),'--output',str(out),'--resume',str(args.resume.resolve())]
        receipt = {'stage':'DT3','supervisor_pid':os.getpid(),'run_id':out.name,
            'command':command,'admission':lease.state,'watchdog_free_floor_mib':2048,
            'max_runtime_s':14400,'dt2_approval_sha256':file_sha(args.approval),
            'source_sha256':{str(x.relative_to(SOURCE_ROOT)):file_sha(x)
                for pattern in ('src/dual_target/*.py','tests/dual_target/test_dt3*.py')
                for x in SOURCE_ROOT.glob(pattern)}}
        (out/'supervisor_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        with (out/'training.log').open('x') as log:
            child = subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            if os.getpgid(child.pid) != child.pid or child.pid == os.getpgrp():
                raise RuntimeError('child process group not isolated')
            status('RUNNING',child_pid=child.pid,child_pgid=child.pid)
            start = time.monotonic()
            with (out/'gpu_samples.jsonl').open('x') as samples:
                while child.poll() is None:
                    row = sample_group(child.pid)
                    samples.write(json.dumps(row)+'\n')
                    samples.flush()
                    if row['free_mib'] < 2048:
                        raise RuntimeError('actual free GPU memory below 2 GiB; stop only owned group')
                    if time.monotonic()-start > 14400:
                        raise TimeoutError('DT3 pilot exceeded 4-hour owned runtime bound')
                    time.sleep(2)
            if child.returncode != 0:
                raise RuntimeError(f'trainer exited {child.returncode}; preserve failure and checkpoints')
        result = json.loads((out/'result.json').read_text())
        if result['status'] != 'ZOH_5000_COMPLETE_AWAITING_REVIEW':
            raise RuntimeError('unexpected training result')
        status('ZOH_5000_COMPLETE_AWAITING_REVIEW',result=result)
    except BaseException as exc:
        status('INTERRUPTED_RESUMABLE' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
               detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    main()

