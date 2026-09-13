"""One episode, owned process-group cleanup, CPU-only GPU watchdog."""
import json
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dual_target.coexistence_watch import sample_group

BASE=Path('/mnt/wxh/go2_short_vln/outputs/r2r_ablation')
RUN_ID='20260910_ep1_continuous_v2'
OUT=BASE/RUN_ID
SOURCE=Path('/home/wxh/go2_short_vln')
EPISODE_ID='1'


def main():
    lock=(BASE/'r2r_continuous_task.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if OUT.exists():raise RuntimeError('Output exists; refusing another run')
    receipt=BASE/f'{RUN_ID}_guard.json'
    if receipt.exists():raise RuntimeError('Prior attempt exists; refusing duplicate')
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME'):env.pop(k,None)
    env['OMNI_KIT_ACCEPT_EULA']='YES'
    command=['/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python',
        str(SOURCE/'scripts/collect_r2r_continuous_v1.py'),'--expert','continuous',
        '--output',str(OUT),'--episode-id',EPISODE_ID,'--seed','20260910','--max-seconds','120',
        '--load_run','2024-09-25_23-22-02','--headless','--enable_cameras']
    status=dict(command=command,min_free_mib=2048,timeout_s=2400,status='preflight')
    # Exclusive creation prevents concurrent duplicate launches.
    with receipt.open('x') as f:json.dump(status,f)
    proc=None
    def interrupt(signum,frame):raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        with (BASE/f'{RUN_ID}_gpu.jsonl').open('x',buffering=1) as samples, (BASE/f'{RUN_ID}.log').open('x') as log:
            for i in range(3):
                pre=sample_group(-1);samples.write(json.dumps(pre)+'\n')
                print(json.dumps({'admission_sample':i+1,'free_mib':pre['free_mib']}),flush=True)
                if pre['free_mib']<8192:raise RuntimeError('admission free memory below 8 GiB')
                if i<2:time.sleep(30)
            proc=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            assert proc.pid>1 and proc.pid!=os.getpgrp()
            status.update(pid=proc.pid,pgid=proc.pid,status='running')
            receipt.write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True);started=time.monotonic()
            while proc.poll() is None:
                row=sample_group(proc.pid);samples.write(json.dumps(row)+'\n')
                if row['free_mib']<2048:raise RuntimeError('watchdog free memory below 2 GiB')
                if time.monotonic()-started>status['timeout_s']:raise RuntimeError('2400 second timeout')
                time.sleep(1)
            status['exit_code']=proc.returncode
            summary=json.loads((OUT/'summary.json').read_text())
            if proc.returncode!=0 or not summary.get('success'):raise RuntimeError('collector did not report successful trajectory')
            status['status']='success'
    except BaseException as e:
        status.update(status='aborted',error=repr(e))
    finally:
        if proc is not None and proc.poll() is None:
            # This PGID was created by our Popen(start_new_session=True).
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            status['owned_cleanup_exit_code']=proc.returncode
        receipt.write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    return 0 if status['status']=='success' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default=RUN_ID)
    parser.add_argument('--episode-id',default=EPISODE_ID)
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('invalid run ID')
    RUN_ID=args.run_id;OUT=BASE/RUN_ID;EPISODE_ID=args.episode_id
    sys.exit(main())
