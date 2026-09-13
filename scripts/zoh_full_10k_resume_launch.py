"""Durable 13.5-GiB queue and watchdog for exact full-policy 2k->10k resume."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

from src.dual_target.gpu_wait import Dt1GpuWaiter,GpuAdmissionPolicy,acquire_live_admission
from src.dual_target.coexistence_watch import sample_group
from src.dual_target.tiny_plan import DATA_ROOT,SOURCE_ROOT,verify_binding
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.zoh_train_core import approved_dataset
from scripts.zoh_full_10k_resume import validate_parent
from scripts.zoh_10k_core import verify_checkpoint

POLICY=GpuAdmissionPolicy('zoh_full_resume10k_13_5g_v1',13824,allow_existing_compute=True)

def write_json(path,value):
    tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def stop(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)

def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','approval','output','resume','tensorboard-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();dataset=args.dataset.resolve();approval=args.approval.resolve();out=args.output.resolve();parent=args.resume.resolve();tb=args.tensorboard_dir.resolve()
    approved_dataset(dataset,approval);validate_parent(parent,dataset);verify_binding(SOURCE_ROOT)
    for path in (out,tb):
        if path.exists():raise FileExistsError(path)
    out.mkdir(parents=True,exist_ok=False);child=bridge=lease=None;bridge_log=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def status(value,**extra):write_json(out/'supervisor_status.json',dict(stage='DT3',status=value,scope='full',**extra))
    try:
        status('WAITING_GPU',required_free_mib=13824,runtime_free_floor_mib=2048)
        waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=out.name,argv=sys.argv,policy=POLICY,required_free_mib=13824)
        if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':raise KeyboardInterrupt
        lease=acquire_live_admission(DATA_ROOT,run_id=out.name,actual_argv=sys.argv,policy=POLICY)
        env={**os.environ,'HF_HOME':str(DATA_ROOT/'cache/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','PYTHONUNBUFFERED':'1',
            'PYTHONPATH':f'{SOURCE_ROOT}:{DATA_ROOT}/third_party/lerobot/src'}
        command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','scripts.zoh_full_10k_resume','--dataset',str(dataset),
            '--approval',str(approval),'--output',str(out),'--resume',str(parent)]
        sources=[SOURCE_ROOT/'scripts/zoh_full_10k_resume.py',SOURCE_ROOT/'scripts/zoh_full_10k_resume_launch.py',
            SOURCE_ROOT/'scripts/zoh_10k_core.py',SOURCE_ROOT/'scripts/zoh_full_support.py']
        write_json(out/'supervisor_receipt.json',dict(stage='DT3',command=command,admission=lease.state,
            resume_checkpoint=str(parent),resume_manifest_sha256=file_sha(parent/'checkpoint_manifest.json'),
            required_free_mib=13824,runtime_free_floor_mib=2048,validation_read=False,evaluation_loss=False,rollout=False,
            source_sha256={str(x.relative_to(SOURCE_ROOT)):file_sha(x) for x in sources}))
        with (out/'training.log').open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        bridge_command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),str(SOURCE_ROOT/'scripts/zoh_scheduled_training_to_tensorboard.py'),
            '--input-log',str(out/'training.log'),'--log-dir',str(tb),'--poll-seconds','2','--follow','--training-only-five-tags']
        bridge_log=(out/'tensorboard.log').open('x');bridge=subprocess.Popen(bridge_command,cwd=SOURCE_ROOT,env=env,
            stdout=bridge_log,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',child_pid=child.pid,tensorboard_bridge_pid=bridge.pid,runtime_free_floor_mib=2048)
        with (out/'gpu_samples.jsonl').open('x') as samples:
            started=time.monotonic()
            while child.poll() is None:
                row=sample_group(child.pid);samples.write(json.dumps(row)+'\n');samples.flush()
                if row['free_mib']<2048:raise RuntimeError('actual free GPU memory below 2 GiB')
                if time.monotonic()-started>43200:raise TimeoutError('resume runtime exceeded 12h')
                time.sleep(2)
        if child.returncode!=0:raise RuntimeError(f'resume trainer exited {child.returncode}')
        result=json.loads((out/'result.json').read_text())
        if result.get('status')!='ZOH_FULL_10000_RESUME_COMPLETE_AWAITING_REVIEW':raise RuntimeError('unexpected result')
        for step in (4000,6000,8000,10000):verify_checkpoint(out/f'checkpoint_{step:06d}',step,dataset,file_sha(out/'sampled_indices.npy'))
        status(result['status'],result=result)
    except BaseException as exc:
        status('INTERRUPTED_RESUMABLE' if isinstance(exc,KeyboardInterrupt) else 'FAILED',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if bridge_log is not None:bridge_log.close()
        stop(bridge);stop(child)
        if lease is not None:lease.close()

if __name__=='__main__':main()
