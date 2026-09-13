"""Fixed fresh 32-trajectory dataset collection, conversion and root-review stop."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .tiny_plan import DATA_ROOT, SOURCE_ROOT, verify_binding, sha
from .zoh_control import ZohSpec
from .zoh_dataset_runtime import POLICY
from .zoh_dataset import make_plan, verify_gate, audit_dataset_episode
from .gpu_wait import Dt1GpuWaiter
from .coexistence_watch import sample_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    verify_gate()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest=make_plan(root)
    tasks=manifest['slots']
    manifest_path=root/'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    child=None
    completed=[]
    def status(value,**extra):
        (root/'queue_status.json').write_text(json.dumps(dict(stage='ZOH_V2_DATASET',status=value,
            pid=os.getpid(),completed=completed,training=False,**extra),indent=2)+'\n')
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for index,task in enumerate(tasks):
            run_id=task['run_id']
            status('WAITING_GPU',current=task)
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=run_id,argv=sys.argv,
                policy=POLICY,required_free_mib=7168)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED': raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','src.dual_target.zoh_dataset_runtime',
                '--live','--headless','--enable_cameras','--zoh-mode','expert','--slot',str(task['index']),
                '--zoh-manifest',str(manifest_path),'--run-id',run_id,'--output-dir',str(root/'runs'),
                '--go2-usd',str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration',str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            with (root/(run_id+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current=task,child_pgid=child.pid)
                start=time.monotonic()
                with (root/(run_id+'_gpu.jsonl')).open('x') as samples:
                    while child.poll() is None:
                        sample=sample_group(child.pid)
                        samples.write(json.dumps(sample)+'\n');samples.flush()
                        if sample['free_mib']<2048 or time.monotonic()-start>1800:
                            raise RuntimeError('owned probe exceeded memory/time bound')
                        time.sleep(2)
                if child.returncode != 0: raise RuntimeError('Isaac runtime failed')
            ep=Path(task['episode_path'])
            evidence=json.loads((ep.parents[1]/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_APPROVED': raise RuntimeError('missing runtime completion')
            if evidence['episode_result']['status']!='success':raise RuntimeError('expert trajectory failed; no automatic retry')
            review,_=audit_dataset_episode(ep,task)
            (ep/'zoh_trace_review.json').write_text(json.dumps(review,indent=2)+'\n')
            completed.append(dict(index=task['index'],split=task['split'],run_id=run_id,
                episode_result=evidence['episode_result'],complete_frames=review['complete_frames'],
                edge_warning_frames=review['edge_warning_frames'],minimum_target_pixels=review['minimum_target_pixels']))
            child=None
        status('CONVERTING_CPU')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HOME=str(DATA_ROOT/'cache/huggingface'),
                 HF_HUB_OFFLINE='1',PYTHONPATH=str(SOURCE_ROOT)+':'+str(DATA_ROOT/'third_party/lerobot/src'))
        env.pop('PYTHONHOME',None)
        command=[str(DATA_ROOT/'envs/conda/smolvla/bin/python'),'-m','src.dual_target.zoh_dataset_convert',
                 '--plan',str(manifest_path),'--output',str(root/'dataset')]
        with (root/'conversion.log').open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child.wait(timeout=1800)
            if child.returncode!=0:raise RuntimeError('dataset conversion failed; inspect conversion.log')
        child=None
        status('DATASET_READY_AWAITING_ROOT_REVIEW')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',detail=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


if __name__=='__main__': main()
