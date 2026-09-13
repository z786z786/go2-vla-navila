"""V2 gate: wait for legacy evaluation, then probe + four expert cases and stop."""
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
from .zoh_runtime import POLICY
from .zoh_review import review_episode
from .gpu_wait import Dt1GpuWaiter
from .coexistence_watch import sample_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-for-queue',type=Path,required=True)
    args=p.parse_args()
    verify_binding(SOURCE_ROOT)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    sources=['src/dual_target/'+name+'.py' for name in
             ('zoh_control','zoh_loop','zoh_runtime','zoh_review','zoh_parking_review','zoh_queue')]
    tasks=[dict(mode='probe',slot=0)]+[dict(mode='expert',slot=i) for i in range(4)]
    manifest=dict(stage='ZOH_V2_GATE',zoh_spec=ZohSpec().metadata(),tasks=tasks,
        source_sha256={name:sha(SOURCE_ROOT/name) for name in sources},
        trajectory_split='development_train_only; never exported as validation/test',
        formal_collection=False,training=False,old_queue=str(args.wait_for_queue))
    manifest_path=root/'gate_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    child=None
    completed=[]
    def status(value,**extra):
        (root/'queue_status.json').write_text(json.dumps(dict(stage='ZOH_V2_GATE',status=value,
            pid=os.getpid(),completed=completed,training=False,**extra),indent=2)+'\n')
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        while True:
            old=json.loads(args.wait_for_queue.read_text())
            if old['status'] not in ('RUNNING','WAITING_GPU'):
                break
            status('WAITING_LEGACY_QUEUE',legacy_current_slot=old.get('current_slot'),legacy_pid=old['pid'])
            try: os.kill(old['pid'],0)
            except ProcessLookupError: raise RuntimeError('legacy status stale; inspect before acquiring GPU')
            time.sleep(15)
        for index,task in enumerate(tasks):
            run_id=f'{root.name}_{index:02d}_{task["mode"]}'
            status('WAITING_GPU',current=task)
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=run_id,argv=sys.argv,
                policy=POLICY,required_free_mib=8192)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED': raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','src.dual_target.zoh_runtime',
                '--live','--headless','--enable_cameras','--zoh-mode',task['mode'],'--slot',str(task['slot']),
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
            evidence=json.loads((root/'runs'/run_id/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_APPROVED': raise RuntimeError('missing runtime completion')
            ep=root/'runs'/run_id/'episodes'/evidence['episode_result']['episode_id']
            review=review_episode(ep,task['mode'])
            (ep/'zoh_trace_review.json').write_text(json.dumps(review,indent=2)+'\n')
            completed.append(dict(task,run_id=run_id,episode_result=evidence['episode_result'],review=review))
            child=None
        status('GATE_TRACES_PASSED_AWAITING_ROOT_REVIEW')
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
