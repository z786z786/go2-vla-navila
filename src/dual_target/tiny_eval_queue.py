"""Fixed DT3 16-task evaluation queue; no retries or automatic training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .tiny_plan import SOURCE_ROOT,DATA_ROOT,verify_binding
from .tiny_closed_loop import POLICY
from .gpu_wait import Dt1GpuWaiter
from .coexistence_watch import sample_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-seed',type=int,default=20260906)
    args=p.parse_args()
    verify_binding(SOURCE_ROOT)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    tasks=[{'slot':i,'policy_seed':args.policy_seed,'run_id':f'{root.name}_s{i:02d}'} for i in range(16)]
    (root/'eval_plan.json').write_text(json.dumps({'stage':'DT3','checkpoint':str(args.checkpoint),
        'tasks':tasks,'pid':os.getpid(),'policy':POLICY.policy_id},indent=2)+'\n')
    child=None
    results=[]
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def status(value,**extra):
        (root/'queue_status.json').write_text(json.dumps({'stage':'DT3','status':value,
            'pid':os.getpid(),'completed':results,**extra},indent=2)+'\n')
    try:
        for task in tasks:
            status('WAITING_GPU',current_slot=task['slot'])
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT,run_id=task['run_id'],argv=[],
                policy=POLICY,required_free_mib=8192)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED':
                raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','src.dual_target.tiny_closed_loop',
                '--live','--headless','--enable_cameras','--slot',str(task['slot']),
                '--policy-seed',str(args.policy_seed),'--policy-checkpoint',str(args.checkpoint),
                '--run-id',task['run_id'],'--output-dir',str(root/'runs'),
                '--go2-usd',str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration',str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None)
            env.pop('PYTHONHOME',None)
            with (root/f'{task["run_id"]}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current_slot=task['slot'],child_pgid=child.pid)
                start=time.monotonic()
                with (root/f'{task["run_id"]}_gpu.jsonl').open('x') as samples:
                    while child.poll() is None:
                        sample=sample_group(child.pid)
                        samples.write(json.dumps(sample)+'\n')
                        samples.flush()
                        if sample['free_mib']<2048 or time.monotonic()-start>1800:
                            raise RuntimeError('owned episode exceeded resource/time bound')
                        time.sleep(2)
                if child.returncode!=0:
                    raise RuntimeError(f'episode runtime failed: {child.returncode}')
            evidence=json.loads((root/'runs'/task['run_id']/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_DT3_APPROVED':
                raise RuntimeError('Isaac exit masked runtime error')
            results.append({**task,**evidence['episode_result']})
            child=None
        status('MATRIX_COMPLETE_NOT_DT3_APPROVED')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',detail=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)


if __name__=='__main__':
    main()
