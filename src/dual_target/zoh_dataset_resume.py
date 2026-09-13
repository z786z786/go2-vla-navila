"""One user-authorized resource retry; preserve frozen collection sources."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .tiny_plan import DATA_ROOT, SOURCE_ROOT, sha
from .zoh_dataset import make_plan, audit_dataset_episode
from .zoh_dataset_runtime import POLICY
from .gpu_wait import Dt1GpuWaiter
from .coexistence_watch import sample_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    plan_path = root / 'split_manifest.json'
    plan = json.loads(plan_path.read_text())
    if plan != make_plan(root):
        raise ValueError('frozen plan/source mismatch')
    old = json.loads((root / 'queue_status.json').read_text())
    if (old['status'] != 'FAILED' or len(old['completed']) != 3
            or old['detail'] != 'owned probe exceeded memory/time bound'):
        raise ValueError('authorization covers only the known slot 3 resource interruption')
    if Path(f"/proc/{old['pid']}").exists():
        raise ValueError('previous parent still alive')
    task = plan['slots'][3]
    samples_path = root / (task['run_id'] + '_gpu.jsonl')
    last = json.loads(samples_path.read_text().splitlines()[-1])
    if last['free_mib'] >= 2048 or Path(f"/proc/{last['owned_pgid']}").exists():
        raise ValueError('resource interruption not proven or child still alive')
    completed = old['completed']
    for i, previous in enumerate(completed):
        slot = plan['slots'][i]
        if previous['index'] != i or previous['run_id'] != slot['run_id']:
            raise ValueError('completed prefix differs from frozen plan')
        review, _ = audit_dataset_episode(Path(slot['episode_path']), slot)
        if review['complete_frames'] != previous['complete_frames']:
            raise ValueError('completed evidence changed')
    # Fail closed on a second invocation: this archive is never overwritten.
    archive = root / 'resource_retry_slot03_attempt0'
    archive.mkdir(exist_ok=False)
    targets = [root/'queue_status.json', root/(task['run_id']+'.log'),
               samples_path, Path(task['episode_path']).parents[1]]
    records = {}
    for path in targets:
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in files:
            if file.is_file():
                records[str(file.relative_to(root))] = sha(file)
    (archive/'authorization.json').write_text(json.dumps(dict(
        reason='User approved one resource retry of slot 3 and continuation; no broader retries',
        interrupted_files_sha256=records, plan_sha256=sha(plan_path),
        resume_source_sha256=sha(Path(__file__)), retained_completed_slots=[0,1,2],
        admission_free_mib=7168, runtime_free_floor_mib=2048), indent=2)+'\n')
    for path in targets:
        path.rename(archive/path.name)
    child = None
    current = None
    def status(value, **extra):
        (root/'queue_status.json').write_text(json.dumps(dict(stage='ZOH_V2_DATASET',
            status=value, pid=os.getpid(), completed=completed, training=False,
            current=current, resource_retry_archive=str(archive), **extra), indent=2)+'\n')
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for current in plan['slots'][3:]:
            run_id = current['run_id']
            status('WAITING_GPU')
            waiter = Dt1GpuWaiter(project_root=DATA_ROOT, run_id=run_id, argv=sys.argv,
                                  policy=POLICY, required_free_mib=7168)
            if waiter.wait()['status'] != 'GPU_READY_LOCKED_RECHECKED':
                raise KeyboardInterrupt
            command = [str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),
                '-m', 'src.dual_target.zoh_dataset_runtime', '--live', '--headless',
                '--enable_cameras', '--zoh-mode', 'expert', '--slot', str(current['index']),
                '--zoh-manifest', str(plan_path), '--run-id', run_id,
                '--output-dir', str(root/'runs'), '--go2-usd',
                str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration', str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
            with (root/(run_id+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=SOURCE_ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                status('RUNNING', child_pgid=child.pid)
                start = time.monotonic()
                with (root/(run_id+'_gpu.jsonl')).open('x') as samples:
                    while child.poll() is None:
                        sample = sample_group(child.pid)
                        samples.write(json.dumps(sample)+'\n'); samples.flush()
                        if sample['free_mib'] < 2048:
                            raise RuntimeError('runtime free memory below 2048 MiB; no automatic retry')
                        if time.monotonic()-start > 1800:
                            raise RuntimeError('episode wall time exceeded 1800 s; no automatic retry')
                        time.sleep(2)
                if child.returncode != 0:
                    raise RuntimeError('Isaac runtime failed; no automatic retry')
            ep = Path(current['episode_path'])
            evidence = json.loads((ep.parents[1]/'stage_status.json').read_text())
            if evidence['status'] != 'EPISODE_COMPLETE_NOT_APPROVED' or evidence['episode_result']['status'] != 'success':
                raise RuntimeError('expert trajectory not successful; no automatic retry')
            review, _ = audit_dataset_episode(ep, current)
            (ep/'zoh_trace_review.json').write_text(json.dumps(review, indent=2)+'\n')
            completed.append(dict(index=current['index'], split=current['split'], run_id=run_id,
                episode_result=evidence['episode_result'], complete_frames=review['complete_frames'],
                edge_warning_frames=review['edge_warning_frames'], minimum_target_pixels=review['minimum_target_pixels']))
            child = None
        current = None
        status('CONVERTING_CPU')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HOME=str(DATA_ROOT/'cache/huggingface'),
            HF_HUB_OFFLINE='1', PYTHONPATH=str(SOURCE_ROOT)+':'+str(DATA_ROOT/'third_party/lerobot/src'))
        env.pop('PYTHONHOME', None)
        with (root/'conversion.log').open('x') as log:
            child = subprocess.Popen([str(DATA_ROOT/'envs/conda/smolvla/bin/python'),
                '-m', 'src.dual_target.zoh_dataset_convert', '--plan', str(plan_path),
                '--output', str(root/'dataset')], cwd=SOURCE_ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=1800)
            if child.returncode != 0:
                raise RuntimeError('conversion failed')
        child = None
        status('DATASET_READY_AWAITING_ROOT_REVIEW')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED', detail=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)


if __name__ == '__main__':
    main()
