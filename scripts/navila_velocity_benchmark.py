"""Sequential official-dataset evaluation with immutable run plan and GPU guard."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SOURCE = Path('/home/wxh/go2_short_vln')
DATA = Path('/mnt/wxh/go2_short_vln')
PYTHON = DATA / 'envs/conda/navila-isaac/bin/python'
NAVILA = DATA / 'third_party/NaVILA-Bench'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def free_memory():
    return int(subprocess.check_output(['nvidia-smi','--id=0',
        '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--initial-episode', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = DATA / 'assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz'
    with gzip.open(dataset, 'rt') as f: episodes = json.load(f)['episodes']
    if not episodes: raise ValueError('empty official dataset')
    sources = list(SOURCE.glob('scripts/navila_velocity*.py')) + [SOURCE/'src/inference/navila_velocity.py']
    sources += [NAVILA/'scripts/navila_eval.py']
    sources += list((NAVILA/'isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils').glob('*.py'))
    sources += [SOURCE/'src/dual_target/zoh_policy_server_no_state.py',
                SOURCE/'src/dual_target/zoh_control.py', args.checkpoint/'checkpoint_manifest.json', dataset]
    frozen = {str(f):digest(f) for f in sources}
    plan = {'scope':'all episodes in official vln_ce_isaac_v1.json.gz',
        'checkpoint':str(args.checkpoint), 'policy_seed':20260906, 'source_sha256':frozen,
        'episodes':[{'index':i,'episode_id':e['episode_id'],'scene_id':e['scene_id']} for i,e in enumerate(episodes)],
        'model_selection':'existing dual-target validation; no official-test-based selection',
        'policy':'5 Hz, chunk first action, zero state; immediate raw low-speed STOP',
        'required_free_mib':12288, 'runtime_free_floor_mib':2048}
    write(args.output/'plan.json', plan)
    results=[]
    if args.initial_episode:
        ep=args.initial_episode
        r=json.loads((ep/'result.json').read_text())
        provenance=json.loads((ep/'provenance.json').read_text())
        expected = {
            'official_source_sha256':digest(NAVILA/'scripts/navila_eval.py'),
            'official_measures_sha256':digest(NAVILA/'isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py'),
            'official_wrappers_sha256':digest(NAVILA/'isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/wrappers.py'),
            'checkpoint_manifest_sha256':digest(args.checkpoint/'checkpoint_manifest.json'),
            'adapter_source_sha256':digest(SOURCE/'scripts/navila_velocity_eval.py'),
        }
        if any(provenance.get(k)!=v for k,v in expected.items()):
            raise ValueError('initial episode source/checkpoint mismatch')
        if (r['status']!='COMPLETE' or r['episode_id']!=episodes[0]['episode_id']
            or r['checkpoint']!=str(args.checkpoint) or r['policy_seed']!=20260906
            or json.loads((ep/'episode.json').read_text())!=episodes[0]):
            raise ValueError('initial episode is not the frozen official first episode')
        results.append({'index':0,**r,'output':str(ep),'result_sha256':digest(ep/'result.json')})
    child=None
    gpu_lock=None
    def status(state, **extra):
        write(args.output/'progress.json', {'status':state,'pid':os.getpid(),
            'completed':len(results),'planned':len(episodes),**extra})
    def interrupt(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for i,e in enumerate(episodes):
            if i < len(results): continue
            for f,h in frozen.items():
                if digest(f)!=h: raise ValueError('frozen source drift: '+f)
            from src.dual_target.gpu_wait import project_gpu_lock
            lock_path=project_gpu_lock(DATA)
            lock_path.parent.mkdir(parents=True,exist_ok=True)
            gpu_lock=lock_path.open('a+')
            while True:
                try:
                    fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status('WAITING_PROJECT_GPU_LOCK',current_index=i)
                    time.sleep(30)
            consecutive=0
            while consecutive<3:
                free=free_memory()
                consecutive=consecutive+1 if free>=12288 else 0
                status('WAITING_GPU',current_index=i,free_mib=free)
                if consecutive<3: time.sleep(30)
            ep=args.output/f'episode_{i:06d}'
            command=[str(PYTHON),'-m','scripts.navila_velocity_eval','--checkpoint',str(args.checkpoint),
                '--output',str(ep),'--task=go2_matterport_vision','--num_envs=1','--history_length=9',
                '--load_run=2024-09-25_23-22-02','--headless','--enable_cameras',f'--episode_idx={i}']
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1',PYTHONPATH=str(SOURCE))
            env.pop('PYTHONHOME',None)
            with (args.output/f'episode_{i:06d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current_index=i,child_pgid=child.pid)
                started=time.monotonic()
                while child.poll() is None:
                    free=free_memory()
                    if free<2048: raise RuntimeError('GPU free below 2048 MiB')
                    if time.monotonic()-started>1800: raise TimeoutError('episode wall time above 1800s')
                    time.sleep(2)
                if child.returncode: raise RuntimeError(f'episode {i} infrastructure exit {child.returncode}')
            r=json.loads((ep/'result.json').read_text())
            if r['status']!='COMPLETE' or r['episode_id']!=e['episode_id']:
                raise ValueError('episode identity/completion mismatch')
            results.append({'index':i,**r,'output':str(ep),'result_sha256':digest(ep/'result.json')})
            keys=results[0]['metrics']
            write(args.output/'results.json', {'complete':len(results)==len(episodes),'results':results,
                'mean_metrics':{k:sum(r['metrics'][k] for r in results)/len(results) for k in keys}})
            child=None
            gpu_lock.close()
            gpu_lock=None
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED_NEEDS_REVIEW',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        if gpu_lock is not None: gpu_lock.close()


if __name__ == '__main__': main()
