"""Fixed AutoDL full-10k checkpoint, all train then validation tasks on neu-3070."""
import json
import os
import signal
import subprocess
import time
from src.dual_target.tiny_plan import SOURCE_ROOT, DATA_ROOT, verify_binding
from src.dual_target.tiny_train_core import file_sha
from src.dual_target.zoh_train_core import approved_dataset
from src.dual_target.zoh_dataset import grouped_tasks
from src.dual_target.zoh_eval_review_v2 import audit_policy_episode
from src.dual_target.gpu_wait import Dt1GpuWaiter
from src.dual_target.coexistence_watch import sample_group
from scripts.zoh_autodl_rollout_runtime import POLICY
from scripts.zoh_compare_video import export_video

BASE = DATA_ROOT/'outputs/dual_target_v2'
ROOT = BASE/'autodl_full10k_train_eval_0908_v3'
CHECKPOINT = BASE/'zoh_full_10000_autodl_0908_v3/checkpoint_010000'
SCOPE = 'autodl-full-10000-train-validation'
SEED = 20260906

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT/'delivery').mkdir()
    results = []
    child = None
    def status(state, **extra):
        write(ROOT/'progress.json', dict(status=state, pid=os.getpid(), completed=len(results),
            planned=24, train_planned=16, validation_planned=8, test_used=False, **extra))
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status('WAITING_CHECKPOINT_TRANSFER')
        deadline = time.monotonic()+21600
        sizes = {'training_state.pt':3148591973, 'pretrained_model/model.safetensors':1800257992,
            'pretrained_model/config.json':2115, 'pretrained_model/policy_preprocessor.json':1543,
            'pretrained_model/policy_postprocessor.json':660,
            'pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors':664,
            'pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors':664}
        while not all((CHECKPOINT/n).is_file() and (CHECKPOINT/n).stat().st_size==s for n,s in sizes.items()):
            if time.monotonic()>deadline: raise TimeoutError('checkpoint transfer exceeded 6h')
            time.sleep(15)
        manifest = json.loads((CHECKPOINT/'checkpoint_manifest.json').read_text())
        if manifest['optimizer_updates']!=10000: raise ValueError('wrong checkpoint step')
        for n,digest in manifest['files_sha256'].items():
            target=(CHECKPOINT/n).resolve()
            if not target.is_relative_to(CHECKPOINT.resolve()) or file_sha(target)!=digest:
                raise ValueError('checkpoint hash mismatch: '+n)
        config=json.loads((CHECKPOINT/'pretrained_model/config.json').read_text())
        if config['train_expert_only'] or config['freeze_vision_encoder']:
            raise ValueError('checkpoint is not full fine-tuned')
        verify_binding(SOURCE_ROOT)
        approved_dataset(BASE/'zoh_dataset_0906_v2/dataset',
            SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json')
        tasks=[]
        for slot,(split,scene,task) in enumerate(grouped_tasks()[:24]):
            expected='train' if slot<16 else 'validation'
            if split!=expected: raise ValueError('task split drift')
            tasks.append(dict(condition='C_full_10k_autodl', slot=slot, step=10000,
                evaluation_split=split, instruction=task.instruction, policy_seed=SEED,
                run_id=f'autodl_full10k_s{slot:02d}', checkpoint=str(CHECKPOINT),
                checkpoint_manifest_sha256=file_sha(CHECKPOINT/'checkpoint_manifest.json')))
        sources=list((SOURCE_ROOT/'src/dual_target').glob('*.py'))
        sources+=list((SOURCE_ROOT/'scripts').glob('zoh_compare*.py'))
        sources+=list((SOURCE_ROOT/'scripts').glob('zoh_autodl*.py'))
        sources.append(SOURCE_ROOT/'scripts/zoh_full_support.py')
        plan=dict(scope=SCOPE, policy=POLICY.policy_id, tasks=tasks, training=False,
            validation_used=True, test_used=False, required_free_mib=9216, runtime_floor_mib=2048,
            source_sha256={str(p.relative_to(SOURCE_ROOT)):file_sha(p) for p in sources})
        write(ROOT/'eval_plan.json',plan)
        for task in tasks:
            status('WAITING_GPU', current_slot=task['slot'], required_free_mib=12288)
            waiter=Dt1GpuWaiter(project_root=DATA_ROOT, run_id=task['run_id'], argv=[],
                policy=POLICY, required_free_mib=POLICY.required_free_mib)
            if waiter.wait()['status']!='GPU_READY_LOCKED_RECHECKED': raise KeyboardInterrupt
            command=[str(DATA_ROOT/'envs/conda/navila-isaac/bin/python'),'-m','scripts.zoh_autodl_rollout_runtime',
                '--live','--headless','--enable_cameras','--slot',str(task['slot']),
                '--policy-seed',str(SEED),'--policy-checkpoint',str(CHECKPOINT),
                '--eval-plan',str(ROOT/'eval_plan.json'),'--run-id',task['run_id'],'--output-dir',str(ROOT/'runs'),
                '--go2-usd',str(DATA_ROOT/'assets/isaac_sim_4_1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'),
                '--motion-calibration',str(DATA_ROOT/'outputs/dual_target_v1/dt1_motion_20260905_r1/motion_precheck/motion_calibration.json')]
            env=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
            with (ROOT/(task['run_id']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE_ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',current_slot=task['slot'],child_pgid=child.pid)
                started=time.monotonic()
                with (ROOT/(task['run_id']+'_gpu.jsonl')).open('x') as gpu:
                    while child.poll() is None:
                        sample=sample_group(child.pid); gpu.write(json.dumps(sample)+'\n');gpu.flush()
                        if sample['free_mib']<2048: raise RuntimeError('GPU free below 2048 MiB')
                        if time.monotonic()-started>1800: raise TimeoutError('rollout exceeded 1800s')
                        time.sleep(2)
                if child.returncode: raise RuntimeError('runtime exited '+str(child.returncode))
            evidence=json.loads((ROOT/'runs'/task['run_id']/'stage_status.json').read_text())
            if evidence['status']!='EPISODE_COMPLETE_NOT_DT3_APPROVED': raise ValueError('runtime incomplete')
            result=evidence['episode_result']; ep=ROOT/'runs'/task['run_id']/'episodes'/result['episode_id']
            review=audit_policy_episode(ep,result);write(ep/'policy_trace_review.json',review)
            video=export_video(ep,task,result,ROOT/'delivery/videos')
            results.append({**task,**result,'review':review,**video})
            write(ROOT/'delivery/results.json',dict(complete=len(results)==24,results=results,
                metrics={s:dict(completed=sum(r['evaluation_split']==s for r in results),
                    success=sum(r['evaluation_split']==s and r['status']=='success' for r in results))
                    for s in ('train','validation')}))
            child=None
        status('COMPLETE_AWAITING_USER')
    except BaseException as exc:
        status('FAILED_NEEDS_REVIEW',detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

if __name__=='__main__':main()
