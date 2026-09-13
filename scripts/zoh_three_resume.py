"""Durable recovery after the zero-result evaluation-v1 loader startup failure."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.zoh_compare_common import ROOT,BASE,SOURCE,RUNS,write_json,sha,verify_training,publish_training_evidence

PY=str(BASE.parents[1]/'envs/conda/navila-isaac/bin/python')
ATTEMPT=ROOT/'evaluation_v2'
FAILED_ATTEMPT=ROOT/'evaluation'


def main():
    if ATTEMPT.exists():raise FileExistsError(ATTEMPT)
    previous=json.loads((FAILED_ATTEMPT/'queue_status.json').read_text())
    if previous.get('status')!='FAILED' or previous.get('completed')!=[]:
        raise ValueError('recovery only applies to the preserved zero-result v1 startup failure')
    failure=json.loads((FAILED_ATTEMPT/'runs/evaluation_A_constant_expert_s00/stage_status.json').read_text())
    if failure.get('detail')!='model pipe closed; inspect model.log':
        raise ValueError('unexpected prior failure evidence')
    checkpoints={name:verify_training(name) for name in RUNS}
    publish_training_evidence() if not (ROOT/'delivery/training').exists() else None
    write_json(ROOT/'recovery_plan.json',dict(pid=os.getpid(),attempt='evaluation_v2',
        prior_attempt=str(FAILED_ATTEMPT),prior_completed=0,reason='fresh-process SmolVLA config registration order',
        checkpoints=checkpoints,test_used=False,episodes=72,
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'scripts').glob('zoh_*.py')}))
    child=None
    def status(value,**extra):
        row=dict(status=value,pid=os.getpid(),attempt='evaluation_v2',wall_time_s=time.time(),**extra)
        write_json(ROOT/'pipeline_status.json',row);write_json(ROOT/'delivery/progress.json',row)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        env=dict(os.environ,PYTHONPATH=str(SOURCE),PYTHONUNBUFFERED='1')
        with (ROOT/'comparison_queue_v2.log').open('x') as log:
            child=subprocess.Popen([PY,'-m','scripts.zoh_compare_queue','--output',str(ATTEMPT)],
                cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            path=ATTEMPT/'queue_status.json'
            state=json.loads(path.read_text()) if path.is_file() else {}
            status('COMPARISON_RUNNING',child_pid=child.pid,completed=len(state.get('completed',[])),
                current_slot=state.get('current_slot'),current_condition=state.get('current_condition'),
                queue_status=state.get('status'))
            time.sleep(20)
        if child.returncode!=0:raise RuntimeError('comparison v2 failed; preserve evidence for review')
        child=None
        result=json.loads((ROOT/'delivery/results.json').read_text())
        if not result['complete'] or len(result['results'])!=72:raise RuntimeError('comparison v2 final result incomplete')
        write_json(ROOT/'delivery/artifacts_sha256.json',{str(p.relative_to(ROOT/'delivery')):sha(p)
            for p in (ROOT/'delivery').rglob('*') if p.is_file() and p.name not in ('progress.json','artifacts_sha256.json')})
        status('COMPLETE',completed=72,report=str(ROOT/'delivery/README.md'),
            note='Recovered from preserved zero-result loader startup failure; all v2 results are fresh.')
    except BaseException as exc:
        status('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_NEEDS_REVIEW',
            detail=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:status('CLEANUP_NEEDS_REVIEW',child_pid=child.pid)


if __name__=='__main__':main()
