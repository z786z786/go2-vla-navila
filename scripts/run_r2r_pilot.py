"""Bounded six-route pilot, sequential guarded collection and audit."""
import fcntl,json,subprocess,sys
from pathlib import Path
SOURCE=Path('/home/wxh/go2_short_vln')
BASE=Path('/mnt/wxh/go2_short_vln/outputs/r2r_ablation')
PY='/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python'
batch=BASE/'20260910_pilot6_v1'
lock=(BASE/'r2r_pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
batch.mkdir(exist_ok=False)
plan=json.loads(subprocess.check_output([sys.executable,str(SOURCE/'scripts/select_r2r_pilot.py')]))
(batch/'plan.json').write_text(json.dumps(plan,indent=2))
results=[]
for item in plan:
    run='20260910_pilot6_ep'+item['episode_id']
    print('START '+run,flush=True)
    rc=subprocess.call([sys.executable,str(SOURCE/'scripts/run_r2r_pilot_guard.py'),'--run-id',run,'--episode-id',item['episode_id']])
    receipt=json.loads((BASE/(run+'_guard.json')).read_text())
    summary_path=BASE/run/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result={**item,'run_id':run,'guard':receipt,'summary':summary}
    # Expected navigation failure remains in the denominator; infrastructure errors stop the batch.
    if summary.get('status')=='complete' and summary.get('termination_reason') in ('success','timeout','environment_termination'):
        audit_rc=subprocess.call([PY,str(SOURCE/'scripts/audit_r2r_ablation.py'),str(BASE/run)])
        result['audit_exit_code']=audit_rc
        if (BASE/run/'audit.json').exists():result['audit']=json.loads((BASE/run/'audit.json').read_text())
    results.append(result)
    (batch/'results.json').write_text(json.dumps(results,indent=2))
    print('RESULT '+json.dumps(result),flush=True)
    if not summary or summary.get('status')!='complete':
        raise SystemExit('Stopped: infrastructure/admission/watchdog error; preserve remaining routes')
(batch/'complete.json').write_text(json.dumps({'attempted':len(results),'successes':sum(r['summary'].get('success',False) for r in results)}))
