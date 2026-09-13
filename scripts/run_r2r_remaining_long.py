"""Resume only the two outstanding pilot routes; never overwrite attempts."""
import fcntl,json,subprocess,sys
from pathlib import Path
BASE=Path('/mnt/wxh/go2_short_vln/outputs/r2r_ablation')
SRC=Path('/home/wxh/go2_short_vln')
PY='/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python'
batch=BASE/'20260910_pilot6_long_v2'
lock=(BASE/'r2r_pilot.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
batch.mkdir(exist_ok=False)
results=[]
for ep in ['6908','7601']:
    run=f'20260910_pilot6_ep{ep}_long_v2'
    rc=subprocess.call([sys.executable,str(SRC/'scripts/run_r2r_pilot_guard.py'),'--run-id',run,'--episode-id',ep])
    root=BASE/run
    receipt=BASE/(run+'_guard.json')
    guard=json.loads(receipt.read_text()) if receipt.exists() else {'status':'missing_receipt'}
    summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
    item={'episode_id':ep,'run_id':run,'guard_exit_code':rc,'guard':guard,'summary':summary}
    if summary.get('status')=='complete':
        item['audit_exit_code']=subprocess.call([PY,str(SRC/'scripts/audit_r2r_ablation.py'),str(root)])
        if (root/'audit.json').exists():item['audit']=json.loads((root/'audit.json').read_text())
    results.append(item)
    (batch/'results.json').write_text(json.dumps(results,indent=2))
    if summary.get('status')!='complete' or 'error' in guard:
        raise SystemExit('Stopped after failed attempt; see results.json')
    if item.get('audit_exit_code')!=0:
        raise SystemExit('Stopped after audit failure; see results.json')
(batch/'complete.json').write_text(json.dumps({'attempted':len(results),'successes':sum(x['summary'].get('success',False) for x in results)}))
