"""Read-only process/lock check, recording completion evidence outside sources."""
import fcntl
import json
import os
from pathlib import Path
import subprocess

base=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v1')
batch=base/'dt2_tiny_0906_v2'
receipts=[json.loads(p.read_text()) for p in batch.glob('*_launcher_receipt.json')]
assert len(receipts)==8
pgids={r['pgid'] for r in receipts}
parent=json.loads((batch/'collection_receipt.json').read_text())['pid']
table=subprocess.run(['ps','-eo','pid=,pgid=,args='],capture_output=True,text=True,check=True).stdout
remaining=[]
for line in table.splitlines():
    pid,pgid,args=line.strip().split(None,2)
    if int(pgid) in pgids or int(pid)==parent:
        remaining.append(line)
assert not remaining,remaining
path=base/'.dt1_gpu_wait.lock'
with path.open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(lock,fcntl.LOCK_UN)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
result={'stage':'DT2','collection_parent_absent':parent,'owned_pgids_absent':sorted(pgids),
        'project_lock_available':True,'lock_path':str(path),'remaining_owned_processes':[],
        'gpu_processes_read_only':gpu.strip(),'foreign_processes_signaled':False}
with (batch/'root_cleanup_review.json').open('x') as f:
    f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
