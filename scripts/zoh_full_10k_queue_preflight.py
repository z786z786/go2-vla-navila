"""Executable preflight for the fresh C-only 10k queue."""
import ast
import json

import numpy as np

from src.dual_target.tiny_plan import SOURCE_ROOT
from scripts.zoh_full_10k_queue import BASE,ROOT,RUN,OLD_RUN,DATASET,TB_DIR
from scripts.zoh_10k_core import training_samples


def main():
    for path in (ROOT,RUN,TB_DIR):
        if path.exists():raise FileExistsError(path)
    old=json.loads((OLD_RUN/'supervisor_status.json').read_text())
    events=[json.loads(x) for x in (OLD_RUN/'events.jsonl').read_text().splitlines()]
    updates=[x for x in events if x.get('event')=='update']
    if (old.get('status')!='FAILED' or len(updates)!=1 or updates[0].get('optimizer_updates')!=1
            or 'first pre-update loss differs' not in (OLD_RUN/'training.log').read_text()):
        raise ValueError('old C attempt is not the preserved wrong-baseline step-1 failure')
    sampled=training_samples(DATASET).numpy();b=np.load(BASE/'zoh_scheduler_10000_0907_v1/sampled_indices.npy')
    if not np.array_equal(sampled,b):raise ValueError('B/C sample order mismatch')
    source=(SOURCE_ROOT/'scripts/zoh_full_10k_queue.py').read_text();tree=ast.parse(source);imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    if any(token in name.lower() for name in imports for token in ('zoh_eval','isaac','compare_runtime')):
        raise ValueError('C-only training queue imports evaluation/runtime code')
    print(json.dumps({'status':'C_FULL_10K_QUEUE_PREFLIGHT_PASSED','fresh_output':str(RUN),
        'old_attempt_status':old['status'],'old_attempt_updates':1,
        'old_attempt_first_loss':updates[0]['loss'],'same_sample_order_as_B':True,
        'samples':len(sampled),'required_free_mib':13824,'runtime_free_floor_mib':2048,
        'validation_read':False,'evaluation_loss':False,'rollout':False,
        'tensorboard_run':TB_DIR.name},indent=2))


if __name__=='__main__':main()
