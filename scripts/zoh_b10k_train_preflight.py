"""Executable preflight for the B-10k train-only rollout queue."""
import ast
import json

from src.dual_target.tiny_plan import SOURCE_ROOT
from scripts.zoh_b10k_train_common import ROOT,checkpoint,tasks_for
from scripts.zoh_b10k_train_runtime import POLICY,SCOPE


def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    value=checkpoint();tasks=tasks_for(ROOT/'rollout',value)
    if (SCOPE!='b-10000-train-rollout' or POLICY.required_free_mib!=9216 or len(tasks)!=16
            or [x['slot'] for x in tasks]!=list(range(16)) or any(x['evaluation_split']!='train' for x in tasks)):
        raise ValueError('B-10k rollout contract failed')
    for name in ('scripts/zoh_b10k_train_runtime.py','scripts/zoh_b10k_train_queue.py'):
        tree=ast.parse((SOURCE_ROOT/name).read_text());imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if any(token in module.lower() for module in imports for token in ('zoh_eval_queue','validation')):
            raise ValueError('forbidden validation queue import')
    print(json.dumps({'status':'B_10K_TRAIN_ROLLOUT_PREFLIGHT_PASSED','scope':SCOPE,
        'checkpoint':value,'tasks':len(tasks),'slots':[x['slot'] for x in tasks],
        'validation_used':False,'test_used':False,'admission_free_mib':POLICY.required_free_mib,
        'runtime_free_floor_mib':2048},indent=2))


if __name__=='__main__':main()
