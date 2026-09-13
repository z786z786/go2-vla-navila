"""Executable gate for the user-authorized low-memory B-10k rollout."""
import json

from scripts.zoh_b10k_floor1_queue import ROOT
from scripts.zoh_b10k_floor1_runtime import POLICY,SCOPE
from scripts.zoh_b10k_train_common import checkpoint,tasks_for


def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    value=checkpoint();tasks=tasks_for(ROOT/'rollout',value)
    if (POLICY.required_free_mib!=7168 or POLICY.allow_existing_compute is not True
            or SCOPE!='b-10000-train-rollout-floor1-exception'):
        raise ValueError('low-memory policy contract changed')
    if len(tasks)!=16 or [x['slot'] for x in tasks]!=list(range(16)) or any(x['evaluation_split']!='train' for x in tasks):
        raise ValueError('train-only task contract changed')
    print(json.dumps({'status':'B_10K_FLOOR1_PREFLIGHT_PASSED','root':str(ROOT),'checkpoint':value,
        'tasks':16,'validation_used':False,'test_used':False,'admission_free_mib':7168,
        'runtime_free_floor_mib':1024,'risk':'single user-authorized low-memory exception'},indent=2))


if __name__=='__main__':main()
