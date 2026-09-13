"""Executable gate for the B-10k validation rollout."""
import json

from scripts.zoh_b10k_eval_common import ROOT,checkpoint,tasks_for
from scripts.zoh_b10k_eval_runtime import POLICY,SCOPE


def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    value=checkpoint();tasks=tasks_for(ROOT/'rollout',value)
    if (POLICY.required_free_mib!=7168 or POLICY.allow_existing_compute is not True
            or SCOPE!='b-10000-validation-rollout-floor1-exception'):
        raise ValueError('validation resource/scope contract changed')
    if len(tasks)!=8 or [x['slot'] for x in tasks]!=list(range(16,24)) or any(x['evaluation_split']!='validation' for x in tasks):
        raise ValueError('validation task contract changed')
    print(json.dumps({'status':'B_10K_VALIDATION_PREFLIGHT_PASSED','root':str(ROOT),'checkpoint':value,
        'tasks':8,'slots':[x['slot'] for x in tasks],'validation_used':True,'test_used':False,
        'admission_free_mib':7168,'runtime_free_floor_mib':1024},indent=2))


if __name__=='__main__':main()
