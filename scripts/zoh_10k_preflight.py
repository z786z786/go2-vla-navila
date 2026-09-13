"""Dependency-free executable preflight for the fresh B/C 10k pipeline."""
import ast
import json
from pathlib import Path

import numpy as np
import torch

from src.dual_target.tiny_plan import DATA_ROOT,SOURCE_ROOT,verify_binding
from src.dual_target.zoh_train_core import approved_dataset
from scripts.zoh_10k_core import (TOTAL,EFFECTIVE_BATCH,SAMPLES,CHECKPOINT_STEPS,
    make_scheduler,training_samples)


def main():
    dataset=DATA_ROOT/'outputs/dual_target_v2/zoh_dataset_0906_v2/dataset'
    approval=SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json'
    verify_binding(SOURCE_ROOT);manifest,_=approved_dataset(dataset,approval)
    if (TOTAL,EFFECTIVE_BATCH,SAMPLES,CHECKPOINT_STEPS)!=(10000,16,160000,(2000,4000,6000,8000,10000)):
        raise ValueError('10k/checkpoint constants changed')
    samples=training_samples(dataset)
    parameter=torch.nn.Parameter(torch.zeros(()));optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    scheduler=make_scheduler(optimizer);used=[]
    for _ in range(TOTAL):
        used.append(float(optimizer.param_groups[0]['lr']));optimizer.step();scheduler.step()
    expected={1:2.9940119760479047e-7,334:9.973347576037485e-5,
        2000:9.069857861578909e-5,4000:6.632914341393507e-5,
        6000:3.6199987949191975e-5,8000:1.1819425556762053e-5,
        10000:2.500002405716051e-6}
    if int(np.argmax(used))+1!=334:raise ValueError('official scheduler maximum step changed')
    for step,value in expected.items():
        if not np.isclose(used[step-1],value,rtol=1e-12,atol=1e-12):
            raise ValueError(f'official scheduler LR changed at {step}')
    forbidden=('zoh_compare','zoh_eval','isaac','navila')
    checked=[]
    for name in ('scripts/zoh_train_10k.py','scripts/zoh_train_10k_launch.py',
                 'scripts/zoh_train_bc_10k_pipeline.py'):
        tree=ast.parse((SOURCE_ROOT/name).read_text());imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if any(token in module.lower() for module in imports for token in forbidden):
            raise ValueError(f'forbidden evaluation/runtime import in {name}')
        checked.append(name)
    from scripts.zoh_scheduled_training_to_tensorboard import SCALARS
    tags={SCALARS[k] for k in ('loss','lr','gradient_norm','seconds','peak_allocated_mib')}
    expected_tags={'Loss/train','Optimization/learning_rate','Optimization/gradient_norm_before_clip',
        'Performance/step_time_s','Memory/peak_allocated_MiB'}
    if tags!=expected_tags:raise ValueError('TensorBoard five-tag contract changed')
    print(json.dumps({'status':'ZOH_10K_PREFLIGHT_PASSED','train_frames':manifest['frames'],
        'sample_count':samples.numel(),'sample_min':int(samples.min()),'sample_max':int(samples.max()),
        'reviewed_prefix_samples':80000,'scheduler_max_used_step':334,
        'scheduler_step_10000_used_lr':used[-1],'checkpoints':list(CHECKPOINT_STEPS),
        'train_only_modules':checked,'tensorboard_tags':sorted(tags)},indent=2))


if __name__=='__main__':main()
