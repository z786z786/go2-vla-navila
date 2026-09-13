"""Read-only preflight for the exact full-policy 2k->10k resume queue."""
import ast,json
from pathlib import Path
from src.dual_target.tiny_plan import DATA_ROOT,SOURCE_ROOT,verify_binding
from src.dual_target.zoh_train_core import approved_dataset
from scripts.zoh_full_10k_resume import validate_parent
from scripts.zoh_10k_core import training_samples

def main():
    base=DATA_ROOT/'outputs/dual_target_v2';dataset=base/'zoh_dataset_0906_v2/dataset'
    approval=SOURCE_ROOT/'reports/dual_target_v1/zoh_v2_dataset_approval.json';parent=base/'zoh_full_10000_0907_v4/checkpoint_002000'
    out=base/'zoh_full_10000_resume_0907_v1';tb=base/'zoh_train_1000_0906_v1/tensorboard/zoh_full_10000_resume_v1'
    verify_binding(SOURCE_ROOT);approved_dataset(dataset,approval);_,receipt=validate_parent(parent,dataset)
    if out.exists() or tb.exists():raise FileExistsError('resume output is not fresh')
    if training_samples(dataset).numel()!=160000:raise ValueError('wrong sample plan')
    for name in ('scripts/zoh_full_10k_resume.py','scripts/zoh_full_10k_resume_launch.py'):
        tree=ast.parse((SOURCE_ROOT/name).read_text());imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if any(x in m.lower() for m in imports for x in ('eval','isaac','rollout')):raise ValueError('train-only import gate failed')
    print(json.dumps(dict(status='FULL_10K_RESUME_PREFLIGHT_PASSED',parent_step=2000,
        parent_checkpoint_verified=receipt['passed'],target_step=10000,new_updates=8000,
        required_free_mib=13824,runtime_free_floor_mib=2048,gpu_reservation=False,
        validation_read=False,evaluation_loss=False,rollout=False,output=str(out),tensorboard_dir=str(tb)),indent=2))

if __name__=='__main__':main()
