"""Fresh-process CPU smoke test for frozen and full comparison checkpoints."""
import gc
import json
from pathlib import Path

from scripts.zoh_full_support import load_for_inference


def main():
    import torch
    roots={
        'A_constant_expert':Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_5000_0906_v2/checkpoint_005000/pretrained_model'),
        'C_scheduler_full':Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_full_5000_0907_v1/checkpoint_005000/pretrained_model'),
    }
    rows=[]
    for name,path in roots.items():
        model=load_for_inference(path)
        rows.append(dict(condition=name,type=type(model).__name__,device=str(next(model.parameters()).device),
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            chunk_size=model.config.chunk_size,n_action_steps=model.config.n_action_steps,
            train_expert_only=model.config.train_expert_only,
            freeze_vision_encoder=model.config.freeze_vision_encoder))
        del model;gc.collect()
    print(json.dumps(dict(event='fresh_process_loader_preflight',cuda_available=torch.cuda.is_available(),rows=rows),indent=2))


if __name__=='__main__':main()
