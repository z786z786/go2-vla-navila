"""CPU-only real checkpoint/model checks before admitting the automatic queue."""
import json
from scripts.zoh_compare_common import verify_training


def main():
    import torch
    from src.dual_target.smolvla_probe_client import DEFAULT_BASE_MODEL,_build_new_3d_config
    from scripts.zoh_full_support import load_training_policy,checked_parameters,selected_weights
    torch.set_num_threads(4)
    config=_build_new_3d_config(DEFAULT_BASE_MODEL)
    config.device='cpu';config.train_expert_only=False;config.freeze_vision_encoder=False
    config.chunk_size=5;config.n_action_steps=1
    policy=load_training_policy(DEFAULT_BASE_MODEL,config)
    rows=checked_parameters(policy)
    selected=selected_weights(policy)
    vision=policy.model.vlm_with_expert.get_vlm_model().vision_model
    assert vision.is_gradient_checkpointing
    print(json.dumps(dict(event='full_cpu_model_preflight',parameters=sum(r['count'] for r in rows),
        trainable=sum(r['count'] for r in rows if r['trainable']),
        selected_parameters={k:v[0] for k,v in selected.items()},vision_checkpointing=True,
        cuda_initialized=torch.cuda.is_initialized()),indent=2),flush=True)
    print(json.dumps(verify_training('A_constant_expert'),indent=2),flush=True)


if __name__=='__main__':main()
