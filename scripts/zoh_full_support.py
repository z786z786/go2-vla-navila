"""Explicit full-policy FP32 masters and compatible inference loading."""
from pathlib import Path

FULL_BASELINE=Path('/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_full_5000_0907_v1')


def baseline_first_loss(root):
    """Return the deterministic full-policy loss, never the expert-only baseline."""
    import json
    from src.dual_target.tiny_train_core import file_sha
    result=json.loads((FULL_BASELINE/'result.json').read_text())
    config=json.loads((FULL_BASELINE/'training_config.json').read_text())
    if (result.get('status')!='ZOH_FULL_5000_COMPLETE_AWAITING_REVIEW'
            or result.get('optimizer_updates')!=5000 or config.get('full_finetuning') is not True
            or config.get('train_expert_only') is not False or config.get('freeze_vision_encoder') is not False):
        raise ValueError('reviewed full-policy 5k baseline required')
    root=Path(root)
    if (config.get('dataset_manifest_sha256')!=file_sha(root/'dataset_manifest.json')
            or config.get('normalizer_sha256')!=file_sha(root/'normalizer.json')):
        raise ValueError('full-policy baseline dataset binding mismatch')
    rows=[]
    for line in (FULL_BASELINE/'events.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('event')=='update' and row.get('optimizer_updates')==1:rows.append(row)
    if len(rows)!=1:raise ValueError('full-policy baseline must contain exactly one step-1 event')
    return float(rows[0]['loss'])

def load_training_policy(path,config=None):
    from lerobot.configs import PreTrainedConfig
    # Importing the concrete config registers the ``smolvla`` draccus choice.
    # A fresh inference process has not necessarily imported modeling code yet.
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_model
    if config is None:config=PreTrainedConfig.from_pretrained(path,local_files_only=True)
    if config.train_expert_only or config.freeze_vision_encoder:
        raise ValueError('full run requires explicit VLM and vision unfreeze')
    policy=SmolVLAPolicy(config)
    for p in policy.parameters():
        p.requires_grad_(True)
        p.data=p.data.float()
    load_model(policy,str(Path(path)/'model.safetensors'),strict=True,device='cpu')
    vision=policy.model.vlm_with_expert.get_vlm_model().vision_model
    # Activation recomputation only: same effective batch/objective, RNG preserved.
    vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return policy.to(config.device).train()

def load_for_inference(path):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    config=PreTrainedConfig.from_pretrained(path,local_files_only=True)
    if config.train_expert_only:
        from src.dual_target.zoh_train import load_training_policy as frozen
        return frozen(path,config)
    return load_training_policy(path,config)

def checked_parameters(policy):
    rows=[dict(name=n,count=p.numel(),dtype=str(p.dtype),trainable=p.requires_grad) for n,p in policy.named_parameters()]
    if not all(r['trainable'] and r['dtype']=='torch.float32' for r in rows):
        raise ValueError('all full-run policy parameters must be enabled FP32 masters')
    for fragment in ('vision_model','text_model','lm_expert','state_proj','action_out_proj'):
        if not any(fragment in r['name'] for r in rows):raise ValueError('missing full scope '+fragment)
    return rows

def selected_weights(policy):
    # Use the first actual parameter of each named component; gradients checked below.
    return {key:next((n,p.detach().cpu().clone()) for n,p in policy.named_parameters() if key in n)
        for key in ('vision_model','text_model.layers.0.self_attn.q_proj','action_out_proj')}

def full_gradient_audit(policy):
    import torch
    rows=[]
    for key in ('vision_model','text_model','lm_expert'):
        found=[]
        for name,p in policy.named_parameters():
            if key in name and p.grad is not None:
                norm=float(p.grad.detach().float().norm())
                if not bool(torch.isfinite(p.grad).all()):raise ValueError('nonfinite full gradient '+name)
                if norm>0:found.append(dict(name=name,norm=norm))
        if not found:raise ValueError('no nonzero gradient in '+key)
        rows.append(dict(component=key,nonzero_parameter_tensors=len(found),example=found[0]))
    return dict(groups=rows,unused_parameter_names=[n for n,p in policy.named_parameters() if p.grad is None],
        note='All parameters enabled; architecture-unused output heads/norms may have no gradients.')

def weight_updates(policy,before):
    params=dict(policy.named_parameters());result={}
    for key,(name,value) in before.items():
        delta=float((params[name].detach().cpu()-value).abs().max())
        if delta<=0:raise ValueError('full scope did not update '+name)
        result[key]=dict(parameter=name,max_abs_update=delta)
    return result
