"""DT3-only masked objective and bounded, reproducible training utilities.

No simulator, target geometry or oracle fields enter this module.
"""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def approved_dataset(root, approval_path):
    root, approval_path = Path(root).resolve(), Path(approval_path).resolve()
    approval = json.loads(approval_path.read_text())
    if approval.get('stage') != 'DT2' or approval.get('status') != 'APPROVED':
        raise ValueError('DT3 requires an explicit completed DT2 approval')
    manifest_path = root/'dataset_manifest.json'
    if (approval['dataset_root'] != str(root)
            or file_sha(manifest_path) != approval['dataset_manifest_sha256']):
        raise ValueError('DT2 approval does not bind this dataset')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['episode_count'] != 16 or manifest['split'] != 'train'
            or manifest['status'] != 'CONVERTED_NOT_DT2_APPROVED'):
        raise ValueError('not the reviewed full 16-episode tiny dataset')
    for name, expected in manifest['files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha(path) != expected:
            raise ValueError(f'dataset artifact drift: {name}')
    normalizer = json.loads((root/'normalizer.json').read_text())
    if (normalizer['source_split'] != 'train'
            or normalizer['action_loss_active_channels'] != [True, False, True]
            or normalizer['action']['mean'][1] != 0
            or normalizer['action']['std'][1] != 1):
        raise ValueError('wrong tiny normalizer or constant-channel policy')
    return manifest, normalizer


def masked_anchor_losses(element_losses, action_is_pad):
    """Equal weight per sampled anchor, then mean over its valid vx/wz.

    This preserves episode/anchor-balanced sampling even for terminal chunks.
    Equal micro-batches therefore give the exact effective-batch objective.
    """
    import torch
    if (element_losses.ndim != 3 or element_losses.shape[-1] < 3
            or action_is_pad.shape != element_losses.shape[:2]
            or action_is_pad.dtype != torch.bool):
        raise ValueError('expected B,T,D element losses and boolean B,T padding')
    valid = ~action_is_pad
    if not bool(valid.any(dim=1).all()):
        raise ValueError('all-padding anchor is not a valid training example')
    selected = element_losses[:, :, [0, 2]]
    return selected.masked_fill(~valid.unsqueeze(-1), 0).sum((1, 2)) / (2*valid.sum(1))


def flow_loss(policy, batch, *, noise=None, time=None):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    if policy.config.adapt_to_pi_aloha:
        raise ValueError('Aloha action remapping is forbidden for Go2')
    images, image_masks = policy.prepare_images(batch)
    elements = policy.model.forward(
        images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        policy.prepare_state(batch), policy.prepare_action(batch), noise, time)
    return masked_anchor_losses(elements, batch['action_is_pad']).mean()


def resource_candidates(effective_batch=16):
    if effective_batch != 16:
        raise ValueError('the approved DT3 effective batch is 16')
    return (1, 2, 4, 8, 16)


def choose_microbatch(profiles):
    good = [p for p in profiles if p.get('passed')]
    if not good:
        raise RuntimeError('no micro-batch passed the live memory/finite-gradient profile')
    fastest = max(p['samples_per_second'] for p in good)
    # Within 5% of measured best throughput, prefer lower shared-GPU pressure.
    return min(p['micro_batch'] for p in good if p['samples_per_second'] >= fastest*.95)
