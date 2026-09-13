"""Strict approved v2 train-split binding; no validation/test frame loading."""
import json
from pathlib import Path
from .tiny_train_core import file_sha


def approved_dataset(root, approval_path):
    root, approval_path = Path(root).resolve(), Path(approval_path).resolve()
    approval = json.loads(approval_path.read_text())
    if (approval.get('stage') != 'ZOH_V2_DATASET' or approval.get('status') != 'APPROVED'
            or not approval.get('formal_dataset_approved')):
        raise ValueError('explicit v2 dataset approval required')
    if str(root.parent) != approval['remote_dataset_root'] or root.name != 'dataset':
        raise ValueError('wrong dataset root; validation/test roots forbidden')
    if file_sha(root/'dataset_manifest.json') != approval['dataset_manifest_sha256']:
        raise ValueError('dataset manifest drift')
    if file_sha(root.parent/'split_manifest.json') != approval['split_manifest_sha256']:
        raise ValueError('split plan drift')
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    if manifest['test_exported'] or manifest['training']:
        raise ValueError('unexpected export semantics')
    for name, digest in manifest['files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or file_sha(path)!=digest:
            raise ValueError('dataset artifact drift: '+name)
    train = [s for s in manifest['splits'] if s['split']=='train']
    if len(train)!=1 or train[0]['episodes']!=16 or train[0]['fps']!=5 or train[0]['chunk_size']!=5:
        raise ValueError('wrong train split / time grid')
    normalizer=json.loads((root/'normalizer.json').read_text())
    if (normalizer['source_split']!='train' or normalizer['action_loss_active_channels']!=[True,False,True]
            or normalizer['plan_sha256']!=approval['split_manifest_sha256']
            or normalizer['action']['mean'][1]!=0 or normalizer['action']['std'][1]!=1):
        raise ValueError('normalizer leakage or fixed-vy drift')
    return train[0], normalizer
