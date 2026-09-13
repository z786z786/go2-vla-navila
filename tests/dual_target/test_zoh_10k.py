import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.zoh_10k_core as core


def test_checkpoint_contract_is_exactly_every_2k():
    assert core.TOTAL==10000
    assert core.EFFECTIVE_BATCH==16
    assert core.SAMPLES==160000
    assert core.CHECKPOINT_STEPS==(2000,4000,6000,8000,10000)


def test_scheduler_uses_observed_official_curve():
    parameter=torch.nn.Parameter(torch.zeros(()))
    optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    scheduler=core.make_scheduler(optimizer)
    used=[]
    for _ in range(10000):
        used.append(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
    assert used[0]==pytest.approx(2.9940119760479047e-7)
    assert int(np.argmax(used))+1==334
    assert used[333]==pytest.approx(9.973347576037485e-5)
    assert used[1999]==pytest.approx(9.069857861578909e-5)
    assert used[3999]==pytest.approx(6.632914341393507e-5)
    assert used[5999]==pytest.approx(3.6199987949191975e-5)
    assert used[7999]==pytest.approx(1.1819425556762053e-5)
    assert used[9999]==pytest.approx(2.500002405716051e-6)


def test_sampler_extends_reviewed_prefix(monkeypatch,tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    weights=np.full(770,1/770,dtype=np.float64);np.save(root/'episode_balanced_sampler_weights.npy',weights)
    generator=torch.Generator().manual_seed(20260906)
    expected=torch.multinomial(torch.from_numpy(weights),160000,replacement=True,generator=generator).numpy()
    old=tmp_path/'old';old.mkdir();np.save(old/'sampled_indices.npy',expected[:80000])
    monkeypatch.setattr(core,'BASELINE_5K',old)
    assert np.array_equal(core.training_samples(root).numpy(),expected)


def test_new_pipeline_is_statically_train_only():
    repo=Path(__file__).resolve().parents[2]
    files=('scripts/zoh_train_10k.py','scripts/zoh_train_10k_launch.py','scripts/zoh_train_bc_10k_pipeline.py')
    forbidden=('zoh_compare','zoh_eval','isaac','navila')
    for name in files:
        source=(repo/name).read_text();tree=ast.parse(source)
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        assert not any(token in module.lower() for module in imports for token in forbidden)
    pipeline=(repo/'scripts/zoh_train_bc_10k_pipeline.py').read_text()
    assert "'validation_read':False" in pipeline
    assert "'evaluation_loss':False" in pipeline
    assert "'rollout':False" in pipeline


def test_tensorboard_strict_profile_has_exact_tags():
    source=Path('scripts/zoh_scheduled_training_to_tensorboard.py').read_text()
    assert '--training-only-five-tags' in source
    expected={'Loss/train','Optimization/learning_rate','Optimization/gradient_norm_before_clip',
        'Performance/step_time_s','Memory/peak_allocated_MiB'}
    tree=ast.parse(source)
    mapping=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SCALARS' for t in n.targets))
    values={v.value for v in mapping.value.values}
    assert expected<=values
