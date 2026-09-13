# Cloud Training Runbook

This runbook describes the intended way to move the Go2 velocity-control training
lane from a local workstation to a remote GPU server.

## What moves to the cloud

The cloud training path expects **processed training datasets**, not raw
`real_sessions/` or `sim_sessions/` trees.

See `docs/runbooks/processed_dataset_contract.md` for the exact dataset shape
expected by the training loader.

Minimum required inputs on the cloud side:

- repo checkout: `llada-vla-go2`
- model weights root
- processed train dataset file
- processed eval dataset file
- image root referenced by those datasets
- output directory for checkpoints/logs

## 1. Prepare a cloud workspace file

Copy the template and fill in real paths on the server:

```bash
cp configs/workspace.cloud.example.yaml /tmp/llada-vla-go2-workspace.yaml
```

Expected keys:

- `paths.model_root`
- `paths.train_data_path`
- `paths.eval_data_path`
- `paths.image_root`
- `paths.outputs_root`
- `paths.checkpoints_root`

Then export it:

```bash
export LLADA_VLA_GO2_WORKSPACE=/tmp/llada-vla-go2-workspace.yaml
```

You can validate the file with:

```bash
python3 -m workspace.config validate
```

## 2. Sync processed datasets if needed

If the processed dataset is still local, sync it first:

```bash
REMOTE_HOST=user@server \
REMOTE_ROOT=/mnt/data/llada-vla-go2 \
bash scripts/sync_train_dataset.sh
```

After syncing, update the cloud workspace file so that:

- `train_data_path -> <remote_root>/train.jsonl`
- `eval_data_path -> <remote_root>/val.jsonl`
- `image_root -> <remote_root>/images`

## 3. Rebuild the training environment

The training lane depends on a compatible `numpy` / `scikit-learn` /
`transformers` combination. Rebuild the environment on the cloud server with:

For RTX 5090 / Blackwell cloud hosts, prefer a dedicated `python=3.10` conda
environment before running the setup script. This project currently pins
`scikit-learn==1.2.2`, which is safer on Python 3.10 than on system-default
Python 3.12 images. On the current 5090 host, the verified runtime is
`/root/miniconda3/envs/llada-vla/bin/python` with `torch 2.9.1+cu128`.

```bash
bash scripts/setup_cloud_train_env.sh
```

This script will:

- install `LLaDA-V/train` in editable mode
- install `webdataset`
- reinstall pinned cloud constraints from `requirements/cloud_train_constraints.txt`
- run a post-check via `scripts/check_train_env.sh`

## 4. Verify imports before training

Always run the explicit environment check before starting a long job:

```bash
bash scripts/check_train_env.sh
```

Expected result:

- `numpy`, `sklearn`, `pandas`, `transformers`, and `torch` all import cleanly
- `transformers.Trainer` imports without ABI errors

If this step fails with a `numpy.dtype size changed` message, the Python
environment is broken and must be rebuilt before training.

## 5. Launch training

```bash
bash scripts/train_llada_vla_go2.sh
```

Useful overrides:

```bash
MODEL_ROOT=/mnt/models/LLaDA-V \
TRAIN_DATA=/mnt/data/processed/train.jsonl \
EVAL_DATA=/mnt/data/processed/val.jsonl \
IMAGE_FOLDER=/mnt/data/processed/images \
OUTPUT_DIR=/mnt/outputs/llada_vla_go2 \
bash scripts/train_llada_vla_go2.sh
```

The script will:

- validate the workspace config
- run the train-environment check
- resolve dataset/model/image paths from the workspace file unless overridden
- pass those resolved paths to `LLaDA-V/train/robotics/train_velocity.py`

## 6. Launch offline evaluation

```bash
bash scripts/eval_llada_vla_go2.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/outputs/llada_vla_go2 \
DATA_PATH=/mnt/data/processed/val.jsonl \
IMAGE_FOLDER=/mnt/data/processed/images \
bash scripts/eval_llada_vla_go2.sh
```

## Failure checklist

If training does not start, check these in order:

1. `bash scripts/check_train_env.sh`
2. `python3 -m workspace.config validate`
3. model path exists
4. train/eval dataset files exist
5. image root exists
6. output directory parent is writable

## Non-goals of this runbook

This runbook does **not** cover:

- preparing processed datasets from raw collector sessions on the cloud
- Isaac Sim raw-data collection on the cloud
- real collector release to the Go2 upper computer

Those flows stay separate from the cloud training path.
