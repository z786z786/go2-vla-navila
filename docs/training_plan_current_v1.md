# current_v1 Velocity Training Plan

## Goal

Train the current LLaDA-VLA velocity-control lane as a high-level Go2 policy:

- input: front camera image, robot state, text instruction
- output: future `K=6` velocity chunk, flattened as `3K` bins for `vx`, `vy`, `wz`
- control assumption: 10 Hz, 0.6 s horizon, receding-horizon execution

This plan intentionally does not train a general-purpose VLA. It only targets high-level velocity commands.

## Available Data

Primary dataset:

- `datasets/current_v1/train.jsonl`
- `datasets/current_v1/val.jsonl`
- `datasets/current_v1/images`

Current scale:

- total: 83,927 samples
- train: 74,306 samples
- val: 9,621 samples
- real: 37,868 samples
- sim: 46,059 samples

Real-only debug dataset:

- `datasets/real_small_current_v1/train.jsonl`
- `datasets/real_small_current_v1/val.jsonl`
- 1,808 real samples from session `20260418_204747`

Lane naming reserved for parallel experiments:

- `datasets/current_v1_real_only`
- `datasets/current_v1_sim_only`
- `datasets/current_v1_mixed` (reserved)

## Data Quality Gate

Run the audit before training:

```bash
cd /home/wxh/llada-vla-go2
conda run -n llada python scripts/audit_velocity_dataset.py \
  --dataset-root datasets/current_v1 \
  --output-dir outputs/data_audit/current_v1_real_vs_sim \
  --filter-source real \
  --filtered-output-root datasets/current_v1_real_only
```

The audit must check:

- sample schema: `image`, `instruction`, `state`, `action_chunk`, `action_mask`
- image path existence and sampled PIL decoding
- action shape, finite values, `K=6` masks, velocity range and clamp ratio
- state finite values and useful `mode/gait_type` variation
- train/val leakage by image path
- real-vs-sim distribution differences for instruction, target, `vx/vy/wz`

Do not start a main run if:

- any required real split is empty
- real images are missing or unreadable
- real action chunks are malformed
- train/val image leakage is nonzero
- `vx/wz` finite-value checks fail

Each prepared lane dataset must write metadata into `stats.json`:

- `lane`
- `source_filter`
- `upstream_dataset_root`
- `config_name`
- `velocity_profile_name`

## Training Stages

### Stage 0: Real-Only Smoke

Purpose: verify the current model/data contract and confirm loss is finite.

Use:

- data: `datasets/current_v1_real_only`
- train mode: `action_heads_only`
- max steps: small smoke run
- trainable: `action_embeddings`, `vx_head`, `vy_head`, `wz_head`
- frozen: LLaDA backbone, vision tower, projector

Command:

```bash
cd /home/wxh/llada-vla-go2
conda run -n llada bash scripts/train_current_v1_real_smoke.sh
```

Pass criteria:

- training starts without data or image errors
- loss is finite
- at least one masked action token is supervised per batch

### Stage 1: Real-Only Projector + Action Heads

Purpose: build the first stable real-data multimodal baseline.

Use:

- data: `datasets/current_v1_real_only`
- train mode: `projector_only`
- trainable: `action_embeddings`, action heads, `mm_projector`, optional `vision_resampler`
- frozen: vision tower, LLaDA language backbone

Command:

```bash
cd /home/wxh/llada-vla-go2
conda run -n llada bash scripts/train_current_v1_real_projector_only.sh
```

Pass criteria:

- train loss trends downward despite diffusion-mask noise
- eval loss does not diverge
- dequantized `vx/wz` MAE improves over the initial checkpoint
- predicted `vy` stays near zero because current data contains no lateral commands

Current completed run:

- server: 5090 cloud GPU
- output: `/root/autodl-tmp/outputs/train_current_v1_real_projector_only_main`
- best checkpoint: `checkpoint-1023`
- train summary: `global_step=1023`, `best_eval_loss=0.627869`
- runtime: about 3 h 14 m train + about 24 m eval
- note: the output directory contains a full `model.safetensors` plus the lightweight `robotics_velocity_head.bin`; local lightweight-head inference is only a data-flow smoke if the full `model.safetensors` is not pulled.

### Stage 1.5: Full-Checkpoint Validation Gate

Run this gate before any Stage 2 adaptation. The goal is to distinguish a real
multimodal policy from a collapsed action-prior model.

Use the full Stage 1 checkpoint on the 5090 server:

- model path: `/root/autodl-tmp/outputs/train_current_v1_real_projector_only_main`
- backbone path: `/root/autodl-tmp/models/LLaDA-V`
- eval data: `/root/project/llada-vla-go2-debug/datasets/current_v1_real_only/val.jsonl`
- image root: `/root/project/llada-vla-go2-debug/datasets/current_v1_real_only/images`

Required checks:

- full real-val evaluation with token accuracy, per-dim accuracy, dequantized MAE, and per-step MAE
- sampled prediction-vs-ground-truth rollout plots for at least 20 validation samples
- baseline comparison against constant-mean action, previous-action hold, and first-step hold
- image ablation: replace or shuffle images and verify metrics degrade
- instruction ablation: shuffle instructions and verify metrics degrade
- episode/session leakage check, especially near-duplicate adjacent frames across train/val

Pass criteria:

- full-checkpoint eval improves over simple constant/hold baselines on `vx` and `wz`
- sampled plots show non-trivial variation across time or across scenes, not only one fixed velocity chunk
- image/instruction ablations degrade at least one meaningful metric; otherwise the model is likely relying mostly on state/action priors
- no train/val image path leakage and no obvious session-level split bug
- no NaN/inf predictions and all dequantized actions remain inside tokenizer ranges

If this gate fails, do not start Stage 2. Instead inspect data split, action
distribution, prompt construction, and whether projector updates are being
loaded during inference.

### Stage 1.6: Delta + Full-Mask Policy Training

Run this when Stage 1.5 shows good teacher-forced denoising metrics but collapsed
full-mask policy decoding.

Purpose:

- align training with deployment-time full-mask action generation
- reduce absolute-action dominant-mode collapse
- weaken previous-action shortcut without removing useful state conditioning
- measure event subsets instead of only global averages

Default configuration:

- data: `datasets/current_v1_real_only`
- init checkpoint: Stage 1 full checkpoint
- target: delta action chunk
- train mode: `projector_only`
- trainable: projector, action embeddings, action heads
- frozen: vision tower and LLaDA backbone
- mask schedule: high-mask curriculum with dedicated full-mask samples
- previous-action dropout: mild first, not aggressive

Delta target contract:

- train labels use `delta[0] = action_chunk[0] - previous_action`
- train labels use `delta[i] = action_chunk[i] - action_chunk[i-1]`
- inference recovers `pred_action[0] = previous_action + pred_delta[0]`
- inference recovers `pred_action[i] = pred_action[i-1] + pred_delta[i]`
- inference must not use GT future actions for cumulative recovery

Default command:

```bash
cd /home/wxh/llada-vla-go2
INIT_CHECKPOINT_PATH=outputs/train_current_v1_real_projector_only_main \
conda run -n llada bash scripts/train_current_v1_real_stage1_6_delta.sh
```

5090 command shape:

```bash
cd /root/project/llada-vla-go2-debug
MODEL_ROOT=/root/autodl-tmp/models/LLaDA-V \
INIT_CHECKPOINT_PATH=/root/autodl-tmp/outputs/train_current_v1_real_projector_only_main \
OUTPUT_DIR=/root/autodl-tmp/outputs/train_current_v1_real_stage1_6_delta \
PYTHONPATH=$PWD:$PWD/LLaDA-V:$PWD/LLaDA-V/train \
/root/miniconda3/envs/llada-vla/bin/python LLaDA-V/train/robotics/train_velocity.py \
  --config configs/llada_vla_go2_real_stage1_6_delta.yaml \
  --model-path "$MODEL_ROOT" \
  --init-checkpoint-path "$INIT_CHECKPOINT_PATH" \
  --train-data-path datasets/current_v1_real_only/train.jsonl \
  --eval-data-path datasets/current_v1_real_only/val.jsonl \
  --image-folder datasets/current_v1_real_only/images \
  --output-dir "$OUTPUT_DIR"
```

Pass criteria:

- delta clip rate stays below warning threshold: train per-dim `<1%`, event subsets per-dim `<3%`
- full-mask policy decoding no longer has near-zero prediction std
- prediction std increase must coincide with event-subset MAE improvement
- turning / first-step-change / wz-change subsets improve over the current fixed `wz≈0` behavior
- image shuffle, instruction shuffle, or no-prev-action ablations cause visible degradation on event subsets
- do not proceed to Stage 2 if gains only appear in teacher-forced denoising eval

### Stage 1.7A: Event-Heavy Real Data Preparation

Run this after Stage 1.6 if the model no longer collapses but still loses to
the previous-action hold baseline.

Purpose:

- fix previous-action reconstruction edge cases at episode boundaries and control-time gaps
- surface `prev_action_valid/source/reason` in audit outputs
- prioritize real samples where visual feedback should force deviation from hold
- prepare a recovery-first priority manifest for Stage 1.7A training

Default command:

```bash
cd /home/wxh/llada-vla-go2
PYTHON_BIN=/home/wxh/miniconda3/envs/llada/bin/python \
bash scripts/prepare_stage1_7a_data.sh
```

Generated artifacts:

- filtered real-only dataset: `datasets/current_v1_real_only`
- audit summary with `prev_action_reasons`: `outputs/data_audit/current_v1_real_vs_sim`
- recovery-first priority manifest: `outputs/data_audit/event_rich_candidates/train/priority_samples.jsonl`

Priority ordering for new real collection:

- `recovery_correction`: highest priority
- `turning_onset`: second priority
- `start/stop/restart`: third priority

Suggested collection mix for the next real batch:

- `35%` recovery / correction
- `30%` turning onset / offset
- `20%` start / stop / restart
- `15%` generic event-heavy overflow only if the first three buckets are saturated

Do not add generic same-view contrast work to this stage.

### Stage 2: Real-Only Longer Adaptation

Only run after Stage 1.7A data prep and a refreshed validation pass show event-subset improvement.

Options:

- keep `projector_only` and train longer
- enable LoRA on the LLaDA backbone while keeping the vision tower frozen
- avoid full finetune until real validation metrics justify it

Recommended first Stage 2 pilot:

- keep the vision tower frozen
- keep action heads and action embeddings trainable
- keep projector trainable
- add LoRA to selected LLaDA backbone modules, or unfreeze only the last 2-4 LLaDA blocks
- start with a short pilot before a long run
- keep real-only data first; add sim only after Stage 2 real-val behavior is understood

### Stage 3: Controlled Sim Mixing

Do not mix sim into the first baseline. Current sim actions are more aggressive than real actions:

- real observed `vx`: about `0.0..0.3`
- sim observed `vx`: about `0.0..0.9`
- real observed `wz`: about `-0.12..0.12`
- sim observed `wz`: about `-1.4..1.4`

Only add sim after a real-only baseline exists. Compare:

- real-only
- real + low-ratio sim, for example `real:sim = 4:1`
- real + sim with explicit sim down-weighting

Accept sim only if real-val metrics do not regress.

### Sim-Only Side Lane

Use this lane only to answer whether the current VLA contract can run closed-loop
inside the simulation domain.

Data preparation:

```bash
cd /home/wxh/llada-vla-go2
PYTHON_BIN=/home/wxh/miniconda3/envs/llada/bin/python \
bash scripts/prepare_sim_only_data.sh
```

Generated artifacts:

- `datasets/current_v1_sim_only`
- `outputs/data_audit/current_v1_sim_only_prepared`

Training entrypoints:

- `bash scripts/train_current_v1_sim_smoke.sh`
- `bash scripts/train_current_v1_sim_projector_only.sh`
- `bash scripts/train_current_v1_sim_stage1_6_delta.sh`
- default sim training outputs now prefer `/mnt/simdata/llada-vla-go2/outputs`; if the mounted disk is unavailable, they fall back to repo-local `outputs/`

Evaluation entrypoint:

- `bash scripts/eval_sim_policy_validation.sh`

Hard rule:

- sim-only success only shows that the model/contract is closed-loop usable in the simulation domain
- sim-only success does **not** constitute direct evidence of real-robot usability
- do not use sim-only results to replace the real-only Stage 1.x gate

Parallel-management safety rules:

- sim wrappers print `LANE`, `DATA_ROOT`, `CONFIG_PATH`, `OUTPUT_DIR`, `MODEL_NAME`
- real wrappers print the same metadata
- real wrappers only write to `outputs/train_current_v1_real_*`
- sim wrappers only write to `outputs/train_current_v1_sim_*` or `/mnt/simdata/llada-vla-go2/outputs/train_current_v1_sim_*`
- output directories without `lane_metadata.json` are treated as unsafe and require explicit overwrite

## Known Current Data Limits

- `vy` is zero in current real and sim data, so this dataset cannot validate lateral velocity learning.
- `mode/gait_type` are currently constant in available processed data, so they are condition fields but not informative training signals yet.
- Sim `wz` can exceed the current tokenizer range `[-1.2, 1.2]`, so sim mixing can increase clamp-at-boundary supervision.
