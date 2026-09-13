# Processed Dataset Contract

This runbook defines the **processed dataset contract** expected by the Go2
velocity-control training lane.

Use this contract when preparing training data for:

- `bash scripts/train_llada_vla_go2.sh`
- `bash scripts/eval_llada_vla_go2.sh`
- cloud training described in `docs/runbooks/cloud_training.md`

This is the recommended training input format. Raw collector session trees such
as `real_sessions/` and `sim_sessions/` are **not** the training contract.

For the upstream raw-session tree shared by real and sim collectors, see
`docs/runbooks/raw_session_contract.md`.

## Accepted container formats

The loader currently accepts:

- `.jsonl`
- `.json` containing a list of samples

Recommended convention:

- `train.jsonl`
- `val.jsonl`
- shared image root directory

## Required sample fields

Each sample must include:

- `instruction`
- `state`
- `action_chunk`

Recommended sample shape:

```json
{
  "sample_id": "real-000123",
  "instruction": "go to the black box",
  "image_path": "images/scene_01/frame_000123.jpg",
  "state": {
    "vx": 0.0919,
    "vy": 0.0496,
    "wz": -0.0703,
    "yaw": 0.1865,
    "roll": 0.0,
    "pitch": 0.0,
    "mode": 3,
    "gait_type": 1,
    "vx_prev": 0.5366,
    "vy_prev": 0.0,
    "wz_prev": 0.0
  },
  "previous_action": {
    "vx": 0.5366,
    "vy": 0.0,
    "wz": 0.0
  },
  "prev_action_valid": true,
  "prev_action_source": "reconstructed",
  "prev_action_reason": "reconstructed",
  "action_chunk": [
    {"vx": 0.5366, "vy": 0.0, "wz": 0.0},
    {"vx": 0.5542, "vy": 0.0, "wz": 0.0},
    {"vx": 0.5548, "vy": 0.0, "wz": 0.0},
    {"vx": 0.5566, "vy": 0.0, "wz": 0.0},
    {"vx": 0.5200, "vy": 0.0, "wz": 0.1},
    {"vx": 0.4800, "vy": 0.0, "wz": 0.2}
  ],
  "action_mask": [1, 1, 1, 1, 1, 1]
}
```

## Field-level contract

### `instruction`

- type: string
- required: yes
- should be non-empty natural-language task text

### `state`

- type: object preferred
- required: yes
- recommended keys:
  - `vx`
  - `vy`
  - `wz`
  - `yaw`
  - `roll`
  - `pitch`
  - `mode`
  - `gait_type`
  - `vx_prev`
  - `vy_prev`
  - `wz_prev`

Previous-action contract:

- `vx_prev / vy_prev / wz_prev` must refer to the executed action at `t-1`
- they must not be copied from the current-step action at `t`
- recommended companion fields:
- `previous_action`
- `prev_action_valid`
- `prev_action_source`
- `prev_action_reason`

Important loader behavior:

- if `yaw_speed` is missing but `wz` exists, the loader creates `yaw_speed <- wz`
- if `mode` / `gait_type` are missing, they fall back to `None`
- dict-form state is the recommended format

Legacy-compatible but not recommended:

- list/tuple state like `[vx, vy, yaw_speed]`

### `action_chunk`

- type: list
- required: yes
- semantic shape: horizon `K` of `(vx, vy, wz)` actions
- recommended representation: list of dicts with explicit keys

Accepted representations:

- `[{"vx": ..., "vy": ..., "wz": ...}, ...]`
- `[[vx, vy, wz], ...]`

Legacy-compatible but not recommended:

- `[[vx, wz], ...]`
- loader compatibility path maps this to `[vx, 0.0, wz]`

### `action_mask`

- type: list of length `K`
- recommended: yes
- required by contract: yes

Semantics:

- `1` means this step is supervised / valid
- `0` means padding step

Current default horizon comes from `velocity_control.horizon_k` and is `6` in
`configs/llada_vla_go2.yaml`.

If `action_mask` is omitted, the loader falls back to `chunk_length` or the raw
chunk length, but this is compatibility behavior only.

### `previous_action`

- type: object
- recommended: yes
- semantic meaning: explicit previous executed command `(vx, vy, wz)` for the current sample

Recommended shape:

```json
{"vx": 0.5366, "vy": 0.0, "wz": 0.0}
```

Companion fields:

- `prev_action_valid: bool`
- `prev_action_source: "logged" | "reconstructed" | "warmup_zero"`
- `prev_action_reason: "logged" | "reconstructed" | "episode_boundary" | "index_gap" | "timestamp_gap"`

Recommended semantics:

- `logged`: previous action already existed in the raw collector record
- `reconstructed`: previous action was reconstructed from the previous valid control step within the same episode
- `warmup_zero`: no trustworthy previous action was available, so zeros were emitted and `prev_action_valid=false`

Important export rule:

- reconstruction must not cross episode boundaries
- reconstruction must not bridge clear control-time gaps / dropped-step gaps
- when invalid, `prev_action_reason` should explain whether the zero came from episode start, index discontinuity, or timestamp discontinuity

### Image path fields

Accepted lookup order:

- `image`
- `image_path`
- `source_image_path`

Recommended field:

- `image_path`

Recommended image-root strategy:

- dataset rows contain image paths relative to `image_root`
- training scripts set `IMAGE_FOLDER` / `paths.image_root` to that root

The loader also supports:

- absolute image paths
- collector-like paths containing `sessions/`, `episodes/`, or `images/`

### Optional depth fields

Depth is currently treated as an optional extension field for exploration and is
not part of the default velocity-control training contract.

Recommended raw-session fields when per-frame depth exists:

- `depth_image`
- `meta.depth_format`
- `meta.depth_scale`
- `meta.depth_min`
- `meta.depth_max`

For real D435i collection, depth may also exist as a session-level sidecar:

- episode-level `d435i_capture`
- session-level `d435i/d435i_capture.json`
- session-level `d435i/d435i_rgbd.bag`

Important alignment note:

- For Go2, the public `unitree_sdk2` repository currently exposes the RGB video API shape via `go2::VideoClient::GetImageSample(...)`
- Depth is **not** currently exposed in that upstream repo as a public Go2 SDK2 camera API
- Therefore any `depth_image` field should be treated as a project-local extension (for example sim-only RGB-D probes), not as an official SDK2-compatible field

## Normalization behavior used by the loader

The training loader converts each sample into:

- prompt text containing instruction + serialized state + action placeholders
- fixed `K x 3` continuous action chunk
- flattened action labels of shape `3K`
- `action_mask` of shape `[K]`
- `action_valid_mask` of shape `[3K]`

Prompt-side action targets are represented as a tail block of mask tokens, so
training loss is applied to the action-token region only.

## Recommended dataset layout

Recommended processed dataset directory on a server:

```text
/mnt/data/processed/
  train.jsonl
  val.jsonl
  images/
```

Recommended workspace entries:

```yaml
paths:
  train_data_path: /mnt/data/processed/train.jsonl
  eval_data_path: /mnt/data/processed/val.jsonl
  image_root: /mnt/data/processed/images
```

## Minimal validation checklist

Before launching training, confirm:

1. `train.jsonl` exists
2. `val.jsonl` exists
3. every row has `instruction`, `state`, `action_chunk`
4. image paths resolve under `image_root`
5. `action_mask` length matches `horizon_k`
6. action rows are explicit 3D `(vx, vy, wz)` whenever possible
7. `previous_action` and `vx_prev / vy_prev / wz_prev` refer to `t-1`, not `t`
8. `prev_action_source` and `prev_action_reason` distributions are visible in exported dataset stats

## Compatibility notes

The loader currently still accepts older collector-derived shapes so migration
can be incremental, including:

- `control_action` / `raw_action` fallback for previous command extraction when explicit previous-action fields are absent
- `chunk_length` fallback when `action_mask` is absent
- `source_image_path` fallback for image resolution
- compact state arrays
- old 2D `[vx, wz]` action chunks

These are migration paths only. New processed datasets should target the
recommended contract above.
