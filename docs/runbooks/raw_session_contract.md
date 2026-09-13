# Unified Raw Session Contract

This document defines the raw-session tree that both:

- `collectors/real_go2/native/go2_collector`
- future `collectors/sim_go2/*`

should write before any downstream conversion.

The goal is simple: **sim raw sessions must look like real raw sessions first**.

## Directory layout

Required layout:

```text
<session_root>/
  index.json
  episodes/
    ep_000001.json
  images/
    ep_000001/
      20260422_120001_001.jpg
```

Optional extensions:

```text
<session_root>/
  images_depth/
    ep_000001/
      frame_000001.png
  d435i/
    d435i_rgbd.bag
    d435i_capture.json
```

Rules:

- `index.json` is the session manifest
- each `episodes/ep_xxxxxx.json` is a complete trajectory payload
- RGB frame paths are stored relative to `session_root`
- depth may be stored either as per-frame `depth_image` assets or as a session-level D435i sidecar bag

## `index.json` contract

Required top-level fields:

- `schema_version`
- `session_id`
- `source_type`
- `episodes`

Each episode summary entry must include:

- `episode_id`
- `instruction`
- `capture_mode`
- `task_family`
- `target_type`
- `target_label`
- `target_description`
- `collector_notes`
- `instruction_source`
- `segment_status`
- `success`
- `termination_reason`
- `scene_id`
- `operator_id`
- `source_type`
- `num_frames`
- `start_timestamp`
- `end_timestamp`

Optional episode-summary extensions:

- `active_target_id`
- `contrast_group_id`
- `contrast_variant`

## Episode payload contract

Required episode-level fields:

- `schema_version`
- `session_id`
- `episode_id`
- `instruction`
- `capture_mode`
- `task_family`
- `target_type`
- `target_label`
- `target_description`
- `collector_notes`
- `instruction_source`
- `segment_status`
- `success`
- `termination_reason`
- `scene_id`
- `operator_id`
- `source_type`
- `frames`

Allowed `source_type` values for now:

- `real_go2`
- `isaac_sim`

Optional episode-level extensions:

- `d435i_capture`
- sim-specific layout / visibility fields
- `active_target_id`
- `contrast_group_id`
- `contrast_variant`
- `scene_targets`

### `d435i_capture`

For real sessions with D435i enabled, store a session-sidecar reference like:

```json
{
  "capture_metadata_path": "d435i/d435i_capture.json",
  "bag_path": "d435i/d435i_rgbd.bag",
  "depth_topic_semantics": "raw_unaligned_depth"
}
```

This is a session-level extension. It is **not** a replacement for per-frame `depth_image`.

Allowed `depth_topic_semantics` values for real D435i capture:

- `raw_unaligned_depth`
- `aligned_depth_to_color`

## Frame contract

Each frame must include:

- `timestamp`
- `image`
- `instruction`
- `state`
- `raw_action`
- `control_action`
- `previous_action`
- `meta`

### `state`

Required keys:

- `vx`
- `vy`
- `vz`
- `wz`
- `yaw`
- `roll`
- `pitch`
- `x`
- `y`
- `z`
- `body_height`
- `error_code`
- `mode`
- `gait_type`

### `raw_action`

Required keys:

- `vx`
- `vy`
- `wz`
- `camera_pitch`
- `keys`

### `control_action`

Required keys:

- `vx`
- `vy`
- `wz`

### `previous_action`

Required keys:

- `vx`
- `vy`
- `wz`

Semantics:

- it must refer to executed command at `t-1`
- it must not be copied from current-step command at `t`

### `meta`

Required keys:

- `schema_version`
- `session_id`
- `episode_id`
- `source_type`
- `camera_interface_source`
- `capture_mode`
- `task_family`
- `target_type`
- `target_label`
- `target_description`
- `instruction_source`
- `segment_status`
- `success`
- `termination_reason`
- `scene_id`
- `operator_id`
- `state_timestamp`
- `action_timestamp`
- `raw_action_timestamp`
- `control_action_timestamp`
- `image_timestamp`

Optional frame-level extensions:

- `depth_image`
- `meta.depth_interface_source`
- `meta.depth_format`
- `meta.depth_scale`
- `meta.depth_min`
- `meta.depth_max`
- sim-specific labels such as layout / visibility / turn buckets
- `meta.active_target_id`
- `meta.contrast_group_id`
- `meta.contrast_variant`

## Depth rules

Two depth modes are allowed by contract:

1. Per-frame depth asset
   - frame carries `depth_image`
   - image file lives under `images_depth/`
2. Session-sidecar depth capture
   - episode carries `d435i_capture`
   - session carries `d435i/d435i_capture.json` + rosbag

Sim collectors should prefer mode 1.
Real D435i collection currently uses mode 2.

## Sim alignment requirements

Any future sim raw collector must:

- reuse `schema_version = go2_local_dataset_v1`
- write `session_id` at both session and episode scope
- write `source_type = isaac_sim`
- write `target_label` explicitly, not just `target_type`
- keep `state/raw_action/control_action/previous_action/meta` field names identical to real collector
- use optional extensions only under clearly named keys, without changing required keys
