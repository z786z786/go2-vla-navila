# Sim Go2 Collector

Isaac Sim raw-data collection code kept under the main workspace.

All sim raw-session output should follow `docs/runbooks/raw_session_contract.md`
so it stays contract-compatible with the real collector before downstream
processing.

Primary entrypoints:

- `scripts/collect_sim_box.sh`
- `scripts/collect_sim_door.sh`
- `scripts/collect_sim_box_depth_probe.sh`
- `scripts/collect_sim_door_depth_probe.sh`
- `scripts/collect_sim_dual_target_contrast_room.sh`
- `scripts/collect_sim_dual_target_contrast_main.sh`
- `scripts/collect_sim_dual_target_contrast_conference.sh`
- `scripts/collect_sim_dual_target_contrast_full.sh`
- `scripts/prepare_sim_dual_target_contrast_data.sh`
- `collectors/sim_go2/scripts/collect_raw_trajectories.py`

Environment hints:

- if Isaac Lab is not importable from the current shell, set `ISAACLAB_ROOT`
- if Unitree RL Lab is not importable from the current shell, set `UNITREE_RL_LAB_ROOT`
- `collect_raw_trajectories.py` also checks the legacy default locations:
  - `/home/zxq/zxq/IsaacLab`
  - `/home/zxq/zxq/unitree_rl_lab`

Depth probe workflow:

- use the `*_depth_probe.sh` wrappers to collect RGB + depth raw sessions
- depth assets are written under `images_depth/`
- run `collectors/sim_go2/scripts/compute_depth_stats.py --input <session_root>` to inspect value ranges and generate preview panels
- the default sim RGB-D camera now matches the D435i `aligned_depth_to_color`
  setup at `640x480` with the color-stream FOV and `depth_min=0.20`

Dual-target contrast workflow:

- the `collect_sim_dual_target_contrast_*.sh` wrappers collect office-scene RGB-D sessions with both `door` and `dark gray suitcase` present
- each contrast group reuses the same scene layout and robot start for two episodes, changing only the active target and instruction
- `prepare_sim_dual_target_contrast_data.sh` merges the four office-scene raw roots into one processed sim dataset and runs the standard audit

Interface alignment note:

- RGB is aligned to the public Go2 `unitree_sdk2` video interface shape, i.e. `go2::VideoClient::GetImageSample(std::vector<uint8_t>&)` returning compressed image samples
- depth is **not** a public `unitree_sdk2` Go2 camera interface in the current upstream repo; it is recorded here as a sim-only extension for RGB-D exploration
