# Pre-M1 Storage Preflight

Audit date: 2026-08-30  
Project workspace: `/home/wxh/go2_short_vln`  
Heavy-storage root: `/mnt/wxh/go2_short_vln`  
Status: **PREFLIGHT COMPLETE — M1 NOT STARTED**

## Actions Taken

- Created `/mnt/wxh/go2_short_vln` with owner/group `wxh:wxh` and mode `775`.
- Performed read-only inspection of the existing Isaac installation, project references, disk usage, and caches.
- Created this report under the lightweight project workspace.
- Did **not** move, delete, reinstall, clean, relink, or modify the existing Isaac Sim, Isaac Lab, environments, projects, datasets, checkpoints, or caches.

At the final storage snapshot, `/` had 71 GiB free and was 92% used; `/mnt` had approximately 2.1 TiB free and was 41% used. These are dynamic values on a shared machine.

## Existing Isaac Installation

### Isaac Sim 5.1

| Property | Finding |
|---|---|
| Path | `/home/wxh/isaacsim` |
| Version | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| Disk usage | 20 GiB |
| Installation form | NVIDIA standalone/binary bundle, evidenced by bundled `kit/`, `exts/`, `extscache/`, `python.sh`, `isaac-sim.sh`, `setup_conda_env.sh`, and a self-contained Python runtime |
| Bundled Python | `/home/wxh/isaacsim/kit/python/bin/python3`, Python 3.11.13 |
| Conda status | Not installed in `base`, `llada`, or `openvla`; bundled Python reports no active `CONDA_PREFIX` |
| Pip status | No `isaacsim` distribution registered in bundled Python, system Python, or the audited Conda environments |
| System package status | No Isaac/Omniverse Debian package was found |
| Git status | Not a Git checkout |

Major on-disk components:

| Component | Size | Classification |
|---|---:|---|
| `/home/wxh/isaacsim/extscache` | 9.1 GiB | Bundled extension payload; treat as installation content, not disposable user cache |
| `/home/wxh/isaacsim/exts` | 7.1 GiB | Installed extensions; do not reclaim |
| `/home/wxh/isaacsim/kit` | 2.8 GiB | Kit runtime, including 1.6 GiB under `kit/cache`; do not alter during this preflight |
| `/home/wxh/isaacsim/standalone_examples` | 35 MiB | Installation content |
| `/home/wxh/isaacsim/docs` | 193 MiB | Installation content |
| `/home/wxh/isaacsim/extsDeprecated` | 127 MiB | Installation content; do not remove without separate compatibility review |

### Isaac Lab 2.3.1

| Property | Finding |
|---|---|
| Path | `/home/wxh/IsaacLab` |
| Repository-level version | `2.3.1` |
| Disk usage | 57 MiB |
| Installation form | Source tree copied/extracted into place; not a Git checkout |
| Conda status | No separate Isaac Lab Conda environment; none of the audited Conda environments contains `isaaclab` |
| Bundled-Python status | Editable packages are registered in the Isaac Sim bundled Python |
| Editable core version | `isaaclab 0.48.0`, plus editable `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, and `isaaclab_tasks` packages |

The repository version (`2.3.1`) and the internal extension package version (`0.48.0`) are different version namespaces, not evidence of two separate current installations.

### Symlinks and Editable-Install Chain

```text
/home/wxh/IsaacLab/_isaac_sim
    -> /home/wxh/isaacsim

/home/zxq/zxq/IsaacLab
    -> /home/wxh/IsaacLab

Isaac Sim bundled Python editable metadata
    -> /home/zxq/zxq/IsaacLab/source/...
    -> resolves through the compatibility symlink to /home/wxh/IsaacLab/source/...
```

The editable import was tested from `/tmp` and resolved to `/home/zxq/zxq/IsaacLab/source/isaaclab/isaaclab/__init__.py`, which is the current `/home/wxh/IsaacLab` tree through the compatibility symlink. Therefore, moving or deleting `/home/wxh/IsaacLab` or `/home/zxq/zxq/IsaacLab` would break the current bundled-Python Isaac Lab imports.

No other symlink under `/home/wxh` was found pointing directly at `/home/wxh/isaacsim` or `/home/wxh/IsaacLab`.

### Existing Project Dependencies

The following existing projects contain Isaac Sim/Isaac Lab API imports or configuration and must be treated as possible consumers of the current installation:

- `/home/wxh/unitree_rl_lab`
- `/home/wxh/llada-vla-go2/collectors/sim_go2`

Representative dependencies include `isaaclab`, `isaacsim`, `omni.isaac`, task registration, simulator collectors, and Go2 scene/environment code. The bounded scan found no hard-coded `/home/wxh/isaacsim` or `/home/wxh/IsaacLab` path in these two project trees, but API-level dependency is sufficient reason not to disturb the installation.

No active executable using `/home/wxh/isaacsim` was found at the final check. This is only a point-in-time observation and is not cleanup authorization.

## Cache and Reclaimability Audit

### User Cache Summary

`/home/wxh/.cache` occupied approximately 19 GiB:

| Location | Size | Reclaimability | Recommendation |
|---|---:|---|---|
| `~/.cache/uv` | 9.0 GiB | Regenerable package cache | Redirect future project use. `uv cache clean` is available but this installed uv has no `--dry-run`; do not clean without explicit authorization. |
| `~/.cache/pip` | 4.3 GiB | Regenerable HTTP/wheel cache | Safe to purge when no package installation is running; future packages may need re-download/rebuild. Redirect with `PIP_CACHE_DIR`. |
| `~/.cache/ov` | 3.3 GiB | Regenerable Omniverse client/texture cache | 2.4 GiB texture cache + 932 MiB client cache. Clear or migrate only while all Isaac/Kit processes are stopped. Redirect future project use. |
| `~/.cache/camoufox` | 1.3 GiB | Re-downloadable browser runtime/cache | Unrelated to this project. Reclaim only if the browser tool is not needed. |
| `~/.cache/ms-playwright` | 631 MiB | Re-downloadable browser runtime/cache | Unrelated to this project. Reclaim only if Playwright is not needed. |
| `~/.cache/nvidia` | 88 MiB | Regenerable GL shader cache | Low-value reclaim; clear only with GPU applications stopped. |
| `~/.cache/huggingface` | 6.1 MiB | Regenerable/model metadata cache | Too small to matter; redirect future project downloads with `HF_HOME`. |
| `~/.cache/openpi` | 4.1 MiB | Project cache | Do not mix with this project; too small to reclaim. |
| `~/.cache/warp` | 8 KiB | Regenerable | Negligible. |
| `~/.cache/torch_extensions` | 12 KiB | Regenerable | Negligible. |

Additional Omniverse/NVIDIA state:

| Location | Size | Recommendation |
|---|---:|---|
| `~/.nvidia-omniverse` | 6.8 MiB | Mostly logs; reclaimable after relevant processes stop, but too small to prioritize |
| `~/.nv/ComputeCache` | 4 KiB | Negligible |
| `/home/wxh/isaacsim/kit/cache` | 1.6 GiB | Inside the standalone installation; do not treat as ordinary `~/.cache` and do not remove during storage cleanup without a dedicated Isaac recovery test |
| `/home/wxh/isaacsim/kit/logs` | 25 MiB | Installation-local logs; low-value reclaim only after separate review |

### Conda Package Cache

| Location | Size | Status |
|---|---:|---|
| `/home/wxh/miniconda3/pkgs` | 3.0 GiB | Active shared package cache for existing environments |
| `/home/wxh/.conda/pkgs` | Missing | No storage to reclaim |

`conda clean --all --dry-run -y` reported only the following safely removable set:

- 188 tarballs: 245.2 MiB
- 43 unused packages: 296.7 MiB
- 1 index cache
- Total quantified candidate: approximately 541.9 MiB plus the small index cache

Do not manually delete the remaining Conda package cache; packages may be hard-linked or needed by existing environments. If cleanup is later authorized, use Conda's own cleanup command and re-run the dry-run immediately beforehand.

### Prioritized Reclaim Plan

No cleanup was performed. If cleanup is separately approved later, use this order:

1. Conda's dry-run-confirmed set: approximately 542 MiB.
2. Pip cache: approximately 4.3 GiB.
3. uv cache: approximately 9.0 GiB; note that no dry-run mode is supported by the installed uv CLI.
4. Omniverse user cache: approximately 3.3 GiB, only with Isaac/Kit stopped.
5. Optional unrelated browser caches: approximately 1.9 GiB, only if Camoufox/Playwright can be re-downloaded.
6. NVIDIA shader cache and logs: under 100 MiB; low priority.

The first four categories represent roughly 17 GiB of regenerable or tool-confirmed cache candidates. This estimate is not a deletion instruction and may overlap dynamically changing files. Do **not** count `isaacsim/extscache` as reclaimable cache.

## Exact Future Storage Layout

The project source, configuration, wrappers, and reports stay on `/home`; all large or regenerable material goes under `/mnt`.

### Lightweight workspace on `/home`

```text
/home/wxh/go2_short_vln/
├── .git/
├── .gitignore
├── README.md
├── configs/
│   └── storage.env
├── reports/
│   ├── ENVIRONMENT.md
│   ├── MILESTONES.md
│   └── STORAGE_PREFLIGHT.md
├── scripts/
├── src/
├── third_party/
│   ├── NaVILA-Bench/
│   ├── IsaacLab/
│   └── lerobot/
├── assets      -> /mnt/wxh/go2_short_vln/assets
├── data        -> /mnt/wxh/go2_short_vln/datasets
├── checkpoints -> /mnt/wxh/go2_short_vln/checkpoints
├── outputs     -> /mnt/wxh/go2_short_vln/outputs
├── envs        -> /mnt/wxh/go2_short_vln/envs
├── cache       -> /mnt/wxh/go2_short_vln/caches
├── downloads   -> /mnt/wxh/go2_short_vln/downloads
└── tmp         -> /mnt/wxh/go2_short_vln/tmp
```

The `.git/` entry is a future M1 project action; it does not exist yet and was not created by this preflight.

### Heavy root on `/mnt`

```text
/mnt/wxh/go2_short_vln/
├── assets/
│   ├── navila_bench/
│   │   ├── vln_ce_isaac_v1.json.gz
│   │   └── matterport_usd/
│   └── go2/
├── envs/
│   ├── conda/
│   │   ├── navila-isaac/
│   │   └── smolvla/
│   └── isaacsim/
│       └── 4.1.0/
├── datasets/
│   ├── source/
│   ├── short_vln_v1/
│   ├── expert_rollouts/
│   └── lerobot/
├── checkpoints/
│   ├── pretrained/
│   ├── go2_low_level/
│   └── smolvla/
├── outputs/
│   ├── m1/
│   ├── m2/
│   ├── m3/
│   ├── m4/
│   ├── m5/
│   ├── m6/
│   ├── m7/
│   └── m8/
├── caches/
│   ├── conda/pkgs/
│   ├── pip/
│   ├── uv/
│   ├── huggingface/
│   ├── torch/
│   ├── xdg/
│   ├── omniverse/ov/
│   ├── nvidia/
│   └── wandb/
├── downloads/
└── tmp/
```

Only `/mnt/wxh/go2_short_vln` itself exists now. None of the proposed children or workspace symlinks were created during this preflight.

## Required M1 Linkage and Environment Rules

When M1 is explicitly authorized, create the layout atomically and verify every link before downloading large content.

Project-level links:

```text
/home/wxh/go2_short_vln/assets      -> /mnt/wxh/go2_short_vln/assets
/home/wxh/go2_short_vln/data        -> /mnt/wxh/go2_short_vln/datasets
/home/wxh/go2_short_vln/checkpoints -> /mnt/wxh/go2_short_vln/checkpoints
/home/wxh/go2_short_vln/outputs     -> /mnt/wxh/go2_short_vln/outputs
/home/wxh/go2_short_vln/envs        -> /mnt/wxh/go2_short_vln/envs
/home/wxh/go2_short_vln/cache       -> /mnt/wxh/go2_short_vln/caches
/home/wxh/go2_short_vln/downloads   -> /mnt/wxh/go2_short_vln/downloads
/home/wxh/go2_short_vln/tmp         -> /mnt/wxh/go2_short_vln/tmp
```

NaVILA-specific links after its repositories and exact expected paths are confirmed from the checked-out source:

```text
/home/wxh/go2_short_vln/third_party/IsaacLab/_isaac_sim
    -> /mnt/wxh/go2_short_vln/envs/isaacsim/4.1.0

/home/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets
    -> /mnt/wxh/go2_short_vln/assets/navila_bench

/home/wxh/go2_short_vln/third_party/NaVILA-Bench/logs/rsl_rl
    -> /mnt/wxh/go2_short_vln/checkpoints/go2_low_level
```

Do not change either existing link:

```text
/home/wxh/IsaacLab/_isaac_sim -> /home/wxh/isaacsim
/home/zxq/zxq/IsaacLab        -> /home/wxh/IsaacLab
```

Create future Conda environments by explicit prefix, not by creating them in `/home/wxh/miniconda3/envs` and moving them afterward:

```text
/mnt/wxh/go2_short_vln/envs/conda/navila-isaac
/mnt/wxh/go2_short_vln/envs/conda/smolvla
```

The future `/home/wxh/go2_short_vln/configs/storage.env` should be sourced only by this project and define:

```bash
export GO2_VLN_HOME=/home/wxh/go2_short_vln
export GO2_VLN_HEAVY=/mnt/wxh/go2_short_vln
export CONDA_PKGS_DIRS=/mnt/wxh/go2_short_vln/caches/conda/pkgs
export PIP_CACHE_DIR=/mnt/wxh/go2_short_vln/caches/pip
export UV_CACHE_DIR=/mnt/wxh/go2_short_vln/caches/uv
export HF_HOME=/mnt/wxh/go2_short_vln/caches/huggingface
export HF_DATASETS_CACHE=/mnt/wxh/go2_short_vln/caches/huggingface/datasets
export TORCH_HOME=/mnt/wxh/go2_short_vln/caches/torch
export XDG_CACHE_HOME=/mnt/wxh/go2_short_vln/caches/xdg
export WANDB_CACHE_DIR=/mnt/wxh/go2_short_vln/caches/wandb
export WANDB_DIR=/mnt/wxh/go2_short_vln/outputs/wandb
export TMPDIR=/mnt/wxh/go2_short_vln/tmp
```

These variables must be project-scoped; do not add them globally to `~/.bashrc` or change existing projects. If the future Isaac 4.1 standalone ignores `XDG_CACHE_HOME` for `~/.cache/ov`, first verify its actual cache path during a smoke launch. Only then, with Isaac stopped and separate approval, migrate `~/.cache/ov` and replace it with a symlink to `/mnt/wxh/go2_short_vln/caches/omniverse/ov`.

## Storage Gates Before M1 Downloads

Before every large M1 download or installation:

1. Report expected download size, expanded size, destination, and current free space on `/mnt`.
2. Confirm destination resolves under `/mnt/wxh/go2_short_vln` with `readlink -f`.
3. Confirm `/home/wxh/isaacsim`, `/home/wxh/IsaacLab`, and their two existing symlinks are unchanged.
4. Confirm no download, package cache, model cache, dataset, checkpoint, or generated video defaults to `/home`.
5. Confirm no Isaac/Kit process is active before any cache migration or cleanup.
6. Do not begin cleanup merely because cache is reclaimable; cleanup requires separate explicit authorization.

## Preflight Acceptance

- [x] Project workspace remains `/home/wxh/go2_short_vln`.
- [x] Heavy-storage root `/mnt/wxh/go2_short_vln` exists.
- [x] Existing Isaac Sim 5.1 path, installation form, size, Python, Conda/Pip status, and internal layout inspected.
- [x] Existing Isaac Lab 2.3.1 path, size, editable-install chain, and symlinks inspected.
- [x] Possible dependencies from existing projects identified.
- [x] User, Pip, Conda, Isaac/Omniverse, and NVIDIA cache sizes inspected.
- [x] Reclaimable versus protected storage classified.
- [x] Exact `/home` to `/mnt` layout and future link rules proposed.
- [x] No existing installation, cache, environment, or project was moved, deleted, reinstalled, or modified.
- [x] M1 was not started.

Stop here and wait for explicit authorization to begin M1.
