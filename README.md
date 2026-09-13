# Go2-SmolVLA-NaVILA

A unified vision-language navigation and high-level velocity-control framework for Unitree Go2. The repository brings simulation task generation, teleoperation collection, shared episode contracts, data review, SmolVLA action interfaces, R2R/NaVILA conversion, and LLaDA-V diffusion action decoding into one project.

## Highlights

- Isaac Sim task, scene, expert trajectory, and episode-recording interfaces
- Go2 state, velocity, safety, and teleoperation adapters
- One versioned schema across simulation, real-robot capture, and mock runs
- Deterministic two-box visual PoC (`python -m go2_nav.cli demo`)
- R2R-VLNCE/NaVILA history-frame loading and split-aware data tooling
- Action clipping, slew-rate limiting, terminal hysteresis, and stop modeling
- Study configuration for historical vision, state injection, legacy/continuous experts, and Flow Matching denoising steps
- SmolVLA and model/backend contracts

## Quickstart

```bash
PYTHONPATH=. python -m go2_nav.cli demo --output artifacts/two_box
PYTHONPATH=. python -m go2_nav.cli review --input artifacts/two_box/episodes.json --output artifacts/two_box/clean.json
PYTHONPATH=. python -m go2_nav.cli study --output artifacts/study.json
```

The demo runs entirely on CPU and writes generated PPM observations plus a versioned episode record. External Isaac Sim, Unitree, checkpoint, and dataset integrations are explicit adapters selected at runtime.

## Layout

`go2_nav/` portable contracts, mock runtime, quality review, and study CLI · `collectors/` simulation and robot collection · `deploy/` Go2 adapters · `src/`, `datasets/`, `evaluation/` navigation and evaluation code · `configs/`, `docs/`, `tests/` configuration, design notes, and checks.

## Design matrix

| Dimension | Configurable variants |
|---|---|
| Expert | legacy / continuous |
| Action | clipping / slew-rate / explicit stop |
| Context | current frame / historical frames |
| State | vision-language only / state injection |
| Diffusion | configurable Flow Matching denoising steps |
| Backbone | SmolVLA / LLaDA-V |

## Licensing

Project additions are Apache-2.0. Files imported from upstream projects retain their upstream licensing and notices in `licenses/` and `THIRD_PARTY_NOTICES.md`. Cite SmolVLA, NaVILA, LLaDA-V, SigLIP2, R2R/VLNCE, Matterport3D, Isaac Sim, and Unitree SDK when using corresponding components.

## Visual demonstrations

The gallery below uses captured assets from the Go2 collection platform and the current simulation workspace. Every item is labeled by source and role.

### Real Go2 data collection platform

The collector console is shown with a real RGB frame inserted into the camera image region. The page is an offline snapshot of the operator UI; its live controls are disabled in the static preview.

![Go2 teleoperation console with a real RGB capture](media/screenshots/collector.png)

The same repaired session is encoded as a short first-person RGB capture sequence. It is real sensor data collected by the Go2 platform, not an autonomous-success claim.

<img src="media/previews/go2_first_person_rgb_capture.gif" alt="First-person Go2 RGB collection preview" width="720">

<video controls muted loop width="720"><source src="media/real/go2_first_person_rgb_capture.mp4" type="video/mp4"></video>

### Data visualization and review

The review page is available as a screenshot and an offline HTML view. Its frame slots are populated with repository-contained real capture thumbnails so the page can be inspected without the original data mount.

![Simulation and data review interface](media/screenshots/simulation-review.png)

![Action distribution from the collected dataset](media/screenshots/data-action-distribution.png)

![Episode length distribution](media/screenshots/data-episode-length.png)

[Open the offline review console](media/real/simulation_dataset_review.html)

### Current workspace simulation environment

The following images come directly from the current Isaac Sim/Go2 workspace outputs.

| Dual-box scene | NaVILA benchmark scene |
|---|---|
| ![Dual-box scene](media/simulation/dual_box_scene.png) | ![NaVILA benchmark scene](media/simulation/navila_benchmark_scene.png) |

### Dual-box instruction-following experiment

Two instruction variants are shown separately so the visual-language target switch is explicit: `A=red, B=blue → red` and `A=blue, B=red → blue`.

<img src="media/previews/dual_box_red_target.gif" alt="Dual-box red target rollout" width="520">
<img src="media/previews/dual_box_blue_target.gif" alt="Dual-box blue target rollout" width="520">

<video controls muted loop width="720"><source src="media/simulation/dual_box_red_target.mp4" type="video/mp4"></video>

<video controls muted loop width="720"><source src="media/simulation/dual_box_blue_target.mp4" type="video/mp4"></video>

### SmolVLA visual-language PoC trace

The workspace policy trace uses the same instruction → RGB observation → high-level velocity action interface exposed by `go2_nav/`.

<img src="media/previews/dual_box_policy_rollout.gif" alt="Dual-box policy rollout preview" width="520">

<video controls muted loop width="720"><source src="media/simulation/dual_box_policy_rollout.mp4" type="video/mp4"></video>

### NaVILA benchmark scene

The benchmark rollout and scene frame are included as a separate navigation demonstration.

<img src="media/previews/navila_benchmark_rollout.gif" alt="NaVILA benchmark rollout preview" width="520">

<video controls muted loop width="720"><source src="media/simulation/navila_benchmark_rollout.mp4" type="video/mp4"></video>

The SmolVLA PoC path can also be exercised locally with the deterministic two-box interface:

```bash
PYTHONPATH=. python -m go2_nav.cli demo --episodes 2 --output artifacts/two_box
```
