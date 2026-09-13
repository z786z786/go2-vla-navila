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

All media below is arranged as compact galleries so the platform and experiments can be inspected without opening separate links.

### Real Go2 data collection platform

<div align="center"><img src="media/screenshots/collector.png" alt="Go2 collection console with a real RGB frame" width="860"></div>

The screenshot is a static collector-console preview with a real RGB frame inserted into the camera panel. The corresponding first-person RGB capture is shown as both an inline GIF and a playable MP4.

<div align="center"><img src="media/previews/go2_first_person_rgb_capture.gif" alt="first-person RGB capture" width="420"> <img src="media/real/sample_frames/ep_000001.jpg" alt="real capture frame" width="260"></div>

<div align="center"><video controls muted loop width="680"><source src="media/real/go2_first_person_rgb_capture.mp4" type="video/mp4"></video></div>

### Data visualization and review

<div align="center"><img src="media/screenshots/data-action-distribution.png" alt="action distribution" width="420"> <img src="media/screenshots/data-episode-length.png" alt="episode lengths" width="420"></div>

<div align="center"><img src="media/screenshots/simulation-review.png" alt="data review interface" width="860"></div>

The offline review console is populated with real capture thumbnails for offline inspection.

### Current workspace simulation environment

<div align="center"><img src="media/simulation/dual_box_scene.png" alt="dual-box simulation scene" width="420"> <img src="media/simulation/navila_benchmark_scene.png" alt="NaVILA benchmark scene" width="420"></div>

### Dual-box instruction-following experiment

The two clips below are paired target variants from the current workspace: A=red, B=blue to red and A=blue, B=red to blue.

<div align="center"><img src="media/previews/dual_box_red_target.gif" alt="red target" width="360"> <img src="media/previews/dual_box_blue_target.gif" alt="blue target" width="360"></div>
<div align="center"><video controls muted loop width="420"><source src="media/simulation/dual_box_red_target.mp4" type="video/mp4"></video> <video controls muted loop width="420"><source src="media/simulation/dual_box_blue_target.mp4" type="video/mp4"></video></div>

### SmolVLA visual-language PoC

<div align="center"><img src="media/previews/dual_box_policy_rollout.gif" alt="dual-box policy trace" width="420"></div>
<div align="center"><video controls muted loop width="680"><source src="media/simulation/dual_box_policy_rollout.mp4" type="video/mp4"></video></div>

### NaVILA benchmark scene

<div align="center"><img src="media/previews/navila_benchmark_rollout.gif" alt="NaVILA benchmark rollout diagnostic" width="420"> <img src="media/simulation/navila_benchmark_scene.png" alt="NaVILA benchmark frame" width="420"></div>

<div align="center"><video controls muted loop width="680"><source src="media/simulation/navila_benchmark_rollout.mp4" type="video/mp4"></video></div>

The included benchmark rollout is a diagnostic trace: this checkpoint produced near-zero movement and timed out (success=0, SPL=0). It is retained to make the current evaluation behavior inspectable rather than presenting it as a successful result.

Run the deterministic local PoC with:

```bash
PYTHONPATH=. python -m go2_nav.cli demo --episodes 2 --output artifacts/two_box
```
