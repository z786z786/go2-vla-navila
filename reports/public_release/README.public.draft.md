# Go2-SmolVLA-NaVILA

A research platform for vision-language navigation and high-level velocity control on Unitree Go2.

This repository presents the complete system design: Isaac Sim task generation, Go2 teleoperation collection, a shared simulation/real-data interface, data review, SmolVLA dual-box PoC, R2R-VLNCE and NaVILA-compatible navigation interfaces, and exploratory LLaDA-V (Mask Diffusion VLM) integration.

## What is included

- **Simulation and robot data platform** — Isaac Sim scene/task deployment, expert trajectory generation, Go2 state and velocity adapters, and a unified episode format.
- **Data closed loop** — collection, normalization, cleaning, review, split guards, and dataset sanity checks.
- **SmolVLA baseline** — visual-language observations mapped to Go2 high-level velocity commands; the dual-box test is the compact PoC example.
- **Navigation interface** — R2R-VLNCE conversion, NaVILA-style episode layout, action decoding, clipping, slew-rate limiting, and stop modeling.
- **Architecture studies** — historical vision, state injection, legacy/continuous experts, Flow Matching denoising steps, and the LLaDA-V multimodal backbone.

## Repository map

- `src/`, `scripts/`, `datasets/`, `evaluation/`, `tests/` — core implementation and validation.
- `deploy/`, `tools/`, `configs/` — Go2 platform adapters and runnable configuration examples.
- `docs/` and `reports/` — design notes, protocol documentation, and reproducibility records.

## Design dimensions

| Area | Variants exposed by the platform |
|---|---|
| Expert trajectory | legacy / continuous |
| Action interface | absolute velocity, clipping, slew-rate limiting |
| Observation context | current frame / historical visual context |
| Robot state | vision-language only / state injection |
| Termination | explicit stop action and terminal hysteresis |
| Flow Matching | configurable denoising step count |
| Multimodal base | SmolVLA / LLaDA-V |

## Scope

The repository documents the completed platform architecture and PoC workflow. The real-robot portion covers platform construction and data interfaces; it does not claim a successful real-robot navigation campaign. Dataset files, model weights, private machine paths, and credentials are intentionally excluded from the public release.

## License and citations

Original project code is released under Apache-2.0. Components and datasets from upstream projects retain their original licenses. Please cite SmolVLA, NaVILA, LLaDA-V, SigLIP2, R2R/VLNCE, and Matterport3D when using this work.
