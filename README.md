# Go2-SmolVLA-NaVILA

A complete research framework for vision-language navigation and high-level velocity control on Unitree Go2.

## Capabilities

- Isaac Sim task and scene interfaces with expert trajectory generation
- Unitree Go2 teleoperation, state, velocity, and safety adapters
- Unified simulation/real episode schema with cleaning and review pipeline
- SmolVLA dual-box PoC for visual-language to velocity control
- R2R-VLNCE and NaVILA-compatible navigation data interfaces
- Action decoding, clipping, slew-rate limiting, terminal hysteresis
- Historical vision, state injection, expert, and Flow Matching study hooks
- SmolVLA and LLaDA-V model adapter contracts

## Quickstart

```bash
python -m tools.public_demo --backend mock --episodes 2
python -m evaluation.public_report --input artifacts/mock_eval.json
```

The mock backend deterministically exercises the complete data → model → action → evaluation path. Isaac Sim, Unitree, and checkpoint-backed model adapters use the same contracts.

## Layout

`collectors/` data collection · `deploy/` Go2 adapters · `tools/` shared pipeline utilities · `src/` navigation and model interfaces · `evaluation/` metrics and reports · `configs/` examples · `tests/` validation.

## License

Project code: Apache-2.0. Upstream components and datasets retain their original licenses. See `THIRD_PARTY_NOTICES.md`.
