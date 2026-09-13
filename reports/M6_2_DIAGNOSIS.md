# M6.2 Navigation Diagnosis

## D0 — Freeze Current Baseline

**Stage:** D0  
**Status:** PASS  
**Frozen at:** 2026-09-01  
**Scope:** Freeze only. No model, dataset, codec, training, or rollout was modified.

### Purpose

This document fixes the M6.1/R4 reference point for subsequent M6.2 diagnostics. Future stages must treat every path and SHA-256 listed below as immutable input. New experiments may write only to a new, timestamped directory under `/mnt/wxh/go2_short_vln/outputs/m6_2/`; they must never overwrite the checkpoint, its training root, or the retained R4 rollout.

### Frozen checkpoint and processors

| Item | Value |
|---|---|
| checkpoint | `/mnt/wxh/go2_short_vln/outputs/m6_1/smoke_20260901T104846+0800/checkpoints/step_002000` |
| `model.safetensors` SHA-256 | `facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7` |
| `config.json` SHA-256 | `f484918159fbe8684754042df9a09f500145284c75f8f283e157bd1db9cece73` |
| preprocessor JSON SHA-256 | `78423ba2bf38e113f59139327ea7d009272f97c60855a02da2e39f5cdf11a80a` |
| postprocessor JSON SHA-256 | `47dbdc6d1be74f738990bbf2a3c66d5ddf639481482f3984fc766e3bf144e3aa` |
| preprocessor normalizer SHA-256 | `a68a196e11abbd9a8204f0665cced86776917b815053e6f408ba225800c17cb1` |
| postprocessor unnormalizer SHA-256 | `3664cb7a946d368a49a747328dc43d320e1e48643690b5a902e2d34ad582ab7a` |
| action codec SHA-256 | `143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27` |
| source M6 model SHA-256 | `8e72db4f1bf46c12dc3eb0f6e58b2cfc18d97e448dea1ecd9116d439a03145dd` |

The checkpoint is a `go2-bounded-action-codec-v1` model. Its codec uses epsilon `1e-4` and decodes to `vx=[0,0.5]`, `vy=0`, and `wz=[-0.5,0.5]`. It accepts exactly one current `512x512` front RGB observation and a 30-D state, and emits an action chunk of shape `50x3`.

### Frozen training configuration

| Field | Value |
|---|---|
| training root | `/mnt/wxh/go2_short_vln/outputs/m6_1/smoke_20260901T104846+0800` |
| source model/revision | M6 `step_002000` / `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| stage | smoke, 2,000 optimizer steps |
| seed / batch / workers | `20260831` / `1` / `2` |
| optimizer | AdamW, LR `1e-4`, betas `[0.9,0.95]`, weight decay `1e-10`, clip norm `10` |
| schedule | warmup 1,000; decay 30,000; final LR `2.5e-6` |
| action treatment | physical actions encoded before the action normalizer; bounded codec epsilon `1e-4` |
| frozen/trainable modules | vision encoder frozen; expert-only and state projection trainable |
| M6.1 report SHA-256 | `9926f6cdd726e691d967cec8bd8692778aaf385ca4e28248b55fcb6dc20e600f` |

### Frozen dataset version

| Item | Value |
|---|---|
| dataset root | `/mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1` |
| manifest format | `go2-short-vln-lerobot-m5-v1` |
| manifest SHA-256 | `d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402` |
| LeRobot | `v0.6.0`, commit `30da8e687a6dfc617fcd94afc367ac7071c376ce` |
| storage rate | 50 Hz; one physical action per source frame; chunks sampled at load time |
| train set | `short_vln_v1_0000` (427 frames) and `short_vln_v1_0004` (420 frames): 2 episodes / 847 frames |
| short-task JSON SHA-256 | `88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6` |

### Frozen inference and control configuration

| Field | Value |
|---|---|
| server checkpoint | the frozen M6.1 checkpoint above |
| policy inputs | current rotated RGB JPEG, M5 30-D state, short instruction only |
| forbidden policy inputs | reference path, waypoint, goal pose/direction/distance, PD state |
| control frequency | 50 Hz (`dt=0.02 s`) |
| policy/replan frequency | 5 Hz: one request after every 10 control frames |
| action chunk horizon | 50 frames / 1.0 s |
| execution horizon | first 10 frames / 0.2 s; remaining 40 frames discarded |
| socket / timeout / JPEG quality | `/tmp/go2_smolvla_m7.sock` / 10 s / 90 |
| episode cap / stop rule | 1,500 frames / 10 consecutive evaluator frames within 0.5 m |
| action safety bounds | `vx=[0,0.5]`, `vy=0`, `wz=[-0.5,0.5]` |
| inference seed rule | base seed `20260831`, deterministically derived from episode ID and replan index |

The policy's `num_steps=10` is its diffusion/inference configuration and is distinct from the 10-frame command execution horizon.

### Frozen failing rollout

| Item | Value |
|---|---|
| retained rollout root | `/mnt/wxh/go2_short_vln/outputs/m7/gate_m61_20260901T112400+0800` |
| failed episode | `short_vln_v1_0004` (training split) |
| episode summary SHA-256 | `04e2e0bbeedbaf27302d671d6d5b6d014b38bb2cc31370b849edc1b337a934c8` |
| first-gate report SHA-256 | `69aa5fb94f5a8f882234dd515f7cbdce3abcb1a7556f945583ede510d6f15ae2` |
| termination | `client_max_steps`, 1,500 frames / 150 requests |
| invalid raw actions | `0 / 7,500` |
| collision / pose abnormality | `0 / 0` |
| final navigation error | `1.0300504478084738 m` |
| success radius / required streak | `0.5 m` / 10 frames |
| official success / SPL | `0` / `0` |
| PD oracle result | success; final navigation error `0.45413894101248947 m` |
| conclusion | `FAIL`: infrastructure and action-range checks passed, but the policy did not enter the success radius. |

### Reproducibility chain

The retained evidence contains the exact `client_config.json`, `server_ready.json`, `preflight.json`, request/step JSONL, diagnostics plot, and a decodable 1,500-frame MP4. The original fixed-seed preflight passed with 50/50 finite, range-compliant actions, checkpoint hash `facc4a73...48e7`, and seed `2012673106`.

The reproduction command, to be used only by the later closed-loop diagnostic stage and only with a new output directory, is:

```text
bash /home/wxh/go2_short_vln/scripts/m7_run_gate.sh \
  /mnt/wxh/go2_short_vln/outputs/m6_2/<new_timestamped_run>
```

With the frozen baseline, the expected first-stage result is a nonzero gate exit after `short_vln_v1_0004` fails official success at the 1,500-frame limit. The command must not target the retained R4 directory.

### Frozen implementation hashes

| Component | SHA-256 |
|---|---|
| `src/inference/server.py` | `e9b10225a02a5641e09139d80d4d16edc54975eb61db250945c497799feb1643` |
| `src/inference/isaac_client.py` | `438534e41693edc28b17dc9667c0b62f2db3b57482360379570c1aa0ad3e2bb4` |
| `src/inference/preflight.py` | `2757d416545125f70383f9a32c68c3eebbf2ce6340886034e478e71311f99b20` |
| `src/inference/check_rollout.py` | `73f4d4deab4de23dd196f1f48a07fef3f4e2cd96b7947eb824c4e16b03cb92d7` |
| `src/inference/action_audit.py` | `e05a5daf1a8f950209d11717f6e00c7abee181c025c93818932a07220f455f31` |
| `scripts/m61_run_stage.sh` | `b14286f22da43bd1aa597fa07d67711f04ba2e9a465727d513dd923cfb2c693b` |
| `scripts/m7_run_server.sh` | `34f343621838cd0747231bfbf86918822f69836fc49e5acd2947a98dd5bc7647` |
| `scripts/m7_run_episode.sh` | `5adc9f12d7452bff048c69bc2691ee66a868c3ca7b0c06bb74d0887cdc0599b6` |
| `scripts/m7_run_gate.sh` | `7b35ebf4d76b7d5c1f561723956e1f0978c81412bb12505c9bd9d6f1af24d767` |

### D0 acceptance evidence

- Checkpoint, processors, codec, train report, dataset manifest, client config, summary, and first-gate report were all re-located and SHA-256 verified.
- The retained rollout has complete request/step logs, plot, and decoded MP4; its gate report confirms infrastructure checks passed.
- The baseline values required for D0 are recorded above.
- No current checkpoint, dataset, codec, or rollout evidence was changed. Future stages must use a new output root and compare their input hashes with this document before reporting a conclusion.

## Next recommended stage

**D1 — Action Codec Round-Trip Test.** Do not begin until the user explicitly provides `CONTINUE D1`.

## D1 — Action Codec Round-Trip Test

**Stage:** D1  
**Status:** PASS  
**Input:** all 847 frozen train actions; no model, Isaac, GPU, or checkpoint write.

The production `float32` codec executed `raw -> encode -> decode` with epsilon `0.0001`. `vx` MAE/RMSE/max-absolute-error were `2.95252347e-06` / `8.59152729e-06` / `2.50041485e-05`; `wz` values were `1.66497957e-05` / `2.88552548e-05` / `5.0008297e-05`. `vy` max error was `0`. The `vx`/`wz` absolute-magnitude ratios were `1.000015559` and `0.999937626`; `wz` sign inversions were `0/847`. Exact physical boundary actions are intentionally reconstructed inward by the finite-`atanh` epsilon contract, with permitted maxima `vx=2.6e-05` and `wz=5.1e-05`.

Artifacts: `/mnt/wxh/go2_short_vln/outputs/m6_2/d1_codec/20260901T141308+0800`; model/codec/dataset hashes remained `facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7`, `143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27`, and `d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402`. The codec preserves the training action distribution within the configured epsilon boundary contract; it is not the source of the M7 navigation failure.

**Next recommended stage:** D2 — Offline Expert vs Model Action Analysis; wait for `CONTINUE D2`.

## D2 — Offline Expert vs Model Action Analysis

**Stage:** D2  
**Status:** PASS  
**Input:** the hash-verified three-seed offline prediction archive; no Isaac, socket, planner, model inference, training, checkpoint, dataset, or codec write.

The primary comparison is the three-seed ensemble mean at the first action offset for all 847 GT training observations. Global `vx` MAE/Pearson/R²/magnitude ratio were `0.037369` / `0.861076` / `0.728022` / `0.964526`; `wz` values were `0.118126` / `0.856329` / `0.675405` / `1.068373`. `vy` remains exactly zero in both targets and predictions. `wz` sign accuracy at `|GT|>=0.05` was `0.707143`, above the majority-sign baseline `0.678571`. The global `vx`/`wz` standard-deviation ratios were `0.953344` and `1.067374`; classification flags were `{"magnitude_ratio_limits": [0.9, 1.1], "vx_magnitude_amplification": false, "vx_magnitude_shrinkage": false, "vx_variance_amplification": false, "vx_variance_shrinkage": false, "wz_magnitude_amplification": false, "wz_magnitude_shrinkage": false, "wz_variance_amplification": false, "wz_variance_shrinkage": false}`.

Artifacts: `/mnt/wxh/go2_short_vln/outputs/m6_2/d2_offline/20260901T143520+0800`. The frozen prediction archive SHA-256 remained `834cb049f2dda4bd74498ebd68129f3c3c160cbec98b45f091d9ade775d0292e` and frozen model/codec/dataset SHA-256 remained `facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7`, `143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27`, and `d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402`. The model has meaningful time-aligned training-set action signal without global action-magnitude shrinkage; D3 should isolate closed-loop execution and drift.

**Next recommended stage:** D3 — Exact Train-Episode Closed-Loop Test; wait for `CONTINUE D3`.

## D3 — Exact Train-Episode Closed-Loop Test
**Stage:** D3  **Status:** PASS  **Evidence root:** `/mnt/wxh/go2_short_vln/outputs/m6_2/d3_closed_loop/20260901T173617+0800`
Three fixed policy-noise seeds (`20260831/20260832/20260833`) were evaluated on each exact train episode with Isaac seed `20260831`. Inputs remained current RGB, 30-D state, and instruction only; no route, waypoint, goal, or PD state was sent to the policy.
- `short_vln_v1_0000`: SmolVLA official success `2/3`; final distance mean±std `1.038±1.060 m`; PD oracle success `True`, final distance `0.455 m`.
- `short_vln_v1_0004`: SmolVLA official success `0/3`; final distance mean±std `1.109±0.138 m`; PD oracle success `True`, final distance `0.454 m`.

**Classification:** `A`. At least one exact train episode was not 3/3 official-successful; prioritize underfit, action scaling, timing, and chunk execution.
Artifacts include all six request/step streams, raw and applied actions, RGB frames, decodable rollout videos, per-run M7 sanity files, seed audits, oracle metrics, and train-route comparison plots.

**Stop here.** Wait for `CONTINUE D4`.

## D4 — Observation / Action Temporal Alignment

**Stage:** D4  
**Status:** PASS  
**Evidence root:** `/mnt/wxh/go2_short_vln/outputs/m6_2/d4_alignment/20260901T192814+0800`

No hard M4→M5→loader mapping mismatch or consistent non-zero model lag was found. M4/M5 source hashes, every one of 847 anchors, 50-step chunks, padding masks, RGB fingerprint lags and D2 prediction lags are retained in the D4 artifact report.

**Stop here.** Wait for explicit direction before D5.
