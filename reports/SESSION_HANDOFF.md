# Session handoff — go2_short_vln

Written 2026-09-13 before a context compaction. Authoritative detail lives in
`CODEX_VLN_PLATFORM_MILESTONES.md` (sections v3.1–v3.25); this file is the index.

## Direction (v3)

Changed from "compare two VLN models" to **"validate VLM navigation first, then study
the action interface."** Phase 1 is **SmolVLA only** (LLaDA-V parked): SmolVLM2-500M +
LoRA + `lm_head` generating NaVILA macro-action text, trained on public NaVILA R2R data.
Phase 2 (Flow-Matching action expert on Go2 rollouts) is deferred until phase 1 results.
No numeric capability gate: the user judges rollout videos.

## Milestone status

| Item | Status |
|---|---|
| P0 foundation freeze | PASS |
| P3 data probe + P3-T1E 100% leakage join | PASS — NO-LEAKAGE; video_id ↔ train episode_id bijection |
| P4-T1 extract / T2 index+split / T3 parser / T4 frame survey / T5 gitignore | PASS |
| P1-DEV100 | **PASS, frozen** — `configs/benchmark_dev100.json`, 100 distinct routes |
| P4-T6v2 throughput | **PASS** — sizing question (v3.1 #12) closed |
| Formal phase 1 training | **RUNNING since 2026-09-13 21:53** on the remote — run `p5_phase1_lora_r2r_2ep_0913`, 2 epochs / 39,790 steps; see milestone doc §v3.25 |
| Evaluator CPU core (P1E-CPU) | **Partial** — episodes + measures pass a differential test vs official code (300 traj / 5,265 steps / 0 mismatches); coverage strengthening, loop, controls, results format still to do |
| Evaluator GPU / closed loop / video | Not started |

## Frozen numbers

- Data: 353,894 records / 288,594 distinct video_ids / 10,819 videos / 61 scenes;
  601,125 frames, all 512×512 RGB. Oversampling kept (STOP ×3, 30°/45° turns ×2).
- Majority-class prior 29.96% (oversampled distribution — cite the distribution).
- Trainable 48.947 M = LoRA 1.638 M (text tower only) + `lm_head` 47.309 M.
- Action tokens are 25.3% of text tokens; loss is masked to action tokens only.
- Throughput (remote 3090): batch 16, workers 2, online → 14.95 samples/s,
  **6.58 h/epoch**. Batch 8/workers 2 is 1.7% slower at half the memory.
  Batch 32 OOMs. Workers >2 no gain. First-run loss 7.654 → 3.446.
- NavCommand: vx∈[0,0.5], vy=0, wz∈[−π/6,+π/6], hold_steps∈{0,25,50,75}, stop.
  Duration restored (5 Hz fixed rate withdrawn). `tokenizer_max_length=160`, longest padding.

## Machines

| | Local `neurt-Precision-7920-Tower` | Remote AutoDL |
|---|---|---|
| GPU | RTX 3090, **shared** (CARLA + another user's job hold ~17 GB) | RTX 3090, **idle** |
| Role | Evaluator development | **Training only** |
| Data | `/mnt/wxh/go2_short_vln/...` | `/root/autodl-tmp/go2_short_vln/...` — archive + frames hash-identical. **Index was NOT** until 2026-09-13 20:00: the remote held a minimal rebuild with `scene_id`/`action_id` null (kept as `t2_records.jsonl.minimal_rebuild_0913_bak`); replaced by the authoritative index, SHA-256 `d8948551…a09f` verified on the remote |
| Code | this repo | `/root/autodl-tmp/go2_short_vln_repo` |
| TensorBoard | — | `/root/tf-logs` → AutoPanel, port 6007 |

**Reaching the remote:** local IPv4 egress is broken (IPv6 fine; gateway session still
online, root cause unknown). Tunnel through the Mac proxy:
`ssh -o "ProxyCommand=nc -X connect -x 127.0.0.1:17890 %h %p" -p 37141 root@connect.nmb2.seetacloud.com`.
Credentials were given in conversation and are **not** stored in any file.

## Environment traps (each has cost real time)

1. `.bashrc` returns early for non-interactive shells; the PATH export was moved above it.
   If codex reports "not installed", check `bash -c 'source ~/.bashrc; node --version'`.
2. codex sandbox is a **container with no GPU** (`Pid: 3`, rc=100). `writable_roots` must
   contain directories only — adding device files wiped all write access (reverted).
3. Local package is named `datasets`, colliding with HuggingFace `datasets`; scripts must
   put the repo root first on `sys.path`.
4. `R2R/train.tar.gz` is an **uncompressed** tar. Verify with `tar -tf`, never `gzip -t`.
5. huggingface.co is unreachable; use `hf-mirror.com` and `HF_HUB_OFFLINE=1`.
6. An idle 3090 sits at P8/210 MHz; benchmarks without warmup read ~18× slow.

## Codex dispatch

Default is now `gpt-6-astra` at `medium` effort. Rules learned:
- Don't dispatch pure-I/O, zero-decision steps.
- Don't ping a running turn.
- An author who cannot execute their code should not write GPU code; split execution from review.
- Verify any codex config change with a minimal probe before real work.
- "Runs clean" ≠ "measures correctly" — acceptance must read code, not only run it.

## Formal run (2026-09-13)

- Launch / resume: `bash /root/autodl-tmp/run_p5_phase1.sh` (copy of `scripts/p5_run_phase1_remote.sh`). It refuses to
  start if a trainer for the run is alive or the GPU already holds >1 GiB (a `kill -9` can orphan DataLoader
  workers that keep 20 GiB of CUDA memory).
- Log `/root/autodl-tmp/p5_phase1_lora_r2r_2ep_0913.log`; checkpoints `/root/autodl-tmp/go2_short_vln/checkpoints/p5_phase1_lora_r2r_2ep_0913/`
  (`step_*` weights every 0.4 epoch, all kept; `latest/` with optimizer + scheduler); TensorBoard run of the same name.
- Evaluator must import `src/smolvla/phase1_prompt.py` (prompt, 8-frame history rule, pixel normalization).

## Next steps (as of 2026-09-13 22:10)

1. Watch the formal run: a checkpoint + full-holdout eval every 0.4 epoch (~3 h apart); compare exact match
   against the shuffled-instruction control and the 28.54% holdout majority prior.
2. Evaluator CPU core: strengthen the differential test (varied routes, exact boundaries, noisy dev100,
   2 m / 1 m radii), then the decision loop (`navila_eval.py` semantics), controls, and results format.
3. Evaluator GPU integration on the local 3090, then dev100 closed-loop rollouts with video for the user's review.
