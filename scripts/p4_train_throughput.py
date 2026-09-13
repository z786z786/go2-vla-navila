#!/usr/bin/env python3
"""P4-T6v2: phase 1 training throughput, measured by actually training.

Deliberately NOT a synthetic probe. Rounds 1 and 2 of this task failed because
synthetic stand-ins measure the wrong thing without erroring: random tensors
replaced text embeddings, and a no_grad region stood in for a training step.
Here every timed step is a real one - real frames, real tokenizer, real
backward and optimizer step - and the loss curve doubles as a self-check: if
loss does not fall, something in the pipeline is broken.

Frozen config (milestone doc v3.1 #5/#6/#11, #17):
  LoRA r=16 alpha=32 on text-tower q_proj/v_proj, plus a trainable lm_head;
  vision tower frozen; tokenizer_max_length=160 with longest padding;
  8 history frames at 64 tokens each; bf16; AdamW; cross-entropy on the
  action_text tokens only.

Authorship: implemented by the main agent after three dispatched rounds failed,
because the codex sandbox has no GPU and could not execute what it wrote. See
milestone doc v3.20. Reviewed independently by a codex subagent.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

# The project root must precede site-packages: the local package is named
# `datasets`, which collides with the installed HuggingFace `datasets` library.
# Running this file as a script puts scripts/ on sys.path[0], not the root, so
# without this the import resolves to the HF library and navila_r2r is missing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HOME", "/mnt/wxh/go2_short_vln/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
# Data root differs between the local box (/mnt/wxh/go2_short_vln) and the remote
# GPU host (/root/autodl-tmp/go2_short_vln), so one code path serves both. The
# archive and its extraction were verified byte-identical on both. The index was
# not: this probe ran against a remote minimal rebuild (scene_id/action_id null),
# which does not affect throughput; it was replaced on 2026-09-13 (see P5 notes).
DATA_ROOT = Path(os.environ.get("GO2_DATA_ROOT", "/mnt/wxh/go2_short_vln"))
INDEX = Path(os.environ.get("GO2_INDEX",
             DATA_ROOT / "data/navila_dataset/R2R/index/t2_records.jsonl"))
FRAMES = Path(os.environ.get("GO2_FRAMES", DATA_ROOT / "data/navila_dataset/R2R/train"))
MODEL = Path(os.environ.get("GO2_MODEL",
             DATA_ROOT / "cache/huggingface/hub/"
             "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
             "7b375e1b73b11138ff12fe22c8f2822d8fe03467"))
MAX_LEN, HISTORY, TOKENS_PER_FRAME = 160, 8, 64
# AutoPanel serves TensorBoard from /root/tf-logs, a symlink to the project's
# tensorboard/ directory, so writing tensorboard/<run>/ is all it takes to appear
# alongside lora10k, zoh_full_10000_autodl_0908_v3 and no_state_expert10k_0909.
# Scalar tags below match those runs exactly so the curves overlay.
TB_ROOT = Path(os.environ.get("GO2_TB_ROOT", "/root/tf-logs"))
EXPECTED_TRAINABLE_M = 48.947   # LoRA 1.638 M + lm_head 47.31 M, measured


def smi() -> dict:
    """One nvidia-smi sample. Called outside timed regions only."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        k = ["clock_sm_mhz", "power_w", "temperature_c", "pstate"]
        return dict(zip(k, [v.strip() for v in out.split(",")]))
    except Exception as exc:
        return {"status": "MISSING", "error": str(exc)}


def build_model():
    """Load SmolVLM2, attach LoRA to the text tower, freeze everything else.

    task_type is deliberately omitted from LoraConfig: model.model.text_model is a
    bare LlamaModel, and TaskType.CAUSAL_LM makes PEFT reach for
    prepare_inputs_for_generation, which that class does not have. LoRA is applied
    to the text tower only - the vision tower also exposes q_proj/v_proj and must
    stay frozen, so wrapping the whole model would train it too.
    """
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16)
    model.model.text_model = get_peft_model(
        model.model.text_model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj"]))

    # Freeze everything, then enable exactly LoRA + lm_head. PEFT's own freezing
    # is not relied on: round 3 failed by re-enabling the whole text tower here.
    for p in model.parameters():
        p.requires_grad_(False)
    lora_n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            lora_n += p.numel()
    for p in model.lm_head.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_vision = sum(p.numel() for n, p in model.named_parameters()
                        if "vision_model" in n and not p.requires_grad)
    assert abs(trainable / 1e6 - EXPECTED_TRAINABLE_M) < 0.5, (
        f"trainable {trainable/1e6:.3f} M != expected {EXPECTED_TRAINABLE_M} M; "
        "the freeze contract is broken")
    assert frozen_vision > 80e6, "vision tower is not frozen"
    return processor, model, {
        "lora_params_m": round(lora_n / 1e6, 3),
        "lm_head_params_m": round(model.lm_head.weight.numel() / 1e6, 3),
        "trainable_params_m": round(trainable / 1e6, 3),
        "frozen_vision_params_m": round(frozen_vision / 1e6, 1),
        "total_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }


def encode(tok, instructions, actions):
    """Tokenize prompt+target and mark which label positions are action tokens.

    The instruction and the action are tokenized separately so the split point is
    exact. Round 2 concatenated them and took cross-entropy over every text token,
    which put 75.3% of the loss on predicting the instruction - the contract asks
    for cross-entropy on action_text.
    """
    seqs, splits, prompt_lens = [], [], []
    for instr, act in zip(instructions, actions):
        p = tok(instr + "\n", add_special_tokens=True)["input_ids"]
        a = tok(act, add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(p))
        # Truncate the PROMPT, never the action. Measured on the training index:
        # 5 of 10,815 distinct instructions tokenize to >= MAX_LEN and 7 leave no
        # room for the shortest action, which would empty the supervision mask and
        # take cross-entropy over an empty tensor. The target must always survive.
        a = a[:MAX_LEN - 1] if len(a) >= MAX_LEN else a
        keep = MAX_LEN - len(a)
        p = p[:keep]
        ids = p + a
        seqs.append(ids)
        splits.append(len(p))                     # first action position
    width = max(len(s) for s in seqs)             # longest padding, from the batch
    pad = tok.pad_token_id or 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    sup = torch.zeros((len(seqs), width), dtype=torch.bool)   # action positions
    for i, (s, sp) in enumerate(zip(seqs, splits)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        sup[i, sp:len(s)] = True
    assert bool(sup[:, 1:].any()), "no supervised action tokens in this batch"
    return ids, attn, sup, prompt_lens


def vision_features(model, pixels, batch_size):
    """Real vision tower -> connector. Never pooler_output."""
    flat = pixels.reshape(-1, *pixels.shape[2:])
    hidden = model.model.vision_model(pixel_values=flat, return_dict=True).last_hidden_state
    feats = model.model.connector(hidden)
    return feats.reshape(batch_size, -1, model.config.text_config.hidden_size)


def train_step(model, opt, batch, device, cached_vision=None):
    """One real training step. No no_grad anywhere in here."""
    ids, attn, sup = batch["ids"].to(device), batch["attn"].to(device), batch["sup"].to(device)
    bs = ids.shape[0]
    if cached_vision is None:
        pixels = batch["frames"].to(device, dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0
        vis = vision_features(model, pixels, bs)
    else:
        vis = cached_vision.to(device, non_blocking=True)

    text_emb = model.model.text_model.get_input_embeddings()(ids)
    embeds = torch.cat([vis, text_emb], dim=1)
    full_attn = torch.cat(
        [torch.ones(vis.shape[:2], device=device, dtype=attn.dtype), attn], dim=1)
    out = model.model.text_model(inputs_embeds=embeds, attention_mask=full_attn,
                                 use_cache=False, return_dict=True)
    # logits at text position i predict token i+1; supervise only action positions
    logits = model.lm_head(out.last_hidden_state[:, vis.shape[1]:-1]).float()
    labels, mask = ids[:, 1:], sup[:, 1:]
    n_sup, n_tok = int(mask.sum()), int(attn[:, 1:].sum())
    loss = F.cross_entropy(logits[mask], labels[mask])
    loss.backward()
    # Measured before step() so it reflects the gradients actually applied;
    # matches the Optimization/gradient_norm_before_clip tag of the earlier runs.
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None],
        max_norm=float("inf")))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach()), n_sup, n_tok, gnorm


def collate(tok):
    def fn(items):
        ids, attn, sup, plens = encode(tok, [x["instruction_raw"] for x in items],
                                       [x["action_text"] for x in items])
        return {"ids": ids, "attn": attn, "sup": sup, "prompt_lens": plens,
                "frames": torch.stack([x["frames"] for x in items])}
    return fn


def run(model, opt, tok, ds, device, batch_size, workers, steps, warmup, use_cache,
        writer=None, tag_offset=0, lr=2e-5):
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True,
                        collate_fn=collate(tok), drop_last=True, shuffle=False)
    it = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    times, e2e_times, losses, sup_tokens, all_tokens = [], [], [], 0, 0
    prompt_lens, clocks = [], []
    cache = None
    for step in range(warmup + steps):
        # End-to-end timing starts BEFORE the fetch so num_workers has something to
        # move; step timing below still isolates the GPU work. Reporting only the
        # GPU slice would make the num_workers comparison meaningless.
        t_fetch = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        prompt_lens.extend(batch["prompt_lens"])
        if use_cache:
            # Precompute this batch's frozen vision features outside the timed
            # region, mirroring an offline feature cache. Recomputed per batch so
            # features always match their own text - round 3 cached batch 0 forever.
            with torch.no_grad():
                px = batch["frames"].to(device, dtype=torch.bfloat16) / 127.5 - 1.0
                cache = vision_features(model, px, batch["ids"].shape[0]).detach()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss, n_sup, n_tok, gnorm = train_step(model, opt, batch, device, cached_vision=cache)
        torch.cuda.synchronize()
        now = time.perf_counter()
        dt, dt_e2e = now - t0, now - t_fetch
        losses.append(loss)
        if writer is not None:
            gs = tag_offset + step
            writer.add_scalar("Loss/train", loss, gs)
            writer.add_scalar("Optimization/learning_rate", lr, gs)
            writer.add_scalar("Optimization/gradient_norm_before_clip", gnorm, gs)
            writer.add_scalar("Performance/step_time_s", dt, gs)
            writer.add_scalar("Memory/peak_allocated_MiB",
                              torch.cuda.max_memory_allocated() / 2**20, gs)
        if step >= warmup:
            times.append(dt); e2e_times.append(dt_e2e)
            sup_tokens += n_sup; all_tokens += n_tok
            if len(times) % 50 == 1:          # periodic, outside the measured window
                clocks.append(smi())
    times.sort(); e2e_times.sort()
    n = len(times)
    plens = sorted(prompt_lens)
    m = len(plens)
    return {
        "batch_size": batch_size, "num_workers": workers,
        "vision_path": "precomputed_cache" if use_cache else "online_encoding",
        "warmup_steps": warmup, "timed_steps": steps,
        "mean_step_s": sum(times) / n, "p50_step_s": times[n // 2],
        "p95_step_s": times[min(n - 1, int(0.95 * n))],
        "samples_per_s": batch_size / (sum(times) / n),
        "mean_step_s_end_to_end": sum(e2e_times) / n,
        "p50_step_s_end_to_end": e2e_times[n // 2],
        "samples_per_s_end_to_end": batch_size / (sum(e2e_times) / n),
        "data_loading_share": round(1 - (sum(times) / sum(e2e_times)), 4),
        "prompt_token_lengths": {"min": plens[0], "p50": plens[m // 2],
                                 "p90": plens[min(m - 1, int(0.9 * m))],
                                 "max": plens[-1], "count": m},
        "torch_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        # Self-proof that the instruction tokens really are masked out.
        "action_token_ratio": round(sup_tokens / max(1, all_tokens), 4),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "loss_curve": [round(x, 4) for x in losses],
        "gpu_samples_during": clocks,
        "gpu_sample_after": smi(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p4/t6_throughput.json")
    ap.add_argument("--run-name", default="p4_t6v2_throughput",
                    help="TensorBoard run directory under TB_ROOT, shown in AutoPanel")
    ap.add_argument("--configs", default=None,
                    help="Explicit configs instead of the sweep, as bs:workers:path "
                         "comma-separated, e.g. 16:2:cache,16:2:online. Used to measure "
                         "combinations the sweep leaves untested - the sweep varies one "
                         "axis at a time, so their intersection is never observed.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: this script must run on a real GPU")

    from datasets.navila_r2r import NavilaR2RDataset
    device = "cuda"
    processor, model, params = build_model()
    model = model.to(device)
    tok = processor.tokenizer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01)
    ds = NavilaR2RDataset(INDEX, FRAMES)

    writer = None
    tb_dir = TB_ROOT / args.run_name
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"tensorboard: {tb_dir}")
    except Exception as exc:                      # logging must never fail the run
        print(f"tensorboard disabled: {exc}")

    report = {
        "task_id": "P4-T6v2", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_command": "python scripts/p4_train_throughput.py",
        "authorship": "implemented by the main agent (v3.20); reviewed by a codex subagent",
        "config": {"tokenizer_max_length": MAX_LEN, "pad_language_to": "longest",
                   "history_frames": HISTORY, "tokens_per_frame": TOKENS_PER_FRAME,
                   "precision": "bf16", "lora": {"r": 16, "alpha": 32,
                   "target_modules": ["q_proj", "v_proj"], "applied_to": "text_model only"},
                   "lm_head_trainable": True, "vision_encoder_frozen": True,
                   "optimizer": "AdamW lr=2e-5 betas=(0.9,0.999) wd=0.01",
                   "objective": "next-token CE on action_text tokens only"},
        "parameters": params,
        "caveats": [
            "All runs share one model and optimizer: reloading per run would cost more "
            "than the measurement. Only the first run's loss curve starts from pristine "
            "weights; later curves continue training and are not independent.",
            "shuffle=False keeps the measurement reproducible but means every run sees "
            "the same records in the same order.",
            "drop_last=True excludes the final partial batch from throughput.",
        ],
        "gpu_sample_before": smi(),
        "runs": [], "oom_boundary": {},
    }

    if args.configs:
        gstep = 0
        for spec in args.configs.split(","):
            bs_s, wk_s, path_s = spec.strip().split(":")
            bs_i, wk_i = int(bs_s), int(wk_s)
            use_cache = path_s.lower().startswith("cache")
            try:
                report["runs"].append(run(model, opt, tok, ds, device, bs_i, wk_i,
                                          args.steps, args.warmup, use_cache=use_cache,
                                          writer=writer, tag_offset=gstep))
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                report["runs"].append({"batch_size": bs_i, "num_workers": wk_i,
                                       "vision_path": "precomputed_cache" if use_cache
                                       else "online_encoding", "status": "OOM"})
            gstep += args.steps + args.warmup
        report["oom_boundary"] = {"note": "explicit --configs mode; no sweep performed"}
        finish(report, args, model, writer, tb_dir)
        return

    # 1) batch-size sweep to the OOM boundary, online encoding, workers=0
    bs, largest, gstep = 1, None, 0
    while bs <= args.max_batch:
        try:
            report["runs"].append(run(model, opt, tok, ds, device, bs, 0,
                                      args.steps, args.warmup, use_cache=False,
                                      writer=writer, tag_offset=gstep))
            gstep += args.steps + args.warmup
            largest = bs
        except torch.cuda.OutOfMemoryError:
            # An OOM can land mid-step, after backward and before step(), leaving
            # partial gradients on a model that later runs reuse. Clear them.
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            report["oom_boundary"] = {"first_oom_batch_size": bs, "largest_usable": largest}
            break
        bs *= 2
    report["oom_boundary"].setdefault("largest_usable", largest)
    report["oom_boundary"].setdefault("first_oom_batch_size", None)

    probe_bs = max(1, (largest or 1) // 2)
    # 2) online vs precomputed vision features, same batch size
    report["runs"].append(run(model, opt, tok, ds, device, probe_bs, 0,
                              args.steps, args.warmup, use_cache=True,
                              writer=writer, tag_offset=gstep))
    gstep += args.steps + args.warmup
    # 3) num_workers comparison on the online path
    report["runs"].append(run(model, opt, tok, ds, device, probe_bs, 2,
                              args.steps, args.warmup, use_cache=False,
                              writer=writer, tag_offset=gstep))
    gstep += args.steps + args.warmup

    finish(report, args, model, writer, tb_dir)


def finish(report, args, model, writer, tb_dir):
    hidden = model.config.text_config.hidden_size
    per_frame = TOKENS_PER_FRAME * hidden * 2
    report["vision_cache"] = {
        "bytes_per_frame": per_frame,
        "formula": f"{TOKENS_PER_FRAME} tokens x {hidden} hidden x 2 bytes (bf16)",
        "frames_total": 601125,
        "total_bytes_all_frames": per_frame * 601125,
        "total_gib_all_frames": round(per_frame * 601125 / 2**30, 1),
    }
    # Basis is the fastest END-TO-END run, not the fastest GPU-only one. Using
    # mean_step_s would ignore data loading, which the sweep measured at up to 52%
    # of wall time when num_workers=0.
    timed = [r for r in report["runs"] if "samples_per_s_end_to_end" in r]
    base = max(timed, key=lambda r: r["samples_per_s_end_to_end"]) if timed else report["runs"][0]
    sps = base["samples_per_s_end_to_end"]
    report["epoch_wall_clock_hours"] = {
        "samples_per_s_used": round(sps, 3), "total_samples": 353894,
        "basis": "fastest end-to-end run (includes data loading)",
        "formula": "hours = 353894 * epochs / samples_per_s_end_to_end / 3600",
        "epochs": {str(e): round(353894 * e / sps / 3600, 2) for e in (1, 2, 3)},
        "basis_run": {"batch_size": base["batch_size"], "num_workers": base["num_workers"],
                      "vision_path": base["vision_path"],
                      "gpu_only_samples_per_s": base.get("samples_per_s"),
                      "data_loading_share": base.get("data_loading_share")},
    }
    if writer is not None:
        # Text panel so the AutoPanel view is self-describing next to the older runs.
        writer.add_text("config", json.dumps(report["config"], indent=1), 0)
        writer.add_text("parameters", json.dumps(report["parameters"], indent=1), 0)
        writer.add_text("caveats", "\n".join(report["caveats"]), 0)
        writer.flush(); writer.close()
        report["tensorboard"] = {"log_dir": str(tb_dir), "run_name": args.run_name,
                                 "autopanel_path": f"/root/tf-logs/{args.run_name}",
                                 "tags": ["Loss/train", "Optimization/learning_rate",
                                          "Optimization/gradient_norm_before_clip",
                                          "Performance/step_time_s",
                                          "Memory/peak_allocated_MiB"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"out": str(args.out),
                      "largest_usable_batch": report["oom_boundary"].get("largest_usable"),
                      "action_token_ratio": base["action_token_ratio"],
                      "loss_decreased": base["loss_decreased"],
                      "epoch_hours": report["epoch_wall_clock_hours"]["epochs"]}, indent=1))


if __name__ == "__main__":
    main()
