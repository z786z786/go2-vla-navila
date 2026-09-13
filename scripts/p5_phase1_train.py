#!/usr/bin/env python3
"""P5-phase1 training: SmolVLM2-500M + LoRA + lm_head generating NaVILA action text.

Grown from scripts/p4_train_throughput.py, whose model construction, freeze
contract and loss masking passed P4-T6v2 and an independent review. What this
adds is what a real run needs: the shared prompt contract, mixed precision with
fp32 master weights, a shuffled resumable sampler, a warmup+cosine schedule,
checkpoints with atomic writes and resume, and a scene-disjoint holdout eval.

Frozen by the milestone doc (v3.1 #5/#6/#11/#17): LoRA r=16 alpha=32 dropout 0.05
on text-tower q_proj/v_proj + trainable lm_head; vision tower and connector
frozen; 8 history frames x 64 tokens; instruction capped at 160 tokens; loss on
the answer tokens only; NaVILA oversampling kept.

Two corrections relative to the throughput probe, both found while converting it:
1. Prompt format. The probe fed [512 vision embeddings][instruction]\\n[action]
   with no image delimiters, no chat template and no end-of-answer token, so
   greedy decoding would never stop by itself. Formal training uses
   src/smolvla/phase1_prompt.py, which the evaluator must import too.
2. Precision. The probe loaded everything in bf16 and stepped AdamW on bf16
   lm_head weights. bf16 has 7 fraction bits (~0.8% relative resolution), so an
   update of lr * O(1) on weights of magnitude ~0.03 mostly rounds to zero:
   lm_head was barely training. Trainable parameters are now fp32 master weights
   and the forward runs under bf16 autocast.

Data split: reports/p4/t2_scene_split.json (P4-T2). Train = 56 scenes / 318,331
records, holdout = 5 other scenes / 35,563 records. The closed-loop benchmark
(configs/benchmark_dev100.json) is a third, separate set of scenes.
"""
from __future__ import annotations

import argparse, json, math, os, random, signal, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # local `datasets` pkg

os.environ.setdefault("HF_HOME", "/mnt/wxh/go2_short_vln/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Tokenization happens inside DataLoader workers; a fast tokenizer that already
# spun its own thread pool in the parent can deadlock after fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GO2_DATA_ROOT", "/mnt/wxh/go2_short_vln"))
INDEX = Path(os.environ.get("GO2_INDEX", DATA_ROOT / "data/navila_dataset/R2R/index/t2_records.jsonl"))
FRAMES = Path(os.environ.get("GO2_FRAMES", DATA_ROOT / "data/navila_dataset/R2R/train"))
MODEL = Path(os.environ.get("GO2_MODEL",
             DATA_ROOT / "cache/huggingface/hub/"
             "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
             "7b375e1b73b11138ff12fe22c8f2822d8fe03467"))
SPLIT = ROOT / "reports/p4/t2_scene_split.json"
TB_ROOT = Path(os.environ.get("GO2_TB_ROOT", "/root/tf-logs"))
EXPECTED_TRAINABLE_M = 48.947          # LoRA 1.638 M + lm_head 47.309 M (P4-T6v2)
EXPECTED_RECORDS = {"train": 318331, "holdout": 35563}

from src.smolvla.phase1_prompt import Phase1PromptBuilder, frames_to_pixel_values   # noqa: E402

STOP = False


def _on_signal(signum, _frame):
    global STOP
    STOP = True
    print(f"signal {signum}: will checkpoint after the current step and exit", flush=True)


# ----------------------------------------------------------------------------- model

def build_model():
    """Same freeze contract as P4-T6v2, with fp32 master weights for what trains."""
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16)
    # task_type omitted on purpose: text_model is a bare LlamaModel (see P4-T6v2).
    model.model.text_model = get_peft_model(
        model.model.text_model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))

    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
    for p in model.lm_head.parameters():
        p.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for _, p in trainable)
    assert abs(n_train / 1e6 - EXPECTED_TRAINABLE_M) < 0.5, f"trainable {n_train/1e6:.3f} M"
    assert all(p.dtype == torch.float32 for _, p in trainable), "master weights must be fp32"
    assert sum(p.numel() for n, p in model.named_parameters()
               if "vision_model" in n and not p.requires_grad) > 80e6, "vision not frozen"
    assert not any(p.requires_grad for p in model.model.connector.parameters()), "connector trains"
    info = {"trainable_params_m": round(n_train / 1e6, 3),
            "lora_params_m": round(sum(p.numel() for n, p in trainable if "lora_" in n) / 1e6, 3),
            "lm_head_params_m": round(model.lm_head.weight.numel() / 1e6, 3),
            "trainable_dtype": "float32", "frozen_dtype": "bfloat16"}
    return processor, model, info


def forward_answer_logits(model, batch, device, image_token_id):
    """Logits at the positions that predict answer tokens, plus their labels.

    Vision features replace the <image> placeholder embeddings in order, which is
    exactly what SmolVLMModel.inputs_merger does; doing it here lets lm_head run
    on the ~12 answer positions instead of all ~650 (49,280-way logits).
    """
    ids = batch["ids"].to(device, non_blocking=True)
    attn = batch["attn"].to(device, non_blocking=True)
    sup = batch["sup"].to(device, non_blocking=True)
    with torch.no_grad():
        vis_dtype = next(model.model.vision_model.parameters()).dtype
        px = frames_to_pixel_values(batch["frames"].to(device, non_blocking=True)).to(vis_dtype)
        hidden = model.model.vision_model(pixel_values=px.flatten(0, 1)).last_hidden_state
        vis = model.model.connector(hidden)                       # [B*8, 64, 960]
    emb = model.model.text_model.get_input_embeddings()(ids)
    img = ids == image_token_id
    if int(img.sum()) != vis.shape[0] * vis.shape[1]:
        raise AssertionError(f"{int(img.sum())} <image> slots vs {vis.shape[0] * vis.shape[1]} features")
    emb = emb.masked_scatter(img.unsqueeze(-1), vis.to(emb.dtype))
    out = model.model.text_model(inputs_embeds=emb, attention_mask=attn, use_cache=False)
    mask = sup[:, 1:]                                             # position t predicts t+1
    logits = model.lm_head(out.last_hidden_state[:, :-1][mask]).float()
    labels = ids[:, 1:][mask]
    rows = mask.nonzero()[:, 0]
    return logits, labels, rows


# ------------------------------------------------------------------------------ data

def collate_fn(builder, pad_id, shuffle_instructions=False):
    """Right-padded batch. With shuffle_instructions, each row gets the next row's
    instruction (a within-batch roll) - the offline form of the v3.1 #14 control."""
    def fn(items):
        instrs = [x["instruction_raw"] for x in items]
        differs = [True] * len(items)
        if shuffle_instructions:
            rolled = instrs[1:] + instrs[:1]
            differs = [a.strip() != b.strip() for a, b in zip(instrs, rolled)]
            instrs = rolled
        seqs, starts, truncated = [], [], 0
        for instr, x in zip(instrs, items):
            p = builder.prompt(instr)
            truncated += p.instruction_truncated
            seqs.append(p.ids + builder.target(x["action_text"]))
            starts.append(len(p.ids))
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        sup = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
            sup[i, st:len(s)] = True
        return {"ids": ids, "attn": attn, "sup": sup, "frames": torch.stack([x["frames"] for x in items]),
                "action_id": torch.tensor([int(x["record"]["action_id"]) for x in items]),
                "action_text": [x["action_text"] for x in items], "instruction": instrs,
                "differs": torch.tensor(differs), "truncated": truncated, "width": width}
    return fn


class ResumableEpochSampler(Sampler):
    """Seeded permutation per epoch; can start mid-epoch so resume never repeats data."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_position(self, epoch, start):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def load_splits(ds_cls):
    split = json.loads(SPLIT.read_text())
    train_v, hold_v = set(map(str, split["train"]["videos"])), set(map(str, split["holdout"]["videos"]))
    assert not train_v & hold_v, "split leaks videos"
    train = ds_cls(INDEX, FRAMES, exclude_videos=hold_v)
    holdout = ds_cls(INDEX, FRAMES, exclude_videos=train_v)
    got = {"train": len(train), "holdout": len(holdout)}
    assert got == EXPECTED_RECORDS, f"split sizes {got} != {EXPECTED_RECORDS}"
    return train, holdout, split["holdout"]["majority_class_share_percent"]


# ------------------------------------------------------------------------------ eval

@torch.no_grad()
def teacher_forced_eval(model, loader, device, image_token_id, max_batches=None):
    """Answer loss and exact match. Exact match under teacher forcing equals greedy
    decoding reproducing the target: greedy emits the target iff argmax is right at
    every answer position given the correct prefix. greedy_check verifies this."""
    model.eval()
    tot_loss = tot_tok = 0.0
    n = exact = 0
    per_class = torch.zeros(10, 2)                     # [correct, count] by action_id
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        keep = batch["differs"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, labels, rows = forward_answer_logits(model, batch, device, image_token_id)
        tot_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        tot_tok += labels.numel()
        wrong = torch.zeros(batch["ids"].shape[0], device=device).index_add_(
            0, rows, (logits.argmax(-1) != labels).float())
        ok = (wrong == 0).cpu() & keep
        n += int(keep.sum()); exact += int(ok.sum())
        per_class.index_add_(0, batch["action_id"][keep], torch.stack([ok[keep].float(),
                             torch.ones(int(keep.sum()))], 1))
    model.train()
    return {"loss": tot_loss / max(1, tot_tok), "exact_match": exact / max(1, n), "n": n,
            "per_class_exact": {int(i): round(float(c[0] / c[1]), 4) for i, c in enumerate(per_class) if c[1] > 0}}


@torch.no_grad()
def greedy_check(model, dataset, indices, builder, device, max_new_tokens=24):
    """Real autoregressive decoding on a few holdout records, parsed by the frozen
    parser. No KV cache: slow but has no second code path that could disagree."""
    from actions.language_parser import LanguageParser
    parser = LanguageParser("layered")
    model.eval()
    rows = []
    for idx in indices:
        item = dataset[idx]
        ids = torch.tensor([builder.prompt(item["instruction_raw"]).ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            px = frames_to_pixel_values(item["frames"].to(device))[None].to(
                next(model.model.vision_model.parameters()).dtype)
            vis = model.model.connector(model.model.vision_model(pixel_values=px.flatten(0, 1)).last_hidden_state)
            emb_fn = model.model.text_model.get_input_embeddings()
            emb = emb_fn(ids)
            emb = emb.masked_scatter((ids == builder.image_token_id).unsqueeze(-1), vis.to(emb.dtype))
            out_ids = []
            for _ in range(max_new_tokens):
                h = model.model.text_model(inputs_embeds=emb, use_cache=False).last_hidden_state[:, -1]
                nxt = int(model.lm_head(h).float().argmax(-1))
                out_ids.append(nxt)
                if nxt == builder.eou_id:
                    break
                emb = torch.cat([emb, emb_fn(torch.tensor([[nxt]], device=device))], 1)
        text = builder.decode_answer(out_ids)
        res = parser.parse(text)
        # Same record through the teacher-forced path: the two must agree, or the
        # exact_match reported on the whole subset does not mean what it claims.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batch = collate_fn(builder, pad_id=0)([item])
            logits, labels, _ = forward_answer_logits(model, batch, device, builder.image_token_id)
        tf_exact = bool((logits.argmax(-1) == labels).all())
        greedy_exact = out_ids == builder.target(item["action_text"])
        rows.append({"video_id": item["video_id"], "target": item["action_text"], "generated": text,
                     "terminated_by_eos": builder.eou_id in out_ids,
                     "match": greedy_exact, "teacher_forced_exact": tf_exact,
                     "parse": res.classification})
    model.train()
    n = max(1, len(rows))
    return {"n": len(rows), "match_rate": sum(r["match"] for r in rows) / n,
            "greedy_teacher_forced_agreement": sum(r["match"] == r["teacher_forced_exact"] for r in rows) / n,
            "eos_rate": sum(r["terminated_by_eos"] for r in rows) / n,
            "parse_error_rate": parser.parse_error_rate, "examples": rows}


# ------------------------------------------------------------------------ checkpoint

def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _atomic_dir(tmp: Path, final: Path):
    """Swap tmp into final so that final is always either the old or the new whole dir."""
    prev = final.with_name(".prev_" + final.name)
    shutil.rmtree(prev, ignore_errors=True)
    if final.exists():
        final.rename(prev)
    tmp.rename(final)
    shutil.rmtree(prev, ignore_errors=True)


def _write_weights(dir_: Path, model, state, step):
    from safetensors.torch import save_file
    save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad},
              str(dir_ / "trainable.safetensors"))
    (dir_ / "trainer_state.json").write_text(json.dumps({**state, "step": step,
                                                         "trainable_names": trainable_names(model)}, indent=1))


def save_checkpoint(out_dir, model, opt, sched, step, state, keep_step_dir):
    """Two kinds of checkpoint (user decision 2026-09-13):

    step_XXXXXXX/  weights + trainer_state, written every 0.4 epoch, ALL kept.
    latest/        weights + optimizer + scheduler + RNG, rewritten at every save
                   and on SIGTERM; the only directory resume reads.
    Keeping optimizer state only in latest/ saves ~365 MB per kept checkpoint.
    """
    written = []
    if keep_step_dir:
        tmp = out_dir / f".tmp_step_{step:07d}"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        _write_weights(tmp, model, state, step)
        _atomic_dir(tmp, out_dir / f"step_{step:07d}")
        written.append(out_dir / f"step_{step:07d}")
    tmp = out_dir / ".tmp_latest"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    _write_weights(tmp, model, state, step)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "python": random.getstate()}}, tmp / "optim.pt")
    _atomic_dir(tmp, out_dir / "latest")
    written.append(out_dir / "latest")
    return written


def find_resume_dir(out_dir):
    """latest/, or .prev_latest/ if a crash landed between the two renames."""
    for name in ("latest", ".prev_latest"):
        d = out_dir / name
        if (d / "optim.pt").exists() and (d / "trainer_state.json").exists():
            return d
    return None


def load_checkpoint(ckpt, model, opt, sched):
    from safetensors.torch import load_file
    st = json.loads((ckpt / "trainer_state.json").read_text())
    if st["trainable_names"] != trainable_names(model):
        raise ValueError("checkpoint parameter order differs; optimizer state would misalign")
    weights = load_file(str(ckpt / "trainable.safetensors"))
    params = dict(model.named_parameters())
    for n, t in weights.items():
        params[n].data.copy_(t.to(params[n].device))
    o = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(o["optimizer"]); sched.load_state_dict(o["scheduler"])
    torch.set_rng_state(o["rng"]["torch"]); torch.cuda.set_rng_state_all(o["rng"]["cuda"])
    random.setstate(o["rng"]["python"])
    return st


def build_optimizer(model, lr_lora, lr_head, weight_decay):
    """Two param groups. lm_head must get its own lr, never LoRA's; the preflight
    check (scripts/p5_preflight_checks.py) verifies this on the returned object."""
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = list(model.lm_head.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": lr_lora, "name": "lora"},
                             {"params": head, "lr": lr_head, "name": "lm_head"}],
                            betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-8)
    return opt, lora, head


def make_train_loader(train_ds, sampler, batch_size, workers, collate):
    """drop_last=True: steps_per_epoch = len(train_ds) // batch_size exactly."""
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=False,
                      prefetch_factor=4 if workers else None)


def resolve_schedule(steps_per_epoch, epochs, max_steps, save_every_epochs, eval_every_epochs,
                     save_every=None, eval_every=None):
    total = int(epochs * steps_per_epoch)
    if max_steps is not None:
        total = min(total, max_steps)
    save_every = save_every or round(save_every_epochs * steps_per_epoch)
    eval_every = eval_every or round(eval_every_epochs * steps_per_epoch)
    if save_every % eval_every:
        raise ValueError(f"save_every {save_every} must be a multiple of eval_every {eval_every}")
    return total, save_every, eval_every


# ------------------------------------------------------------------------------ main

def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "MISSING"
    except Exception:
        return "MISSING"


def code_sha256():
    """The remote copy has no .git, so git_rev alone cannot say what code ran."""
    import hashlib
    files = ["scripts/p5_phase1_train.py", "src/smolvla/phase1_prompt.py", "datasets/navila_r2r.py",
             "actions/language_parser.py", "actions/nav_command.py", "reports/p4/t2_scene_split.json"]
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "checkpoints")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="cap for smoke tests")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--save-every-epochs", type=float, default=0.4,
                    help="weights-only checkpoint, all kept; full holdout eval at each")
    ap.add_argument("--eval-every-epochs", type=float, default=0.2, help="subset eval cadence")
    ap.add_argument("--save-every", type=int, default=None, help="step override (smoke tests)")
    ap.add_argument("--eval-every", type=int, default=None, help="step override (smoke tests)")
    ap.add_argument("--eval-records", type=int, default=2048, help="fixed holdout subset per eval")
    ap.add_argument("--greedy-records", type=int, default=32)
    ap.add_argument("--full-eval-limit", type=int, default=None,
                    help="cap the full holdout eval (smoke tests only; formal runs use all records)")
    ap.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: this script must run on a real GPU")

    signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGINT, _on_signal)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"
    out_dir = args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets.navila_r2r import NavilaR2RDataset
    processor, model, param_info = build_model()
    model.to(device).train()
    tok = processor.tokenizer
    builder = Phase1PromptBuilder(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    train_ds, hold_ds, hold_majority = load_splits(NavilaR2RDataset)
    steps_per_epoch = len(train_ds) // args.batch_size
    total, save_every, eval_every = resolve_schedule(
        steps_per_epoch, args.epochs, args.max_steps, args.save_every_epochs, args.eval_every_epochs,
        args.save_every, args.eval_every)
    warmup = max(1, int(args.warmup_ratio * total))   # schedule spans ALL epochs, not one

    opt, lora, head = build_optimizer(model, args.lr_lora, args.lr_head, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, total))

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup,
                   "save_every_steps": save_every, "eval_every_steps": eval_every,
                   "train_records": len(train_ds), "holdout_records": len(hold_ds),
                   "holdout_majority_prior_percent": hold_majority, "git_rev": git_rev(),
                   "code_sha256": code_sha256(),
                   "prompt_template_tokens": builder.template_tokens, "params": param_info,
                   "model": str(MODEL), "index": str(INDEX)})

    step = 0
    resume_dir = find_resume_dir(out_dir) if args.resume == "auto" else None
    if resume_dir is not None:
        st = load_checkpoint(resume_dir, model, opt, sched)
        step = st["step"]
        for k in ("total_steps", "batch_size", "save_every_steps", "eval_every_steps", "warmup_steps"):
            if st["config"][k] != config[k]:
                raise SystemExit(f"resume config mismatch on {k}: {st['config'][k]} != {config[k]}")
        print(f"resumed from {resume_dir} at step {step}", flush=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(TB_ROOT / args.run_name), purge_step=step if step else None)
    if step == 0:
        writer.add_text("config", "```\n" + json.dumps(config, indent=1) + "\n```", 0)

    sampler = ResumableEpochSampler(len(train_ds), args.seed)
    g = torch.Generator().manual_seed(args.seed)
    eval_idx = torch.randperm(len(hold_ds), generator=g)[:args.eval_records].tolist()
    greedy_idx = eval_idx[:args.greedy_records]
    eval_loader = lambda shuffled: DataLoader(
        Subset(hold_ds, eval_idx), batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_fn(builder, pad_id, shuffle_instructions=shuffled), shuffle=False)
    # On resume, keep earlier evals: rewriting the file from an empty list would
    # silently drop every evaluation before the interruption.
    hist_path = out_dir / "eval_history.json"
    history = json.loads(hist_path.read_text()) if step and hist_path.exists() else []
    history = [h for h in history if h["step"] <= step]

    def evaluate(tag_step, full=False):
        """Every eval point: fixed 2048-record subset, its shuffled-instruction control,
        and greedy decoding. Checkpoint points (full=True) add the whole holdout, so
        every kept checkpoint carries its own full-holdout numbers."""
        t0 = time.time()
        res = {"subset": teacher_forced_eval(model, eval_loader(False), device, builder.image_token_id),
               "subset_shuffled_instruction": teacher_forced_eval(model, eval_loader(True), device,
                                                                  builder.image_token_id),
               "greedy": greedy_check(model, hold_ds, greedy_idx, builder, device)}
        if full:
            full_set = hold_ds if args.full_eval_limit is None else Subset(hold_ds, range(args.full_eval_limit))
            loader = DataLoader(full_set, batch_size=args.batch_size, num_workers=args.workers,
                                collate_fn=collate_fn(builder, pad_id), shuffle=False)
            res["full_holdout"] = teacher_forced_eval(model, loader, device, builder.image_token_id)
            writer.add_scalar("Eval/full_holdout_loss", res["full_holdout"]["loss"], tag_step)
            writer.add_scalar("Eval/full_holdout_exact_match", res["full_holdout"]["exact_match"], tag_step)
            writer.add_text("Eval/full_holdout_per_class_exact",
                            json.dumps(res["full_holdout"]["per_class_exact"]), tag_step)
        res.update({"step": tag_step, "epoch": round(tag_step / steps_per_epoch, 4),
                    "eval_seconds": round(time.time() - t0, 1), "majority_prior": hold_majority / 100})
        writer.add_scalar("Eval/subset_loss", res["subset"]["loss"], tag_step)
        writer.add_scalar("Eval/subset_exact_match", res["subset"]["exact_match"], tag_step)
        writer.add_scalar("Eval/majority_class_prior", hold_majority / 100, tag_step)
        writer.add_scalar("Eval/subset_shuffled_instruction_exact_match",
                          res["subset_shuffled_instruction"]["exact_match"], tag_step)
        writer.add_scalar("Eval/greedy_match_rate", res["greedy"]["match_rate"], tag_step)
        writer.add_scalar("Eval/greedy_eos_rate", res["greedy"]["eos_rate"], tag_step)
        writer.add_scalar("Eval/greedy_teacher_forced_agreement",
                          res["greedy"]["greedy_teacher_forced_agreement"], tag_step)
        writer.add_scalar("Eval/greedy_parse_error_rate", res["greedy"]["parse_error_rate"], tag_step)
        writer.add_text("Eval/greedy_examples", "\n".join(
            f"- `{r['generated']}` | target `{r['target']}` | {r['parse']}"
            for r in res["greedy"]["examples"][:8]), tag_step)
        history.append(res)
        tmp = hist_path.with_suffix(".json.tmp")         # rename is atomic; a kill mid-write
        tmp.write_text(json.dumps(history, indent=1))    # must not leave unparseable JSON
        tmp.replace(hist_path)
        print(json.dumps({k: v for k, v in res.items() if k != "greedy"} |
                         {"greedy_match_rate": res["greedy"]["match_rate"],
                          "greedy_tf_agreement": res["greedy"]["greedy_teacher_forced_agreement"]}),
              flush=True)
        return res

    is_save_point = lambda s_: s_ % save_every == 0 or s_ == total
    is_eval_point = lambda s_: s_ % eval_every == 0 or s_ == total
    if step == 0:
        evaluate(0)                                     # untrained reference point
    elif is_eval_point(step) and not any(
            h["step"] == step and ("full_holdout" in h or not is_save_point(step)) for h in history):
        # Killed after the checkpoint was written but before its eval finished:
        # the loop resumes at step+1, so this eval would otherwise never happen.
        evaluate(step, full=is_save_point(step))

    def state():
        return {"config": config, "epoch": step // steps_per_epoch}

    t_last = time.time()
    truncated_seen = 0
    saved_at = step if resume_dir is not None else None
    while step < total and not STOP:
        epoch, offset = divmod(step, steps_per_epoch)
        sampler.set_position(epoch, offset * args.batch_size)
        loader = make_train_loader(train_ds, sampler, args.batch_size, args.workers,
                                   collate_fn(builder, pad_id))
        torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, labels, _ = forward_answer_logits(model, batch, device, builder.image_token_id)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}; last checkpoint left intact")
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(lora + head, args.grad_clip))
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            truncated_seen += batch["truncated"]
            dt = time.perf_counter() - t0
            writer.add_scalar("Loss/train", float(loss), step)
            writer.add_scalar("Optimization/learning_rate", opt.param_groups[0]["lr"], step)
            writer.add_scalar("Optimization/learning_rate_lm_head", opt.param_groups[1]["lr"], step)
            writer.add_scalar("Optimization/gradient_norm_before_clip", gnorm, step)
            writer.add_scalar("Performance/step_time_s", dt, step)
            writer.add_scalar("Memory/peak_allocated_MiB", torch.cuda.max_memory_allocated() / 2**20, step)
            if step % 50 == 0:
                now = time.time()
                sps = 50 * args.batch_size / (now - t_last)
                t_last = now
                writer.add_scalar("Performance/samples_per_s_end_to_end", sps, step)
                eta_h = (total - step) * args.batch_size / sps / 3600
                print(f"step {step}/{total} ep {epoch} loss {float(loss):.4f} gnorm {gnorm:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {sps:.1f} samples/s eta {eta_h:.2f} h "
                      f"width {batch['width']} truncated_instr {truncated_seen}", flush=True)
            if is_save_point(step) or STOP:
                # Save BEFORE eval: a kill during a 20-minute full eval must not lose steps.
                paths = save_checkpoint(out_dir, model, opt, sched, step, state(),
                                        keep_step_dir=is_save_point(step))
                saved_at = step
                print(f"checkpoint {[str(p) for p in paths]}", flush=True)
            if is_eval_point(step) and not STOP:
                evaluate(step, full=is_save_point(step))
            if STOP and saved_at != step:
                # SIGTERM arrived after the save check (typically during an eval):
                # without this, the loop exits at `step` while latest/ is still at an
                # earlier step, silently losing up to one eval interval of training.
                paths = save_checkpoint(out_dir, model, opt, sched, step, state(), keep_step_dir=False)
                saved_at = step
                print(f"checkpoint {[str(p) for p in paths]}", flush=True)
            if step >= total or STOP:
                break

    if STOP:
        if saved_at != step:                            # STOP before any step ran in this process
            save_checkpoint(out_dir, model, opt, sched, step, state(), keep_step_dir=False)
        print(f"stopped at step {step}; latest/ holds step {step}; resume with the same command", flush=True)
    else:
        (out_dir / "DONE").write_text(json.dumps({"step": step, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
    writer.flush(); writer.close()


if __name__ == "__main__":
    main()
