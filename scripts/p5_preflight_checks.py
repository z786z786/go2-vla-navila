#!/usr/bin/env python3
"""Three pre-launch checks requested by the user (2026-09-13), run against the
objects the training script itself builds, not against a re-implementation.

1. steps_per_epoch = 19,895 comes from drop_last=True
   (318,331 / 16 = 19,895.69; without drop_last it would be 19,896).
2. lm_head is in its own param group at 2e-5 and does not receive LoRA's 1e-4
   - checked structurally (group membership) AND behaviourally (one AdamW step
   with unit gradients moves each parameter by exactly its group's lr).
3. Holdout (5 scenes) and train (56 scenes) are scene-disjoint, computed from the
   scene_id of every record the two Dataset objects actually serve.

GPU host (needs transformers for check 2). Writes reports/p5/preflight_checks.json.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import BatchSampler


def load_train_module():
    spec = importlib.util.spec_from_file_location("p5_phase1_train", ROOT / "scripts/p5_phase1_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p5/preflight_checks.json")
    args = ap.parse_args()
    T = load_train_module()
    from datasets.navila_r2r import NavilaR2RDataset
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_sha256": T.code_sha256()}

    # ---------------------------------------------------------------- check 1
    train_ds, hold_ds, _ = T.load_splits(NavilaR2RDataset)
    bs = args.batch_size
    sampler = T.ResumableEpochSampler(len(train_ds), seed=20260913)
    sampler.set_position(0, 0)
    loader = T.make_train_loader(train_ds, sampler, bs, 2, collate=lambda x: x)
    len_loader_epoch_start = len(loader)       # measured BEFORE the sampler is moved below
    batches_drop = list(BatchSampler(sampler, bs, drop_last=True))
    batches_keep = list(BatchSampler(sampler, bs, drop_last=False))
    spe = len(train_ds) // bs
    total, save_every, eval_every = T.resolve_schedule(spe, args.epochs, None, 0.4, 0.2)
    sampler.set_position(1, 7 * bs)                       # resume 7 batches into epoch 1
    c1 = {
        "train_records": len(train_ds),
        "records_div_batch": len(train_ds) / bs,
        "script_steps_per_epoch": spe,
        "train_loader_drop_last": loader.drop_last,
        "len(train_loader)_epoch_start": len_loader_epoch_start,
        "full_batches_iterated_drop_last_true": len(batches_drop),
        "batches_if_drop_last_false": len(batches_keep),
        "last_batch_size_if_drop_last_false": len(batches_keep[-1]),
        "records_dropped_per_epoch": len(train_ds) - len(batches_drop) * bs,
        "resumed_epoch1_after_7_batches_len": len(T.make_train_loader(train_ds, sampler, bs, 2, lambda x: x)),
        "schedule": {"total_steps": total, "save_every_steps": save_every, "eval_every_steps": eval_every,
                     "save_points": [s for s in range(1, total + 1) if s % save_every == 0 or s == total],
                     "warmup_steps": max(1, int(0.03 * total))},
    }
    c1["PASS"] = (loader.drop_last is True and spe == len_loader_epoch_start == len(batches_drop) == 19895
                  and len(batches_keep) == 19896 and c1["resumed_epoch1_after_7_batches_len"] == 19895 - 7)
    report["check1_drop_last"] = c1
    print(json.dumps(c1, indent=1), flush=True)

    # ---------------------------------------------------------------- check 2
    _, model, _ = T.build_model()
    opt, lora, head = T.build_optimizer(model, args.lr_lora, args.lr_head, 0.0)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [{"name": g["name"], "lr": g["lr"], "n_tensors": len(g["params"]),
               "numel": sum(p.numel() for p in g["params"]),
               "all_names_contain_lora_": all("lora_" in names[id(p)] for p in g["params"]),
               "names_if_few": [names[id(p)] for p in g["params"]] if len(g["params"]) <= 2 else None}
              for g in opt.param_groups]
    ids = [set(map(id, g["params"])) for g in opt.param_groups]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, T.lr_lambda(warm, total))
    lr_trace = {}
    for s in range(total + 1):
        if s in (0, warm - 1, warm, total // 2, total - 1):
            lr_trace[s] = [g["lr"] for g in opt.param_groups]
        if s < total:
            sched.step()
    # Behavioural: fresh optimizer at base lr, unit grads, one step. For AdamW with
    # wd=0, step 1 moves each weight by lr * g/(|g|+eps) = lr (to ~1e-8 relative).
    opt2, lora2, head2 = T.build_optimizer(model, args.lr_lora, args.lr_head, 0.0)
    before_head = head2[0].detach().clone()
    before_lora = [p.detach().clone() for p in lora2[:4]]
    for p in lora2 + head2:
        p.grad = torch.ones_like(p)
    opt2.step()
    d_head = (head2[0].detach() - before_head).abs()
    d_lora = torch.cat([(p.detach() - b).abs().flatten() for p, b in zip(lora2[:4], before_lora)])
    c2 = {
        "param_groups": groups,
        "groups_disjoint": not (ids[0] & ids[1]),
        "groups_cover_all_trainable": (ids[0] | ids[1]) == trainable_ids,
        "lm_head_weight_in_group": [g["name"] for g, s in zip(opt.param_groups, ids)
                                    if id(model.lm_head.weight) in s],
        "lr_trace_step_to_[lora,lm_head]": lr_trace,
        "lm_head_to_lora_lr_ratio_constant": all(abs(v[1] / v[0] - args.lr_head / args.lr_lora) < 1e-12
                                                 for v in lr_trace.values() if v[0] > 0),
        "one_step_update_abs": {"lm_head_mean": float(d_head.mean()), "lm_head_max": float(d_head.max()),
                                "lora_mean": float(d_lora.mean()), "lora_max": float(d_lora.max())},
    }
    c2["PASS"] = (c2["groups_disjoint"] and c2["groups_cover_all_trainable"]
                  and c2["lm_head_weight_in_group"] == ["lm_head"]
                  and groups[1]["lr"] == args.lr_head and groups[0]["lr"] == args.lr_lora
                  and groups[0]["all_names_contain_lora_"] and groups[1]["n_tensors"] == 1
                  and abs(c2["one_step_update_abs"]["lm_head_mean"] / args.lr_head - 1) < 1e-3
                  and abs(c2["one_step_update_abs"]["lora_mean"] / args.lr_lora - 1) < 1e-3
                  and c2["lm_head_to_lora_lr_ratio_constant"])
    report["check2_lm_head_lr"] = c2
    print(json.dumps(c2, indent=1), flush=True)

    # ---------------------------------------------------------------- check 3
    def scenes(ds):
        sc, vids, null = set(), set(), 0
        for i in range(len(ds)):
            r = ds._read_record(i)
            if not r.get("scene_id"):
                null += 1
            sc.add(r.get("scene_id")); vids.add(str(r["video"]))
        return sc, vids, null
    tr_sc, tr_v, tr_null = scenes(train_ds)
    ho_sc, ho_v, ho_null = scenes(hold_ds)
    split = json.loads(T.SPLIT.read_text())
    c3 = {
        "train": {"records": len(train_ds), "scenes": len(tr_sc), "videos": len(tr_v), "null_scene_id": tr_null},
        "holdout": {"records": len(hold_ds), "scenes": len(ho_sc), "videos": len(ho_v), "null_scene_id": ho_null,
                    "scene_ids": sorted(ho_sc)},
        "scene_intersection": sorted(tr_sc & ho_sc),
        "video_intersection_count": len(tr_v & ho_v),
        "union_scenes": len(tr_sc | ho_sc),
        "matches_split_file": (sorted(ho_sc) == sorted(split["holdout"]["scene_ids"])
                               and sorted(tr_sc) == sorted(split["train"]["scene_ids"])),
    }
    c3["PASS"] = (not c3["scene_intersection"] and c3["video_intersection_count"] == 0
                  and len(tr_sc) == 56 and len(ho_sc) == 5 and tr_null == 0 and ho_null == 0
                  and c3["union_scenes"] == 61 and c3["matches_split_file"])
    report["check3_scene_disjoint"] = c3
    print(json.dumps(c3, indent=1), flush=True)

    report["ALL_PASS"] = c1["PASS"] and c2["PASS"] and c3["PASS"]
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("ALL_PASS", report["ALL_PASS"])


if __name__ == "__main__":
    main()
