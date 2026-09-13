#!/usr/bin/env python3
"""Check that the training forward equals SmolVLM2's own forward (review item B).

The training script bypasses the HF processor and SmolVLMModel.inputs_merger:
it normalizes uint8 frames itself and masked_scatters vision features into the
<image> slots. That is only valid if it computes what the untouched model code
computes. This script feeds the same holdout records through both:

  manual  scripts/p5_phase1_train.forward_answer_logits (uint8 -> x/127.5-1)
  native  processor.image_processor(PIL frames, do_image_splitting=False) ->
          SmolVLMForConditionalGeneration.forward(input_ids, pixel_values, ...)

and compares pixels, answer-position logits and argmax. A checkpoint can be
loaded so LoRA is non-trivial (freshly initialised lora_B is zero, which would
make a LoRA bug invisible). GPU; writes reports/p5/native_forward_check.json.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image


def load_train_module():
    spec = importlib.util.spec_from_file_location("p5_phase1_train", ROOT / "scripts/p5_phase1_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--records", type=int, default=16)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p5/native_forward_check.json")
    ap.add_argument("--fp32", action="store_true",
                    help="whole model in fp32, no autocast: separates logic errors from bf16 rounding")
    args = ap.parse_args()

    T = load_train_module()
    from datasets.navila_r2r import NavilaR2RDataset
    from src.smolvla.phase1_prompt import Phase1PromptBuilder
    from safetensors.torch import load_file

    processor, model, _ = T.build_model()
    if args.checkpoint:
        params = dict(model.named_parameters())
        for n, t in load_file(str(args.checkpoint / "trainable.safetensors")).items():
            params[n].data.copy_(t)
    model.to("cuda").eval()
    if args.fp32:
        model.float()
    import contextlib
    amp = contextlib.nullcontext if args.fp32 else (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
    dt = torch.float32 if args.fp32 else torch.bfloat16
    b = Phase1PromptBuilder(processor.tokenizer)
    _, hold, _ = T.load_splits(NavilaR2RDataset)
    g = torch.Generator().manual_seed(7)
    idx = torch.randperm(len(hold), generator=g)[:args.records].tolist()

    rows = []
    for i in idx:
        item = hold[i]
        batch = T.collate_fn(b, pad_id=processor.tokenizer.pad_token_id)([item])
        with torch.no_grad(), amp():
            man_logits, labels, _ = T.forward_answer_logits(model, batch, "cuda", b.image_token_id)

            pil = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in item["frames"]]
            proc = processor.image_processor(pil, do_image_splitting=False, return_tensors="pt")
            pv = proc["pixel_values"]
            mine_px = item["frames"].float() / 127.5 - 1.0
            px_diff = float((pv.reshape(mine_px.shape) - mine_px).abs().max()) if pv.numel() == mine_px.numel() else None
            ids = batch["ids"].cuda()
            out = model(input_ids=ids, attention_mask=batch["attn"].cuda(),
                        pixel_values=pv.cuda().to(dt),
                        pixel_attention_mask=proc.get("pixel_attention_mask", None).cuda()
                        if proc.get("pixel_attention_mask", None) is not None else None,
                        use_cache=False)
            mask = batch["sup"].cuda()[:, 1:]
            nat_logits = out.logits[:, :-1][mask].float()
            # Native forward again, fed the training script's own pixels. This
            # isolates the merge/attention logic from the pixel pipeline: the
            # processor resamples 512 -> 2048 -> 512 even with splitting disabled.
            out2 = model(input_ids=ids, attention_mask=batch["attn"].cuda(),
                         pixel_values=mine_px.cuda().to(dt)[None], use_cache=False)
            nat_same_px = out2.logits[:, :-1][mask].float()
        d = (man_logits - nat_logits).abs()
        d2 = (man_logits - nat_same_px).abs()
        rows.append({"video_id": item["video_id"], "n_frames": item["record"]["n_frames"],
                     "pixel_values_shape": list(pv.shape), "pixel_max_abs_diff": px_diff,
                     "answer_tokens": int(labels.numel()),
                     "logit_max_abs_diff": float(d.max()),
                     "logit_max_abs": float(nat_logits.abs().max()),
                     "argmax_equal": bool((man_logits.argmax(-1) == nat_logits.argmax(-1)).all()),
                     "same_pixels_logit_max_abs_diff": float(d2.max()),
                     "same_pixels_argmax_equal": bool((man_logits.argmax(-1) == nat_same_px.argmax(-1)).all()),
                     "loss_manual": float(torch.nn.functional.cross_entropy(man_logits, labels)),
                     "loss_native": float(torch.nn.functional.cross_entropy(nat_logits, labels))})
        print(json.dumps(rows[-1]), flush=True)

    summary = {"checkpoint": str(args.checkpoint), "precision": "fp32" if args.fp32 else "bf16-autocast",
               "records": len(rows),
               "all_argmax_equal": all(r["argmax_equal"] for r in rows),
               "same_pixels_all_argmax_equal": all(r["same_pixels_argmax_equal"] for r in rows),
               "same_pixels_max_logit_abs_diff": max(r["same_pixels_logit_max_abs_diff"] for r in rows),
               "max_logit_abs_diff": max(r["logit_max_abs_diff"] for r in rows),
               "max_pixel_abs_diff": max((r["pixel_max_abs_diff"] or 0.0) for r in rows),
               "max_loss_abs_diff": max(abs(r["loss_manual"] - r["loss_native"]) for r in rows),
               "rows": rows}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
