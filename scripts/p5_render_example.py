#!/usr/bin/env python3
"""Render real training records exactly as phase 1 feeds them to the model.

CPU only. Everything shown comes from the same code paths training uses:
NavilaR2RDataset (history frames), Phase1PromptBuilder (token ids), and the
frozen LanguageParser (answer -> NavCommand). Nothing here is retyped by hand,
so the example cannot drift from what the model actually receives.

Writes reports/p5/EXAMPLE_IO.md plus one contact sheet per record.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GO2_DATA_ROOT", "/mnt/wxh/go2_short_vln"))
INDEX = DATA_ROOT / "data/navila_dataset/R2R/index/t2_records.jsonl"
FRAMES = DATA_ROOT / "data/navila_dataset/R2R/train"
MODEL = DATA_ROOT / ("cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
                     "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467")


def contact_sheet(item, out: Path, thumb=192):
    sheet = Image.new("RGB", (thumb * 8 + 7 * 4, thumb + 22), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, frame in enumerate(item["frames"]):
        img = Image.fromarray(frame.permute(1, 2, 0).numpy()).resize((thumb, thumb))
        x = k * (thumb + 4)
        sheet.paste(img, (x, 22))
        path = item["frame_paths"][k]
        label = "black pad" if not path else Path(path).stem.replace("frame_", "f")
        draw.text((x + 4, 4), f"{k}: {label}" + ("  (current)" if k == 7 else ""), fill=(0, 0, 0))
    sheet.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids", default="914-23,SHORT",
                    help="comma-separated video_id; SHORT picks the first turn record with n_frames<8")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p5/EXAMPLE_IO.md")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets.navila_r2r import NavilaR2RDataset
    from src.smolvla.phase1_prompt import Phase1PromptBuilder, navila_history_layout
    from actions.language_parser import parse_nav_command

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    b = Phase1PromptBuilder(tok)
    ds = NavilaR2RDataset(INDEX, FRAMES)

    wanted = args.video_ids.split(",")
    picks = {}
    with INDEX.open() as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if r["video_id"] in wanted and r["video_id"] not in picks:
                picks[r["video_id"]] = i
            if "SHORT" in wanted and "SHORT" not in picks and r["n_frames"] < 8 and "turn" in r["action_text"]:
                picks["SHORT"] = i
            if len(picks) == len(wanted):
                break

    md = ["# Phase 1 模型输入输出示例（由训练代码路径实际生成）", "",
          f"生成命令：`python scripts/p5_render_example.py`。模板 token 数（不含指令）：**{b.template_tokens}**，"
          f"其中图像占位 {8 * 64}。", ""]
    for key in wanted:
        item = ds[picks[key]]
        r = item["record"]
        p = b.prompt(r["instruction_raw"])
        t = b.target(r["action_text"])
        res = parse_nav_command(b.decode_answer(t))
        sheet = args.out.parent / f"example_frames_{r['video_id']}.jpg"
        contact_sheet(item, sheet)
        cmd = res.command
        md += [f"## 记录 `{r['video_id']}`（scene `{r['scene_id']}`，n_frames={r['n_frames']}）", "",
               f"**8 帧历史**（NaVILA 规则 `navila_history_layout({r['n_frames']})` = "
               f"`{navila_history_layout(r['n_frames'])}`，None=黑帧）：见 `{sheet.name}`", "",
               "**模型看到的完整输入**（每个 `<image>x64` 会被该帧的 64 个视觉 token 替换）：", "",
               "```text", b.render(r["instruction_raw"]), "```", "",
               f"- prompt 总 token：**{len(p.ids)}**（指令 {p.instruction_tokens} token，截断={p.instruction_truncated}）",
               f"- 监督目标（loss 只算这些）：`{tok.decode(t)}`",
               f"- 目标 token ids（{len(t)} 个）：`{t}`",
               f"- 训练序列长度：{len(p.ids) + len(t)}", "",
               "**模型应输出**（贪心解码到 `<end_of_utterance>` 停止）→ parser → 控制命令：", "",
               "```text", r["action_text"], "```", "",
               f"`NavCommand(vx={cmd.vx}, vy={cmd.vy}, wz={cmd.wz:.6f}, hold_steps={cmd.hold_steps}, stop={cmd.stop})`"
               f"，parse 分类 `{res.classification}`", ""]
    args.out.write_text("\n".join(md) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
