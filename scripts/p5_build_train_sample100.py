"""Stratified 100-record sample of NaVILA TRAINING decisions — NOT benchmark dev100.

WHAT THIS IS: 100 decision records drawn round-robin from the 61 R2R *training*
scenes in t2_records.jsonl, one JSON object per record.

WHAT THIS IS NOT: the frozen benchmark dev100. Section 4 of the milestone
contract defines dev100 as 100 *episodes* stratified by scene from the
1077-episode VLN-CE-Isaac evaluation set over its 11 unseen scenes, with the id
list frozen into configs/benchmark.yaml. This file differs in source (training
vs benchmark), granularity (decision records vs episodes) and purpose (audit vs
evaluation). Never evaluate against it and never call its contents dev100.

Renamed from p5_build_dev100.py on 2026-09-13 because the old name invited
exactly that confusion; see reports/P5_PREFLIGHT_CORRECTION.md.
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = Path('/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl')
OUT = ROOT / 'reports/p5_train_sample100.json'
JSONL = ROOT / 'reports/p5_train_sample100.jsonl'
SEED = 20260913

rows = [json.loads(line) for line in INDEX.open() if line.strip()]
groups = defaultdict(list)
for row in rows:
    scene = row.get('scene_id') or row.get('scene_stem')
    if scene is None:
        raise ValueError('record missing scene_id/scene_stem')
    groups[scene].append(row)
rng = random.Random(SEED)
for values in groups.values():
    rng.shuffle(values)
scenes = sorted(groups)
chosen = []
for i in range(100):
    scene = scenes[i % len(scenes)]
    chosen.append(groups[scene][i // len(scenes)])
chosen.sort(key=lambda x: (x.get('scene_id', x.get('scene_stem','')), x.get('video_id', x.get('episode_id', ''))))
payload = {'task_id':'P5-train-sample100','seed':SEED,'source_index':str(INDEX),'record_count':len(chosen),
           'NOT_benchmark_dev100':('100 decision records from the 61 R2R TRAINING scenes, not 100 episodes from '
               'the 1077-episode benchmark over its 11 unseen scenes. Audit only; never use for evaluation. '
               'Renamed from p5_dev100 on 2026-09-13; see reports/P5_PREFLIGHT_CORRECTION.md'),
           'scene_intersection_with_eval11': [],
           'scene_count':len({x.get('scene_id', x.get('scene_stem')) for x in chosen}),
           'records':chosen}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
JSONL.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in chosen))
print(json.dumps({'output':str(OUT),'records':len(chosen),'scenes':payload['scene_count']}))
