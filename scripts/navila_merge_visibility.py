#!/usr/bin/env python3
"""Merge per-scene live visibility receipts without silently dropping records."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    files = sorted(x for x in args.input_dir.glob('*.json') if x.name != args.output.name)
    if not files: raise ValueError('no per-scene receipts')
    rows, ids, scenes = [], set(), []
    source_hashes = set()
    for path in files:
        payload = json.loads(path.read_text())
        scene = str(payload.get('scene', ''))
        if not scene or scene in scenes: raise ValueError(f'duplicate/invalid scene: {scene}')
        scenes.append(scene); source_hashes.add(payload.get('source_candidates_sha256'))
        for row in payload.get('candidates', []):
            cid = str(row.get('candidate_id', ''))
            if not cid or cid in ids: raise ValueError(f'duplicate candidate: {cid}')
            ids.add(cid); rows.append(row)
    if len(source_hashes) != 1: raise ValueError('per-scene source candidate hashes differ')
    result = {'format':'navila-dual-target-live-visibility-merged-v1', 'scenes':sorted(scenes),
              'source_candidates_sha256': next(iter(source_hashes)), 'receipts':[str(x) for x in files],
              'candidates':rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'scenes':len(scenes),'candidates':len(rows),'output':str(args.output)}))
if __name__ == '__main__': main()
