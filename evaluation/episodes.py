"""Load the frozen benchmark and its configured dev subset, without simulation."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz")
DEFAULT_CONFIG = ROOT / "configs/benchmark_dev100.json"
SOURCE_SHA256 = "ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec"
EXPECTED_EPISODES = 1077


def load_episodes(source=DEFAULT_SOURCE, *, expected_sha256=SOURCE_SHA256):
    """Verify compressed bytes before decoding; reject a non-frozen population.

    Already decoded dict/list fields are accepted alongside Python-literal strings.
    The explicit checksum override supports independently checksummed test fixtures.
    """
    raw = Path(source).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    episodes = json.loads(gzip.decompress(raw))["episodes"]
    assert len(episodes) == EXPECTED_EPISODES, (
        f"Expected {EXPECTED_EPISODES} episodes, got {len(episodes)}"
    )
    bad = [(e["episode_id"], e["goals"][0].get("radius"))
           for e in episodes if e["goals"][0].get("radius") != 3.0]
    assert not bad, f"Expected radius == 3.0 for all episodes; offending (id, radius): {bad}"
    for episode in episodes:
        for field, expected_type in (("instruction", dict), ("reference_path", list),
                                     ("gt_locations", list)):
            if isinstance(episode[field], str):
                episode[field] = ast.literal_eval(episode[field])
            if not isinstance(episode[field], expected_type):
                raise ValueError(f"Episode {episode['episode_id']}: invalid {field}")
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs in source")
    return episodes


def load_dev100(source=DEFAULT_SOURCE, config=DEFAULT_CONFIG):
    """Select original episode IDs in exactly the manifest's order."""
    manifest = json.loads(Path(config).read_text())
    episodes = load_episodes(source, expected_sha256=manifest["input_sha256"])
    ids = manifest["episode_ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("dev100 must contain exactly 100 distinct episode IDs")
    by_id = {e["episode_id"]: e for e in episodes}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"dev100 episode IDs absent from source: {missing}")
    return [by_id[i] for i in ids]
