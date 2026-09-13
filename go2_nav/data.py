"""Episode validation, review decisions and deterministic scene-level splits."""
import hashlib
import json
from pathlib import Path
from .contracts import validate_episode


def read_episodes(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise ValueError("expected an object containing an episodes list")
    return value["episodes"]


def audit(episodes, root, rejected=()):
    root = Path(root).resolve()
    kept, issues, seen = [], [], set()
    for index, episode in enumerate(episodes):
        identity = episode.get("episode_id", f"row-{index}") if isinstance(episode, dict) else f"row-{index}"
        try:
            validate_episode(episode)
            if identity in seen:
                raise ValueError("duplicate episode_id")
            seen.add(identity)
            if identity in rejected:
                raise ValueError("rejected by manual review")
            for step in episode["steps"]:
                path = (root/step["observation"]["rgb_path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("RGB path escapes dataset root")
                if not path.is_file():
                    raise ValueError("missing RGB frame")
            kept.append(episode)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            issues.append({"episode_id": identity, "reason": str(exc)})
    return kept, issues


def split_for_scene(scene, seed=0):
    """All episodes from one scene share a split; no episode-level leakage."""
    if not isinstance(scene, str) or not scene:
        raise ValueError("scene is required")
    bucket = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def summarize(episodes):
    """Structural diagnostics only. Never interpret mock termination as success."""
    sources = sorted({e["source"] for e in episodes})
    return {"sources": sources, "episode_count": len(episodes),
            "step_count": sum(len(e["steps"]) for e in episodes),
            "stopped": sum(e["outcome"] == "stopped" for e in episodes),
            "timeouts": sum(e["outcome"] == "timeout" for e in episodes)}
