"""Build a conservative *preflight* bank for the minimal visual-language task.

This stage does not invent or shorten language.  It only selects complete
official NaVILA episodes whose original annotation is plausibly a single,
short command, then leaves the decisive ``initial_target_visible`` judgement
pending a marker-free RGB preview in Isaac.  The preview images are selection
evidence only and can never be added to training data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.navila_full.contracts import DATASET_SCHEMA_VERSION
from src.navila_full.selection import (
    CATEGORIES,
    _hash,
    instruction_text,
    load_official_episodes,
    normalized_word_count,
    path_length_m,
    route_category,
    route_id_for,
    scene_name,
    sha256,
)


EASY_VISIBLE_PREFLIGHT_FORMAT = "navila-easy-visible-preflight-v1"
EASY_VISIBLE_RULES: dict[str, float | int] = {
    # vln_ce_isaac_v1 has no complete source episode under 5.05 m.  This is
    # therefore the shortest faithful range possible without slicing an
    # episode or synthesising a new task.
    "minimum_reference_length_m": 5.0,
    "maximum_reference_length_m": 6.5,
    "maximum_gt_length_m": 7.0,
    "maximum_instruction_words": 16,
    "maximum_major_turns": 1,
}
# These terms make the annotation sequential even when it is very short.  The
# check is deliberately lexical: target visibility must be demonstrated from a
# real initial RGB image, not inferred from language or path geometry.
SEQUENTIAL_TERMS = frozenset({"then", "once", "after", "before", "until", "past", "around", "through"})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())


def is_single_stage_instruction(text: str) -> bool:
    """Accept short, direct commands and reject coordination/route sequences."""
    tokens = _tokens(text)
    if not tokens or any(token in SEQUENTIAL_TERMS for token in tokens):
        return False
    # A semicolon or multiple sentence-ending punctuation is an unambiguous
    # compound command.  A final period is normal and remains accepted.
    if ";" in text or len(re.findall(r"[.!?]", text)) > 1:
        return False
    # "and stop/wait" is a conventional single navigation instruction; other
    # conjunctions are treated as a second requested behaviour/landmark.
    if "and" in tokens:
        positions = [index for index, token in enumerate(tokens) if token == "and"]
        if any(tokens[index + 1:index + 2] not in (["stop"], ["wait"]) for index in positions):
            return False
    return True


def _goal_distance_m(episode: Mapping[str, Any]) -> float:
    start = episode.get("start_position")
    goals = episode.get("goals")
    if not isinstance(start, Sequence) or len(start) != 3:
        raise ValueError("official episode has no xyz start_position")
    if not isinstance(goals, Sequence) or len(goals) != 1 or not isinstance(goals[0], Mapping):
        raise ValueError("official episode must have exactly one goal")
    goal = goals[0].get("position")
    if not isinstance(goal, Sequence) or len(goal) != 3:
        raise ValueError("official goal has no xyz position")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(start, goal)) ** 0.5


def candidate_from_episode(episode: Mapping[str, Any]) -> dict[str, Any] | None:
    instruction = instruction_text(episode)
    reference_length = path_length_m(episode["reference_path"], "reference_path")
    gt_length = path_length_m(episode["gt_locations"], "gt_locations")
    words = normalized_word_count(instruction)
    try:
        category, turns = route_category(episode["reference_path"])
    except ValueError:
        return None
    if not (
        float(EASY_VISIBLE_RULES["minimum_reference_length_m"])
        <= reference_length
        <= float(EASY_VISIBLE_RULES["maximum_reference_length_m"])
        and gt_length <= float(EASY_VISIBLE_RULES["maximum_gt_length_m"])
        and words <= int(EASY_VISIBLE_RULES["maximum_instruction_words"])
        and len(turns) <= int(EASY_VISIBLE_RULES["maximum_major_turns"])
        and is_single_stage_instruction(instruction)
    ):
        return None
    route_id = route_id_for(episode)
    return {
        "route_id": route_id,
        "annotation_id": _hash({"route_id": route_id, "episode_id": str(episode["episode_id"]), "instruction": instruction.strip()})[:24],
        "source_episode_id": str(episode["episode_id"]),
        "source_trajectory_id": str(episode["trajectory_id"]),
        "scene_id": str(episode["scene_id"]),
        "scene_name": scene_name(str(episode["scene_id"])),
        "instruction": instruction,
        "reference_path": episode["reference_path"],
        "gt_locations": episode["gt_locations"],
        "start_position": episode.get("start_position"),
        "start_rotation": episode.get("start_rotation"),
        "goals": episode.get("goals"),
        "category": category,
        "turn_audit": turns,
        "route_metrics": {
            "reference_length_m": reference_length,
            "gt_length_m": gt_length,
            "initial_goal_distance_m": _goal_distance_m(episode),
            "instruction_word_count": words,
        },
        "visibility_status": "pending_marker_free_initial_rgb_review",
    }


def candidate_routes(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate paraphrases without ever rewriting the selected instruction."""
    choices: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        candidate = candidate_from_episode(episode)
        if candidate is not None:
            choices.setdefault(str(candidate["route_id"]), []).append(candidate)
    records: list[dict[str, Any]] = []
    for route_id in sorted(choices):
        options = choices[route_id]
        selected = sorted(options, key=lambda row: (str(row["annotation_id"]), str(row["source_episode_id"])))[0].copy()
        selected["source_annotation_count"] = len(options)
        selected["source_annotation_ids"] = sorted(str(row["annotation_id"]) for row in options)
        records.append(selected)
    return records


def build_preflight_manifest(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    episodes = load_official_episodes(dataset_path)
    manifest = {
        "format": EASY_VISIBLE_PREFLIGHT_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "stage": "metadata_prefilter_pending_marker_free_initial_rgb_review",
        "source_provenance": {
            "official_dataset_path": str(dataset_path),
            "official_dataset_sha256": sha256(dataset_path),
            "source_episode_count": len(episodes),
            "source_fields_preserved": ["instruction.instruction_text", "reference_path", "gt_locations"],
        },
        "selection_contract": {
            "complete_episode_only": True,
            "original_instruction_only": True,
            "metadata_rules": EASY_VISIBLE_RULES,
            "reject_sequential_language_terms": sorted(SEQUENTIAL_TERMS),
            "initial_target_visible": "must be manually confirmed in marker-free reset RGB; pending candidates are not training data",
            "candidate_balance_goal": {category: 2 for category in CATEGORIES},
        },
        "routes": candidate_routes(episodes),
    }
    errors = validate_preflight_manifest(manifest)
    if errors:
        raise RuntimeError("invalid easy-visible preflight manifest:\n" + "\n".join(errors))
    return manifest


def validate_preflight_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != EASY_VISIBLE_PREFLIGHT_FORMAT:
        errors.append("wrong easy-visible preflight format")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong full-episode schema version")
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        return errors + ["routes must be a list"]
    route_ids: set[str] = set()
    for route in routes:
        route_id = str(route.get("route_id", "")) if isinstance(route, Mapping) else ""
        if not route_id or route_id in route_ids:
            errors.append("route ids are missing or duplicated")
        route_ids.add(route_id)
        if not isinstance(route, Mapping) or route.get("category") not in CATEGORIES:
            errors.append(f"{route_id}: invalid route category")
            continue
        if not isinstance(route.get("instruction"), str) or not is_single_stage_instruction(str(route["instruction"])):
            errors.append(f"{route_id}: instruction is not a single-stage original command")
        if route.get("visibility_status") != "pending_marker_free_initial_rgb_review":
            errors.append(f"{route_id}: invalid visibility review state")
        if not route.get("reference_path") or not route.get("gt_locations"):
            errors.append(f"{route_id}: source paths must be retained")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_preflight_manifest(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(str(route["category"]) for route in manifest["routes"])
    print(json.dumps({"output": str(args.output), "route_count": len(manifest["routes"]), "categories": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
