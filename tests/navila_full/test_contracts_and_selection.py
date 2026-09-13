from __future__ import annotations

import unittest

from src.navila_full.contracts import POLICY_CONTRACT, required_training_steps, validate_training_plan
from src.navila_full.finalize_selection import select_canary
from src.navila_full.selection import (
    DEFAULT_UNSEEN_SCENE,
    assign_splits,
    attach_spatial_clusters,
    candidate_routes,
    normalized_word_count,
    validate_split_records,
)


def _path(category: str, offset: float) -> list[list[float]]:
    if category == "straight":
        return [[offset, 0.0, 0.0], [offset + 3.0, 0.0, 0.0], [offset + 6.0, 0.0, 0.0]]
    if category == "left_turn":
        return [[offset, 0.0, 0.0], [offset + 3.0, 0.0, 0.0], [offset + 3.0, 3.0, 0.0]]
    return [[offset, 0.0, 0.0], [offset + 3.0, 0.0, 0.0], [offset + 3.0, -3.0, 0.0]]


def _episode(index: int, category: str, scene: str, *, duplicate: bool = False) -> dict[str, object]:
    path = _path(category, index * 10.0)
    return {
        "episode_id": f"episode_{index}_{'b' if duplicate else 'a'}",
        "trajectory_id": f"trajectory_{index}",
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "instruction": {"instruction_text": "Walk to the indicated location and stop." if not duplicate else "Go to the target and stop."},
        "reference_path": path,
        "gt_locations": path,
        "start_position": path[0],
        "start_rotation": [1.0, 0.0, 0.0, 0.0],
        "goals": [{"position": path[-1], "radius": 0.5}],
    }


def valid_plan() -> dict[str, object]:
    return {
        **POLICY_CONTRACT,
        "dataset_schema_version": "navila_full_episode_go2_v1",
        "dataset_root": "/new/full_episode/lerobot",
        "source_checkpoint": None,
        "selection_manifest": "/new/full_episode/selection.json",
        "acceptance_reference": "/new/full_episode/acceptance.json",
        "policy_signals": ["body_vx", "body_vy", "body_yaw_rate"],
        "n_train_samples": 6000,
        "effective_batch_size": 8,
        "training_steps": required_training_steps(6000, 8),
    }


class FullEpisodeContractTests(unittest.TestCase):
    def test_contract_accepts_only_fresh_3d_training(self) -> None:
        self.assertEqual(validate_training_plan(valid_plan()), [])
        plan = valid_plan()
        plan["state"] = {"shape": [30], "names": ["rpy"]}
        plan["latency_p95_seconds_max"] = 0.2
        plan["policy_signals"] = ["body_vx", "reference_path"]
        plan["source_checkpoint"] = "/outputs/m6/checkpoint"
        errors = validate_training_plan(plan)
        self.assertTrue(any("state must" in error for error in errors))
        self.assertTrue(any("latency" in error for error in errors))
        self.assertTrue(any("prohibited" in error for error in errors))
        self.assertTrue(any("legacy policy checkpoint" in error for error in errors))

    def test_word_count_is_normalized_not_whitespace_dependent(self) -> None:
        self.assertEqual(normalized_word_count("Go---left, then don't stop!"), 5)


class FullEpisodeSplitTests(unittest.TestCase):
    def test_candidate_preserves_exact_official_instruction_whitespace(self) -> None:
        episode = _episode(999, "straight", "seen_house")
        original = "  Walk to the indicated location and stop.  "
        episode["instruction"]["instruction_text"] = original
        selected = candidate_routes([episode])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["instruction"], original)

    def test_deduplicates_paraphrases_and_assigns_30_12_12_without_leakage(self) -> None:
        episodes: list[dict[str, object]] = []
        index = 0
        # 14 route groups/category supports the 10 train + 4 seen-val quota.
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(14):
                episodes.append(_episode(index, category, "seen_house"))
                if index == 0:
                    episodes.append(_episode(index, category, "seen_house", duplicate=True))
                index += 1
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(4):
                episodes.append(_episode(index, category, DEFAULT_UNSEEN_SCENE))
                index += 1
        candidates = candidate_routes(episodes)
        self.assertEqual(len(candidates), 54)
        duplicate = [row for row in candidates if row["source_trajectory_id"] == "trajectory_0"][0]
        self.assertEqual(duplicate["source_annotation_count"], 2)
        assigned = assign_splits(attach_spatial_clusters(candidates), seed=7)
        self.assertEqual(len(assigned), 54)
        self.assertEqual(validate_split_records(assigned), [])
        self.assertEqual(sum(row["split"] == "train" for row in assigned), 30)
        self.assertEqual(sum(row["split"] == "seen-val" for row in assigned), 12)
        self.assertEqual(sum(row["split"] == "unseen-test" for row in assigned), 12)
        self.assertTrue(all(row["scene_name"] == DEFAULT_UNSEEN_SCENE for row in assigned if row["split"] == "unseen-test"))

    def test_no_extra_annotation_can_cross_a_route_boundary(self) -> None:
        rows = attach_spatial_clusters(candidate_routes([_episode(1, "straight", "seen_house")]))
        duplicate = dict(rows[0])
        duplicate["split"] = "train"
        rows[0]["split"] = "seen-val"
        self.assertTrue(any("physical route" in error for error in validate_split_records([*rows, duplicate])))

    def test_canary_is_seen_only_and_reserves_a_viable_final_split(self) -> None:
        episodes: list[dict[str, object]] = []
        index = 0
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(16):
                episodes.append(_episode(index, category, "seen_house"))
                index += 1
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(4):
                episodes.append(_episode(index, category, DEFAULT_UNSEEN_SCENE))
                index += 1
        canary = select_canary(attach_spatial_clusters(candidate_routes(episodes)), seed=19)
        self.assertEqual(len(canary), 6)
        self.assertTrue(all(row["scene_name"] != DEFAULT_UNSEEN_SCENE for row in canary))
