import json
import math
from pathlib import Path
import tempfile
import unittest

from src.collector.d5_collection import (
    D5_SPLIT_RULES,
    apply_large_orientation_recovery,
    collection_ruleset,
    _features_by_split,
    _write_json_atomically,
    IMMUTABLE_OFFICIAL_PLANNER_SHA256,
    next_blocked_snapshot_path,
    next_collection_decision,
    next_d5_selection,
    official_large_orientation,
    promote_accepted_attempt,
    reselect_d5_splits,
    write_assignment_sidecar,
)


def _episode(identifier, split, category, ordinal):
    import math

    yaw = ((ordinal % 8) + 0.5) * math.pi / 4.0
    return {
        "short_episode_id": identifier,
        "split": split,
        "scene_id": f"scene_{ordinal % 9}",
        "path_length": (1.8, 2.5, 3.5)[ordinal % 3],
        "start_pose": {"rotation_wxyz": [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]},
        "instruction_metadata": {"category": category},
    }


def _d5_synthetic_episodes():
    episodes = []
    ordinal = 0
    for category, ids in (
        ("straight", ["short_vln_v1_0000", "short_vln_v1_0004", *[f"train_straight_{i}" for i in range(8)]]),
        ("left_turn", [f"train_left_{i}" for i in range(10)]),
        ("right_turn", ["short_vln_v1_0072", *[f"train_right_{i}" for i in range(10)]]),
    ):
        for identifier in ids:
            episodes.append(_episode(identifier, "train", category, ordinal))
            ordinal += 1
    for category, ids in (
        ("straight", ["short_vln_v1_0001", "short_vln_v1_0003", "seen_straight_2", "seen_straight_3"]),
        ("left_turn", [f"seen_left_{i}" for i in range(4)]),
        ("right_turn", [f"seen_right_{i}" for i in range(4)]),
    ):
        for identifier in ids:
            episodes.append(_episode(identifier, "seen-val", category, ordinal))
            ordinal += 1
    return episodes


class D5CollectionFilesystemTests(unittest.TestCase):
    def test_official_large_orientation_uses_strict_point_six_radians(self):
        def quaternion_for_roll(roll):
            return [math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0]

        self.assertFalse(official_large_orientation(quaternion_for_roll(0.6)))
        self.assertTrue(official_large_orientation(quaternion_for_roll(0.6001)))

    def test_recovers_only_a_verified_large_orientation_last_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_dir = root / "attempts" / "short_vln_v1_0878"
            frames = attempt_dir / "front_rgb"
            frames.mkdir(parents=True)
            records = []
            for index, roll in enumerate((0.1, 0.61)):
                relative = f"front_rgb/{index:06d}.jpg"
                (attempt_dir / relative).write_bytes(b"jpeg")
                records.append({
                    "frame_index": index,
                    "timestamp": index * 0.02,
                    "control_dt_s": 0.02,
                    "front_rgb": relative,
                    "planner_command": [0.1, 0.0, 0.5],
                    "locomotion_command": [0.1, 0.0, 0.5],
                    "robot_state": [0.0] * 33,
                    "next_robot_pose": {
                        "position_w": [0.0, 0.0, 0.3],
                        "quaternion_wxyz": [math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0],
                    },
                })
            (attempt_dir / "steps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            manifest = {
                "status": "blocked_infrastructure_failure",
                "attempt_root": str(root / "attempts"),
                "rejected_ids": {"train": [], "seen-val": []},
                "attempts": [{
                    "short_episode_id": "short_vln_v1_0878",
                    "split": "train",
                    "decision": {"kind": "infrastructure_failure", "reason": "missing_or_invalid_attempt_artifacts"},
                    "attempt_dir": str(attempt_dir),
                }],
            }

            evidence = apply_large_orientation_recovery(manifest, attempt_dir)

            self.assertEqual(evidence["reason"], "official_large_orientation")
            self.assertIn("short_vln_v1_0878", manifest["rejected_ids"]["train"])
            self.assertEqual(manifest["attempts"][-1]["initial_decision"]["kind"], "infrastructure_failure")
            self.assertEqual(manifest["attempts"][-1]["decision"]["kind"], "expert_rollout_reject")
            self.assertTrue((attempt_dir / "recovery_evidence.json").is_file())

    def test_recovery_refuses_missing_orientation_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_dir = root / "attempts" / "short_vln_v1_0878"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "steps.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "status": "blocked_infrastructure_failure",
                "attempt_root": str(root / "attempts"),
                "rejected_ids": {"train": [], "seen-val": []},
                "attempts": [{
                    "short_episode_id": "short_vln_v1_0878", "split": "train",
                    "decision": {"kind": "infrastructure_failure"}, "attempt_dir": str(attempt_dir),
                }],
            }
            with self.assertRaises(ValueError):
                apply_large_orientation_recovery(manifest, attempt_dir)

    def test_recovers_verified_zero_frame_official_orientation_from_runner_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_dir = root / "attempts" / "short_vln_v1_1050"
            (attempt_dir / "front_rgb").mkdir(parents=True)
            (attempt_dir / "steps.jsonl").write_text("", encoding="utf-8")
            (attempt_dir / "collector_config.json").write_text(
                json.dumps(
                    {
                        "status": "collecting",
                        "short_episode_id": "short_vln_v1_1050",
                        "official_planner_sha256": IMMUTABLE_OFFICIAL_PLANNER_SHA256,
                    }
                ),
                encoding="utf-8",
            )
            logs = root / "m4_logs"
            logs.mkdir()
            (logs / "20260902T132813+0800-short_vln_v1_1050.log").write_text(
                "\n".join(
                    [
                        f"m4_episode_dir={attempt_dir.resolve()}",
                        "Warmup step 99/100...",
                        "Large orientation:  tensor([-0.3257])   tensor([-0.6511])",
                        "M4 timeline STOP current=0.0 end=10000.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {
                "status": "blocked_infrastructure_failure",
                "attempt_root": str(root / "attempts"),
                "rejected_ids": {"train": [], "seen-val": []},
                "attempts": [
                    {
                        "short_episode_id": "short_vln_v1_1050",
                        "split": "train",
                        "decision": {
                            "kind": "infrastructure_failure",
                            "reason": "missing_or_invalid_attempt_artifacts",
                        },
                        "attempt_dir": str(attempt_dir),
                    }
                ],
            }

            evidence = apply_large_orientation_recovery(
                manifest, attempt_dir, m4_log_root=logs
            )

            self.assertEqual(evidence["kind"], "expert_reset_reject")
            self.assertEqual(
                evidence["reason"], "official_large_orientation_before_first_action"
            )
            self.assertGreater(
                evidence["conservative_abs_lower_bound_rad"]["pitch"], 0.6
            )
            self.assertEqual(
                manifest["attempts"][-1]["decision"]["kind"], "expert_reset_reject"
            )
            self.assertIn("short_vln_v1_1050", manifest["rejected_ids"]["train"])

    def test_zero_frame_recovery_refuses_unverifiable_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_dir = root / "attempts" / "short_vln_v1_1050"
            (attempt_dir / "front_rgb").mkdir(parents=True)
            (attempt_dir / "steps.jsonl").write_text("", encoding="utf-8")
            (attempt_dir / "collector_config.json").write_text(
                json.dumps(
                    {
                        "short_episode_id": "short_vln_v1_1050",
                        "official_planner_sha256": IMMUTABLE_OFFICIAL_PLANNER_SHA256,
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "status": "blocked_infrastructure_failure",
                "rejected_ids": {"train": [], "seen-val": []},
                "attempts": [
                    {
                        "short_episode_id": "short_vln_v1_1050",
                        "split": "train",
                        "decision": {"kind": "infrastructure_failure"},
                        "attempt_dir": str(attempt_dir),
                    }
                ],
            }
            logs = root / "m4_logs"
            logs.mkdir()

            with self.assertRaises(ValueError):
                apply_large_orientation_recovery(manifest, attempt_dir, m4_log_root=logs)

    def test_blocked_snapshots_are_incrementing_and_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "collection_manifest.json"
            legacy = manifest.with_name("collection_manifest.blocked_before_resume.json")
            legacy.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                next_blocked_snapshot_path(manifest).name,
                "collection_manifest.blocked_before_resume.0002.json",
            )
            (manifest.with_name("collection_manifest.blocked_before_resume.0002.json")).write_text(
                "{}\n", encoding="utf-8"
            )
            self.assertEqual(
                next_blocked_snapshot_path(manifest).name,
                "collection_manifest.blocked_before_resume.0003.json",
            )

    def test_promote_moves_successful_attempt_out_of_attempt_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempts" / "short_vln_v1_0100"
            attempt.mkdir(parents=True)
            (attempt / "summary.json").write_text(json.dumps({"success": True}))
            accepted_root = root / "accepted"

            accepted = promote_accepted_attempt(attempt, accepted_root)

            self.assertEqual(accepted, accepted_root / "short_vln_v1_0100")
            self.assertFalse(attempt.exists())
            self.assertTrue((accepted / "summary.json").is_file())

    def test_promote_refuses_existing_accepted_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempts" / "short_vln_v1_0100"
            accepted = root / "accepted" / "short_vln_v1_0100"
            attempt.mkdir(parents=True)
            accepted.mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                promote_accepted_attempt(attempt, accepted.parent)

    def test_reselection_replaces_0072_without_losing_required_routes(self):
        episodes = _d5_synthetic_episodes()
        base_ids = {
            "train": {
                "short_vln_v1_0000", "short_vln_v1_0004",
                *[f"train_straight_{i}" for i in range(8)],
                *[f"train_left_{i}" for i in range(10)],
                "short_vln_v1_0072", *[f"train_right_{i}" for i in range(9)],
            },
            "seen-val": {item["short_episode_id"] for item in episodes if item["split"] == "seen-val"},
        }

        selected = reselect_d5_splits(
            episodes,
            base_ids=base_ids,
            accepted_ids={"train": {"short_vln_v1_0000", "short_vln_v1_0004"}, "seen-val": set()},
            rejected_ids={"train": {"short_vln_v1_0072"}, "seen-val": set()},
            seed=20260831,
        )

        train_ids = {item["short_episode_id"] for item in selected["train"]}
        seen_val_ids = {item["short_episode_id"] for item in selected["seen-val"]}
        self.assertEqual(len(train_ids), 30)
        self.assertEqual(len(seen_val_ids), 12)
        self.assertTrue({"short_vln_v1_0000", "short_vln_v1_0004"} <= train_ids)
        self.assertNotIn("short_vln_v1_0072", train_ids)
        self.assertEqual(base_ids["train"] - train_ids, {"short_vln_v1_0072"})
        self.assertEqual(len(train_ids - base_ids["train"]), 1)
        self.assertTrue({"short_vln_v1_0001", "short_vln_v1_0003"} <= seen_val_ids)

    def test_next_selection_is_single_route_and_prefers_coverage_gain(self):
        episodes = [
            _episode(f"accepted_{category}_{index}", "train", category, 0)
            for category in ("straight", "left_turn", "right_turn") for index in range(10)
        ]
        # The accepted minimum has deliberately degenerate coverage; this route
        # is the sole untried candidate and must be selected as a supplement.
        episodes.append(_episode("coverage_repair", "train", "straight", 17))
        accepted = {item["short_episode_id"] for item in episodes if item["short_episode_id"] != "coverage_repair"}
        decision = next_d5_selection(
            episodes, split="train", base_ids=set(), accepted_ids=accepted,
            rejected_ids=set(), seed=20260831,
        )
        self.assertEqual(decision["kind"], "candidate")
        self.assertEqual(decision["phase"], "coverage_supplement")
        self.assertEqual(decision["feature"]["short_episode_id"], "coverage_repair")
        self.assertGreater(decision["coverage_gain"], 0)

    def test_train_coverage_accepts_eight_distinct_scenes_under_r5_ruleset(self):
        episodes = []
        accepted = set()
        ordinal = 0
        for category in ("straight", "left_turn", "right_turn"):
            for index in range(10):
                identifier = f"accepted_{category}_{index}"
                episodes.append(_episode(identifier, "train", category, ordinal % 8))
                accepted.add(identifier)
                ordinal += 1
        decision = next_d5_selection(
            episodes, split="train", base_ids=set(), accepted_ids=accepted,
            rejected_ids=set(), seed=20260831,
        )
        self.assertEqual(D5_SPLIT_RULES["train"]["min_scenes"], 8)
        self.assertEqual(decision["kind"], "complete")
        self.assertEqual(decision["summary"]["scene_count"], 8)
        self.assertTrue(decision["summary"]["coverage_met"])

    def test_ruleset_records_the_authorized_train_scene_threshold_change(self):
        ruleset = collection_ruleset()
        self.assertEqual(ruleset["train_min_scenes_change"]["previous"], 9)
        self.assertEqual(ruleset["train_min_scenes_change"]["current"], 8)
        self.assertEqual(ruleset["split_rules"]["train"]["min_scenes"], 8)

    def test_collection_decision_reports_infeasible_instead_of_search_limit(self):
        episodes = _d5_synthetic_episodes()
        accepted = {
            "train": {
                item["short_episode_id"] for item in episodes
                if item["split"] == "train" and item["instruction_metadata"]["category"] != "right_turn"
            },
            "seen-val": set(),
        }
        rejected = {
            "train": {item["short_episode_id"] for item in episodes if item["split"] == "train" and item["instruction_metadata"]["category"] == "right_turn"},
            "seen-val": set(),
        }
        decision = next_d5_selection(
            episodes, split="train", base_ids=set(), accepted_ids=accepted["train"],
            rejected_ids=rejected["train"], seed=20260831,
        )
        self.assertEqual(decision["kind"], "blocked")
        self.assertEqual(decision["reason"], "selection_infeasible")

    def test_assignment_override_remains_excluded_from_seen_val_after_r5_threshold_change(self):
        episodes = []
        accepted = set()
        for category in ("straight", "left_turn", "right_turn"):
            for index in range(10):
                identifier = f"accepted_{category}_{index}"
                episodes.append(_episode(identifier, "train", category, index % 8))
                accepted.add(identifier)
        episodes.append(_episode("short_vln_v1_1051", "seen-val", "straight", 8))
        policy = {"assignments": {
            "short_vln_v1_1051": {
                "short_episode_id": "short_vln_v1_1051", "source_split": "seen-val",
                "collection_split": "train", "reason": "recover_required_train_scene_coverage",
                "exclude_from_seen_val_evaluation": True,
            }
        }}
        decision = next_d5_selection(
            episodes, split="train", base_ids=set(), accepted_ids=accepted,
            rejected_ids=set(), seed=20260831, assignment_policy=policy,
        )
        self.assertEqual(decision["kind"], "complete")
        self.assertEqual(decision["summary"]["scene_count"], 8)
        self.assertNotIn("short_vln_v1_1051", _features_by_split(episodes, policy)["seen-val"])

    def test_atomic_json_write_serializes_sets_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"old":true}\n', encoding="utf-8")
            _write_json_atomically(path, {"scenes": {"scene_b", "scene_a"}})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"scenes": ["scene_a", "scene_b"]})

            class Unsupported:
                pass

            with self.assertRaises(TypeError):
                _write_json_atomically(path, {"bad": Unsupported()})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"scenes": ["scene_a", "scene_b"]})

    def test_assignment_sidecar_preserves_source_and_collection_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "short_vln_v1_1051"
            attempt.mkdir()
            feature = {
                "short_episode_id": "short_vln_v1_1051",
                "assignment_override": {
                    "source_split": "seen-val", "collection_split": "train",
                    "reason": "recover_required_train_scene_coverage",
                    "exclude_from_seen_val_evaluation": True,
                },
            }
            sidecar = write_assignment_sidecar(
                attempt, feature,
                {"sha256": "policy-hash", "source_dataset_sha256": "dataset-hash"},
            )
            self.assertEqual(sidecar, attempt / "d5_assignment.json")
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["collection_split"], "train")


if __name__ == "__main__":
    unittest.main()
