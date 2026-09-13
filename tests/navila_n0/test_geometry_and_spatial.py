from __future__ import annotations

import unittest

from src.navila_n0.geometry import RouteRules, SurfaceProbe, enumerate_route_candidates, resample_dense_path
from src.navila_n0.spatial import buffered_overlap_length_m, cluster_candidates, validate_train_seen_isolation


class TestRouteGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.path = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 3.0, 0.0)]
        self.probes = [SurfaceProbe("floor", 1.0) for _ in resample_dense_path(self.path)]

    def test_generates_flat_continuous_turn_candidates_with_full_audit(self) -> None:
        candidates = enumerate_route_candidates("parent-1", self.path, self.probes)
        turn = next(candidate for candidate in candidates if candidate["category"] == "left_turn")
        self.assertEqual(turn["parent_episode_id"], "parent-1")
        self.assertLessEqual(turn["path_length_m"], 4.0)
        self.assertGreaterEqual(turn["path_length_m"], 1.5)
        self.assertEqual(len(turn["turn_audit"]["principal_turns"]), 1)
        self.assertTrue(turn["flatness_audit"]["passed"])
        self.assertEqual(turn["parent_dense_path"], [list(point) for point in self.path])
        self.assertEqual(len(turn["dense_index_range"]), 2)
        self.assertGreaterEqual(turn["instruction_anchor_arc_length_m"], 0.0)
        self.assertLessEqual(turn["instruction_anchor_arc_length_m"], turn["path_length_m"])

    def test_rejects_route_with_an_unacceptable_surface(self) -> None:
        probes = list(self.probes)
        probes[20] = SurfaceProbe("stairs", 1.0)
        candidates = enumerate_route_candidates("parent-1", self.path, probes)
        self.assertFalse(
            any(
                any(
                    abs(float(candidate["dense_arclength_range_m"][0]) + float(point["arc_length_m"]) - 2.0) < 1e-9
                    for point in candidate["resampled_path"]
                )
                for candidate in candidates
            )
        )

    def test_rejects_steep_floor(self) -> None:
        probes = [SurfaceProbe("floor", 0.95) for _ in self.probes]
        self.assertEqual(enumerate_route_candidates("parent-1", self.path, probes), [])


class TestSpatialSplitIsolation(unittest.TestCase):
    def test_buffered_overlap_clusters_and_blocks_cross_split(self) -> None:
        path_a = [
            {"xyz": [0.0, 0.0, 0.0], "arc_length_m": 0.0},
            {"xyz": [0.5, 0.0, 0.0], "arc_length_m": 0.5},
            {"xyz": [1.0, 0.0, 0.0], "arc_length_m": 1.0},
        ]
        path_b = [
            {"xyz": [0.0, 0.3, 0.0], "arc_length_m": 0.0},
            {"xyz": [0.5, 0.3, 0.0], "arc_length_m": 0.5},
            {"xyz": [1.0, 0.3, 0.0], "arc_length_m": 1.0},
        ]
        self.assertGreaterEqual(buffered_overlap_length_m(path_a, path_b), 0.5)
        candidates = [
            {"candidate_id": "a", "parent_episode_id": "parent-a", "resampled_path": path_a},
            {"candidate_id": "b", "parent_episode_id": "parent-b", "resampled_path": path_b},
        ]
        clusters = cluster_candidates(candidates)
        self.assertEqual(clusters["a"], clusters["b"])
        records = [
            {"parent_episode_id": "parent-a", "spatial_cluster_id": clusters["a"], "split": "train"},
            {"parent_episode_id": "parent-b", "spatial_cluster_id": clusters["b"], "split": "seen-val"},
        ]
        self.assertTrue(any("spatial cluster" in error for error in validate_train_seen_isolation(records)))

    def test_same_parent_isolation_is_independent_of_geometry(self) -> None:
        records = [
            {"parent_episode_id": "same", "spatial_cluster_id": 1, "split": "train"},
            {"parent_episode_id": "same", "spatial_cluster_id": 2, "split": "seen-val"},
        ]
        self.assertTrue(any("parent dense" in error for error in validate_train_seen_isolation(records)))
