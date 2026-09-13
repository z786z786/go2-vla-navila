"""Pure trajectory replay of NaVILA's six measures and additional SR/OSR radii.

The output sequence includes reset at index 0, followed by each update. STOP is
1-based: stop_called_at=1 means true on the first post-step update and thereafter;
None means never. Replay continues through supplied positions even after STOP.
Input floating dtype is preserved because it affects official Euclidean sums.
"""

import numpy as np
from scipy.spatial import KDTree

METRIC_NAMES = ("path_length", "distance_to_goal", "success", "spl",
                "oracle_navigation_error", "oracle_success")


def euclidean_distance(pos_a, pos_b):
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


def evaluate_trajectory(positions, stop_called_at, episode):
    """Return {'steps': snapshots, 'final': last snapshot} without mutating inputs.

    Supplemental success_2m/oracle_success_2m and success_1m/oracle_success_1m
    share the main DistanceToGoal cache. Threshold comparisons are strictly <.
    Official zero-start-distance SPL produces NaN (including reset); we preserve
    that undefined value rather than silently substituting a different metric.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions must be a nonempty N x 3 array")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if stop_called_at is not None and (
        isinstance(stop_called_at, bool) or not isinstance(stop_called_at, int)
        or not 1 <= stop_called_at < len(positions)
    ):
        raise ValueError("stop_called_at must be None or a 1-based update index")
    waypoints = episode["gt_locations"]
    tree = KDTree(waypoints)
    radius = episode["goals"][0]["radius"]
    cached_position = None
    previous = positions[0]
    path_length = 0.0
    agent_distance = 0.0
    oracle_error = float("inf")
    oracle_success = 0.0
    supplemental_oracles = {2: 0.0, 1: 0.0}
    snapshots = []
    for step, current in enumerate(positions):
        # PathLength precedes DistanceToGoal; SPL has its own accumulation.
        if step:
            path_length += euclidean_distance(current, previous)
        if cached_position is None or not np.allclose(
            cached_position, current, atol=1e-4
        ):
            distance, closest = tree.query(current)
            for i in range(closest, len(waypoints) - 1):
                distance += euclidean_distance(waypoints[i], waypoints[i + 1])
            cached_position = (current[0], current[1], current[2])
        stopped = stop_called_at is not None and step >= stop_called_at
        success = 1.0 if stopped and distance < radius else 0.0
        if step == 0:
            start_distance = distance
        agent_distance += euclidean_distance(current, previous)
        spl = success * (start_distance / np.maximum(start_distance, agent_distance))
        oracle_error = min(oracle_error, distance)
        oracle_success = float(oracle_success or distance < radius)
        snapshot = dict(zip(METRIC_NAMES, (path_length, distance, success, spl,
                                          oracle_error, oracle_success)))
        for extra_radius in supplemental_oracles:
            supplemental_oracles[extra_radius] = float(
                supplemental_oracles[extra_radius] or distance < extra_radius
            )
            snapshot[f"success_{extra_radius}m"] = float(stopped and distance < extra_radius)
            snapshot[f"oracle_success_{extra_radius}m"] = supplemental_oracles[extra_radius]
        snapshots.append(snapshot)
        previous = current
    return {"steps": snapshots, "final": snapshots[-1].copy()}
