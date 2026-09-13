"""Fixed paired 2 geometry × 4 task × 2 repeat DT1 expert schedule.

The schedule is deliberately a planning artifact, not a policy input.  It
pairs red and blue instructions within one physical color configuration so
both runs must consume the same reset seed and can be compared by
``reset_audit`` after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layouts import TaskSpec, build_four_tasks
from .reset_audit import PAIR_SEED_ALGORITHM, ResetAuditError, derive_pair_seed
from .scene import ColorConfiguration, DualTargetSceneSpec, dt1_development_groups


DT1_EXPERT_REPEATS = 2
_CONFIGURATIONS: tuple[ColorConfiguration, ...] = ("A_red_B_blue", "A_blue_B_red")


class ExpertMatrixError(ValueError):
    """The bounded DT1 expert matrix lost its paired-task contract."""


@dataclass(frozen=True)
class ExpertPair:
    """One same-reset red/blue instruction pair in the development matrix."""

    scene: DualTargetSceneSpec
    repeat: int
    pair_seed: int
    red_task: TaskSpec
    blue_task: TaskSpec

    def validate(self) -> None:
        if self.repeat < 0:
            raise ExpertMatrixError("repeat must be non-negative")
        if self.red_task.target_color != "red" or self.blue_task.target_color != "blue":
            raise ExpertMatrixError("each paired expert slot must contain red then blue instructions")
        for task in (self.red_task, self.blue_task):
            if task.geometry_group_id != self.scene.group.geometry_group_id:
                raise ExpertMatrixError("expert task geometry does not match its paired scene")
            if task.color_configuration != self.scene.color_configuration:
                raise ExpertMatrixError("expert task color configuration does not match paired scene")
            if task.start != self.scene.group.start:
                raise ExpertMatrixError("expert task reset start does not match paired scene")
        expected = derive_pair_seed(
            geometry_group_id=self.scene.group.geometry_group_id,
            color_configuration=self.scene.color_configuration,
            repeat=self.repeat,
        )
        if self.pair_seed != expected:
            raise ExpertMatrixError("expert pair seed is not color/instruction independent")

    @property
    def pair_id(self) -> str:
        return f"{self.scene.group.geometry_group_id}__{self.scene.color_configuration}__r{self.repeat}"

    def task_slots(self) -> tuple[tuple[str, TaskSpec], tuple[str, TaskSpec]]:
        self.validate()
        return (("red", self.red_task), ("blue", self.blue_task))


def build_dt1_expert_pairs(*, repeats: int = DT1_EXPERT_REPEATS) -> tuple[ExpertPair, ...]:
    """Return exactly eight pairs for the approved 16-task DT1 precheck."""
    if repeats != DT1_EXPERT_REPEATS:
        raise ExpertMatrixError("DT1 expert matrix is frozen to exactly two paired repeats")
    pairs: list[ExpertPair] = []
    for group in dt1_development_groups():
        all_tasks = build_four_tasks(group)
        for configuration in _CONFIGURATIONS:
            tasks = {task.target_color: task for task in all_tasks if task.color_configuration == configuration}
            if set(tasks) != {"red", "blue"}:
                raise ExpertMatrixError("each color configuration must yield one red and one blue task")
            scene = DualTargetSceneSpec(group, configuration)
            for repeat in range(repeats):
                pair = ExpertPair(
                    scene=scene,
                    repeat=repeat,
                    pair_seed=derive_pair_seed(
                        geometry_group_id=group.geometry_group_id,
                        color_configuration=configuration,
                        repeat=repeat,
                    ),
                    red_task=tasks["red"],
                    blue_task=tasks["blue"],
                )
                pair.validate()
                pairs.append(pair)
    if len(pairs) != 8:
        raise ExpertMatrixError("DT1 expert matrix must have exactly eight reset pairs")
    return tuple(pairs)


def flatten_dt1_expert_pairs(pairs: tuple[ExpertPair, ...] | None = None) -> tuple[tuple[ExpertPair, str, TaskSpec], ...]:
    """Return the exact 16 tasks, paired adjacency retained for audit tooling."""
    selected = build_dt1_expert_pairs() if pairs is None else pairs
    if len(selected) != 8:
        raise ExpertMatrixError("DT1 flattened expert matrix requires exactly eight pairs")
    slots = tuple(slot for pair in selected for color, task in pair.task_slots() for slot in ((pair, color, task),))
    if len(slots) != 16:
        raise ExpertMatrixError("DT1 flattened expert matrix must contain exactly 16 tasks")
    return slots


def expert_matrix_manifest() -> dict[str, object]:
    """CPU-serializable schedule; no simulator or policy dependency."""
    pairs = build_dt1_expert_pairs()
    return {
        "format": "go2-dual-target-dt1-expert-matrix-v1",
        "status": "SCHEDULED_NOT_TASK_APPROVED",
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM,
        "geometry_groups": 2,
        "color_instruction_combinations_per_group": 4,
        "paired_repeats": DT1_EXPERT_REPEATS,
        "pair_count": len(pairs),
        "task_count": len(flatten_dt1_expert_pairs(pairs)),
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "geometry_group_id": pair.scene.group.geometry_group_id,
                "color_configuration": pair.scene.color_configuration,
                "repeat": pair.repeat,
                "pair_seed": pair.pair_seed,
                "task_colors": [color for color, _ in pair.task_slots()],
                "instructions": [task.instruction for _, task in pair.task_slots()],
            }
            for pair in pairs
        ],
        "navigation_success_approved": False,
        "dt1_approved": False,
    }
