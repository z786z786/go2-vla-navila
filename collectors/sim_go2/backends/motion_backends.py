from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from collectors.sim_go2.utils.action_utils import clip_train_action
from collectors.sim_go2.utils.state_utils import yaw_to_quat_wxyz


class SceneWriterProtocol(Protocol):
    def write_data_to_sim(self) -> None: ...
    def update(self, dt: float) -> None: ...


class SimulationProtocol(Protocol):
    def step(self, render: bool = True) -> None: ...


@dataclass
class BackendState:
    x: float
    y: float
    yaw: float


class RootKinematicMotionBackend:
    """Kinematic root-state backend for stable high-level navigation collection."""

    def __init__(
        self,
        *,
        robot: Any,
        scene: SceneWriterProtocol,
        sim: SimulationProtocol,
        physics_dt: float,
        control_decimation: int,
        robot_cfg: Mapping[str, Any],
        controller_cfg: Mapping[str, Any],
        default_joint_pos: Any,
        default_joint_vel: Any,
    ):
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.physics_dt = float(physics_dt)
        self.control_decimation = int(control_decimation)
        self.robot_cfg = robot_cfg
        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        self.action_bounds = {
            "vx_min": float(controller_cfg.get("vx_min", -0.1)),
            "vx_max": float(controller_cfg.get("vx_max", 1.0)),
            "wz_min": float(controller_cfg.get("wz_min", -1.5)),
            "wz_max": float(controller_cfg.get("wz_max", 1.5)),
        }
        self.state = BackendState(x=0.0, y=0.0, yaw=0.0)

    def reset(self, *, x: float, y: float, yaw: float) -> None:
        self.state = BackendState(x=float(x), y=float(y), yaw=float(yaw))
        self._write_robot_root_state(vx=0.0, wz=0.0)

    def apply_action(self, action: list[float] | tuple[float, float]) -> list[float]:
        vx_cmd, wz_cmd = clip_train_action(
            action,
            vx_min=self.action_bounds["vx_min"],
            vx_max=self.action_bounds["vx_max"],
            wz_min=self.action_bounds["wz_min"],
            wz_max=self.action_bounds["wz_max"],
        )
        for _ in range(self.control_decimation):
            self.state.yaw = ((self.state.yaw + wz_cmd * self.physics_dt) + math.pi) % (2.0 * math.pi) - math.pi
            self.state.x += vx_cmd * math.cos(self.state.yaw) * self.physics_dt
            self.state.y += vx_cmd * math.sin(self.state.yaw) * self.physics_dt
            self._write_robot_root_state(vx=vx_cmd, wz=wz_cmd)
            self.robot.set_joint_position_target(self.default_joint_pos)
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(self.physics_dt)
        return [vx_cmd, wz_cmd]

    def warmup(self, warmup_steps: int) -> None:
        for _ in range(int(warmup_steps)):
            self.robot.set_joint_position_target(self.default_joint_pos)
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(self.physics_dt)

    def _write_robot_root_state(self, vx: float, wz: float) -> None:
        root_pose = self.robot.data.default_root_state.clone()
        root_pose[:, 0] = self.state.x
        root_pose[:, 1] = self.state.y
        root_pose[:, 2] = float(self.robot_cfg.get("base_height", 0.4))
        quat = yaw_to_quat_wxyz(self.state.yaw)
        root_pose[:, 3] = quat[0]
        root_pose[:, 4] = quat[1]
        root_pose[:, 5] = quat[2]
        root_pose[:, 6] = quat[3]
        root_velocity = self.robot.data.default_root_state[:, 7:].clone()
        root_velocity[:, 0] = vx * math.cos(self.state.yaw)
        root_velocity[:, 1] = vx * math.sin(self.state.yaw)
        root_velocity[:, 2] = 0.0
        root_velocity[:, 3] = 0.0
        root_velocity[:, 4] = 0.0
        root_velocity[:, 5] = wz
        self.robot.write_root_pose_to_sim(root_pose[:, :7])
        self.robot.write_root_velocity_to_sim(root_velocity)
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)


class LowLevelPolicyMotionBackend:
    def __init__(self, *, robot_cfg: Mapping[str, Any]):
        policy_cfg = robot_cfg.get("low_level_policy", {})
        checkpoint = str(policy_cfg.get("checkpoint", "")).strip()
        if not checkpoint:
            raise NotImplementedError(
                "motion_backend=low_level_policy is reserved but no checkpoint was configured."
            )
        raise NotImplementedError(
            "motion_backend=low_level_policy is reserved for future walking-policy integration."
        )


def build_motion_backend(
    motion_backend_name: str,
    *,
    robot: Any,
    scene: SceneWriterProtocol,
    sim: SimulationProtocol,
    physics_dt: float,
    control_decimation: int,
    robot_cfg: Mapping[str, Any],
    controller_cfg: Mapping[str, Any],
    default_joint_pos: Any,
    default_joint_vel: Any,
):
    backend_name = str(motion_backend_name).strip().lower()
    if backend_name == "root_kinematic":
        return RootKinematicMotionBackend(
            robot=robot,
            scene=scene,
            sim=sim,
            physics_dt=physics_dt,
            control_decimation=control_decimation,
            robot_cfg=robot_cfg,
            controller_cfg=controller_cfg,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
        )
    if backend_name == "low_level_policy":
        return LowLevelPolicyMotionBackend(robot_cfg=robot_cfg)
    raise ValueError(f"Unsupported motion backend: {motion_backend_name}")
