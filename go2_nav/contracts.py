"""Versioned observation/action contracts, shared by simulation and robot adapters."""
from dataclasses import asdict, dataclass
import math

SCHEMA_VERSION = 1


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Action:
    vx: float = 0.0
    wz: float = 0.0
    stop: bool = False

    def __post_init__(self):
        finite(self.vx, "vx")
        finite(self.wz, "wz")
        if type(self.stop) is not bool:
            raise ValueError("stop must be boolean")
        if self.stop and (self.vx != 0 or self.wz != 0):
            raise ValueError("stop action must have zero velocity")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    timestamp: float
    instruction: str
    rgb_path: str
    state: tuple[float, float, float]

    def __post_init__(self):
        if finite(self.timestamp, "timestamp") < 0:
            raise ValueError("negative timestamp")
        if not self.instruction.strip() or not self.rgb_path:
            raise ValueError("instruction and RGB path are required")
        if len(self.state) != 3:
            raise ValueError("state must contain vx, vy, wz")
        for v in self.state:
            finite(v, "state")

    def to_dict(self):
        return asdict(self)


def validate_episode(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if record.get("source") not in {"mock", "simulation", "real"}:
        raise ValueError("source must identify mock, simulation, or real")
    if not isinstance(record.get("episode_id"), str) or not record["episode_id"]:
        raise ValueError("episode_id is required")
    if record.get("outcome") not in {"stopped", "timeout"}:
        raise ValueError("invalid outcome")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("episode must contain steps")
    previous = -1.0
    for i, step in enumerate(steps):
        obs = Observation(**step["observation"])
        action = Action(**step["action"])
        if obs.timestamp <= previous:
            raise ValueError("timestamps must increase strictly")
        previous = obs.timestamp
        if action.stop and i != len(steps) - 1:
            raise ValueError("steps after stop are invalid")
    if (record["outcome"] == "stopped") != steps[-1]["action"]["stop"]:
        raise ValueError("terminal action and outcome disagree")
    return record


@dataclass
class ActionFilter:
    max_vx: float = 0.35
    max_wz: float = 0.5
    acceleration: float = 0.8
    angular_acceleration: float = 1.5
    clipping: bool = True
    slew: bool = True
    previous: Action = Action()

    def __post_init__(self):
        for name in ("max_vx", "max_wz", "acceleration", "angular_acceleration"):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")

    def apply(self, action, dt):
        if finite(dt, "dt") <= 0:
            raise ValueError("dt must be positive")
        if action.stop:
            self.previous = Action(stop=True)
            return self.previous
        vx, wz = action.vx, action.wz
        if self.clipping:
            vx = min(self.max_vx, max(0.0, vx))
            wz = min(self.max_wz, max(-self.max_wz, wz))
        if self.slew:
            vx = min(self.previous.vx + self.acceleration*dt, max(self.previous.vx-self.acceleration*dt, vx))
            wz = min(self.previous.wz + self.angular_acceleration*dt, max(self.previous.wz-self.angular_acceleration*dt, wz))
        self.previous = Action(vx, wz)
        return self.previous
