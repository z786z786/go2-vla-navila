from .base import BaseRobotAdapter, RobotObservation, TrajectoryStatus, TrajectoryWaypoint
from .mock_adapter import MockRobotAdapter
from .ros2_adapter import Ros2LikeAdapter

try:
    from .bridge_adapter import Go2BridgeAdapter, Go2BridgeProcessTransport
except ImportError:  # Optional runtime-only adapter not present in this workspace.
    Go2BridgeAdapter = None
    Go2BridgeProcessTransport = None

__all__ = [
    "BaseRobotAdapter",
    "RobotObservation",
    "TrajectoryStatus",
    "TrajectoryWaypoint",
    "Go2BridgeAdapter",
    "Go2BridgeProcessTransport",
    "MockRobotAdapter",
    "Ros2LikeAdapter",
]
