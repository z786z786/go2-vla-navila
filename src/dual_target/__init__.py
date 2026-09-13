"""CPU-only contracts shared by the dual-target Go2 experiment.

Isaac and SmolVLA are deliberately not imported here.  DT0 must be runnable on
an ordinary CPU host so that the task contract can be reviewed before any
simulator or GPU process is started.
"""

from .contracts import ACTION_NAMES, POLICY_INPUTS, STATE_NAMES

__all__ = ("ACTION_NAMES", "POLICY_INPUTS", "STATE_NAMES")
