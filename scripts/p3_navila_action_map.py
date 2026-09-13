"""CPU-only P3-T2 probe: execution ASSUMPTION, NOT supervision.

Source root: /mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/
isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py:
  forward 25/50/75 cm: 79-80 / 77-78 / 75-76;
  left 15/30/45 deg: 63-64 / 61-62 / 59-60;
  right 15/30/45 deg: 71-72 / 69-70 / 67-68; stop: 82-83.
Units follow the velocity*time integral, not an explicit upstream unit parser.

Requests use floor(duration / 0.2) full-speed ticks, each held for 10
low-level steps. STOP emits one zero-velocity stop-intent tick. These are
raw NavCommand-shaped requests, before the platform's shared slew limiter;
integrals/residuals are ideal command arithmetic, not measured robot motion.
This probe does not implement the P1 stateful adapter or STOP braking/latch.
"""

import argparse
import json
import math
import re


TICK_S = 0.2
LOW_LEVEL_STEPS_PER_TICK = 10
MAX_VX = 0.5
MAX_WZ = 0.5
FORWARD_CM = (25, 50, 75)
TURN_DEG = (15, 30, 45)


def expand_action(text: str) -> dict:
    """Strict local grammar; reject unknown values rather than upstream fallback.

    Accept `move forward {25,50,75} cm`, `turn {left,right} {15,30,45}
    degrees`, or `stop` (case/whitespace normalized, optional final period).
    This spelling is a probe interface, not a verified training serialization.
    """
    if not isinstance(text, str):
        raise ValueError("action must be a string")
    normalized = " ".join(text.lower().split())
    if normalized.endswith("."):
        normalized = normalized[:-1]
    vx = wz = target = 0.0
    stop = normalized == "stop"
    unit = "m"
    if stop:
        kind = "stop"
    elif match := re.fullmatch(r"move forward (25|50|75) cm", normalized):
        kind = "forward"
        target = int(match[1]) / 100.0
        vx = MAX_VX
    elif match := re.fullmatch(r"turn (left|right) (15|30|45) degrees", normalized):
        kind = match[1]
        target = math.radians(int(match[2]))
        wz = MAX_WZ if kind == "left" else -MAX_WZ
        unit = "rad"
    else:
        raise ValueError("unsupported or malformed NaVILA benchmark macro-action")

    speed = MAX_VX if kind == "forward" else MAX_WZ
    ideal_duration = target / speed
    motion_ticks = math.floor(ideal_duration / TICK_S)
    tick_count = 1 if stop else motion_ticks
    executed = motion_ticks * TICK_S * speed
    command = {"vx": vx, "vy": 0.0, "wz": wz, "stop": stop}
    return {
        "action": normalized,
        "semantics": "EXECUTION_ASSUMPTION_NOT_SUPERVISION",
        "command_stage": "raw_before_shared_slew_limiter",
        "rounding": "floor_full_speed_ticks",
        "tick_s": TICK_S,
        "low_level_steps_per_tick": LOW_LEVEL_STEPS_PER_TICK,
        "tick_count": tick_count,
        "motion_tick_count": motion_ticks,
        "low_level_steps": tick_count * LOW_LEVEL_STEPS_PER_TICK,
        "ideal_duration_s": ideal_duration,
        "emitted_duration_s": tick_count * TICK_S,
        "truncated_time_s": ideal_duration - motion_ticks * TICK_S,
        "target_magnitude": target,
        "integrated_request_magnitude": executed,
        "truncation_residual": target - executed,
        "residual_unit": unit,
        "commands": [dict(command) for _ in range(tick_count)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", help='e.g. "turn left 30 degrees"')
    args = parser.parse_args()
    try:
        result = expand_action(args.action)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
