"""Model-only velocity/STOP contract for official NaVILA episodes."""
import math

IMAGE_KEY = 'observation.images.front'
STATE_KEY = 'observation.state'
STOP_THRESHOLDS = (0.03, 0.03, 0.05)


def validate_policy_input(payload):
    if set(payload) != {IMAGE_KEY, STATE_KEY, 'task'}:
        raise ValueError('only RGB, zero state and original instruction are allowed')
    if payload[STATE_KEY] != [0.0, 0.0, 0.0]:
        raise ValueError('no-state checkpoint requires a constant zero placeholder')
    if not isinstance(payload['task'], str) or not payload['task'].strip():
        raise ValueError('nonempty official instruction required')
    if not isinstance(payload[IMAGE_KEY], str) or not payload[IMAGE_KEY]:
        raise ValueError('RGB file required')
    return dict(payload)  # Preserve official instruction whitespace verbatim.


def velocity_stop(raw):
    if len(raw) != 3 or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in raw):
        raise ValueError('three finite model velocities required')
    return all(abs(float(v)) < threshold for v, threshold in zip(raw, STOP_THRESHOLDS))


def hold_steps(control_dt):
    count = round(0.2 / control_dt)
    if count < 1 or not math.isclose(count * control_dt, 0.2, abs_tol=1e-9):
        raise ValueError('5 Hz must be an integer number of official control steps')
    return count
