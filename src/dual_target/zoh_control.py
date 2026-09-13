"""Versioned 5 Hz command boundary shared by expert and learned controllers.

No simulator or model dependencies. Rate limits below are probe candidates,
not approved physical calibration. The old locked contract is untouched.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ZohSpec:
    physics_dt_s: float = .005
    decimation: int = 4
    hold_control_steps: int = 10
    chunk_size: int = 5
    execute_steps: int = 1
    bounds: tuple = ((0., .5), (0., 0.), (-.5, .5))
    rate_limits: tuple = (.5, 0., 1.)

    def __post_init__(self):
        if (self.physics_dt_s != .005 or self.decimation != 4
                or self.hold_control_steps != 10 or self.execute_steps != 1
                or type(self.chunk_size) is not int or self.chunk_size < 1):
            raise ValueError('keep approved physics and 5 Hz ZOH timing; chunk is configurable')
        if self.bounds != ((0., .5), (0., 0.), (-.5, .5)):
            raise ValueError('do not expand approved command bounds')
        if (len(self.rate_limits) != 3 or self.rate_limits[1] != 0
                or any(not math.isfinite(v) or v < 0 for v in self.rate_limits)
                or self.rate_limits[0] <= 0 or self.rate_limits[2] <= 0):
            raise ValueError('finite positive vx/wz rate limits and fixed-zero vy required')

    @property
    def action_dt_s(self):
        return self.physics_dt_s * self.decimation * self.hold_control_steps

    def metadata(self):
        return dict(asdict(self), action_dt_s=self.action_dt_s,
                    reset_previous_applied=[0., 0., 0.],
                    order='finite_check -> absolute_clip -> slew_from_previous_applied -> ZOH',
                    rate_limit_status='CANDIDATE_REQUIRES_LIVE_APPROVAL')


class CommandBoundary:
    def __init__(self, spec=ZohSpec()):
        self.spec = spec
        self.reset()

    def reset(self):
        self.previous = (0., 0., 0.)

    def update(self, raw):
        if len(raw) != 3 or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in raw):
            raise ValueError('expected finite three-dimensional command')
        clipped = tuple(max(lo, min(hi, float(v))) for v, (lo, hi) in zip(raw, self.spec.bounds))
        applied = tuple(max(p-rate*self.spec.action_dt_s, min(p+rate*self.spec.action_dt_s, v))
                        for p, rate, v in zip(self.previous, self.spec.rate_limits, clipped))
        self.previous = applied
        return applied


class ZohClock:
    """Exactly one source call/limiter update per ten low-level actions."""
    def __init__(self, spec=ZohSpec()):
        self.spec = spec
        self.boundary = CommandBoundary(spec)
        self.step = 0
        self.raw = self.applied = (0., 0., 0.)

    @property
    def needs_command(self):
        return self.step % self.spec.hold_control_steps == 0

    def advance(self, command=None):
        if self.needs_command:
            if command is None:
                raise ValueError('missing command at ZOH boundary')
            applied = self.boundary.update(command)
            self.raw, self.applied = tuple(map(float, command)), applied
        elif command is not None:
            raise ValueError('source must not update between ZOH boundaries')
        self.step += 1
        return self.raw, self.applied


def probe_command(index):
    """Fixed forward/stop, left/stop, right/stop sequence; never uses GT."""
    for count, command in ((10,(0.,0.,0.)), (5,(.3,0.,0.)), (15,(0.,0.,0.)),
                           (5,(0.,0.,.4)), (15,(0.,0.,0.)),
                           (5,(0.,0.,-.4)), (15,(0.,0.,0.))):
        if index < count:
            return command
        index -= count
    raise ValueError('probe exceeded fixed 70 command budget')


def validate_trajectory_splits(records):
    """Every whole trajectory and related geometry lineage belongs to one split."""
    owners = {}
    for row in records:
        split = row['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        for field in ('trajectory_id', 'geometry_group_id', 'lineage_root_id'):
            value = row[field]
            if not isinstance(value, str) or not value:
                raise ValueError('missing grouping provenance')
            key = field, value
            if key in owners and owners[key] != split:
                raise ValueError('cross-split trajectory/geometry/lineage leakage')
            owners[key] = split
    return True
