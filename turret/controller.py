"""Gimbal control: per-axis controllers and backend selection (C++ core or Python)."""
from __future__ import annotations

import logging

from .kalman import KalmanCV2D as PyKalman
from .pid import PID as PyPID

log = logging.getLogger(__name__)

try:  # Built from cpp/ with CMake; see README "Build the C++ core".
    import turret_core as _core  # type: ignore

    HAVE_CPP = True
except ImportError:  # pragma: no cover - depends on build
    _core = None
    HAVE_CPP = False


class PyAxisController:
    """Python twin of cpp/include/turret_core/axis.hpp."""

    def __init__(self, pid, min_angle, max_angle, max_accel, feedforward_gain=1.0):
        self.pid = pid
        self.min_angle, self.max_angle = min_angle, max_angle
        self.max_accel = max_accel
        self.feedforward_gain = feedforward_gain
        self.setpoint = 0.0
        self.rate = 0.0
        self._have_setpoint = False

    def step(self, target, target_rate, position, dt):
        if dt <= 0:
            return self.setpoint
        if not self._have_setpoint:
            self.setpoint = position
            self._have_setpoint = True
        error = target - position
        rate = self.pid.update(error, position, dt)
        if abs(error) > self.pid.deadband:
            rate += self.feedforward_gain * target_rate
        lim = self.pid.out_limit
        rate = max(-lim, min(lim, rate))
        max_dv = self.max_accel * dt
        rate = max(self.rate - max_dv, min(self.rate + max_dv, rate))
        nxt = self.setpoint + rate * dt
        if nxt > self.max_angle or nxt < self.min_angle:
            nxt = max(self.min_angle, min(self.max_angle, nxt))
            rate = 0.0
        self.rate = rate
        self.setpoint = nxt
        return nxt

    def hold(self, position):
        self.setpoint = position
        self._have_setpoint = True
        self.rate = 0.0
        self.pid.reset()


def resolve_backend(name: str) -> str:
    name = (name or "auto").lower()
    if name == "cpp" and not HAVE_CPP:
        raise RuntimeError("control.backend=cpp but turret_core is not built/importable")
    if name == "auto":
        return "cpp" if HAVE_CPP else "python"
    return name


def make_kalman(cfg, backend: str):
    k = cfg.kalman
    args = (k.process_accel_std, k.meas_std, k.init_vel_std)
    return _core.KalmanCV2D(*args) if backend == "cpp" else PyKalman(*args)


def _make_pid(p, backend):
    args = (p.kp, p.ki, p.kd, p.out_limit, p.i_limit, p.deadband, p.d_filter_hz)
    return _core.PID(*args) if backend == "cpp" else PyPID(*args)


def make_axis(axis_cfg, backend: str):
    pid = _make_pid(axis_cfg.pid, backend)
    cls = _core.AxisController if backend == "cpp" else PyAxisController
    return cls(pid, axis_cfg.min_angle, axis_cfg.max_angle, axis_cfg.max_accel,
               axis_cfg.feedforward_gain)


class GimbalController:
    """Pan + tilt axes, each with its own independently tuned PID."""

    def __init__(self, cfg, backend: str):
        self.pan = make_axis(cfg.control.pan, backend)
        self.tilt = make_axis(cfg.control.tilt, backend)
        self.return_home_rate = cfg.control.return_home_rate
        self.lost_hold = cfg.tracker.lost_hold_time
        sc = cfg.control.search
        self.search_enabled = bool(sc.enabled)
        self.search_rate = sc.rate
        self.search_rows = list(sc.rows)
        self.search_pan = tuple(sc.pan_limits)
        self._row = 0
        self._dir = 1.0

    def step(self, target, target_rate, position, dt):
        """target/target_rate/position are (pan, tilt) tuples. Returns new (pan, tilt) setpoint."""
        return (
            self.pan.step(target[0], target_rate[0], position[0], dt),
            self.tilt.step(target[1], target_rate[1], position[1], dt),
        )

    def hold(self, position):
        self.pan.hold(position[0])
        self.tilt.hold(position[1])

    def home_step(self, position, dt, home=(0.0, 0.0)):
        """Slew slowly back to home after a track is lost."""
        out = []
        for p, h in zip(position, home):
            d = h - p
            step = max(-self.return_home_rate * dt, min(self.return_home_rate * dt, d))
            out.append(p + step)
        self.hold(tuple(out))
        return tuple(out)

    def lost_step(self, position, dt, time_since_lost, last_target):
        """What to do without a track: aim at the last estimate for a moment (the target
        often reappears right there), then scan, or return home if scanning is off."""
        if last_target is not None and time_since_lost < self.lost_hold:
            return self.step(last_target, (0.0, 0.0), position, dt)
        if self.search_enabled:
            return self.search_step(position, dt)
        return self.home_step(position, dt)

    def begin_search(self, position, pan_rate_hint=0.0):
        self._dir = 1.0 if pan_rate_hint >= 0 else -1.0
        # start on the scan row nearest the current tilt
        self._row = min(range(len(self.search_rows)),
                        key=lambda i: abs(self.search_rows[i] - position[1]))

    def search_step(self, position, dt):
        """Boustrophedon raster: sweep pan across search.pan_limits, step tilt each pass."""
        pan, tilt = position
        lo, hi = self.search_pan
        if (self._dir > 0 and pan >= hi) or (self._dir < 0 and pan <= lo):
            self._dir = -self._dir
            self._row = (self._row + 1) % len(self.search_rows)
        step = self.search_rate * dt
        new_pan = max(lo, min(hi, pan + self._dir * step))
        row = self.search_rows[self._row]
        new_tilt = tilt + max(-step, min(step, row - tilt))
        self.hold((new_pan, new_tilt))
        return new_pan, new_tilt

