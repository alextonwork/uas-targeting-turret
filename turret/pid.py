"""Single-axis PID with deadband, anti-windup and filtered derivative-on-measurement.

The controller output is a *rate* command (deg/s). Hobby servos are position-controlled,
so the actuation loop integrates this rate into a position setpoint. Commanding rate
rather than position directly is what lets the PID shape the approach and remove the
overshoot/hunting you get from slamming the servo straight to the target angle.

Pure-Python reference; cpp/include/turret_core/pid.hpp is the C++ twin.
"""
from __future__ import annotations

import math


class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        out_limit: float = 300.0,
        i_limit: float = 50.0,
        deadband: float = 0.0,
        d_filter_hz: float = 15.0,
    ) -> None:
        """
        out_limit: |output| clamp (deg/s); keep at or below the servo's real slew rate.
        i_limit: |integral term| clamp in output units (anti-windup).
        deadband: |error| (deg) below which the axis is considered on-target. Inside it
            the output is zero and the integrator bleeds off, which stops servo hunting
            caused by detector jitter and gear backlash.
        d_filter_hz: first-order low-pass cutoff on the derivative term.
        """
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.deadband = deadband
        self.d_filter_hz = d_filter_hz
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.d_state = 0.0
        self.prev_meas: float | None = None
        self.last_error = 0.0
        self.last_output = 0.0

    def update(self, error: float, measurement: float, dt: float) -> float:
        """
        error: setpoint - measurement (deg).
        measurement: the controlled variable (deg). Used for the D-term so a sudden
            jump in setpoint (new target, re-acquisition) causes no derivative kick.
        dt: seconds since last call.
        """
        if dt <= 0:
            return self.last_output
        self.last_error = error

        # Derivative on measurement, low-pass filtered.
        if self.prev_meas is None:
            raw_d = 0.0
        else:
            raw_d = -(measurement - self.prev_meas) / dt
        self.prev_meas = measurement
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.d_filter_hz * dt)
        self.d_state += alpha * (raw_d - self.d_state)

        if abs(error) <= self.deadband:
            # On target: hold still and let the integrator decay instead of winding.
            self.integral *= math.exp(-dt / 0.25)
            self.last_output = 0.0
            return 0.0

        # Shift the error so the output is continuous at the deadband edge.
        e = error - math.copysign(self.deadband, error)

        p = self.kp * e
        d = self.kd * self.d_state
        candidate_i = self.integral + self.ki * e * dt
        candidate_i = max(-self.i_limit, min(self.i_limit, candidate_i))
        out = p + candidate_i + d

        # Conditional integration: don't integrate further into saturation.
        if abs(out) > self.out_limit and math.copysign(1, out) == math.copysign(1, e):
            out = math.copysign(self.out_limit, out)
        else:
            self.integral = candidate_i
            out = max(-self.out_limit, min(self.out_limit, out))

        self.last_output = out
        return out
