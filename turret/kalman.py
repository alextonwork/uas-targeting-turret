"""2D constant-velocity Kalman filter (pure-Python/numpy reference implementation).

State is x = [az, el, v_az, v_el] in degrees and degrees/second, expressed in the
*absolute* gimbal frame (not image pixels). Tracking in the absolute frame means the
filter is unaffected by the camera's own motion: a target that is stationary in the
world stays stationary in state space even while the gimbal slews.

Process noise uses the continuous white-noise-acceleration model, so irregular frame
intervals (dropped frames, variable inference time) are handled correctly.

The C++ version in cpp/include/turret_core/kalman.hpp implements the same math and is
checked against this file in tests/test_cpp_parity.py.
"""
from __future__ import annotations

import numpy as np


class KalmanCV2D:
    def __init__(
        self,
        process_accel_std: float = 60.0,
        meas_std: float = 0.4,
        init_vel_std: float = 30.0,
    ) -> None:
        """
        process_accel_std: std-dev of unmodelled target acceleration (deg/s^2).
            Higher = filter trusts measurements more and reacts faster to manoeuvres.
        meas_std: std-dev of an angular measurement (deg). Roughly detector jitter
            in pixels divided by pixels-per-degree.
        init_vel_std: velocity uncertainty when a track is first created (deg/s).
        """
        self.q = float(process_accel_std) ** 2
        self.r = float(meas_std) ** 2
        self.init_vel_var = float(init_vel_std) ** 2
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * self.r

    # ---- model matrices -------------------------------------------------
    @staticmethod
    def F(dt: float) -> np.ndarray:
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )

    def Q(self, dt: float) -> np.ndarray:
        dt2, dt3 = dt * dt, dt * dt * dt
        q = self.q
        return np.array(
            [
                [dt3 / 3 * q, 0, dt2 / 2 * q, 0],
                [0, dt3 / 3 * q, 0, dt2 / 2 * q],
                [dt2 / 2 * q, 0, dt * q, 0],
                [0, dt2 / 2 * q, 0, dt * q],
            ]
        )

    # ---- filter ops -----------------------------------------------------
    def reset(self, az: float, el: float) -> None:
        self.x = np.array([az, el, 0.0, 0.0])
        self.P = np.diag([self.r, self.r, self.init_vel_var, self.init_vel_var])
        self.initialized = True

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q(dt)

    def update(self, az: float, el: float) -> float:
        """Fuse a measurement. Returns the squared Mahalanobis distance of the innovation."""
        z = np.array([az, el])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        # Joseph form keeps P symmetric positive-definite under round-off.
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(y @ S_inv @ y)

    def mahalanobis2(self, az: float, el: float) -> float:
        """Squared Mahalanobis distance of a candidate measurement (for gating)."""
        y = np.array([az, el]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)

    def extrapolate(self, dt: float) -> np.ndarray:
        """State at now+dt without mutating the filter (used for latency compensation)."""
        return self.F(max(dt, 0.0)) @ self.x

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def position_std(self) -> tuple[float, float]:
        return float(np.sqrt(self.P[0, 0])), float(np.sqrt(self.P[1, 1]))
