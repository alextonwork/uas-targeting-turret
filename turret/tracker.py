"""Single-target tracker: detection association, Kalman fusion, coast-through-occlusion.

State machine:
    SEARCH    no target. Gimbal holds, then returns home.
    TENTATIVE detections seen but not yet `confirm_hits` in a row; gimbal not moving yet.
    TRACK     fresh measurement within the last few frames.
    COAST     no gated detection; the Kalman filter is predicting the trajectory.
    LOST      coasted longer than `coast_timeout`; drops back to SEARCH.

Inference thread calls update(); actuation thread calls query(). Both are guarded
by one lock, and each holds it for microseconds.
"""
from __future__ import annotations

import math
import threading

from .controller import make_kalman
from .geometry import CameraModel
from .types import Detection, Frame, TrackEstimate

SEARCH, TENTATIVE, TRACK, COAST, LOST = "SEARCH", "TENTATIVE", "TRACK", "COAST", "LOST"


class Tracker:
    def __init__(self, cfg, backend: str, camera: CameraModel):
        self.cfg = cfg
        self.backend = backend
        self.cam = camera
        self.use_kalman = bool(cfg.kalman.enabled)
        self.gate = cfg.tracker.gate_chi2
        self.max_gate_deg = cfg.tracker.max_gate_deg
        self.gate_growth = cfg.tracker.gate_growth
        self.max_rate = cfg.tracker.max_target_rate
        self.coast_tau = cfg.tracker.coast_velocity_tau
        self.last_target = None   # last aim point handed to the controller
        self.confirm_hits = cfg.tracker.confirm_hits
        self.coast_timeout = cfg.tracker.coast_timeout
        self.latency_comp = cfg.tracker.latency_compensation and self.use_kalman
        self.lock = threading.Lock()
        self.track_frames = 0
        self.coast_frames = 0
        self.reacquisitions = 0
        self._new_track()

    def _new_track(self):
        self.kf = make_kalman(self.cfg, self.backend)
        self.state = SEARCH
        self.hits = 0
        self.t_meas = 0.0        # capture time of the last fused measurement
        self.t_filter = 0.0      # time the Kalman state currently refers to
        self.meas_seq = getattr(self, "meas_seq", 0)
        self.raw = None          # last raw measurement (baseline mode)
        self.alt = None          # out-of-gate re-acquisition candidate ((az, el), t)
        self.lost_at = None

    # ---- inference side ---------------------------------------------------
    def update(self, detections: list[Detection], frame: Frame) -> None:
        cands = [
            (self.cam.pixel_to_absolute(d.u, d.v, frame.pan, frame.tilt), d)
            for d in detections
        ]
        t = frame.t_capture
        with self.lock:
            self._expire(t)
            if self.state in (SEARCH, LOST):
                if cands:
                    (az, el), _ = max(cands, key=lambda c: c[1].conf)
                    self.kf.reset(az, el)
                    self.x_meas = (az, el, 0.0, 0.0)
                    self.raw = (az, el)
                    self.t_meas = self.t_filter = t
                    self.hits = 1
                    self.state = TENTATIVE if self.confirm_hits > 1 else TRACK
                    if self.lost_at is not None:
                        self.reacquisitions += 1
                    self.lost_at = None
                return

            # Associate: predict to this frame's capture time and gate.
            if self.use_kalman:
                self.kf.predict(t - self.t_filter)
                self.t_filter = t
                px, py = self.kf.position
                cap = self.max_gate_deg + self.gate_growth * (t - self.t_meas)
                # Statistical gate, plus a hard cap in degrees: after a long coast the
                # covariance is large enough that the chi2 gate alone would happily
                # accept a clutter detection on the far side of the frame. The cap
                # widens with time since the last hit, because a target that turned
                # while hidden reappears off the predicted path.
                scored = [
                    (self.kf.mahalanobis2(az, el), (az, el))
                    for (az, el), _ in cands
                    if math.hypot(az - px, el - py) <= cap
                ]
                scored = [s for s in scored if s[0] <= self.gate]
            else:
                # Baseline: nearest detection to the last raw measurement.
                scored = [
                    ((az - self.raw[0]) ** 2 + (el - self.raw[1]) ** 2, (az, el))
                    for (az, el), _ in cands
                ]
            if scored:
                self.alt = None
                _, (az, el) = min(scored, key=lambda s: s[0])
                if self.use_kalman:
                    self.kf.update(az, el)
                    self.x_meas = tuple(self.kf.extrapolate(0.0))
                self.raw = (az, el)
                self.t_meas = t
                self.meas_seq += 1
                self.hits += 1
                if self.state != TENTATIVE or self.hits >= self.confirm_hits:
                    self.state = TRACK
                self.track_frames += 1
            elif self.state == TENTATIVE:
                self._new_track()
            else:
                # No gated detection: keep predicting. The filter has already been
                # propagated to t, so its covariance grows while coasting, which
                # widens the gate for re-acquisition after the occlusion.
                self.state = COAST
                self.coast_frames += 1
                if self.use_kalman and cands:
                    self._try_reinit(cands, t)

    def _try_reinit(self, cands, t) -> None:
        """Re-acquire a target that reappeared outside the gate.

        A target that turned while occluded comes back away from the prediction, and
        the gate (correctly) rejects it. Rather than coast blindly until timeout, keep
        the strongest out-of-gate detection as a candidate; if the next frame has a
        detection consistent with it, restart the filter there. Two consistent hits in
        a row is very unlikely from random clutter.
        """
        (az, el), det = max(cands, key=lambda c: c[1].conf)
        if self.alt is not None:
            (a0, e0), t0 = self.alt
            dt = t - t0
            if 0 < dt < 0.2 and math.hypot(az - a0, el - e0) <= 2.0 + self.max_rate * dt:
                self.kf.reset(az, el)
                self.x_meas = (az, el, 0.0, 0.0)
                self.raw = (az, el)
                self.t_meas = self.t_filter = t
                self.meas_seq += 1
                self.state = TRACK
                self.reacquisitions += 1
                self.alt = None
                return
        self.alt = ((az, el), t)

    def _expire(self, now: float) -> None:
        if self.state in (TRACK, COAST) and now - self.t_meas > self.coast_timeout:
            self.state = LOST
            self.lost_at = now
        elif self.state == TENTATIVE and now - self.t_meas > 0.3:
            self._new_track()

    # ---- actuation side ---------------------------------------------------
    def query(self, now: float) -> TrackEstimate:
        with self.lock:
            self._expire(now)
            if self.state not in (TRACK, COAST):
                return TrackEstimate(self.state, None, (0.0, 0.0), self.t_meas, self.meas_seq)
            if not self.use_kalman:
                self.last_target = self.raw
                return TrackEstimate(self.state, self.raw, (0.0, 0.0), self.t_meas, self.meas_seq)
            # The filter state refers to the capture time of the last processed frame.
            x = self.kf.extrapolate(0.0)
            vaz = max(-self.max_rate, min(self.max_rate, x[2]))
            vel = max(-self.max_rate, min(self.max_rate, x[3]))
            if self.state == COAST and self.coast_tau > 0:
                # Coasting: extrapolate from the last real measurement with a velocity
                # that decays over coast_tau. Pure constant-velocity extrapolation
                # overshoots badly when the target turns while hidden; a damped model
                # bounds the worst-case error and still beats holding still.
                xm = self.x_meas
                vaz = max(-self.max_rate, min(self.max_rate, xm[2]))
                vel = max(-self.max_rate, min(self.max_rate, xm[3]))
                age = (now if self.latency_comp else self.t_filter) - self.t_meas
                k = math.exp(-age / self.coast_tau)
                travel = self.coast_tau * (1.0 - k)
                target = (xm[0] + vaz * travel, xm[1] + vel * travel)
                rate = (vaz * k, vel * k)
            else:
                # Latency compensation: extrapolate to "now" so the gimbal aims where
                # the target is, not where it was when the frame was captured.
                dt = max(now - self.t_filter, 0.0) if self.latency_comp else 0.0
                target = (x[0] + vaz * dt, x[1] + vel * dt)
                rate = (vaz, vel)
            self.last_target = target
            return TrackEstimate(self.state, target, rate, self.t_meas, self.meas_seq)
