"""Hardware-free simulation: synthetic aerial target, occlusions, noisy detector, servo lag.

Two ways to use it:
  * Real-time, threaded: SimCamera + SimDetector plug into the same pipeline as the
    real camera and ONNX model, with MockDriver as the gimbal (python -m turret --mode sim).
  * Offline, deterministic: run_offline() steps a simulated clock with the same Tracker
    and GimbalController code. Used by scripts/benchmark.py and the tests; a 60 s
    scenario runs in well under a second.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np

from .controller import GimbalController, resolve_backend
from .geometry import CameraModel
from .servo import SlewModel
from .tracker import COAST, TRACK, Tracker
from .types import Detection, Frame


class SimScene:
    """Target trajectory in absolute (az, el) degrees with abrupt manoeuvres and occlusions."""

    DT = 0.001

    def __init__(self, cfg, duration: float | None = None, seed: int | None = None):
        s = cfg.sim
        self.duration = duration or s.duration
        rng = random.Random(s.seed if seed is None else seed)
        n = int(self.duration / self.DT) + 2
        az = np.empty(n)
        el = np.empty(n)
        a, e = rng.uniform(-20, 20), rng.uniform(15, 35)
        lo_az, hi_az = -70.0, 70.0
        lo_el, hi_el = 5.0, 60.0
        i = 0
        while i < n:
            seg = int(rng.uniform(*s.segment_duration) / self.DT)
            speed = rng.uniform(*s.target_speed)
            heading = rng.uniform(0, 2 * math.pi)
            va, ve = speed * math.cos(heading), 0.6 * speed * math.sin(heading)
            for _ in range(min(seg, n - i)):
                a += va * self.DT
                e += ve * self.DT
                if not lo_az <= a <= hi_az:
                    va = -va
                    a = min(hi_az, max(lo_az, a))
                if not lo_el <= e <= hi_el:
                    ve = -ve
                    e = min(hi_el, max(lo_el, e))
                az[i], el[i] = a, e
                i += 1
        self.az, self.el = az, el

        self.occlusions = []
        t = rng.uniform(1.5, 4.0)
        rate = s.occlusions_per_min / 60.0
        while t < self.duration and rate > 0:
            d = rng.uniform(*s.occlusion_duration)
            self.occlusions.append((t, t + d))
            t += d + rng.expovariate(rate) + 0.5

    def truth(self, t: float) -> dict:
        i = min(max(int(t / self.DT), 0), len(self.az) - 1)
        occ = any(a <= t < b for a, b in self.occlusions)
        return {"az": float(self.az[i]), "el": float(self.el[i]), "occluded": occ, "t": t}


class SimSensor:
    """Turns ground truth + gimbal pose into the detections a real detector would emit."""

    def __init__(self, cfg, cam: CameraModel, seed: int = 0):
        self.s = cfg.sim
        self.cam = cam
        self.rng = random.Random(seed + 1)

    def detect(self, truth: dict, pan: float, tilt: float) -> list[Detection]:
        out = []
        d_az, d_el = truth["az"] - pan, truth["el"] - tilt
        visible = (not truth["occluded"]) and self.cam.in_view(d_az, d_el, margin=0.5)
        if visible and self.rng.random() > self.s.miss_prob:
            u, v = self.cam.offset_to_pixel(d_az, d_el)
            n = self.s.meas_noise_px
            out.append(Detection(u + self.rng.gauss(0, n), v + self.rng.gauss(0, n),
                                 24, 16, self.rng.uniform(0.5, 0.9)))
        if self.rng.random() < self.s.false_pos_rate:
            out.append(Detection(self.rng.uniform(0, self.cam.width),
                                 self.rng.uniform(0, self.cam.height), 20, 20,
                                 self.rng.uniform(0.35, 0.6)))
        return out


# ---- real-time sources used by the threaded pipeline --------------------------

class SimCamera:
    """Frame source driven by the wall clock; reads the mock gimbal's current pose."""

    def __init__(self, cfg, servo):
        self.scene = SimScene(cfg, duration=max(cfg.sim.duration, 1.0) + 5.0)
        self.servo = servo
        self.period = 1.0 / cfg.camera.fps
        self.t0 = None
        self.index = 0
        self.next_t = None

    def read(self) -> Frame | None:
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = self.next_t = now
        delay = self.next_t - now
        if delay > 0:
            time.sleep(delay)
        self.next_t += self.period
        t = time.perf_counter()
        pan, tilt = self.servo.get_angles(t)
        self.index += 1
        return Frame(self.index, t, pan, tilt, None, self.scene.truth(t - self.t0))

    def close(self):
        pass


class SimDetector:
    """Stands in for the ONNX model: sleeps for the configured inference time."""

    def __init__(self, cfg, cam: CameraModel):
        self.cfg = cfg
        self.sensor = SimSensor(cfg, cam, cfg.sim.seed)
        self.rng = random.Random(cfg.sim.seed + 2)

    def detect(self, frame: Frame) -> list[Detection]:
        ms = max(1.0, self.rng.gauss(self.cfg.sim.inference_ms, self.cfg.sim.inference_jitter_ms))
        time.sleep(ms / 1000.0)
        return self.sensor.detect(frame.truth, frame.pan, frame.tilt)


# ---- offline deterministic run -------------------------------------------------

@dataclass
class OfflineResult:
    duration: float
    frames: int
    lost_frac: float          # fraction of post-acquisition frames with target outside the FOV
    off_center_frac: float    # ... outside the central 50% of the FOV
    rms_error_deg: float      # pointing error RMS while target in view
    drops: int                # tracker LOST events
    occluded_frames: int
    coast_frames: int

    def as_dict(self):
        return self.__dict__.copy()


def run_offline(cfg, duration: float | None = None, seed: int | None = None) -> OfflineResult:
    backend = resolve_backend(cfg.control.backend)
    cam = CameraModel.from_cfg(cfg.camera)
    scene = SimScene(cfg, duration, seed)
    sensor = SimSensor(cfg, cam, cfg.sim.seed if seed is None else seed)
    tracker = Tracker(cfg, backend, cam)
    ctrl = GimbalController(cfg, backend)
    rng = random.Random((cfg.sim.seed if seed is None else seed) + 3)

    pan_m, tilt_m = SlewModel(cfg.servo.slew_rate), SlewModel(cfg.servo.slew_rate)
    pan_m.t = tilt_m.t = 0.0
    dt_ctrl = 1.0 / cfg.control.rate_hz
    frame_period = 1.0 / cfg.camera.fps

    t, next_frame = 0.0, 0.0
    busy_until, pending = 0.0, None     # single inference worker, latest-frame policy
    queued = None
    frames = post = lost = off = occl = 0
    err2, err_n = 0.0, 0
    acquired = False
    drops, prev_state, lost_since = 0, "SEARCH", None
    half_w, half_h = cam.hfov / 2, cam.vfov / 2

    while t < scene.duration:
        pos = (pan_m.advance(t), tilt_m.advance(t))

        if t >= next_frame:
            next_frame += frame_period
            truth = scene.truth(t)
            queued = Frame(frames, t, pos[0], pos[1], None, truth)  # overwrite = drop stale
            frames += 1
            if acquired:
                post += 1
                d_az, d_el = truth["az"] - pos[0], truth["el"] - pos[1]
                if truth["occluded"]:
                    occl += 1
                if abs(d_az) > half_w or abs(d_el) > half_h:
                    lost += 1
                else:
                    err2 += d_az * d_az + d_el * d_el
                    err_n += 1
                if abs(d_az) > half_w / 2 or abs(d_el) > half_h / 2:
                    off += 1

        if pending is not None and t >= busy_until:
            tracker.update(sensor.detect(pending.truth, pending.pan, pending.tilt), pending)
            pending = None
        if pending is None and queued is not None:
            pending, queued = queued, None
            lat = max(1.0, rng.gauss(cfg.sim.inference_ms, cfg.sim.inference_jitter_ms))
            busy_until = t + lat / 1000.0

        est = tracker.query(t)
        if est.state in (TRACK, COAST):
            acquired = True
            lost_since = None
            sp = ctrl.step(est.target, est.rate, pos, dt_ctrl)
        else:
            if prev_state in (TRACK, COAST):
                drops += 1
            if prev_state in (TRACK, COAST) or lost_since is None:
                lost_since = t
                ctrl.begin_search(pos, ctrl.pan.rate)
            sp = ctrl.lost_step(pos, dt_ctrl, t - lost_since, tracker.last_target)
        prev_state = est.state
        lo_p, hi_p = cfg.control.pan.min_angle, cfg.control.pan.max_angle
        lo_t, hi_t = cfg.control.tilt.min_angle, cfg.control.tilt.max_angle
        pan_m.cmd = min(hi_p, max(lo_p, sp[0]))
        tilt_m.cmd = min(hi_t, max(lo_t, sp[1]))
        t += dt_ctrl

    counted = max(1, post)
    return OfflineResult(
        duration=scene.duration,
        frames=frames,
        lost_frac=lost / counted,
        off_center_frac=off / counted,
        rms_error_deg=math.sqrt(err2 / err_n) if err_n else float("nan"),
        drops=drops,
        occluded_frames=occl,
        coast_frames=tracker.coast_frames,
    )
