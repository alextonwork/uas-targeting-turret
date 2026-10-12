"""Three-thread pipeline: capture -> inference -> actuation.

    CaptureThread    grabs frames as fast as the camera delivers, stamps time + gimbal pose,
                     and drops them into a 1-slot mailbox (newer frame overwrites older).
    InferenceThread  takes the newest frame, runs the detector, updates the tracker.
    ActuationThread  runs at control.rate_hz independent of camera/inference rate: asks the
                     tracker for a latency-compensated target estimate, steps both PID axes,
                     writes the servos.

Decoupling matters: inference (~30 ms) is the slowest stage, and if the actuation loop
ran only when a detection arrived, servo updates would be jerky 30 Hz steps. Running it at
100 Hz on Kalman predictions gives smooth motion and lets the gimbal keep moving through
missed frames and occlusions. The 1-slot mailbox means a slow detector never builds a
queue of stale frames, which would silently add latency.
"""
from __future__ import annotations

import logging
import threading
import time

from .controller import GimbalController
from .tracker import COAST, TRACK, Tracker

log = logging.getLogger(__name__)


class LatestSlot:
    """Single-item mailbox. put() overwrites; get() blocks until a new item arrives."""

    def __init__(self):
        self._item = None
        self._cv = threading.Condition()
        self.overwritten = 0

    def put(self, item) -> None:
        with self._cv:
            if self._item is not None:
                self.overwritten += 1
            self._item = item
            self._cv.notify()

    def get(self, timeout: float | None = None):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


class Pipeline:
    def __init__(self, cfg, camera, detector, tracker: Tracker, controller: GimbalController,
                 servo, metrics, on_result=None):
        self.cfg = cfg
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.ctrl = controller
        self.servo = servo
        self.metrics = metrics
        self.on_result = on_result          # optional callback(frame, detections, estimate)
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="inference", daemon=True),
            threading.Thread(target=self._actuation_loop, name="actuation", daemon=True),
        ]
        self.error: BaseException | None = None
        # capture time of each fused measurement, keyed by tracker meas_seq, for e2e latency
        self._pending_e2e: dict[int, float] = {}
        self._e2e_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------------
    def start(self):
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        self.slot.put(None)
        for t in self.threads:
            t.join(timeout=2.0)

    def _guard(self, fn):
        try:
            fn()
        except BaseException as e:  # surface thread crashes to the main thread
            log.exception("thread %s crashed", threading.current_thread().name)
            self.error = e
            self.stop_event.set()

    # ---- threads ------------------------------------------------------------
    def _capture_loop(self):
        def run():
            while not self.stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    log.info("camera stream ended")
                    self.stop_event.set()
                    break
                self.metrics.tick("capture", frame.t_capture)
                before = self.slot.overwritten
                self.slot.put(frame)
                if self.slot.overwritten > before:
                    self.metrics.count("frames_dropped")
        self._guard(run)

    def _inference_loop(self):
        def run():
            while not self.stop_event.is_set():
                frame = self.slot.get(timeout=0.5)
                if frame is None:
                    continue
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                t1 = time.perf_counter()
                seq_before = self.tracker.meas_seq
                self.tracker.update(dets, frame)
                t2 = time.perf_counter()
                if self.tracker.meas_seq != seq_before:
                    with self._e2e_lock:
                        self._pending_e2e[self.tracker.meas_seq] = frame.t_capture
                q_ms, i_ms, k_ms = (t0 - frame.t_capture) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3
                self.metrics.record("queue_ms", q_ms)
                self.metrics.record("infer_ms", i_ms)
                self.metrics.record("track_ms", k_ms)
                self.metrics.tick("inference", t2)
                self.metrics.count("frames_processed")
                self.metrics.frame_row({"frame": frame.index, "t_capture": frame.t_capture,
                                        "queue_ms": q_ms, "infer_ms": i_ms, "track_ms": k_ms,
                                        "n_det": len(dets), "state": self.tracker.state})
                if self.on_result:
                    self.on_result(frame, dets, self.tracker.query(t2))
        self._guard(run)

    def _actuation_loop(self):
        def run():
            period = 1.0 / self.cfg.control.rate_hz
            last = time.perf_counter()
            next_t = last + period
            lost_since = None
            prev_state = None
            while not self.stop_event.is_set():
                time.sleep(max(0.0, next_t - time.perf_counter()))
                next_t += period
                now = time.perf_counter()
                dt, last = now - last, now
                pos = self.servo.get_angles(now)
                est = self.tracker.query(now)
                if est.state in (TRACK, COAST):
                    lost_since = None
                    sp = self.ctrl.step(est.target, est.rate, pos, dt)
                    self.servo.set_angles(*sp)
                    t_sent = time.perf_counter()
                    with self._e2e_lock:
                        t_cap = self._pending_e2e.pop(est.meas_seq, None)
                        # older measurements superseded before actuation ran are discarded
                        for k in [k for k in self._pending_e2e if k < est.meas_seq]:
                            del self._pending_e2e[k]
                    if t_cap is not None:
                        self.metrics.record("e2e_ms", (t_sent - t_cap) * 1e3)
                else:
                    if prev_state in (TRACK, COAST) or lost_since is None:
                        lost_since = now
                        self.ctrl.begin_search(pos, self.ctrl.pan.rate)
                        if prev_state in (TRACK, COAST):
                            log.info("track lost")
                    sp = self.ctrl.lost_step(pos, dt, now - lost_since, self.tracker.last_target)
                    self.servo.set_angles(*sp)
                if est.state != prev_state:
                    log.debug("tracker %s -> %s", prev_state, est.state)
                prev_state = est.state
                self.metrics.tick("actuation", now)
        self._guard(run)
