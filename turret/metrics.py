"""Latency and throughput instrumentation.

Per processed frame we record:
  queue_ms      capture -> inference start (time the frame waited for the detector)
  infer_ms      detector wall time (preprocess + model + NMS)
  track_ms      tracker update
  e2e_ms        capture -> first servo command computed from that frame's measurement
                ("glass to servo"; the number that matters for closed-loop stability)

and FPS for each of the three threads. Percentiles are over a rolling window.
"""
from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict, deque

import numpy as np


class RateCounter:
    def __init__(self, window: float = 2.0):
        self.window = window
        self.times: deque = deque()

    def tick(self, t: float | None = None):
        t = time.perf_counter() if t is None else t
        self.times.append(t)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()

    def rate(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class Metrics:
    def __init__(self, window: int = 300, csv_path: str | None = None):
        self.lock = threading.Lock()
        self.samples = defaultdict(lambda: deque(maxlen=window))
        self.totals = defaultdict(list)
        self.rates = defaultdict(RateCounter)
        self.counters = defaultdict(int)
        self.csv_path = csv_path
        self._csv_rows: list[dict] = []

    def record(self, name: str, ms: float) -> None:
        with self.lock:
            self.samples[name].append(ms)
            self.totals[name].append(ms)

    def tick(self, name: str, t: float | None = None) -> None:
        with self.lock:
            self.rates[name].tick(t)

    def count(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] += n

    def frame_row(self, row: dict) -> None:
        if self.csv_path:
            with self.lock:
                self._csv_rows.append(row)

    def snapshot(self) -> dict:
        with self.lock:
            out = {"fps": {k: round(v.rate(), 1) for k, v in self.rates.items()},
                   "counters": dict(self.counters)}
            for k, v in self.samples.items():
                if v:
                    a = np.fromiter(v, float)
                    out[k] = {"p50": float(np.percentile(a, 50)),
                              "p95": float(np.percentile(a, 95)),
                              "max": float(a.max())}
            return out

    def summary(self) -> dict:
        with self.lock:
            out = {}
            for k, v in self.totals.items():
                if v:
                    a = np.asarray(v)
                    out[k] = {"n": len(a), "mean": float(a.mean()),
                              "p50": float(np.percentile(a, 50)),
                              "p95": float(np.percentile(a, 95)),
                              "p99": float(np.percentile(a, 99)), "max": float(a.max())}
            out["counters"] = dict(self.counters)
            return out

    def format_line(self) -> str:
        s = self.snapshot()
        fps = s["fps"]
        parts = [f"fps cap {fps.get('capture', 0):4.1f} inf {fps.get('inference', 0):4.1f} "
                 f"act {fps.get('actuation', 0):5.1f}"]
        for k in ("infer_ms", "e2e_ms"):
            if k in s:
                parts.append(f"{k[:-3]} p50 {s[k]['p50']:5.1f} p95 {s[k]['p95']:5.1f} ms")
        dropped = s["counters"].get("frames_dropped", 0)
        parts.append(f"dropped {dropped}")
        return " | ".join(parts)

    def write_csv(self) -> None:
        if not self.csv_path or not self._csv_rows:
            return
        keys = sorted({k for r in self._csv_rows for k in r})
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._csv_rows)
