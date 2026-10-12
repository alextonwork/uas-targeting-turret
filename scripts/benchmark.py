#!/usr/bin/env python3
"""Offline benchmarks on the deterministic simulator (no hardware, runs in seconds).

  occlusion   Kalman tracker vs. baseline (steer on raw detections, hold last position
              when the detector loses the target), over many random scenarios with
              occlusions and abrupt velocity changes. Reports tracking loss, i.e. the
              share of post-acquisition frames where the target is outside the camera FOV.
  step        Step response of each axis for PID tuning: overshoot, settling time, and
              how many times the axis reverses direction after settling (hunting).

    python scripts/benchmark.py occlusion --seeds 40
    python scripts/benchmark.py step --set control.pan.pid.kp=8
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turret.config import load_config  # noqa: E402
from turret.controller import make_axis, resolve_backend  # noqa: E402
from turret.servo import SlewModel  # noqa: E402
from turret.sim import run_offline  # noqa: E402


def occlusion(args):
    rows = {}
    for label, enabled in (("baseline", False), ("kalman", True)):
        cfg = load_config(args.config, args.set + [f"kalman.enabled={enabled}"])
        rs = [run_offline(cfg, duration=args.duration, seed=s) for s in range(args.seeds)]
        rows[label] = rs
        lf = [r.lost_frac for r in rs]
        print(f"{label:9s} track loss {100 * st.mean(lf):5.2f}% (median {100 * st.median(lf):5.2f}%)  "
              f"rms err {st.mean(r.rms_error_deg for r in rs):4.2f} deg  "
              f"drops {sum(r.drops for r in rs):3d}  coast frames {sum(r.coast_frames for r in rs)}")
    b = st.mean(r.lost_frac for r in rows["baseline"])
    k = st.mean(r.lost_frac for r in rows["kalman"])
    wins = sum(kr.lost_frac <= br.lost_frac for kr, br in zip(rows["kalman"], rows["baseline"]))
    if b > 0:
        print(f"\ntracking-loss reduction with Kalman: {100 * (b - k) / b:.1f}%  "
              f"(Kalman equal or better in {wins}/{args.seeds} scenarios, "
              f"{args.seeds} x {args.duration:.0f} s)")


def step(args):
    cfg = load_config(args.config, args.set)
    backend = resolve_backend(cfg.control.backend)
    dt = 1.0 / cfg.control.rate_hz
    for name in ("pan", "tilt"):
        ax_cfg = cfg.control[name]
        ax = make_axis(ax_cfg, backend)
        servo = SlewModel(cfg.servo.slew_rate)
        servo.t = 0.0
        target = args.amplitude
        t, peak, settled_at, reversals, last_dir = 0.0, 0.0, None, 0, 0
        band = max(ax_cfg.pid.deadband, 0.05) + 0.25
        while t < args.time:
            pos = servo.advance(t)
            servo.cmd = ax.step(target, 0.0, pos, dt)
            peak = max(peak, pos)
            if abs(pos - target) <= band:
                settled_at = settled_at if settled_at is not None else t
            else:
                settled_at = None
            d = (servo.cmd > pos + 1e-6) - (servo.cmd < pos - 1e-6)
            if settled_at is not None and d and last_dir and d != last_dir:
                reversals += 1
            if d:
                last_dir = d
            t += dt
        overshoot = 100 * max(0.0, peak - target) / target
        st_txt = f"{settled_at:.3f} s" if settled_at is not None else "never"
        print(f"{name:4s} step {target:.0f} deg: overshoot {overshoot:4.1f}%  settle(±{band:.2f}) {st_txt}  "
              f"final err {target - pos:+.3f} deg  hunting reversals {reversals}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=["occlusion", "step"])
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--amplitude", type=float, default=20.0)
    p.add_argument("--time", type=float, default=2.0)
    a = p.parse_args()
    {"occlusion": occlusion, "step": step}[a.bench](a)


if __name__ == "__main__":
    main()
