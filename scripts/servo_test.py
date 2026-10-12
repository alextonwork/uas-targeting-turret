#!/usr/bin/env python3
"""Move the gimbal through a few poses to check wiring, direction, trim and limits.

    python scripts/servo_test.py --set servo.driver=pca9685
    python scripts/servo_test.py --set servo.driver=pigpio --axis pan

Watch the camera: positive pan should turn it RIGHT, positive tilt should point it UP.
If not, flip servo.<axis>.invert. If centre isn't straight ahead, adjust servo.<axis>.trim_deg.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turret.config import load_config  # noqa: E402
from turret.servo import make_driver  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--axis", choices=["pan", "tilt", "both"], default="both")
    p.add_argument("--amplitude", type=float, default=30.0)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    drv = make_driver(cfg)
    poses = [(0, 0)]
    if a.axis in ("pan", "both"):
        poses += [(a.amplitude, 0), (-a.amplitude, 0), (0, 0)]
    if a.axis in ("tilt", "both"):
        poses += [(0, a.amplitude), (0, max(cfg.control.tilt.min_angle, -a.amplitude / 3)), (0, 0)]
    try:
        for pan, tilt in poses:
            drv.set_angles(pan, tilt)
            print(f"pan {pan:+6.1f}  tilt {tilt:+6.1f}  ->  "
                  f"{drv.angle_to_us('pan', pan):.0f} us / {drv.angle_to_us('tilt', tilt):.0f} us")
            time.sleep(1.2)
    finally:
        drv.center()
        time.sleep(0.5)
        drv.close()


if __name__ == "__main__":
    main()
