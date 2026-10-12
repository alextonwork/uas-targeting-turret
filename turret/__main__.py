"""Entry point: python -m turret [--config FILE] [--mode sim|live] [--display] ...

Pan-tilt camera tracking testbench. Points a camera at a detected aerial target and
keeps it centred. There is no effector of any kind in this project.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time

from .capture import make_camera
from .config import load_config
from .controller import GimbalController, resolve_backend
from .detector import make_detector
from .geometry import CameraModel
from .metrics import Metrics
from .pipeline import Pipeline
from .servo import make_driver
from .tracker import Tracker

log = logging.getLogger("turret")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="turret", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="YAML file merged over config/default.yaml")
    p.add_argument("--mode", choices=["sim", "live"], help="override config mode")
    p.add_argument("--duration", type=float, default=None, help="seconds to run (default: sim.duration in sim, forever live)")
    p.add_argument("--display", action="store_true", help="show an annotated preview window (live mode, needs a desktop)")
    p.add_argument("--metrics-csv", default=None, help="write per-frame timing rows here")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override any config key, e.g. --set control.pan.pid.kp=5")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = list(args.set)
    if args.mode:
        overrides.append(f"mode={args.mode}")
    if args.metrics_csv:
        overrides.append(f"metrics.csv={args.metrics_csv}")
    cfg = load_config(args.config, overrides)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")

    backend = resolve_backend(cfg.control.backend)
    cam_model = CameraModel.from_cfg(cfg.camera)
    log.info("mode=%s control backend=%s servo driver=%s", cfg.mode, backend, cfg.servo.driver)

    servo = make_driver(cfg)
    servo.center()
    camera = make_camera(cfg, servo)
    detector = make_detector(cfg, cam_model)
    tracker = Tracker(cfg, backend, cam_model)
    ctrl = GimbalController(cfg, backend)
    metrics = Metrics(csv_path=cfg.metrics.csv)

    preview = Preview(cam_model) if args.display and cfg.mode == "live" else None

    def on_result(frame, dets, est):
        if frame.truth is not None:  # simulation: score pointing error against ground truth
            d_az = frame.truth["az"] - frame.pan
            d_el = frame.truth["el"] - frame.tilt
            if cam_model.in_view(d_az, d_el):
                metrics.record("point_err_deg", math.hypot(d_az, d_el))
            else:
                metrics.count("frames_target_out_of_view")
        if preview:
            preview.push(frame, dets, est)

    pipe = Pipeline(cfg, camera, detector, tracker, ctrl, servo, metrics, on_result)
    duration = args.duration if args.duration is not None else (cfg.sim.duration if cfg.mode == "sim" else None)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    pipe.start()
    t_start = last_log = time.perf_counter()
    try:
        while not stop["flag"] and not pipe.stop_event.is_set():
            if preview:
                if not preview.show():
                    break
            else:
                time.sleep(0.05)
            now = time.perf_counter()
            if now - last_log >= cfg.metrics.log_interval:
                last_log = now
                log.info("%s | %s", tracker.state, metrics.format_line())
            if duration is not None and now - t_start >= duration:
                break
    finally:
        pipe.stop()
        servo.center()
        time.sleep(0.3)
        servo.close()
        camera.close()
        metrics.write_csv()
        if preview:
            preview.close()

    summary = metrics.summary()
    summary["tracker"] = {"track_frames": tracker.track_frames, "coast_frames": tracker.coast_frames,
                          "reacquisitions": tracker.reacquisitions}
    print(json.dumps(_round(summary), indent=2))
    if pipe.error:
        log.error("pipeline error: %r", pipe.error)
        return 1
    return 0


def _round(o):
    if isinstance(o, float):
        return round(o, 2)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    return o


class Preview:
    """Annotated preview: detections, Kalman estimate, crosshair and deadband circle."""

    def __init__(self, cam: CameraModel):
        import cv2

        self.cv2, self.cam = cv2, cam
        self.latest = None

    def push(self, frame, dets, est):
        self.latest = (frame, dets, est)

    def show(self) -> bool:
        cv2 = self.cv2
        if self.latest is None:
            return cv2.waitKey(10) != 27
        frame, dets, est = self.latest
        img = frame.image.copy()
        c = (int(self.cam.cx), int(self.cam.cy))
        cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
        for d in dets:
            p1 = (int(d.u - d.w / 2), int(d.v - d.h / 2))
            p2 = (int(d.u + d.w / 2), int(d.v + d.h / 2))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, f"{d.cls}:{d.conf:.2f}", (p1[0], p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if est.target is not None:
            u, v = self.cam.offset_to_pixel(est.target[0] - frame.pan, est.target[1] - frame.tilt)
            cv2.circle(img, (int(u), int(v)), 8, (0, 165, 255), 2)
        cv2.putText(img, est.state, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("turret", img)
        return (cv2.waitKey(1) & 0xFF) not in (27, ord("q"))

    def close(self):
        self.cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
