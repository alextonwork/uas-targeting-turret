"""Frame sources: OpenCV (USB webcam / video file) and Picamera2 (Pi camera module).

Each frame is stamped with the capture time and the gimbal pose at that instant, so the
tracker can convert pixel positions to absolute bearings even while the gimbal moves.
"""
from __future__ import annotations

import logging
import time

from .types import Frame

log = logging.getLogger(__name__)


class OpenCVCamera:
    def __init__(self, cfg, servo):
        import cv2

        c = cfg.camera
        self.cv2 = cv2
        self.servo = servo
        src = c.source
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera source {src!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        self.cap.set(cv2.CAP_PROP_FPS, c.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # never hand us a stale buffered frame
        self.flip = _flip_code(c)
        self.index = 0
        self.is_file = not str(src).isdigit()
        self.period = 1.0 / c.fps

    def read(self) -> Frame | None:
        t0 = time.perf_counter()
        ok, img = self.cap.read()
        if not ok:
            return None
        t = time.perf_counter()
        if self.flip is not None:
            img = self.cv2.flip(img, self.flip)
        if self.is_file:  # pace video files at their nominal rate
            time.sleep(max(0.0, self.period - (t - t0)))
        pan, tilt = self.servo.get_angles(t)
        self.index += 1
        return Frame(self.index, t, pan, tilt, img)

    def close(self):
        self.cap.release()


class Picamera2Camera:
    def __init__(self, cfg, servo):
        from picamera2 import Picamera2  # apt install python3-picamera2

        import cv2

        c = cfg.camera
        self.cv2 = cv2
        self.servo = servo
        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (c.width, c.height), "format": "RGB888"},
            controls={"FrameRate": c.fps},
            buffer_count=2,
        )
        self.cam.configure(conf)
        self.cam.start()
        self.flip = _flip_code(c)
        self.index = 0

    def read(self) -> Frame | None:
        img = self.cam.capture_array()  # "RGB888" is BGR byte order in memory
        t = time.perf_counter()
        if self.flip is not None:
            img = self.cv2.flip(img, self.flip)
        pan, tilt = self.servo.get_angles(t)
        self.index += 1
        return Frame(self.index, t, pan, tilt, img)

    def close(self):
        self.cam.stop()


def _flip_code(c):
    if c.flip_horizontal and c.flip_vertical:
        return -1
    if c.flip_horizontal:
        return 1
    if c.flip_vertical:
        return 0
    return None


def make_camera(cfg, servo):
    if cfg.mode == "sim":
        from .sim import SimCamera

        return SimCamera(cfg, servo)
    if str(cfg.camera.source).lower() == "picamera2":
        return Picamera2Camera(cfg, servo)
    return OpenCVCamera(cfg, servo)
