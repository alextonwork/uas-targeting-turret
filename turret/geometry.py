"""Pixel <-> angle conversion for a pinhole camera riding on the gimbal."""
from __future__ import annotations

import math


class CameraModel:
    def __init__(self, width: int, height: int, hfov_deg: float, vfov_deg: float):
        self.width, self.height = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.fx = self.cx / math.tan(math.radians(hfov_deg) / 2)
        self.fy = self.cy / math.tan(math.radians(vfov_deg) / 2)
        self.hfov, self.vfov = hfov_deg, vfov_deg

    @classmethod
    def from_cfg(cls, cam) -> "CameraModel":
        return cls(cam.width, cam.height, cam.hfov_deg, cam.vfov_deg)

    def pixel_to_offset(self, u: float, v: float) -> tuple[float, float]:
        """Pixel -> (d_az, d_el) degrees from the optical axis. +az right, +el up."""
        d_az = math.degrees(math.atan((u - self.cx) / self.fx))
        d_el = -math.degrees(math.atan((v - self.cy) / self.fy))
        return d_az, d_el

    def offset_to_pixel(self, d_az: float, d_el: float) -> tuple[float, float]:
        u = self.cx + self.fx * math.tan(math.radians(d_az))
        v = self.cy - self.fy * math.tan(math.radians(d_el))
        return u, v

    def pixel_to_absolute(self, u, v, pan, tilt) -> tuple[float, float]:
        """Absolute bearing of a pixel given the gimbal angles when the frame was taken.

        Small-angle composition (adds offsets to gimbal angles). Error is well under a
        degree inside a ~60 deg FOV at tilts below ~60 deg, which the Kalman R absorbs.
        """
        d_az, d_el = self.pixel_to_offset(u, v)
        return pan + d_az, tilt + d_el

    def in_view(self, d_az: float, d_el: float, margin: float = 0.0) -> bool:
        return abs(d_az) <= self.hfov / 2 - margin and abs(d_el) <= self.vfov / 2 - margin

    def px_per_deg(self) -> float:
        return self.fx * math.pi / 180.0
