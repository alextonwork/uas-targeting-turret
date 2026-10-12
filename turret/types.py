"""Shared data types passed between the capture, inference and actuation threads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Detection:
    u: float          # bbox centre, pixels
    v: float
    w: float
    h: float
    conf: float
    cls: int = 0


@dataclass
class Frame:
    index: int
    t_capture: float                  # time.perf_counter() when the frame was grabbed
    pan: float                        # gimbal angles at capture time (deg)
    tilt: float
    image: Optional[Any] = None       # HxWx3 uint8 BGR, or None in pure simulation
    truth: Optional[dict] = None      # simulation ground truth
    timings: dict = field(default_factory=dict)


@dataclass
class TrackEstimate:
    state: str                        # SEARCH | TENTATIVE | TRACK | COAST | LOST
    target: Optional[tuple] = None    # (az, el) absolute deg, at the queried time
    rate: tuple = (0.0, 0.0)          # (v_az, v_el) deg/s
    t_meas: float = 0.0               # capture time of the last fused measurement
    meas_seq: int = 0                 # increments every fused measurement
