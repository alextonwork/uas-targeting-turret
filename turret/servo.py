"""Servo driver abstraction: PCA9685 (I2C), pigpio (hardware-timed GPIO PWM), and a mock.

All drivers take angles in degrees relative to the mechanical centre (0, 0) and keep a
slew-rate model of where the horn physically is right now. Hobby servos report no
position, so the model is what the capture thread stamps onto each frame and what the
controller uses as "current position".
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class SlewModel:
    """Rate-limited first-order model of where a position servo actually is."""

    def __init__(self, slew_rate: float, initial: float = 0.0):
        self.slew_rate = slew_rate
        self.cmd = initial
        self.pos = initial
        self.t = time.perf_counter()

    def advance(self, now: float) -> float:
        dt = max(0.0, now - self.t)
        self.t = now
        d = self.cmd - self.pos
        step = self.slew_rate * dt
        self.pos += max(-step, min(step, d))
        return self.pos


class ServoDriver(ABC):
    def __init__(self, cfg):
        s = cfg.servo
        self.axes = {"pan": s.pan, "tilt": s.tilt}
        self.limits = {
            "pan": (cfg.control.pan.min_angle, cfg.control.pan.max_angle),
            "tilt": (cfg.control.tilt.min_angle, cfg.control.tilt.max_angle),
        }
        self.models = {"pan": SlewModel(s.slew_rate), "tilt": SlewModel(s.slew_rate)}
        self.lock = threading.Lock()
        self.writes = 0

    def angle_to_us(self, axis: str, angle: float) -> float:
        a = self.axes[axis]
        angle = angle + a.trim_deg
        if a.invert:
            angle = -angle
        mid = (a.min_us + a.max_us) / 2.0
        us = mid + angle * (a.max_us - a.min_us) / a.range_deg
        return max(a.min_us, min(a.max_us, us))

    def set_angles(self, pan: float, tilt: float) -> None:
        now = time.perf_counter()
        with self.lock:
            for axis, val in (("pan", pan), ("tilt", tilt)):
                lo, hi = self.limits[axis]
                val = max(lo, min(hi, val))
                m = self.models[axis]
                m.advance(now)
                m.cmd = val
                self._write(axis, self.angle_to_us(axis, val))
            self.writes += 1

    def get_angles(self, now: float | None = None) -> tuple[float, float]:
        """Estimated physical angles right now (from the slew model)."""
        now = time.perf_counter() if now is None else now
        with self.lock:
            return self.models["pan"].advance(now), self.models["tilt"].advance(now)

    def center(self) -> None:
        self.set_angles(0.0, 0.0)

    @abstractmethod
    def _write(self, axis: str, pulse_us: float) -> None: ...

    def close(self) -> None:
        pass


class MockDriver(ServoDriver):
    """No hardware. The slew model *is* the gimbal; used for sim and laptop runs."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.last_us = {"pan": None, "tilt": None}

    def _write(self, axis, pulse_us):
        self.last_us[axis] = pulse_us


class PCA9685Driver(ServoDriver):
    """Direct register driver for the PCA9685 16-ch PWM board over I2C (smbus2)."""

    MODE1, PRESCALE, LED0_ON_L = 0x00, 0xFE, 0x06

    def __init__(self, cfg):
        super().__init__(cfg)
        from smbus2 import SMBus  # pip install smbus2

        p = cfg.servo.pca9685
        self.addr = int(p.address)
        self.freq = float(p.frequency)
        self.bus = SMBus(int(p.bus))
        prescale = round(25_000_000 / (4096 * self.freq)) - 1
        old = self.bus.read_byte_data(self.addr, self.MODE1)
        self.bus.write_byte_data(self.addr, self.MODE1, (old & 0x7F) | 0x10)  # sleep
        self.bus.write_byte_data(self.addr, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.addr, self.MODE1, old & ~0x10)
        time.sleep(0.005)
        self.bus.write_byte_data(self.addr, self.MODE1, (old & ~0x10) | 0xA0)  # restart + auto-inc
        self.period_us = 1_000_000 / self.freq
        log.info("PCA9685 at 0x%02x, %.0f Hz (prescale %d)", self.addr, self.freq, prescale)

    def _write(self, axis, pulse_us):
        ch = int(self.axes[axis].channel)
        off = int(round(pulse_us / self.period_us * 4096)) & 0x0FFF
        reg = self.LED0_ON_L + 4 * ch
        self.bus.write_i2c_block_data(self.addr, reg, [0, 0, off & 0xFF, off >> 8])

    def close(self):
        try:
            for axis in ("pan", "tilt"):  # full-off: servos go limp, no buzzing at rest
                reg = self.LED0_ON_L + 4 * int(self.axes[axis].channel)
                self.bus.write_i2c_block_data(self.addr, reg, [0, 0, 0, 0x10])
        finally:
            self.bus.close()


class PigpioDriver(ServoDriver):
    """Hardware-timed servo pulses on any GPIO via the pigpio daemon (sudo pigpiod)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        import pigpio  # apt install pigpio python3-pigpio

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running: sudo systemctl start pigpiod")

    def _write(self, axis, pulse_us):
        self.pi.set_servo_pulsewidth(int(self.axes[axis].gpio), int(pulse_us))

    def close(self):
        for axis in ("pan", "tilt"):
            self.pi.set_servo_pulsewidth(int(self.axes[axis].gpio), 0)
        self.pi.stop()


def make_driver(cfg) -> ServoDriver:
    name = cfg.servo.driver.lower()
    return {"mock": MockDriver, "pca9685": PCA9685Driver, "pigpio": PigpioDriver}[name](cfg)
