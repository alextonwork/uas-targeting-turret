import pytest

from turret.config import load_config
from turret.controller import PyAxisController
from turret.pid import PID
from turret.servo import SlewModel


def test_proportional_only():
    pid = PID(kp=2.0)
    assert pid.update(3.0, 0.0, 0.01) == pytest.approx(6.0)
    assert pid.update(-1.5, 0.0, 0.01) == pytest.approx(-3.0)


def test_output_limit():
    pid = PID(kp=100.0, out_limit=50.0)
    assert pid.update(10.0, 0.0, 0.01) == 50.0
    assert pid.update(-10.0, 0.0, 0.01) == -50.0


def test_deadband_zeroes_output_and_is_continuous_at_edge():
    pid = PID(kp=4.0, deadband=0.5)
    assert pid.update(0.49, 0.0, 0.01) == 0.0
    assert pid.update(-0.3, 0.0, 0.01) == 0.0
    # just outside the band, output starts from ~0 instead of jumping to kp*0.5
    assert abs(pid.update(0.51, 0.0, 0.01)) < 0.1


def test_integral_bleeds_off_inside_deadband():
    pid = PID(kp=0.0, ki=10.0, deadband=0.2, i_limit=100.0)
    for _ in range(100):
        pid.update(2.0, 0.0, 0.01)
    assert pid.integral > 5
    for _ in range(200):
        pid.update(0.1, 0.0, 0.01)
    assert abs(pid.integral) < 0.1


def test_anti_windup_clamps_integral():
    pid = PID(kp=1.0, ki=50.0, out_limit=10.0, i_limit=5.0)
    for _ in range(1000):
        pid.update(100.0, 0.0, 0.01)
    assert abs(pid.integral) <= 5.0 + 1e-9
    # recovers immediately once error flips sign: no long wind-down
    out = pid.update(-1.0, 0.0, 0.01)
    assert out < 5.0


def test_no_derivative_kick_on_setpoint_step():
    """D acts on measurement, so a sudden new target produces no derivative spike."""
    pid = PID(kp=1.0, kd=5.0)
    pid.update(0.0, 0.0, 0.01)
    out = pid.update(10.0, 0.0, 0.01)  # setpoint jumped, measurement didn't move
    assert out == pytest.approx(10.0)


def test_derivative_opposes_motion():
    pid = PID(kp=0.0, kd=1.0, d_filter_hz=1000.0)
    pid.update(5.0, 0.0, 0.01)
    out = pid.update(5.0, 1.0, 0.01)  # measurement moving towards setpoint fast
    assert out < 0


def test_integral_removes_steady_state_error_against_disturbance():
    """Axis with a constant drift disturbance (e.g. unbalanced camera); PI should cancel it."""
    pid = PID(kp=3.0, ki=4.0, out_limit=200.0, i_limit=50.0)
    pos, dt = 0.0, 0.01
    for _ in range(1500):
        rate = pid.update(10.0 - pos, pos, dt)
        pos += (rate - 5.0) * dt  # 5 deg/s sag
    assert pos == pytest.approx(10.0, abs=0.1)


@pytest.mark.parametrize("axis", ["pan", "tilt"])
def test_default_tuning_settles_without_overshoot_or_hunting(axis):
    """Closed loop with the default gains against a slew-limited servo model."""
    cfg = load_config()
    a = cfg.control[axis]
    p = a.pid
    ctrl = PyAxisController(PID(p.kp, p.ki, p.kd, p.out_limit, p.i_limit, p.deadband, p.d_filter_hz),
                            a.min_angle, a.max_angle, a.max_accel, a.feedforward_gain)
    servo = SlewModel(cfg.servo.slew_rate)
    servo.t = 0.0
    target, dt, t = 25.0, 0.01, 0.0
    peak, cmds = 0.0, []
    while t < 3.0:
        pos = servo.advance(t)
        servo.cmd = ctrl.step(target, 0.0, pos, dt)
        cmds.append(servo.cmd)
        peak = max(peak, pos)
        t += dt
    assert peak <= target + p.deadband + 0.05            # no overshoot beyond the deadband
    assert abs(target - pos) <= p.deadband + 0.05         # settled inside the deadband
    tail = cmds[-100:]
    assert max(tail) - min(tail) < 1e-6                   # completely still: no hunting
