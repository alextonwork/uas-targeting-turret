"""The C++ core must produce the same numbers as the Python reference."""
import numpy as np
import pytest

from turret.controller import PyAxisController
from turret.kalman import KalmanCV2D
from turret.pid import PID

core = pytest.importorskip("turret_core", reason="C++ core not built (see README)")


def test_kalman_parity():
    rng = np.random.default_rng(0)
    py, cc = KalmanCV2D(40.0, 0.3, 25.0), core.KalmanCV2D(40.0, 0.3, 25.0)
    py.reset(1.0, 2.0)
    cc.reset(1.0, 2.0)
    for _ in range(500):
        dt = rng.uniform(0.01, 0.08)
        py.predict(dt)
        cc.predict(dt)
        if rng.random() > 0.2:
            z = rng.normal(0, 3, 2)
            assert py.mahalanobis2(*z) == pytest.approx(cc.mahalanobis2(*z), rel=1e-9)
            assert py.update(*z) == pytest.approx(cc.update(*z), rel=1e-9)
    np.testing.assert_allclose(py.x, np.array(cc.state), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(py.P, np.array(cc.covariance), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(py.extrapolate(0.1), cc.extrapolate(0.1), atol=1e-9)


def test_pid_parity():
    rng = np.random.default_rng(1)
    args = (5.0, 1.2, 0.2, 150.0, 20.0, 0.3, 12.0)
    py, cc = PID(*args), core.PID(*args)
    meas = 0.0
    for _ in range(2000):
        e = rng.normal(0, 5)
        meas += rng.normal(0, 0.5)
        dt = rng.uniform(0.005, 0.02)
        assert py.update(e, meas, dt) == pytest.approx(cc.update(e, meas, dt), abs=1e-9)
    assert py.integral == pytest.approx(cc.integral, abs=1e-9)


def test_axis_parity():
    args = (6.0, 0.6, 0.12, 300.0, 40.0, 0.35, 15.0)
    py = PyAxisController(PID(*args), -85, 85, 2500, 0.9)
    cc = core.AxisController(core.PID(*args), -85, 85, 2500, 0.9)
    rng = np.random.default_rng(2)
    pos_p = pos_c = 0.0
    for i in range(1000):
        tgt = 40 * np.sin(i * 0.01) + rng.normal(0, 0.2)
        rate = 4 * np.cos(i * 0.01)
        pos_p = py.step(tgt, rate, pos_p, 0.01)
        pos_c = cc.step(tgt, rate, pos_c, 0.01)
        assert pos_p == pytest.approx(pos_c, abs=1e-9)
