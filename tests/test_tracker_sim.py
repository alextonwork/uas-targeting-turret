"""Tracker behaviour, geometry, config, and end-to-end simulation (offline + threaded)."""
import statistics as st

import pytest

from turret.config import load_config
from turret.geometry import CameraModel
from turret.sim import run_offline
from turret.tracker import COAST, LOST, SEARCH, TENTATIVE, TRACK, Tracker
from turret.types import Detection, Frame


@pytest.fixture
def cfg():
    return load_config(overrides=["control.backend=python"])


@pytest.fixture
def cam(cfg):
    return CameraModel.from_cfg(cfg.camera)


def det_at(cam, d_az, d_el, conf=0.8):
    u, v = cam.offset_to_pixel(d_az, d_el)
    return Detection(u, v, 20, 20, conf)


def test_geometry_round_trip(cam):
    for d in [(0, 0), (10, -5), (-25, 20)]:
        u, v = cam.offset_to_pixel(*d)
        assert cam.pixel_to_offset(u, v) == pytest.approx(d, abs=1e-9)
    assert cam.pixel_to_offset(cam.width, cam.cy)[0] == pytest.approx(cam.hfov / 2)


def test_config_overrides():
    c = load_config(overrides=["control.pan.pid.kp=9.5", "servo.driver=pigpio", "metrics.csv=null"])
    assert c.control.pan.pid.kp == 9.5
    assert c.servo.driver == "pigpio"
    assert c.metrics.csv is None
    assert c.control.tilt.pid.kp == load_config().control.tilt.pid.kp


def test_confirm_then_coast_then_lost(cfg, cam):
    tr = Tracker(cfg, "python", cam)
    t = 0.0
    dt = 1 / 30
    tr.update([det_at(cam, 1.0, 0.0)], Frame(0, t, 0.0, 20.0))
    assert tr.state == TENTATIVE
    for i in range(1, 10):
        t = i * dt
        tr.update([det_at(cam, 1.0 + 10 * t, 0.0)], Frame(i, t, 0.0, 20.0))
    assert tr.state == TRACK
    est = tr.query(t)
    assert est.target[0] == pytest.approx(1.0 + 10 * t, abs=0.5)
    # occlusion: no detections
    t += dt
    tr.update([], Frame(99, t, 0.0, 20.0))
    assert tr.state == COAST
    assert tr.query(t).target is not None
    assert tr.query(t + cfg.tracker.coast_timeout + 0.1).state == LOST


def test_gate_rejects_far_clutter(cfg, cam):
    tr = Tracker(cfg, "python", cam)
    dt = 1 / 30
    for i in range(20):
        tr.update([det_at(cam, 0.0, 0.0)], Frame(i, i * dt, 0.0, 20.0))
    before = tr.meas_seq
    # only a far-away false positive this frame
    tr.update([det_at(cam, 25.0, 15.0, conf=0.95)], Frame(21, 21 * dt, 0.0, 20.0))
    assert tr.meas_seq == before
    assert tr.state == COAST
    # clutter plus the real target: the real one is associated
    tr.update([det_at(cam, 25.0, 15.0, 0.95), det_at(cam, 0.1, 0.0, 0.5)], Frame(22, 22 * dt, 0.0, 20.0))
    assert tr.state == TRACK
    assert tr.query(22 * dt).target[0] == pytest.approx(0.1, abs=0.3)


def test_reacquires_target_that_turned_while_hidden(cfg, cam):
    tr = Tracker(cfg, "python", cam)
    dt = 1 / 30
    for i in range(30):
        tr.update([det_at(cam, 20 * i * dt, 0.0)], Frame(i, i * dt, 0.0, 20.0))
    # hidden for 0.5 s, reappears 12 deg away from where CV prediction expects it
    t = 45 * dt
    tr.update([det_at(cam, 1.0, 0.0)], Frame(45, t, 0.0, 20.0))
    tr.update([det_at(cam, 0.9, 0.0)], Frame(46, t + dt, 0.0, 20.0))
    assert tr.state == TRACK
    assert tr.query(t + dt).target[0] == pytest.approx(0.9, abs=0.5)


def test_measurements_use_gimbal_pose_at_capture(cfg, cam):
    """A world-stationary target seen while the gimbal pans must stay stationary in state."""
    tr = Tracker(cfg, "python", cam)
    dt = 1 / 30
    for i in range(40):
        pan = 0.5 * i  # gimbal slewing right at 15 deg/s
        tr.update([det_at(cam, 10.0 - pan, 0.0)], Frame(i, i * dt, pan, 20.0))
    est = tr.query(40 * dt)
    assert est.target[0] == pytest.approx(10.0, abs=0.3)
    assert abs(est.rate[0]) < 1.0


def test_kalman_reduces_tracking_loss_vs_baseline(cfg):
    seeds = range(12)
    base = load_config(overrides=["control.backend=python", "kalman.enabled=false"])
    k = [run_offline(cfg, duration=40, seed=s) for s in seeds]
    b = [run_offline(base, duration=40, seed=s) for s in seeds]
    assert st.mean(r.lost_frac for r in k) < st.mean(r.lost_frac for r in b)
    assert st.mean(r.rms_error_deg for r in k) < st.mean(r.rms_error_deg for r in b)


def test_threaded_pipeline_runs_in_sim(cfg):
    """Smoke test of the real three-thread pipeline against the simulator."""
    from turret.__main__ import main

    assert main(["--mode", "sim", "--duration", "3", "--set", "control.backend=python"]) == 0
