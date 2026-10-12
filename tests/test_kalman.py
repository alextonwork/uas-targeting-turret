import math

import numpy as np
import pytest

from turret.kalman import KalmanCV2D

DT = 1 / 30


def run_cv(kf, vaz, vel, n, noise=0.0, rng=None):
    kf.reset(0.0, 0.0)
    for i in range(1, n + 1):
        kf.predict(DT)
        nz = rng.normal(0, noise, 2) if rng is not None else (0, 0)
        kf.update(vaz * i * DT + nz[0], vel * i * DT + nz[1])


def test_converges_to_true_velocity():
    kf = KalmanCV2D(20.0, 0.3, 30.0)
    run_cv(kf, 15.0, -6.0, 90)
    assert kf.velocity == pytest.approx((15.0, -6.0), abs=0.3)


def test_velocity_unbiased_under_noise():
    # Per-frame velocity estimates are noisy by design (the filter must follow
    # manoeuvres), but they should be centred on the truth.
    rng = np.random.default_rng(0)
    kf = KalmanCV2D(20.0, 0.3, 30.0)
    kf.reset(0, 0)
    est = []
    for i in range(1, 600):
        kf.predict(DT)
        kf.update(15 * i * DT + rng.normal(0, 0.3), -6 * i * DT + rng.normal(0, 0.3))
        if i > 60:
            est.append(kf.velocity)
    m = np.mean(est, axis=0)
    assert m[0] == pytest.approx(15.0, abs=0.5)
    assert m[1] == pytest.approx(-6.0, abs=0.5)


def test_filters_measurement_noise():
    rng = np.random.default_rng(1)
    kf = KalmanCV2D(10.0, 0.5, 30.0)
    kf.reset(0, 0)
    errs_raw, errs_kf = [], []
    for i in range(1, 300):
        t = i * DT
        truth = (10 * t, 3 * t)
        z = (truth[0] + rng.normal(0, 0.5), truth[1] + rng.normal(0, 0.5))
        kf.predict(DT)
        kf.update(*z)
        if i > 30:
            errs_raw.append(math.dist(z, truth))
            errs_kf.append(math.dist(kf.position, truth))
    assert np.mean(errs_kf) < 0.6 * np.mean(errs_raw)


def test_coasts_through_occlusion():
    kf = KalmanCV2D(20.0, 0.2, 30.0)
    run_cv(kf, 20.0, 5.0, 60)
    t_end = 60 * DT
    for _ in range(15):  # 0.5 s with no detections
        kf.predict(DT)
    t = t_end + 15 * DT
    assert kf.position[0] == pytest.approx(20.0 * t, abs=0.5)
    assert kf.position[1] == pytest.approx(5.0 * t, abs=0.5)


def test_uncertainty_grows_while_coasting_and_shrinks_on_update():
    kf = KalmanCV2D()
    run_cv(kf, 10.0, 0.0, 30)
    s0 = kf.position_std[0]
    for _ in range(10):
        kf.predict(DT)
    s1 = kf.position_std[0]
    assert s1 > s0
    kf.update(*kf.position)
    assert kf.position_std[0] < s1


def test_irregular_dt_matches_uniform():
    """Two predicts of dt/2 should equal one predict of dt (Q is integrated, not per-step)."""
    a, b = KalmanCV2D(), KalmanCV2D()
    for k in (a, b):
        k.reset(1.0, 2.0)
        k.update(1.1, 2.1)
    a.predict(0.05)
    b.predict(0.025)
    b.predict(0.025)
    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
    # P differs only by the cross term from F applied twice; must stay close
    np.testing.assert_allclose(a.P, b.P, rtol=0.05, atol=1e-6)


def test_covariance_stays_symmetric_positive_definite():
    rng = np.random.default_rng(2)
    kf = KalmanCV2D(100.0, 0.1, 50.0)
    kf.reset(0, 0)
    for _ in range(2000):
        kf.predict(rng.uniform(0.005, 0.1))
        if rng.random() > 0.3:
            kf.update(*rng.normal(0, 5, 2))
    np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_mahalanobis_gate_rejects_outlier():
    kf = KalmanCV2D(20.0, 0.3)
    run_cv(kf, 10.0, 0.0, 60)
    kf.predict(DT)
    near = kf.mahalanobis2(*kf.position)
    far = kf.mahalanobis2(kf.position[0] + 15, kf.position[1])
    assert near < 1.0
    assert far > 13.8


def test_extrapolate_does_not_mutate():
    kf = KalmanCV2D()
    run_cv(kf, 10.0, 0.0, 30)
    x = kf.x.copy()
    ahead = kf.extrapolate(0.1)
    np.testing.assert_array_equal(kf.x, x)
    assert ahead[0] == pytest.approx(x[0] + 0.1 * x[2])
