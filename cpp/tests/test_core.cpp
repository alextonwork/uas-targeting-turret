// Minimal dependency-free C++ tests, run with ctest.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "turret_core/axis.hpp"
#include "turret_core/kalman.hpp"
#include "turret_core/pid.hpp"

static int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static void kalman_tracks_constant_velocity() {
  turret::KalmanCV2D kf(20.0, 0.2, 30.0);
  const double dt = 1.0 / 30.0, vaz = 12.0, vel = -4.0;
  kf.reset(0.0, 0.0);
  for (int i = 1; i <= 90; ++i) {
    kf.predict(dt);
    kf.update(vaz * i * dt, vel * i * dt);
  }
  CHECK(std::fabs(kf.state()[2] - vaz) < 1.0);
  CHECK(std::fabs(kf.state()[3] - vel) < 1.0);
  // Coast 0.5 s without measurements.
  const double p00_before = kf.covariance()[0][0];
  for (int i = 0; i < 15; ++i) kf.predict(dt);
  CHECK(std::fabs(kf.state()[0] - vaz * 3.5) < 0.5);
  CHECK(kf.covariance()[0][0] > p00_before);
}

static void pid_deadband_and_limits() {
  turret::PID pid(5.0, 1.0, 0.0, 100.0, 10.0, 0.5);
  CHECK(pid.update(0.3, 0.0, 0.01) == 0.0);
  CHECK(std::fabs(pid.update(1000.0, 0.0, 0.01)) <= 100.0);
}

static void axis_settles_without_overshoot() {
  turret::AxisController ax(turret::PID(6.0, 0.5, 0.15, 300.0, 30.0, 0.3), -90, 90, 3000);
  double pos = 0.0, max_pos = 0.0;
  for (int i = 0; i < 300; ++i) {
    pos = ax.step(20.0, 0.0, pos, 0.01);
    max_pos = std::fmax(max_pos, pos);
  }
  CHECK(std::fabs(pos - 20.0) < 0.5);
  CHECK(max_pos < 20.0 + 0.5);
}

int main() {
  kalman_tracks_constant_velocity();
  pid_deadband_and_limits();
  axis_settles_without_overshoot();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("all C++ core tests passed");
  return EXIT_SUCCESS;
}
