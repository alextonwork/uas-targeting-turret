// Single-axis PID: deadband, conditional-integration anti-windup, low-pass
// derivative on measurement. Output is a rate command (deg/s).
// Mirrors turret/pid.py; parity is checked by tests/test_cpp_parity.py.
#pragma once

#include <algorithm>
#include <cmath>

namespace turret {

class PID {
 public:
  PID(double kp, double ki = 0.0, double kd = 0.0, double out_limit = 300.0,
      double i_limit = 50.0, double deadband = 0.0, double d_filter_hz = 15.0)
      : kp(kp), ki(ki), kd(kd), out_limit(out_limit), i_limit(i_limit),
        deadband(deadband), d_filter_hz(d_filter_hz) {}

  void reset() {
    integral_ = 0.0;
    d_state_ = 0.0;
    have_prev_ = false;
    last_output_ = 0.0;
  }

  double update(double error, double measurement, double dt) {
    if (dt <= 0.0) return last_output_;

    double raw_d = have_prev_ ? -(measurement - prev_meas_) / dt : 0.0;
    prev_meas_ = measurement;
    have_prev_ = true;
    const double alpha = 1.0 - std::exp(-2.0 * M_PI * d_filter_hz * dt);
    d_state_ += alpha * (raw_d - d_state_);

    if (std::fabs(error) <= deadband) {
      integral_ *= std::exp(-dt / 0.25);
      last_output_ = 0.0;
      return 0.0;
    }

    const double e = error - std::copysign(deadband, error);
    const double p = kp * e;
    const double d = kd * d_state_;
    double cand_i = std::clamp(integral_ + ki * e * dt, -i_limit, i_limit);
    double out = p + cand_i + d;

    if (std::fabs(out) > out_limit && std::signbit(out) == std::signbit(e)) {
      out = std::copysign(out_limit, out);
    } else {
      integral_ = cand_i;
      out = std::clamp(out, -out_limit, out_limit);
    }
    last_output_ = out;
    return out;
  }

  double integral() const { return integral_; }

  double kp, ki, kd, out_limit, i_limit, deadband, d_filter_hz;

 private:
  double integral_ = 0.0;
  double d_state_ = 0.0;
  double prev_meas_ = 0.0;
  bool have_prev_ = false;
  double last_output_ = 0.0;
};

}  // namespace turret
