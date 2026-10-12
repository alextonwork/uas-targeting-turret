// One gimbal axis: PID rate command -> acceleration limit -> integrate into a
// position setpoint -> soft travel limits. Mirrors turret/controller.py AxisController.
#pragma once

#include <algorithm>
#include <cmath>

#include "pid.hpp"

namespace turret {

class AxisController {
 public:
  AxisController(const PID& pid, double min_angle, double max_angle, double max_accel,
                 double feedforward_gain = 1.0)
      : pid(pid), min_angle(min_angle), max_angle(max_angle), max_accel(max_accel),
        feedforward_gain(feedforward_gain) {}

  // target: desired angle (deg); target_rate: estimated target angular velocity
  // (deg/s, feed-forward); position: current gimbal angle estimate (deg).
  // Returns the new position setpoint (deg).
  double step(double target, double target_rate, double position, double dt) {
    if (dt <= 0.0) return setpoint_;
    if (!have_setpoint_) {
      setpoint_ = position;
      have_setpoint_ = true;
    }
    const double error = target - position;
    double rate = pid.update(error, position, dt);
    // Feed-forward only when outside the deadband, otherwise a slow drifting
    // estimate re-introduces the hunting the deadband is there to remove.
    if (std::fabs(error) > pid.deadband) rate += feedforward_gain * target_rate;
    rate = std::clamp(rate, -pid.out_limit, pid.out_limit);
    const double max_dv = max_accel * dt;
    rate = std::clamp(rate, rate_ - max_dv, rate_ + max_dv);
    double next = setpoint_ + rate * dt;
    if (next > max_angle || next < min_angle) {
      next = std::clamp(next, min_angle, max_angle);
      rate = 0.0;
    }
    rate_ = rate;
    setpoint_ = next;
    return setpoint_;
  }

  void hold(double position) {
    setpoint_ = position;
    have_setpoint_ = true;
    rate_ = 0.0;
    pid.reset();
  }

  double rate() const { return rate_; }
  double setpoint() const { return setpoint_; }

  PID pid;
  double min_angle, max_angle, max_accel, feedforward_gain;

 private:
  double setpoint_ = 0.0;
  double rate_ = 0.0;
  bool have_setpoint_ = false;
};

}  // namespace turret
