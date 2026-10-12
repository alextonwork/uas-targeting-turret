// 2D constant-velocity Kalman filter, header-only, no dependencies.
// State x = [az, el, v_az, v_el] (deg, deg/s) in the absolute gimbal frame.
// Mirrors turret/kalman.py; parity is checked by tests/test_cpp_parity.py.
#pragma once

#include <array>
#include <cmath>

namespace turret {

class KalmanCV2D {
 public:
  using Vec4 = std::array<double, 4>;
  using Mat4 = std::array<std::array<double, 4>, 4>;

  explicit KalmanCV2D(double process_accel_std = 60.0, double meas_std = 0.4,
                      double init_vel_std = 30.0)
      : q_(process_accel_std * process_accel_std),
        r_(meas_std * meas_std),
        init_vel_var_(init_vel_std * init_vel_std) {
    x_.fill(0.0);
    P_ = identity();
  }

  void reset(double az, double el) {
    x_ = {az, el, 0.0, 0.0};
    P_ = zeros();
    P_[0][0] = r_;
    P_[1][1] = r_;
    P_[2][2] = init_vel_var_;
    P_[3][3] = init_vel_var_;
    initialized_ = true;
  }

  void predict(double dt) {
    if (dt <= 0.0) return;
    // x = F x
    x_[0] += dt * x_[2];
    x_[1] += dt * x_[3];
    // P = F P F^T + Q, F = [[I, dt I], [0, I]]
    Mat4 FP = P_;
    for (int j = 0; j < 4; ++j) {
      FP[0][j] += dt * P_[2][j];
      FP[1][j] += dt * P_[3][j];
    }
    Mat4 FPFt = FP;
    for (int i = 0; i < 4; ++i) {
      FPFt[i][0] += dt * FP[i][2];
      FPFt[i][1] += dt * FP[i][3];
    }
    const double dt2 = dt * dt, dt3 = dt2 * dt;
    for (int a = 0; a < 2; ++a) {
      FPFt[a][a] += q_ * dt3 / 3.0;
      FPFt[a][a + 2] += q_ * dt2 / 2.0;
      FPFt[a + 2][a] += q_ * dt2 / 2.0;
      FPFt[a + 2][a + 2] += q_ * dt;
    }
    P_ = FPFt;
  }

  // Returns squared Mahalanobis distance of the innovation.
  double update(double az, double el) {
    const double y0 = az - x_[0], y1 = el - x_[1];
    double Si[2][2];
    innovation_inv(Si);
    // K = P H^T S^-1  (4x2); P H^T is the first two columns of P.
    double K[4][2];
    for (int i = 0; i < 4; ++i) {
      K[i][0] = P_[i][0] * Si[0][0] + P_[i][1] * Si[1][0];
      K[i][1] = P_[i][0] * Si[0][1] + P_[i][1] * Si[1][1];
    }
    for (int i = 0; i < 4; ++i) x_[i] += K[i][0] * y0 + K[i][1] * y1;

    // Joseph form: P = (I-KH) P (I-KH)^T + K R K^T
    Mat4 A = identity();
    for (int i = 0; i < 4; ++i) {
      A[i][0] -= K[i][0];
      A[i][1] -= K[i][1];
    }
    Mat4 AP = mul(A, P_);
    Mat4 APAt = zeros();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        double s = 0.0;
        for (int k = 0; k < 4; ++k) s += AP[i][k] * A[j][k];
        APAt[i][j] = s + r_ * (K[i][0] * K[j][0] + K[i][1] * K[j][1]);
      }
    P_ = APAt;
    return y0 * (Si[0][0] * y0 + Si[0][1] * y1) + y1 * (Si[1][0] * y0 + Si[1][1] * y1);
  }

  double mahalanobis2(double az, double el) const {
    const double y0 = az - x_[0], y1 = el - x_[1];
    double Si[2][2];
    innovation_inv(Si);
    return y0 * (Si[0][0] * y0 + Si[0][1] * y1) + y1 * (Si[1][0] * y0 + Si[1][1] * y1);
  }

  Vec4 extrapolate(double dt) const {
    if (dt < 0.0) dt = 0.0;
    return {x_[0] + dt * x_[2], x_[1] + dt * x_[3], x_[2], x_[3]};
  }

  const Vec4& state() const { return x_; }
  const Mat4& covariance() const { return P_; }
  bool initialized() const { return initialized_; }

 private:
  void innovation_inv(double Si[2][2]) const {
    const double s00 = P_[0][0] + r_, s01 = P_[0][1], s10 = P_[1][0], s11 = P_[1][1] + r_;
    const double det = s00 * s11 - s01 * s10;
    Si[0][0] = s11 / det;
    Si[0][1] = -s01 / det;
    Si[1][0] = -s10 / det;
    Si[1][1] = s00 / det;
  }
  static Mat4 zeros() {
    Mat4 m{};
    for (auto& r : m) r.fill(0.0);
    return m;
  }
  static Mat4 identity() {
    Mat4 m = zeros();
    for (int i = 0; i < 4; ++i) m[i][i] = 1.0;
    return m;
  }
  static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 c = zeros();
    for (int i = 0; i < 4; ++i)
      for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j) c[i][j] += a[i][k] * b[k][j];
    return c;
  }

  double q_, r_, init_vel_var_;
  Vec4 x_;
  Mat4 P_;
  bool initialized_ = false;
};

}  // namespace turret
