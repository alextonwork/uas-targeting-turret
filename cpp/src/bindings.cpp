// pybind11 module exposing the C++ control core as `turret_core`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "turret_core/axis.hpp"
#include "turret_core/kalman.hpp"
#include "turret_core/pid.hpp"

namespace py = pybind11;
using turret::AxisController;
using turret::KalmanCV2D;
using turret::PID;

PYBIND11_MODULE(turret_core, m) {
  m.doc() = "C++ Kalman/PID control core for the pan-tilt tracking testbench";

  py::class_<KalmanCV2D>(m, "KalmanCV2D")
      .def(py::init<double, double, double>(), py::arg("process_accel_std") = 60.0,
           py::arg("meas_std") = 0.4, py::arg("init_vel_std") = 30.0)
      .def("reset", &KalmanCV2D::reset)
      .def("predict", &KalmanCV2D::predict)
      .def("update", &KalmanCV2D::update)
      .def("mahalanobis2", &KalmanCV2D::mahalanobis2)
      .def("extrapolate", &KalmanCV2D::extrapolate)
      .def_property_readonly("initialized", &KalmanCV2D::initialized)
      .def_property_readonly("state", &KalmanCV2D::state)
      .def_property_readonly("covariance", &KalmanCV2D::covariance)
      .def_property_readonly("position",
                             [](const KalmanCV2D& k) {
                               return py::make_tuple(k.state()[0], k.state()[1]);
                             })
      .def_property_readonly("velocity",
                             [](const KalmanCV2D& k) {
                               return py::make_tuple(k.state()[2], k.state()[3]);
                             })
      .def_property_readonly("position_std", [](const KalmanCV2D& k) {
        return py::make_tuple(std::sqrt(k.covariance()[0][0]),
                              std::sqrt(k.covariance()[1][1]));
      });

  py::class_<PID>(m, "PID")
      .def(py::init<double, double, double, double, double, double, double>(),
           py::arg("kp"), py::arg("ki") = 0.0, py::arg("kd") = 0.0,
           py::arg("out_limit") = 300.0, py::arg("i_limit") = 50.0,
           py::arg("deadband") = 0.0, py::arg("d_filter_hz") = 15.0)
      .def("reset", &PID::reset)
      .def("update", &PID::update, py::arg("error"), py::arg("measurement"),
           py::arg("dt"))
      .def_property_readonly("integral", &PID::integral)
      .def_readwrite("kp", &PID::kp)
      .def_readwrite("ki", &PID::ki)
      .def_readwrite("kd", &PID::kd)
      .def_readwrite("deadband", &PID::deadband)
      .def_readwrite("out_limit", &PID::out_limit);

  py::class_<AxisController>(m, "AxisController")
      .def(py::init<const PID&, double, double, double, double>(), py::arg("pid"),
           py::arg("min_angle"), py::arg("max_angle"), py::arg("max_accel"),
           py::arg("feedforward_gain") = 1.0)
      .def("step", &AxisController::step, py::arg("target"), py::arg("target_rate"),
           py::arg("position"), py::arg("dt"))
      .def("hold", &AxisController::hold)
      .def_property_readonly("rate", &AxisController::rate)
      .def_property_readonly("setpoint", &AxisController::setpoint)
      .def_readwrite("pid", &AxisController::pid);
}
