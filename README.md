# Pan-tilt drone tracking testbench

A Raspberry Pi 2-axis camera gimbal that detects aerial targets with an int8-quantized
YOLOv8n (ONNX Runtime), follows them with a 2D constant-velocity Kalman filter that keeps
predicting through occlusions, and drives pan and tilt with two independent PID loops
that have a deadband. Capture, inference and actuation each run on their own thread. The
Kalman/PID control core exists in both C++ (pybind11) and Python, and a test checks that
the two give the same numbers.

**Scope:** this is a camera tracking testbench. It points a camera and nothing else, and
there is no effector or effector interface anywhere in the code.

```
 ┌─────────────┐  latest-frame   ┌──────────────┐   lock    ┌───────────────┐  100 Hz  ┌────────┐
 │  Capture    │ ──── slot ────▶ │  Inference   │ ───────▶  │   Tracker     │ ◀─────── │Actuation│──▶ servos
 │ cam @30 fps │  (drops stale)  │ YOLOv8n int8 │  update   │ Kalman + gate │  query   │ 2 × PID │
 │ +gimbal pose│                 │  ~25-35 ms   │           │ coast/reacq   │          │ deadband│
 └─────────────┘                 └──────────────┘           └───────────────┘          └────────┘
```

## Contents

```
turret/            Python package (python -m turret)
  pipeline.py        the three threads + 1-slot latest-frame mailbox
  tracker.py         association, gating, coast, re-acquisition, latency compensation
  kalman.py pid.py   Python reference implementations
  controller.py      per-axis controller, search pattern, C++/Python backend selection
  servo.py           PCA9685 / pigpio / mock drivers + slew-rate position model
  capture.py         OpenCV (USB cam, video file) and Picamera2 sources
  detector.py        ONNX Runtime YOLOv8 detector: letterbox, decode, NMS
  sim.py             simulated target/occlusions/detector, real-time and offline
  metrics.py         per-stage latency, FPS, end-to-end "glass-to-servo" latency
cpp/               header-only C++ core (kalman.hpp, pid.hpp, axis.hpp), pybind11 module, C++ tests
config/default.yaml  every tunable, documented inline
scripts/           export_onnx.py, bench_detector.py, benchmark.py, servo_test.py
tests/             pytest suite (Kalman, PID, tracker, C++ parity, decode, sim, live path)
docs/WIRING.md     dedicated servo power rail and wiring
```

## Quick start on a laptop (no hardware)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# optional but recommended: build the C++ control core
cmake -S cpp -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build -j && (cd build && ctest)
export PYTHONPATH=$PWD/build        # so `import turret_core` works

pytest                              # 35 tests; C++ parity tests skip if the core isn't built
python -m turret --mode sim         # real-time threaded sim, prints latency/FPS
python scripts/benchmark.py occlusion   # Kalman vs. no-Kalman over 40 random scenarios
python scripts/benchmark.py step        # PID step response (overshoot / settling / hunting)
```

Laptop webcam with mock servos (the gimbal is simulated, the camera and detector are real):

```bash
pip install opencv-python             # the non-headless build, for --display
python -m turret --mode live --display --set camera.source=0 --set servo.driver=mock
```

## Hardware

| Part | Notes |
|---|---|
| Raspberry Pi 5 (4 GB+) or Pi 4 | Pi 5 roughly doubles detector throughput |
| Pi Camera Module 3 / v2, or a USB UVC webcam | set `camera.hfov_deg/vfov_deg` to match your lens |
| Pan-tilt bracket + 2 × MG996R / DS3218 metal-gear servos | |
| PCA9685 16-ch PWM board (I²C) | or skip it and use pigpio on GPIO18/13 |
| 6 V ≥ 5 A UBEC or bench supply + 1000–2200 µF capacitor | **dedicated servo rail**, see [docs/WIRING.md](docs/WIRING.md) |

## Pi setup

```bash
# Raspberry Pi OS Bookworm 64-bit
sudo apt update && sudo apt install -y python3-venv python3-picamera2 python3-opencv \
    pigpio python3-pigpio cmake g++ i2c-tools
sudo raspi-config nonint do_i2c 0           # enable I2C for the PCA9685
i2cdetect -y 1                              # expect 0x40

python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # keeps apt's picamera2
pip install -r requirements-pi.txt pybind11
cmake -S cpp -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DCMAKE_CXX_FLAGS="-mcpu=native"
cmake --build build -j4
export PYTHONPATH=$PWD/build
```

### Get the model

Export on a laptop (ultralytics and torch are too heavy for the Pi), then copy the file:

```bash
pip install ultralytics onnx onnxruntime
python scripts/export_onnx.py --imgsz 320                                  # dynamic int8, no data needed
python scripts/export_onnx.py --imgsz 320 --calib-dir path/to/drone/images # static int8 (faster on ARM)
scp models/yolov8n_int8.onnx pi@turret.local:~/uas-turret/models/
```

Stock COCO weights have **no drone class**. `detector.classes: [4, 14, 33]` (airplane,
bird, kite) works as a rough stand-in for testing. For real use, fine-tune YOLOv8n on a
public drone dataset (for example the Roboflow drone datasets or Anti-UAV), export with
`--weights best.pt`, and set `detector.classes: [0]`.

Then, on the Pi, measure the detector:

```bash
python scripts/bench_detector.py --model models/yolov8n_int8.onnx --runs 200
```

### First run

```bash
python scripts/servo_test.py --set servo.driver=pca9685         # check direction/trim/limits
python -m turret --mode live --set servo.driver=pca9685 --set camera.source=picamera2 \
    --metrics-csv run.csv
```

Put your settings in a file (for example `config/pi.yaml` with just the keys you
change) and run `python -m turret --config config/pi.yaml`.

## How it works

**Tracking in the absolute frame.** Each frame is stamped with the gimbal angles at
capture time (from a slew-rate model of the servos, since hobby servos report no
position). Detections are converted from pixels to absolute bearings with the pinhole
model, so the Kalman filter tracks the target in world angles. A stationary target stays
stationary in state space while the gimbal moves, and the camera's own motion never
looks like target motion.

**Kalman filter.** State `[az, el, v_az, v_el]`, continuous white-noise-acceleration
process model (correct for variable frame intervals), Joseph-form update. Measurements
are gated with a chi-square test plus a hard angular cap that widens the longer the
track coasts.

**Occlusion handling.** With no gated detection the track goes to `COAST` and keeps
predicting from the last measurement, with velocity decaying over `coast_velocity_tau`
so a target that turned while hidden doesn't send the gimbal far off. If the target
reappears outside the gate, two consistent out-of-gate detections in a row re-initialise
the filter there. After `coast_timeout` the track is `LOST`. The gimbal then holds on the
last aim point briefly and falls back to a raster search.

**Latency compensation.** The filter state is as old as the frame (capture + inference
time). The actuation thread extrapolates it to *now* before computing the error, so the
gimbal aims at where the target is rather than where it was. The estimated target
velocity is also fed forward to the axis controllers.

**Control.** Each axis has its own PID producing a *rate* command (deg/s), which is
acceleration-limited and integrated into the servo position setpoint at 100 Hz. The D
term acts on the measurement, so a new target never causes a derivative kick. Anti-windup
uses conditional integration plus an integral clamp. Inside the deadband the output is
zero and the integrator bleeds off. That is what removes the hunting caused by detector
jitter and gear backlash.

**Threads.** Capture never blocks on inference: it overwrites a one-frame mailbox, so a
slow detector drops stale frames rather than building a queue that adds latency.
Actuation runs on its own clock and keeps steering smoothly between detections and
through missed frames.

## Measuring latency and FPS

`python -m turret` logs, every couple of seconds:

```
TRACK | fps cap 30.0 inf 30.0 act 100.0 | infer p50 28.5 p95 37.5 ms | e2e p50 34.6 p95 48.0 ms | dropped 0
```

At exit it prints a JSON summary (mean/p50/p95/p99/max) of:

| metric | meaning |
|---|---|
| `queue_ms` | capture → inference start (time waiting for the detector) |
| `infer_ms` | preprocess + model + NMS |
| `track_ms` | tracker update |
| `e2e_ms` | capture timestamp → first servo command using that frame's measurement |

`--metrics-csv run.csv` writes one row per processed frame. `e2e_ms` starts at the
moment the frame reaches userspace. To get the full photon-to-motion loop, add camera
exposure and readout (about 10–20 ms on the Pi camera) and the servo's own response. You
can measure all of that together by filming an LED and the gimbal with a phone in slow
motion.

## Tuning

Use the offline tools first, because they're quick to iterate with, and then confirm on
hardware.

1. **Geometry.** Set `camera.width/height/hfov_deg/vfov_deg` correctly. A wrong FOV
   shows up as the gimbal consistently over- or under-shooting on re-centring.
2. **Servo direction and trim.** Use `scripts/servo_test.py`. Set `servo.slew_rate` from
   the servo datasheet (for example 0.19 s/60° → about 315 deg/s) and keep each axis's
   `pid.out_limit` at or below it.
3. **P first.** Set `ki=0`, `kd=0`, `feedforward_gain=0`. Raise `kp` until a step
   (`python scripts/benchmark.py step --set control.pan.pid.ki=0 ...`) or a hand-held
   target starts to overshoot, then back off about 30%.
4. **D to damp.** Add `kd` until the overshoot is gone. If the servos buzz, lower
   `d_filter_hz` or `kd`.
5. **I for drift.** Add a small `ki` to remove steady error from an unbalanced camera,
   which mostly matters on tilt. Keep `i_limit` small.
6. **Deadband.** Raise it until the gimbal stops making small corrections when the target
   is still. Start at about 2× the detector jitter in degrees (jitter in pixels ÷
   `fx·π/180`). Typical values are 0.3–0.6°.
7. **Feed-forward.** Raise `feedforward_gain` towards 1.0 to reduce lag on fast crossing
   targets. Back off if it overshoots when the target stops.
8. **Kalman.** `process_accel_std` is the main knob. Higher values follow manoeuvres
   faster but are noisier; lower values are smoother but lag turns. `meas_std` should
   roughly match the detector jitter in degrees. Run `python scripts/benchmark.py
   occlusion` after each change.

Pan and tilt usually end up with different gains. Tilt carries the camera's weight
against gravity and pan carries the whole tilt stage's inertia.

## Results so far (simulation only)

These numbers come from the deterministic simulator on a laptop. **They are not hardware
measurements.** Run the same scripts on your Pi and gimbal and use those numbers instead.

| | Kalman (default config) | Baseline (raw detections, hold on miss) |
|---|---|---|
| Tracking loss (target outside FOV after first lock) | **0.88 %** | 2.64 % |
| RMS pointing error while in view | **2.9°** | 6.6° |
| Track drops | 4 | 17 |

Setup: 40 scenarios × 60 s each, 8 occlusions/min lasting 0.25–0.9 s, abrupt velocity
changes every 0.6–2.5 s at 8–45 deg/s, 5 % missed detections, 3 % clutter. That is a
**66 % reduction in tracking loss**, and Kalman did as well or better in 37 of 40
scenarios. The reduction depends on the scenario: it fell to about 56 % with faster
targets (20–70 deg/s). The default pipeline step response shows 0 % overshoot, 0.66 s
settling for a 20° step on pan, and no hunting reversals.

On the real Pi the loop rate depends on the detector. YOLOv8n int8 at 320 px is
typically 25–35 ms on a Pi 5 with 4 threads, which allows 30 FPS. A Pi 4 is about twice
as slow, so expect closer to 15 FPS there unless you drop to 256 px input.

## Safety notes for the bench

- Keep fingers and cables out of the bracket's sweep. The search pattern moves the
  gimbal on its own when no target is visible. Set `control.search.enabled: false` to
  stop that.
- Soft travel limits are in `control.pan/tilt.min_angle/max_angle`. Set them inside the
  bracket's mechanical stops.
- On exit (Ctrl-C) the gimbal re-centres and the PWM outputs are switched off.

## Status

- Built and tested: the C++ core and its CTest suite, all 35 pytest tests (including C++
  parity, and the live path using a stand-in ONNX model on a video file), the threaded
  sim, and both benchmarks.
- Written but not run here, because it needs hardware or a weights download: the PCA9685
  and pigpio drivers, the Picamera2 source, and `export_onnx.py` against real YOLOv8
  weights. Run `servo_test.py`, `export_onnx.py` and `bench_detector.py` first when you
  bring up the hardware.
