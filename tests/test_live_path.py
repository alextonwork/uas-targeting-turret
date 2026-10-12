"""Exercise the live code path (OpenCV video source + onnxruntime detector + mock servos)
with a tiny stand-in ONNX model that has YOLOv8's input/output shapes. Skipped when
onnx / onnxruntime / OpenCV aren't installed."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
cv2 = pytest.importorskip("cv2")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

SIZE = 320
N = 2100  # anchors for 320x320 input (40^2 + 20^2 + 10^2)


def fake_yolo(path):
    """Output = fixed tensor with one 'airplane' box at the centre + 0 * sum(input)."""
    out = np.zeros((1, 84, N), np.float32)
    out[0, :4, 0] = (SIZE / 2, SIZE / 2, 24, 16)
    out[0, 4 + 4, 0] = 0.9
    const = numpy_helper.from_array(out, "const")
    zero = numpy_helper.from_array(np.zeros((1,), np.float32), "zero")
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["const", "z"], ["output0"]),
    ]
    g = helper.make_graph(
        nodes, "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, N])],
        initializer=[const, zero],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_onnx_detector_maps_boxes_back_to_frame(tmp_path):
    from turret.config import load_config
    from turret.detector import OnnxYoloDetector
    from turret.types import Frame

    model = tmp_path / "fake.onnx"
    fake_yolo(model)
    cfg = load_config(overrides=[f"detector.model={model}"])
    det = OnnxYoloDetector(cfg)
    img = np.zeros((480, 640, 3), np.uint8)
    dets = det.detect(Frame(0, 0.0, 0.0, 0.0, img))
    assert len(dets) == 1
    # network centre (160,160) maps to image centre after undoing the letterbox
    assert dets[0].u == pytest.approx(320, abs=1)
    assert dets[0].v == pytest.approx(240, abs=1)
    assert dets[0].w == pytest.approx(24 * 2, abs=1)


def test_live_mode_end_to_end_on_video_file(tmp_path):
    from turret.__main__ import main

    model = tmp_path / "fake.onnx"
    fake_yolo(model)
    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for i in range(60):
        vw.write(np.full((480, 640, 3), i * 3 % 255, np.uint8))
    vw.release()

    rc = main(["--mode", "live", "--duration", "1.5",
               "--set", f"detector.model={model}", "--set", f"camera.source={video}",
               "--set", "servo.driver=mock", "--set", "control.backend=python"])
    assert rc == 0
