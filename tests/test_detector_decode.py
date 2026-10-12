"""YOLOv8 output decoding and NMS, on synthetic tensors (no model or onnxruntime needed)."""
import numpy as np

from turret.detector import decode_yolov8, nms


def make_output(boxes, num_classes=80):
    """boxes: list of (cx, cy, w, h, cls, score) -> (1, 4+C, N) like the ONNX export."""
    out = np.zeros((1, 4 + num_classes, len(boxes)), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


def test_decode_filters_by_conf_and_class():
    out = make_output([(100, 100, 20, 10, 4, 0.9),   # airplane, keep
                       (200, 50, 20, 10, 4, 0.2),    # below threshold
                       (50, 60, 30, 30, 0, 0.95)])   # person, not in class list
    dets = decode_yolov8(out, 0.35, 0.45, [4, 14, 33], 10)
    assert len(dets) == 1
    cx, cy, w, h, conf, cls = dets[0]
    assert (cx, cy, cls) == (100, 100, 4)
    assert abs(conf - 0.9) < 1e-6


def test_nms_suppresses_overlaps_keeps_separate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_decode_empty():
    assert decode_yolov8(make_output([]), 0.35, 0.45, [], 10) == []
