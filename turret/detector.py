"""YOLOv8 ONNX detector (onnxruntime), with letterbox preprocessing and numpy NMS.

Works with the fp32 or int8-quantized model produced by scripts/export_onnx.py.
YOLOv8 ONNX output is (1, 4 + num_classes, N): cx, cy, w, h, then per-class scores.
"""
from __future__ import annotations

import logging

import numpy as np

from .types import Detection, Frame

log = logging.getLogger(__name__)


def letterbox(img: np.ndarray, size: int):
    """Resize keeping aspect ratio and pad to size x size. Returns (img, scale, pad_x, pad_y)."""
    import cv2

    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = resized
    return out, scale, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """boxes: (N, 4) x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_yolov8(
    output: np.ndarray, conf_thr: float, iou_thr: float, classes, max_det: int
) -> list[tuple]:
    """Returns [(cx, cy, w, h, conf, cls)] in network-input pixel coordinates."""
    pred = np.squeeze(output, 0).T  # (N, 4 + C)
    boxes, scores_all = pred[:, :4], pred[:, 4:]
    if classes:
        mask = np.zeros(scores_all.shape[1], dtype=bool)
        mask[[c for c in classes if c < scores_all.shape[1]]] = True
        scores_all = np.where(mask, scores_all, 0.0)
    cls = scores_all.argmax(1)
    conf = scores_all[np.arange(len(cls)), cls]
    sel = conf >= conf_thr
    if not sel.any():
        return []
    boxes, conf, cls = boxes[sel], conf[sel], cls[sel]
    xyxy = np.column_stack(
        [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2]
    )
    keep = nms(xyxy, conf, iou_thr)[:max_det]
    return [(*boxes[i], float(conf[i]), int(cls[i])) for i in keep]


class OnnxYoloDetector:
    def __init__(self, cfg):
        import onnxruntime as ort

        d = cfg.detector
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(d.num_threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(d.model, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else int(d.input_size)
        self.conf, self.iou = d.conf_threshold, d.iou_threshold
        self.classes = list(d.classes or [])
        self.max_det = int(d.max_detections)
        log.info("ONNX model %s, input %dx%d", d.model, self.size, self.size)

    def detect(self, frame: Frame) -> list[Detection]:
        img, scale, px, py = letterbox(frame.image, self.size)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        dets = []
        for cx, cy, w, h, conf, cls in decode_yolov8(out, self.conf, self.iou, self.classes, self.max_det):
            dets.append(Detection((cx - px) / scale, (cy - py) / scale, w / scale, h / scale, conf, cls))
        return dets


def make_detector(cfg, cam):
    if cfg.mode == "sim":
        from .sim import SimDetector

        return SimDetector(cfg, cam)
    return OnnxYoloDetector(cfg)
