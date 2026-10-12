#!/usr/bin/env python3
"""Time the ONNX detector on this machine (run it on the Pi to get real numbers).

    python scripts/bench_detector.py --model models/yolov8n_int8.onnx --runs 200
    python scripts/bench_detector.py --model models/yolov8n_320_fp32.onnx   # compare

Reports preprocess, model and postprocess time separately, plus p50/p95.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turret.config import load_config  # noqa: E402
from turret.detector import OnnxYoloDetector, decode_yolov8, letterbox  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--image", default=None, help="test image (default: random noise)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args()

    over = []
    if a.model:
        over.append(f"detector.model={a.model}")
    if a.threads:
        over.append(f"detector.num_threads={a.threads}")
    cfg = load_config(overrides=over)
    det = OnnxYoloDetector(cfg)

    if a.image:
        import cv2
        img = cv2.imread(a.image)
    else:
        img = np.random.default_rng(0).integers(0, 255, (cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)

    pre, run, post = [], [], []
    for i in range(a.runs + 10):
        t0 = time.perf_counter()
        lb, *_ = letterbox(img, det.size)
        x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        t1 = time.perf_counter()
        out = det.sess.run(None, {det.input_name: x})[0]
        t2 = time.perf_counter()
        decode_yolov8(out, det.conf, det.iou, det.classes, det.max_det)
        t3 = time.perf_counter()
        if i >= 10:  # warm-up
            pre.append(t1 - t0); run.append(t2 - t1); post.append(t3 - t2)

    def fmt(name, v):
        v = np.asarray(v) * 1e3
        return f"{name:12s} mean {v.mean():6.2f}  p50 {np.percentile(v, 50):6.2f}  p95 {np.percentile(v, 95):6.2f} ms"

    tot = np.asarray(pre) + np.asarray(run) + np.asarray(post)
    print(f"model {cfg.detector.model}  input {det.size}  threads {cfg.detector.num_threads}")
    print(fmt("preprocess", pre)); print(fmt("model", run)); print(fmt("postprocess", post)); print(fmt("total", tot))
    print(f"max sustainable detector rate: {1000 / (tot.mean() * 1e3):.1f} FPS")


if __name__ == "__main__":
    main()
