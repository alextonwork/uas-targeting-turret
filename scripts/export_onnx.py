#!/usr/bin/env python3
"""Export YOLOv8n to ONNX and quantize it to int8 for Raspberry Pi CPU inference.

Run this on a laptop/desktop (ultralytics + torch are heavy), then copy the .onnx file
to the Pi. The Pi itself only needs onnxruntime.

    pip install ultralytics onnx onnxruntime
    python scripts/export_onnx.py                          # stock COCO yolov8n.pt
    python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt \
        --calib-dir datasets/drones/images/val            # your fine-tuned drone model

Quantization modes:
  dynamic  weights int8, activations quantized at runtime. No data needed. ~2x smaller,
           modest speedup on ARM.
  static   weights + activations int8 with QDQ nodes, calibrated on real images. Best
           speed on Pi 4/5 CPUs; needs ~100-300 representative frames (--calib-dir).
           Falls back to dynamic if no calibration images are given.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np


def export_fp32(weights: str, imgsz: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    dst = out_dir / f"{Path(weights).stem}_{imgsz}_fp32.onnx"
    os.replace(path, dst)
    print(f"fp32 model: {dst}")
    return dst


class ImageCalibrationReader:
    """Feeds letterboxed calibration images to onnxruntime's static quantizer."""

    def __init__(self, model_path: Path, image_dir: str, imgsz: int, limit: int):
        import onnxruntime as ort

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from turret.detector import letterbox
        import cv2

        self.input_name = ort.InferenceSession(str(model_path)).get_inputs()[0].name
        files = sorted(
            f for ext in ("jpg", "jpeg", "png", "bmp")
            for f in glob.glob(os.path.join(image_dir, f"**/*.{ext}"), recursive=True)
        )
        random.Random(0).shuffle(files)
        self.files = files[:limit]
        if not self.files:
            raise SystemExit(f"no images found in {image_dir}")
        self.imgsz, self.letterbox, self.cv2 = imgsz, letterbox, cv2
        self.it = iter(self.files)
        print(f"calibrating on {len(self.files)} images")

    def get_next(self):
        f = next(self.it, None)
        if f is None:
            return None
        img, *_ = self.letterbox(self.cv2.imread(f), self.imgsz)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {self.input_name: np.ascontiguousarray(x)}

    def rewind(self):
        self.it = iter(self.files)


def quantize(fp32: Path, out: Path, calib_dir: str | None, imgsz: int, n_calib: int) -> None:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = fp32.with_name(fp32.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    # Keep the detection head in fp32: quantizing the final box/score convs costs a lot
    # of accuracy for almost no speed. Node names follow the ultralytics export.
    import onnx

    g = onnx.load(str(pre)).graph
    head = [n.name for n in g.node if "/model.22/" in n.name and n.op_type in ("Conv", "Concat", "Sigmoid", "Mul", "Add", "Sub", "Div")]

    if calib_dir:
        reader = ImageCalibrationReader(pre, calib_dir, imgsz, n_calib)
        quantize_static(str(pre), str(out), reader, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                        nodes_to_exclude=head)
        print(f"static int8 model: {out}")
    else:
        quantize_dynamic(str(pre), str(out), weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["Conv", "MatMul"], nodes_to_exclude=head)
        print(f"dynamic int8 model: {out}  (pass --calib-dir for static quantization)")
    pre.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--out", default="models/yolov8n_int8.onnx")
    p.add_argument("--calib-dir", default=None, help="folder of representative images for static int8")
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--fp32-only", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = export_fp32(a.weights, a.imgsz, out.parent)
    if not a.fp32_only:
        quantize(fp32, out, a.calib_dir, a.imgsz, a.n_calib)
        print(f"size: fp32 {fp32.stat().st_size / 1e6:.1f} MB -> int8 {out.stat().st_size / 1e6:.1f} MB")
    print("next: python scripts/bench_detector.py --model", out)


if __name__ == "__main__":
    main()
