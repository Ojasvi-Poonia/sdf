"""
Stage 7: INT8 post-training static quantization of the exported ONNX student.

After you train the student and export it to ONNX (see GUIDE Stage 5-6), this
shrinks it ~4x and speeds it up on edge/NPU hardware using a small calibration set.

Run:  python 04_quantize_int8.py
Requires: pip install onnx onnxruntime

!!! TWO THINGS YOU MUST SET CORRECTLY:
    1) INPUT_NAME  -> the model's real input tensor name (printed below on first run)
    2) preprocess() -> must match EXACTLY the normalization used during training
       (e.g. RF-DETR / DETR-family use ImageNet mean/std, NOT just /255).
       If preprocessing here differs from training, INT8 accuracy will tank.
"""
import os
import glob

import numpy as np
from PIL import Image
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

import config

ONNX_FP32 = "output/model.onnx"        # <- path to your exported FP32 ONNX
ONNX_INT8 = "output/model_int8.onnx"

# ImageNet normalization (correct for RF-DETR / DETR-family). Change if your
# student used different stats.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def discover_input_name(path):
    model = onnx.load(path)
    names = [i.name for i in model.graph.input]
    print(f"Model inputs: {names}")
    return names[0]


def preprocess(path, size=config.IMG_SIZE):
    im = Image.open(path).convert("RGB").resize((size, size))
    x = np.asarray(im).astype(np.float32) / 255.0   # HWC, [0,1]
    x = x.transpose(2, 0, 1)                         # CHW
    x = (x - MEAN) / STD                             # normalize
    return np.ascontiguousarray(x[None])             # NCHW


class ApplianceCalibReader(CalibrationDataReader):
    def __init__(self, folder, input_name, limit=200):
        self.input_name = input_name
        self.files = (
            glob.glob(os.path.join(folder, "*.jpg"))
            + glob.glob(os.path.join(folder, "*.jpeg"))
            + glob.glob(os.path.join(folder, "*.png"))
        )[:limit]
        if not self.files:
            raise SystemExit(f"No calibration images in {folder}. "
                             "Copy ~200 representative images there.")
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = preprocess(self.files[self.i])
        self.i += 1
        return {self.input_name: x}


if __name__ == "__main__":
    if not os.path.exists(ONNX_FP32):
        raise SystemExit(f"{ONNX_FP32} not found. Export your trained student to ONNX first.")
    input_name = discover_input_name(ONNX_FP32)
    reader = ApplianceCalibReader(config.CALIB_DIR, input_name)
    quantize_static(
        ONNX_FP32,
        ONNX_INT8,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 model written to {ONNX_INT8}")
    print("Next: convert to your on-device runtime (TFLite / NCNN) -- see GUIDE Stage 8.")
