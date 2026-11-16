"""Post-training quantization for on-device deployment.

Two artefacts are produced:
  * int8 static (QDQ) ONNX  -- smallest, for CPU/GPU/Hexagon via ORT-QNN or
                               TFLite converter. Uses Vimeo-90K calibration data.
  * fp16 ONNX               -- what you normally push to Qualcomm AI Hub for
                               HTP compile (fp16 is the native HTP datatype).

    python -m src.quantize --onnx checkpoints/mobileifnet.onnx \
        --data '/tmp/vfi/data/*.parquet'
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16  # noqa: F401  (optional dep)
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

from .data import InterpDataset, build_sources, split_indices


class VimeoCalib(CalibrationDataReader):
    def __init__(self, data_globs, n: int = 24, derf_dir: Optional[str] = None,
                 val_names=None):
        src = build_sources(data_globs, None, None, derf_dir=derf_dir,
                            val_names=val_names)
        if hasattr(src, "split"):
            _, val_idx = src.split()
        else:
            _, val_idx = split_indices(len(src), 0.08, 0)
        ds = InterpDataset(src, val_idx, crop=None, train=False)
        self.items = []
        for i in range(min(n, len(ds))):
            a, b, _ = ds[i]
            self.items.append({
                "frame0": a.unsqueeze(0).numpy().astype(np.float32),
                "frame1": b.unsqueeze(0).numpy().astype(np.float32),
            })
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        it = self.items[self.i]
        self.i += 1
        return it


def bench(path: str, iters: int = 8) -> float:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    a = np.random.rand(1, 3, 256, 448).astype(np.float32)
    b = np.random.rand(1, 3, 256, 448).astype(np.float32)
    for _ in range(2):
        sess.run(None, {"frame0": a, "frame1": b})
    t0 = time.time()
    for _ in range(iters):
        sess.run(None, {"frame0": a, "frame1": b})
    return (time.time() - t0) / iters * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="checkpoints/mobileifnet.onnx")
    ap.add_argument("--data", nargs="+", default=["/tmp/vfi/data/*.parquet"])
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--derf-dir", default=None)
    ap.add_argument("--val-clips-names", default="")
    args = ap.parse_args()

    base = args.onnx
    fp32_mb = os.path.getsize(base) / 1e6

    # ---- int8 static -----------------------------------------------------
    int8 = os.path.join(args.out_dir, "mobileifnet_int8.onnx")
    quantize_static(
        base,
        int8,
        VimeoCalib(args.data, args.calib, args.derf_dir,
                   [s for s in args.val_clips_names.split(",") if s]),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    int8_mb = os.path.getsize(int8) / 1e6

    # ---- fp16 ------------------------------------------------------------
    fp16_path = os.path.join(args.out_dir, "mobileifnet_fp16.onnx")
    m = onnx.load(base)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, fp16_path)
    fp16_mb = os.path.getsize(fp16_path) / 1e6

    print(f"fp32 : {fp32_mb:6.2f} MB  ort-cpu {bench(base):7.1f} ms/frame")
    print(f"fp16 : {fp16_mb:6.2f} MB  ort-cpu {bench(fp16_path):7.1f} ms/frame")
    print(f"int8 : {int8_mb:6.2f} MB  ort-cpu {bench(int8):7.1f} ms/frame")
    print("\nNote: CPU numbers here are only a sanity check. Real throughput comes "
          "from the phone (QNN HTP / GPU delegate) -- see android/README.md.")


if __name__ == "__main__":
    main()
