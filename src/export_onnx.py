"""Export a checkpoint to ONNX (fixed input size) and sanity-check it.

    python -m src.export_onnx --ckpt checkpoints/ckpt_last.pt \
        --height 256 --width 448 --out checkpoints/mobileifnet.onnx

The exported graph only uses Conv/LeakyRelu/Resize/Concat/GridSample/Sigmoid/Mul
so it is directly consumable by the Qualcomm AI Hub / QNN toolchain, or by
onnxruntime with the QNN Execution Provider on device.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .model import MobileIFNet, count_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_last.pt")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width-px", type=int, default=448)
    ap.add_argument("--out", default="checkpoints/mobileifnet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MobileIFNet(width=tuple(ck.get("width", (24, 48, 64)))).eval()
    model.load_state_dict(ck["model"])

    # single output tensor for deployment: the full-resolution frame
    class Deploy(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, a, b):
            return self.m.interpolate(a, b)

    dep = Deploy(model).eval()
    a = torch.rand(1, 3, args.height, args.width_px)
    b = torch.rand(1, 3, args.height, args.width_px)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.onnx.export(
        dep,
        (a, b),
        args.out,
        input_names=["frame0", "frame1"],
        output_names=["frame_mid"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[onnx] wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB, "
          f"{count_params(model):,} params)")

    # numeric check against pytorch
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ref = dep(a, b).detach().numpy()
    got = sess.run(None, {"frame0": a.numpy(), "frame1": b.numpy()})[0]
    print(f"[check] max abs diff vs torch: {np.abs(ref - got).max():.3e}")


if __name__ == "__main__":
    main()
