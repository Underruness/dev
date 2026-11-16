"""Evaluate a checkpoint on the held-out split; compare to trivial baselines.

    python -m src.eval --ckpt checkpoints/ckpt_last.pt --data '/tmp/vfi/data/*.parquet'
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import InterpDataset, build_sources, split_indices
from .model import MobileIFNet
from .train import psnr, ssim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_last.pt")
    ap.add_argument("--data", nargs="+", default=["/tmp/vfi/data/*.parquet"])
    ap.add_argument("--vimeo-root", default=None)
    ap.add_argument("--vimeo-list", default=None)
    ap.add_argument("--derf-dir", default=None)
    ap.add_argument("--derf-stride", type=int, default=1)
    ap.add_argument("--val-clips", type=int, default=2)
    ap.add_argument("--val-clips-names", default="")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all val samples")
    ap.add_argument("--save-samples", type=int, default=4,
                    help="write x3 side-by-side PNGs (gt | linear | model)")
    ap.add_argument("--out", default="logs/eval.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    width = tuple(ck.get("width", (24, 48, 64)))
    model = MobileIFNet(width=width).eval()
    model.load_state_dict(ck["model"])

    src = build_sources(args.data, args.vimeo_root, args.vimeo_list,
                        derf_dir=args.derf_dir, derf_stride=args.derf_stride,
                        val_clips=args.val_clips,
                        val_names=[s for s in args.val_clips_names.split(",") if s])
    if hasattr(src, "split"):
        _, val_idx = src.split()
    elif args.vimeo_root:
        val_idx = list(range(len(src)))
    else:
        _, val_idx = split_indices(len(src), args.val_frac, args.seed)
    ds = InterpDataset(src, val_idx, crop=None, train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    p_m = p_l = p_c = s_m = s_l = 0.0
    n = 0
    saved = 0
    os.makedirs("logs/samples", exist_ok=True)
    with torch.no_grad():
        for a, b, gt in dl:
            out = model(a, b)[-1].clamp(0, 1)
            lin = 0.5 * a + 0.5 * b
            p_m += psnr(out, gt)
            p_l += psnr(lin, gt)
            p_c += psnr(a, gt)  # "just repeat frame 1" baseline
            s_m += ssim(out, gt)
            s_l += ssim(lin, gt)
            if saved < args.save_samples:
                row = torch.cat((gt[0], lin[0], out[0]), dim=2).clamp(0, 1)
                arr = (row.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(f"logs/samples/cmp_{saved:02d}.png")
                saved += 1
            n += 1
            if args.limit and n >= args.limit:
                break

    n = max(1, n)
    stats = {
        "ckpt": args.ckpt,
        "iter": ck.get("iter"),
        "params": sum(p.numel() for p in model.parameters()),
        "n_samples": n,
        "psnr_model": round(p_m / n, 3),
        "psnr_linear_blend": round(p_l / n, 3),
        "psnr_repeat_frame": round(p_c / n, 3),
        "ssim_model": round(s_m / n, 4),
        "ssim_linear_blend": round(s_l / n, 4),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
