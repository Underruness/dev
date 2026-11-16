"""Training entry point for MobileIFNet (CPU/GPU agnostic).

Features that matter for a long, interruptible training run:
  * resumable: `--resume checkpoints/ckpt_last.pt` (model + optimiser + step + history)
  * periodic checkpoints: `--ckpt-every` iterations, atomic writes
  * wall-clock budget: `--max-minutes` (the supervisor loop runs many slices)
  * CSV logs (`logs/train.csv`, `logs/val.csv`) so progress survives restarts
  * validation PSNR/SSIM against the linear-blend baseline

Example (CPU, short slice):
    python -m src.train --derf-dir /tmp/vfi/derf --val-clips-names Johnny,shields \
        --iters 20000 --crop 192 --batch 8 --threads 6 --max-minutes 25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import InterpDataset, build_sources, split_indices
from .model import MobileIFNet, count_params


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 1e-10 else -10.0 * math.log10(mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5):
    g = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM on [0,1] tensors (NCHW), 11x11 Gaussian, per channel."""
    win = _gaussian_window().to(a.device).expand(a.shape[1], 1, 11, 11)
    mu_a = F.conv2d(a, win, padding=5, groups=a.shape[1])
    mu_b = F.conv2d(b, win, padding=5, groups=a.shape[1])
    saa = F.conv2d(a * a, win, padding=5, groups=a.shape[1]) - mu_a ** 2
    sbb = F.conv2d(b * b, win, padding=5, groups=a.shape[1]) - mu_b ** 2
    sab = F.conv2d(a * b, win, padding=5, groups=a.shape[1]) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return (num / den).mean().item()


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def charbonnier(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - gt) ** 2 + eps ** 2).mean()


def total_loss(preds: List[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    """Deep supervision: weights 0.25 / 0.5 / 1.0 (coarse -> fine)."""
    gts = [
        F.interpolate(gt, scale_factor=0.25, mode="area"),
        F.interpolate(gt, scale_factor=0.5, mode="area"),
        gt,
    ]
    loss = 0.0
    for w, p, g in zip((0.25, 0.5, 1.0), preds, gts):
        loss = loss + w * charbonnier(p, g)
    return loss


# --------------------------------------------------------------------------- #
# train / val
# --------------------------------------------------------------------------- #
def make_loader(ds, batch: int, workers: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle,
        persistent_workers=workers > 0,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate(model, loader, limit: Optional[int] = None) -> dict:
    model.eval()
    p_model, p_blend, s_model, n = 0.0, 0.0, 0.0, 0
    for a, b, gt in loader:
        preds = model(a, b)
        out = preds[-1].clamp(0, 1)
        blend = 0.5 * a + 0.5 * b
        p_model += psnr(out, gt)
        p_blend += psnr(blend, gt)
        s_model += ssim(out, gt)
        n += 1
        if limit and n >= limit:
            break
    model.train()
    n = max(1, n)
    return {
        "val_psnr": p_model / n,
        "val_psnr_linear": p_blend / n,
        "val_ssim": s_model / n,
        "val_n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["/tmp/vfi/data/*.parquet"],
                    help="parquet shards/globs, or a directory")
    ap.add_argument("--vimeo-root", default=None, help="official vimeo_triplet dir")
    ap.add_argument("--vimeo-list", default=None, help="tri_trainlist.txt")
    ap.add_argument("--derf-dir", default=None, help="directory of .npy clips")
    ap.add_argument("--derf-stride", type=int, default=1)
    ap.add_argument("--val-clips", type=int, default=2,
                    help="clips fully held out for validation (derf mode)")
    ap.add_argument("--val-clips-names", default="",
                    help="comma separated substrings of held-out clips")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", default="24,48,64")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-limit", type=int, default=25,
                    help="number of val batches (<=0 = all)")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    width = tuple(int(x) for x in args.width.split(","))
    model = MobileIFNet(width=width)
    print(f"[model] MobileIFNet width={width} params={count_params(model):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter, best = 0, -1e9
    history = []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_iter = int(ck.get("iter", 0))
        best = float(ck.get("best_psnr", -1e9))
        history = ck.get("history", [])
        print(f"[resume] {args.resume} @ iter {start_iter} (best val psnr {best:.3f})",
              flush=True)

    # ---- data -------------------------------------------------------------
    src = build_sources(args.data, args.vimeo_root, args.vimeo_list,
                        derf_dir=args.derf_dir, derf_stride=args.derf_stride,
                        val_clips=args.val_clips,
                        val_names=[s for s in args.val_clips_names.split(",") if s])
    if hasattr(src, "split"):
        train_idx, val_idx = src.split()
    elif args.vimeo_root:
        train_idx, val_idx = list(range(len(src))), list(range(len(src)))
    else:
        train_idx, val_idx = split_indices(len(src), args.val_frac, args.seed)
    print(f"[data] {len(src)} triplets -> {len(train_idx)} train / {len(val_idx)} val",
          flush=True)

    ds_train = InterpDataset(src, train_idx, crop=args.crop, train=True, repeat=1)
    ds_val = InterpDataset(src, val_idx, crop=None, train=False)
    dl_train = make_loader(ds_train, args.batch, args.workers, shuffle=True)
    dl_val = make_loader(ds_val, max(1, min(args.batch, 4)), 0, shuffle=False)

    train_csv = "logs/train.csv"
    val_csv = "logs/val.csv"
    new_log = not os.path.exists(train_csv)
    ft = open(train_csv, "a", newline="")
    fv = open(val_csv, "a", newline="")
    wt = csv.writer(ft)
    wv = csv.writer(fv)
    if new_log:
        wt.writerow(["iter", "loss", "lr", "sec_per_iter", "elapsed_min", "timestamp"])
        wv.writerow(["iter", "val_psnr", "val_psnr_linear", "val_ssim", "val_n"])

    total_iters = args.iters
    t_start = time.time()
    t_last = t_start
    running = 0.0
    it = start_iter
    model.train()

    def save(tag: str, it: int, extra: dict) -> None:
        payload = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "iter": it,
            "width": width,
            "best_psnr": best,
            "history": history[-50:],
            "config": vars(args),
            **extra,
        }
        tmp = os.path.join(args.out, f"{tag}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(args.out, f"{tag}.pt"))

    def slice_over() -> bool:
        return args.max_minutes > 0 and (time.time() - t_start) / 60.0 >= args.max_minutes

    while it < total_iters and not slice_over():
        for a, b, gt in dl_train:
            if it >= total_iters or slice_over():
                break
            preds = model(a, b)
            loss = total_loss(preds, gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # cosine decay over the whole run
            frac = min(1.0, it / max(1, total_iters))
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()

            it += 1
            running += loss.item()
            now = time.time()
            spi = (now - t_last) / max(1, it - start_iter) if it > start_iter else 0.0
            t_last = now

            if it % 10 == 0:
                avg = running / 10.0
                running = 0.0
                elapsed = (now - t_start) / 60.0
                print(f"iter {it:6d} loss {avg:.4f} lr {lr:.2e} "
                      f"{spi*1000:.0f} ms/it elapsed {elapsed:.1f} min", flush=True)
                wt.writerow([it, f"{avg:.5f}", f"{lr:.3e}", f"{spi:.4f}",
                             f"{elapsed:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])
                ft.flush()

            if args.val_every and it % args.val_every == 0:
                m = evaluate(model, dl_val, args.val_limit if args.val_limit > 0 else None)
                print(f"  [val] iter {it} psnr {m['val_psnr']:.3f} dB "
                      f"(linear blend {m['val_psnr_linear']:.3f} dB) "
                      f"ssim {m['val_ssim']:.4f}", flush=True)
                wv.writerow([it, f"{m['val_psnr']:.4f}", f"{m['val_psnr_linear']:.4f}",
                             f"{m['val_ssim']:.5f}", m["val_n"]])
                fv.flush()
                history.append({"iter": it, **m})
                if m["val_psnr"] > best:
                    best = m["val_psnr"]
                    save("ckpt_best", it, {})
                with open("logs/best.json", "w") as fh:
                    json.dump({"iter": it, "best_psnr": best, **m}, fh, indent=2)

            if args.ckpt_every and it % args.ckpt_every == 0:
                save("ckpt_last", it, {})

    save("ckpt_last", it, {})
    ft.close()
    fv.close()
    print(f"[done] iter {it} best val psnr {best:.3f} elapsed "
          f"{(time.time() - t_start)/60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
