#!/usr/bin/env python3
"""Build a *correct* frame-interpolation dataset from Xiph/DERF reference videos.

Why: the official Vimeo-90K mirror (data.csail.mit.edu) answers 403 to scripted
clients, and the HuggingFace re-uploads we tested (`danjacobellis/vimeo90k_triplet`,
`bijinc/vimeo-90k-mini`) store frames that are **not adjacent** — the linear-blend
PSNR there is ~25 dB instead of the ~31 dB of the real benchmark, i.e. the frames
are several frames/pans apart. Training on them teaches the wrong thing.

This script downloads real reference clips (Xiph/DERF, 720p50/60), decodes the
YUV4MPEG2 stream with numpy (no ffmpeg needed), resizes to 256x448 like Vimeo-90K
and stores each clip as a single `.npy` of consecutive RGB frames:

    <out>/<clip>.npy   shape (T, 256, 448, 3) uint8

Consecutive frames are guaranteed by construction, so triplets (t, t+1, t+2) are
true interpolation pairs.

    python scripts/fetch_derf.py --frames 300 --out /tmp/vfi/derf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

BASE = "https://media.xiph.org/video/derf/y4m/"

# 720p clips: real motion, 50/60 fps, 500+ frames each (~70 MB / 100 frames)
CLIPS = [
    "720p50_mobcal_ter.y4m",
    "720p50_parkrun_ter.y4m",
    "720p50_shields_ter.y4m",
    "720p5994_stockholm_ter.y4m",
    "FourPeople_1280x720_60.y4m",
    "Johnny_1280x720_60.y4m",
    "KristenAndSara_1280x720_60.y4m",
]


def y4m_frames(data: bytes, limit: int, max_bytes: int):
    """Yield RGB uint8 frames (H, W, 3) from a (possibly truncated) y4m blob."""
    nl = data.find(b"\n")
    if nl < 0:
        return
    header = data[:nl].decode("ascii", "replace").split()
    if not header or header[0] != "YUV4MPEG2":
        raise ValueError("not a y4m stream")
    w = h = None
    chroma = "420"
    for tok in header[1:]:
        if tok[0] == "W":
            w = int(tok[1:])
        elif tok[0] == "H":
            h = int(tok[1:])
        elif tok[0] == "C":
            chroma = tok[1:].lower()
    if not w or not h:
        raise ValueError("missing W/H in y4m header")
    if not chroma.startswith("420"):
        raise ValueError(f"unsupported chroma {chroma} (8-bit 420 only)")
    pos = nl + 1
    n = 0
    cw, ch = w // 2, h // 2
    while n < limit and pos < len(data):
        nl2 = data.find(b"\n", pos)
        if nl2 < 0 or not data[pos:nl2].startswith(b"FRAME"):
            break
        pos = nl2 + 1
        need = w * h + 2 * cw * ch
        if pos + need > len(data) or pos + need > max_bytes:
            break
        y = np.frombuffer(data, np.uint8, w * h, pos).reshape(h, w)
        u = np.frombuffer(data, np.uint8, cw * ch, pos + w * h).reshape(ch, cw)
        v = np.frombuffer(data, np.uint8, cw * ch, pos + w * h + cw * ch).reshape(ch, cw)
        pos += need
        u = np.repeat(np.repeat(u, 2, 0), 2, 1)
        v = np.repeat(np.repeat(v, 2, 0), 2, 1)
        ycc = np.stack([y, u, v], -1).astype(np.float32)
        # BT.601 limited range YUV -> RGB
        yy = (ycc[..., 0] - 16.0) * 1.164383
        cb = ycc[..., 1] - 128.0
        cr = ycc[..., 2] - 128.0
        r = yy + 1.596027 * cr
        g = yy - 0.391762 * cb - 0.812968 * cr
        b = yy + 2.017232 * cb
        rgb = np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)
        yield rgb
        n += 1


def fetch(url: str, max_bytes: int) -> bytes:
    print(f"  [get ] {url} (max {max_bytes/1e6:.0f} MB)", flush=True)
    out = subprocess.run(
        ["curl", "-sSL", "--max-time", "240", "-r", f"0-{max_bytes}",
         "-A", "Mozilla/5.0", url],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    print(f"  [ok  ] {len(out)/1e6:.0f} MB", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/vfi/derf")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--bytes", type=int, default=420_000_000)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in (args.clips or CLIPS):
        dst = os.path.join(args.out, name.replace(".y4m", ".npy"))
        if os.path.exists(dst):
            print(f"  [skip] {dst}")
            continue
        try:
            blob = fetch(BASE + name, args.bytes)
            frames = []
            for rgb in y4m_frames(blob, args.frames, len(blob)):
                im = Image.fromarray(rgb).resize((args.width, args.height), Image.LANCZOS)
                frames.append(np.array(im, dtype=np.uint8))
            if len(frames) < 8:
                print(f"  [warn] only {len(frames)} frames for {name}")
                continue
            arr = np.stack(frames)
            np.save(dst + ".part.npy", arr)
            os.replace(dst + ".part.npy", dst)
            print(f"  [save] {dst} {arr.shape} {arr.nbytes/1e6:.0f} MB")
        except Exception as exc:  # noqa: BLE001 - keep going on a broken clip
            print(f"  [fail] {name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
