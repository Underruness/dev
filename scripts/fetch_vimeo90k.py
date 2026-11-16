#!/usr/bin/env python3
"""Download a subset of Vimeo-90K triplets as parquet shards (HuggingFace mirror).

The official MIT mirror (data.csail.mit.edu) returns 403 for scripted clients and
the full septuplet archive is ~80 GB. The HF mirror `danjacobellis/vimeo90k_triplet`
stores the triplets as ~71 parquet shards (264 MB each, ~1000 triplets per shard).

WARNING: the frames stored by this mirror are NOT adjacent (see docs/DATASET.md),
so check the linear-blend PSNR (~30-40 dB expected) before training on it.

    python scripts/fetch_vimeo90k.py --shards 0-3 --out /tmp/vfi/data
    python scripts/fetch_vimeo90k.py --shards all --out /data/vimeo90k   # ~19 GB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

API = ("https://huggingface.co/api/datasets/danjacobellis/vimeo90k_triplet/"
       "tree/main/data?recursive=true")
BASE = ("https://huggingface.co/datasets/danjacobellis/vimeo90k_triplet/resolve/main/")


def list_files() -> list[str]:
    with urllib.request.urlopen(API, timeout=60) as fh:
        tree = json.load(fh)
    return sorted(x["path"] for x in tree
                  if x["type"] == "file" and x["path"].endswith(".parquet"))


def parse_range(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0-3", help="e.g. 0-3, 7, 'all'")
    ap.add_argument("--out", default="/tmp/vfi/data")
    args = ap.parse_args()

    files = list_files()
    idx = parse_range(args.shards, len(files))
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(files)} shards available; downloading {len(idx)} -> {args.out}")
    for i in idx:
        dst = os.path.join(args.out, f"shard{i:05d}.parquet")
        if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
            print(f"  [skip] {dst}")
            continue
        url = BASE + files[i]
        print(f"  [get ] shard {i}: {files[i]}", flush=True)
        tmp = dst + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        print(f"  [ok  ] {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
