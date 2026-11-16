"""Data loading for frame-interpolation triplets (consecutive frames).

Three sources are supported:

1. `NpyClips`        -- **used for the training in this repo.** Clips stored as
   (T, H, W, 3) uint8 `.npy` (see `scripts/fetch_derf.py`). Triplets are
   (t, t+stride, t+2*stride) inside ONE clip, so they are true interpolation
   pairs, and whole clips can be held out for validation.

2. `ParquetTriplets` -- HuggingFace mirror `danjacobellis/vimeo90k_triplet`
   (`data/train-XXXXX-of-00071-*.parquet`). NOTE: the frames of these mirrors
   are *not* adjacent (linear-blend PSNR ~19-25 dB instead of ~31 dB), see
   docs/DATASET.md. The loader itself is correct (triplets rebuilt per label).

3. `FolderTriplets` -- the original `vimeo_triplet/` layout after unzipping the
   official archive: sequences/<clip>/<seq>/im1.png, im2.png, im3.png driven by
   `tri_trainlist.txt` / `tri_testlist.txt`.

Target convention (same as every VFI paper): predict `im2` from `im1` and `im3`.
"""

from __future__ import annotations

import glob as _glob
import io
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# sample sources
# --------------------------------------------------------------------------- #
@dataclass
class Triplet:
    src: str          # "parquet" | "folder"
    ref: object       # parquet: (file_idx, row indices) | folder: dir path


class ParquetTriplets:
    """Index of triplets stored inside HuggingFace parquet shards."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self._tables: dict[int, pa.Table] = {}
        self._bytes: dict[int, pa.Array] = {}
        self.triplets: List[Tuple[int, int, int]] = []
        for fi, p in enumerate(self.paths):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing parquet shard: {p}")
            table = pq.read_table(p, memory_map=True)
            names = pc.struct_field(table["image"], "path").to_pylist()
            idx_by_order: List[int] = []
            for row, name in enumerate(names):
                base = (name or "").rsplit("/", 1)[-1]
                idx_by_order.append({"im1.png": 0, "im2.png": 1, "im3.png": 2}.get(base, -1))
            # group consecutive runs of im1,im2,im3
            triple: List[int] = []
            for row, slot in enumerate(idx_by_order):
                if slot == 0:
                    triple = [row]
                elif triple and slot == len(triple):
                    triple.append(row)
                else:
                    triple = []
                    continue
                if len(triple) == 3:
                    self.triplets.append((fi, triple[0], triple[2]))
                    triple = []

    def __len__(self) -> int:
        return len(self.triplets)

    def _columns(self, fi: int) -> Tuple[pa.Table, pa.Array]:
        if fi not in self._tables:
            self._tables[fi] = pq.read_table(self.paths[fi], memory_map=True)
            self._bytes[fi] = pc.struct_field(self._tables[fi]["image"], "bytes").combine_chunks()
        return self._tables[fi], self._bytes[fi]

    def get(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fi, a, b = self.triplets[i]
        _, blobs = self._columns(fi)
        # row a = im1, row b = im3, the middle frame is row a+1 (im2)
        ims = []
        for row in (a, a + 1, b):
            raw = blobs[row].as_py()
            ims.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8))
        return ims[0], ims[1], ims[2]


class FolderTriplets:
    """Official vimeo_triplet folder layout + list file."""

    def __init__(self, root: str, list_file: str):
        self.root = root
        self.triplets: List[str] = []
        with open(list_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = os.path.join(root, "sequences", line)
                if os.path.isdir(d):
                    self.triplets.append(d)

    def __len__(self) -> int:
        return len(self.triplets)

    def get(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.triplets[i]
        out = []
        for n in ("im1.png", "im2.png", "im3.png"):
            out.append(
                np.asarray(Image.open(os.path.join(d, n)).convert("RGB"), dtype=np.uint8)
            )
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# torch dataset
# --------------------------------------------------------------------------- #
def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.array(arr, dtype=np.uint8)).float().div_(255.0)
    return t.permute(2, 0, 1)  # CHW in [0,1]


class InterpDataset(Dataset):
    """Triplet dataset with random crop / flip augmentation for training."""

    def __init__(
        self,
        source,
        indices: Optional[Sequence[int]] = None,
        crop: Optional[int] = 192,
        train: bool = True,
        hflip: bool = True,
        temporal_reverse: bool = True,
        repeat: int = 1,
    ):
        self.source = source
        self.indices = list(range(len(source))) if indices is None else list(indices)
        self.crop = crop
        self.train = train
        self.hflip = hflip
        self.temporal_reverse = temporal_reverse
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.indices) * self.repeat

    def __getitem__(self, i: int):
        k = self.indices[i % len(self.indices)]
        im1, im2, im3 = self.source.get(k)
        a, gt, b = _to_tensor(im1), _to_tensor(im2), _to_tensor(im3)

        if self.train and self.temporal_reverse and random.random() < 0.5:
            a, b = b, a  # still predicts the middle frame

        if self.train and self.crop:
            _, h, w = gt.shape
            ch = min(self.crop, h)
            cw = min(self.crop, w)
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            a = a[:, top : top + ch, left : left + cw]
            gt = gt[:, top : top + ch, left : left + cw]
            b = b[:, top : top + ch, left : left + cw]

        if self.train and self.hflip and random.random() < 0.5:
            a, gt, b = a.flip(-1), gt.flip(-1), b.flip(-1)

        return a, b, gt


class NpyClips:
    """Consecutive-frame clips stored as (T, H, W, 3) uint8 .npy files.

    Triplets are (t, t+stride, t+2*stride) inside ONE clip, so they are true
    interpolation pairs (contrary to the broken HF mirrors of Vimeo-90K).
    Clips whose filename matches `val_names`, or the last `val_clips` files when
    no name is given, are held out entirely: the validation score is therefore
    never contaminated by neighbouring frames of the training set.
    """

    def __init__(self, directory: str, stride: int = 1, val_clips: int = 2,
                 val_names: Optional[Sequence[str]] = None):
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".npy")
        )
        if not files:
            raise FileNotFoundError(f"no .npy clips in {directory}")
        self.files = files
        self.stride = max(1, int(stride))
        self.clips = [np.load(f, mmap_mode="r") for f in files]
        self.pairs: List[Tuple[int, int]] = []
        self.val_pairs: List[Tuple[int, int]] = []
        if val_names:
            holdout = {i for i, f in enumerate(files)
                       if any(n and n in os.path.basename(f) for n in val_names)}
        else:
            n_val = min(val_clips, max(0, len(files) - 1))
            holdout = set(range(len(files) - n_val, len(files))) if n_val else set()
        for ci, clip in enumerate(self.clips):
            last = len(clip) - 2 * self.stride
            dest = self.val_pairs if ci in holdout else self.pairs
            for t in range(0, max(0, last)):
                dest.append((ci, t))
        self._all = self.pairs + self.val_pairs
        self.clip_names = [os.path.basename(f)[:-4] for f in files]

    def __len__(self) -> int:
        return len(self._all)

    def split(self) -> Tuple[List[int], List[int]]:
        return list(range(len(self.pairs))), list(
            range(len(self.pairs), len(self._all)))

    def get(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ci, t = self._all[i]
        clip = self.clips[ci]
        s = self.stride
        return (
            np.asarray(clip[t]),
            np.asarray(clip[t + s]),
            np.asarray(clip[t + 2 * s]),
        )


def build_sources(parquet_globs: Sequence[str], vimeo_root: Optional[str],
                  vimeo_list: Optional[str], derf_dir: Optional[str] = None,
                  derf_stride: int = 1, val_clips: int = 2,
                  val_names: Optional[Sequence[str]] = None):
    """Return the triplet source used for training/validation."""
    if derf_dir:
        return NpyClips(derf_dir, stride=derf_stride, val_clips=val_clips,
                        val_names=val_names)
    if vimeo_root and vimeo_list:
        return FolderTriplets(vimeo_root, vimeo_list), None
    paths: List[str] = []
    for g in parquet_globs:
        if os.path.isdir(g):
            paths += sorted(
                os.path.join(g, f) for f in os.listdir(g) if f.endswith(".parquet")
            )
        else:
            hits = sorted(_glob.glob(g))
            paths.extend(hits if hits else [g])
    if not paths:
        raise SystemExit("no parquet shards found; see scripts/fetch_vimeo90k.py")
    src = ParquetTriplets(sorted(paths))
    return src, None


def split_indices(n: int, val_frac: float = 0.1, seed: int = 0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]
