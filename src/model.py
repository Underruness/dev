"""MobileIFNet: compact coarse-to-fine frame interpolation network.

Design goals (mobile / QNN friendly):
  * only standard ops: Conv2d(stride 1|2), LeakyReLU, concat, bilinear resize,
    sigmoid, element-wise mul/add, and one bilinear sampling (grid_sample).
    -> exports cleanly to ONNX, then to QNN (HTP/NPU) or TFLite-GPU.
  * no dilation, no depthwise 3x3 with exotic padding, no deformable conv,
    no (batched) matmul tricks -> avoids QNN layout fallbacks.
  * ~0.15 M params at default width: fits in L2, runs real-time on a phone NPU.

Regression target: per-pixel bidirectional flow (dx,dy for each input) plus a
blend mask, supervised at 3 scales (coarse -> fine residuals).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Conv3x3 + LeakyReLU (QNN-friendly: plain convolution, no norm layers)."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


def flow_to_grid(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert a pixel-space displacement field into a `grid_sample` grid.

    flow is the displacement (dx, dy) that takes a *target* pixel to its
    *source* location in the input frame.
    """
    dtype, device = flow.dtype, flow.device
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x = gx.unsqueeze(0) + flow[:, 0]
    y = gy.unsqueeze(0) + flow[:, 1]
    gx_n = x / max(width - 1, 1) * 2.0 - 1.0
    gy_n = y / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack((gx_n, gy_n), dim=-1)


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear backward warp of `img` by `flow` (pixels)."""
    b, _, h, w = img.shape
    grid = flow_to_grid(flow, h, w)
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class MobileIFNet(nn.Module):
    """Coarse-to-fine interpolator: IFNet-lite (flow + mask, 3 levels)."""

    def __init__(self, width: Tuple[int, int, int] = (24, 48, 64)):
        super().__init__()
        c0, c1, c2 = width
        self.width = width

        # shared encoder (both frames are stacked on the channel axis)
        self.enc0 = nn.Sequential(ConvBlock(6, c0), ConvBlock(c0, c0))
        self.enc1 = nn.Sequential(ConvBlock(c0, c1, stride=2), ConvBlock(c1, c1))
        self.enc2 = nn.Sequential(ConvBlock(c1, c2, stride=2), ConvBlock(c2, c2))

        # coarse flow (1/4 resolution): 4 flow channels + 1 mask channel
        self.head2 = nn.Conv2d(c2, 5, 3, padding=1, bias=True)

        # mid level (1/2): flow residual + mask
        self.fuse1 = ConvBlock(c1 + 4, c1)
        self.head1 = nn.Conv2d(c1, 5, 3, padding=1, bias=True)

        # full level: flow residual + mask
        self.fuse0 = ConvBlock(c0 + 4, c0)
        self.head0 = nn.Conv2d(c0, 5, 3, padding=1, bias=True)

        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)
        nn.init.zeros_(self.head1.weight)
        nn.init.zeros_(self.head1.bias)
        nn.init.zeros_(self.head0.weight)
        nn.init.zeros_(self.head0.bias)

    @staticmethod
    def _blend(
        img0: torch.Tensor, img1: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        w0 = backward_warp(img0, flow[:, 0:2])
        w1 = backward_warp(img1, flow[:, 2:4])
        a = torch.sigmoid(mask)
        return a * w0 + (1.0 - a) * w1

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> List[torch.Tensor]:
        """Returns predictions from coarse to fine: [p2(1/4), p1(1/2), p0(full)]."""
        f0 = self.enc0(torch.cat((img0, img1), dim=1))
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)

        h2 = self.head2(f2)
        flow2, mask2 = h2[:, 0:4], h2[:, 4:5]

        up2 = F.interpolate(flow2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        h1 = self.head1(self.fuse1(torch.cat((f1, up2), dim=1)))
        flow1 = up2 + h1[:, 0:4]
        mask1 = h1[:, 4:5]

        up1 = F.interpolate(flow1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        h0 = self.head0(self.fuse0(torch.cat((f0, up1), dim=1)))
        flow0 = up1 + h0[:, 0:4]
        mask0 = h0[:, 4:5]

        # deep supervision: blend at each resolution (inputs downsampled to match)
        i0_q = F.interpolate(img0, scale_factor=0.25, mode="area")
        i1_q = F.interpolate(img1, scale_factor=0.25, mode="area")
        i0_h = F.interpolate(img0, scale_factor=0.5, mode="area")
        i1_h = F.interpolate(img1, scale_factor=0.5, mode="area")

        p2 = self._blend(i0_q, i1_q, flow2, mask2)
        p1 = self._blend(i0_h, i1_h, flow1, mask1)
        p0 = self._blend(img0, img1, flow0, mask0)
        return [p2, p1, p0]

    # convenience for deployment / evaluation
    def interpolate(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        return self.forward(img0, img1)[-1]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":  # tiny smoke test
    m = MobileIFNet().eval()
    a = torch.rand(1, 3, 128, 224)
    b = torch.rand(1, 3, 128, 224)
    with torch.no_grad():
        out = m(a, b)
    print("params", count_params(m))
    print([tuple(o.shape) for o in out])
