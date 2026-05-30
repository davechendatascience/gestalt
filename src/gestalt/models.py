"""PyTorch models for the viewpoint-invariance test.

Three things to compare on viewpoint (in-plane rotation) EXTRAPOLATION:

  VanillaCNN  - the baseline. Region-local; learns whatever the training
                viewpoints show. Convolution is equivariant to pixel-translation
                but NOT to rotation, so out-of-range viewpoints are OOD.

  STN         - a localisation net predicts an affine theta; grid_sample
                canonicalises the input. This is AMORTISED coordinate-transform
                inference: a fast feedforward proposer for the "undo the
                viewpoint" step (the differentiable cousin of the analysis-by-
                synthesis settling).

  GestaltNet  - STN canonicaliser + encoder, trained with cross-entropy PLUS a
                multi-view invariance loss (two rotated views of the same image
                must map to the same embedding). This is the 2D proxy of
                Equivariant Neural Rendering's principle: force the
                representation to be invariant across viewpoints of one object,
                using the multi-view structure a 3D sim (Isaac) can generate.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                         nn.ReLU(), nn.MaxPool2d(2))


class VanillaCNN(nn.Module):
    def __init__(self, n_classes=10, ch=16):
        super().__init__()
        self.f = nn.Sequential(_block(1, ch), _block(ch, 2 * ch), _block(2 * ch, 4 * ch))
        self.head = nn.Linear(4 * ch, n_classes)

    def forward(self, x):
        h = F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)
        return self.head(h), h

    def embed(self, x):
        return F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)


class STN(nn.Module):
    """Predict an affine transform and canonicalise the input (grid_sample)."""

    def __init__(self, ch=8):
        super().__init__()
        self.loc = nn.Sequential(
            _block(1, ch), _block(ch, ch), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, 32), nn.ReLU(), nn.Linear(32, 6))
        self.loc[-1].weight.data.zero_()
        self.loc[-1].bias.data.copy_(torch.tensor([1., 0, 0, 0, 1, 0]))  # identity init

    def forward(self, x):
        theta = self.loc(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False), theta


class GestaltNet(nn.Module):
    """STN canonicaliser + encoder; embedding `h` used for the invariance loss."""

    def __init__(self, n_classes=10, ch=16):
        super().__init__()
        self.stn = STN()
        self.enc = nn.Sequential(_block(1, ch), _block(ch, 2 * ch), _block(2 * ch, 4 * ch))
        self.head = nn.Linear(4 * ch, n_classes)

    def forward(self, x):
        xc, theta = self.stn(x)
        h = F.adaptive_avg_pool2d(self.enc(xc), 1).flatten(1)
        return self.head(h), h, xc

    def embed(self, x):
        xc, _ = self.stn(x)
        return F.adaptive_avg_pool2d(self.enc(xc), 1).flatten(1)


def rotate_batch(x, angles_deg):
    """Rotate a batch (N,1,H,W) by per-sample angles (deg) via grid_sample."""
    a = angles_deg * torch.pi / 180.0
    c, s = torch.cos(a), torch.sin(a)
    theta = torch.zeros(x.size(0), 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = c, -s
    theta[:, 1, 0], theta[:, 1, 1] = s, c
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def random_affine_batch(x, severity, rng):
    """Apply a per-sample random SIMILARITY+shear transform (the widened
    viewpoint group): rotation, scale, translation, shear, all scaled by
    `severity` in [0,~1]. severity=0 is identity. theta = scale * R(rot) @ shear,
    plus translation, in normalised [-1,1] grid coords."""
    n = x.size(0)
    u = lambda: torch.tensor(rng.uniform(-1, 1, n), dtype=x.dtype)
    rot = u() * severity * (90 * torch.pi / 180)
    scale = torch.exp(u() * severity * 0.4)
    tx, ty = u() * severity * 0.3, u() * severity * 0.3
    sh = u() * severity * 0.3
    c, s = torch.cos(rot), torch.sin(rot)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = scale * c
    theta[:, 0, 1] = scale * (c * sh - s)
    theta[:, 1, 0] = scale * s
    theta[:, 1, 1] = scale * (s * sh + c)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)
