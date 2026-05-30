"""Neural equivariant autoencoder: image -> canonical 3D voxel latent -> rotate by
the camera pose -> project (depth-collapse) -> rendered view.

The architecture bakes in the 3D inductive bias the factorization proved correct:
  - the latent z is a 3D FEATURE VOLUME (the nonlinear generalisation of the
    rank-3 structure S);
  - "change viewpoint" = ROTATE THE VOLUME in 3D (rho(R), the SO(3) action) --
    not a learned 2D warp;
  - rendering = rotate then PROJECT along depth (the generalisation of M).

Trained render-and-compare: encode one view -> z, then z must re-render EVERY
other view of the same object at its known pose. One z explaining all views
forces z to be the viewpoint-canonical 3D construct -> E(x) is the invariant
feature backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _down(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                         nn.BatchNorm2d(co), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.C, self.D = C, D
        self.conv = nn.Sequential(_down(1, 16), _down(16, 32), _down(32, 64))
        self.fc = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                                nn.Linear(256, C * D * D * D))

    def forward(self, x):
        h = F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)
        return self.fc(h).view(-1, self.C, self.D, self.D, self.D)


def rotate3d(z, R):
    """Rotate voxel feature volume z (N,C,D,D,D) by R (N,3,3) -- the SO(3) action."""
    theta = torch.zeros(z.size(0), 3, 4, device=z.device, dtype=z.dtype)
    theta[:, :3, :3] = R
    grid = F.affine_grid(theta, z.size(), align_corners=False)
    return F.grid_sample(z, grid, align_corners=False, padding_mode="zeros")


class Decoder(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, zR):
        feat = zR.sum(2)                              # project (orthographic depth-collapse)
        img = self.net(feat)
        img = F.interpolate(img, size=(self.H, self.H), mode="bilinear",
                            align_corners=False)
        return torch.sigmoid(img)


class EquivAE(nn.Module):
    def __init__(self, C=8, D=16, H=48):
        super().__init__()
        self.enc = Encoder(C, D)
        self.dec = Decoder(C, H)

    def encode(self, x):
        return self.enc(x)

    def render(self, z, R):
        return self.dec(rotate3d(z, R))

    def code(self, x):
        """Flattened canonical latent -- the viewpoint-invariant feature."""
        return self.enc(x).flatten(1)


class MultiTaskNet(nn.Module):
    """One CNN encoder, two heads: classify AND (optionally) explain the 3D via
    render-and-compare. With the render auxiliary OFF (lam=0) it is exactly a
    vanilla CNN classifier; ON, the SAME encoder is also told to reconstruct
    other views from a rotated 3D voxel -- i.e. equipped with the extra pose +
    3D information the sim provides. Tests: does that extra info help recognition?
    """

    def __init__(self, n_classes, C=8, D=16, H=48):
        super().__init__()
        self.C, self.D = C, D
        self.conv = nn.Sequential(_down(1, 16), _down(16, 32), _down(32, 64))
        self.cls = nn.Linear(64, n_classes)
        self.to_vox = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                                    nn.Linear(256, C * D * D * D))
        self.dec = Decoder(C, H)

    def feat(self, x):
        return F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)

    def forward(self, x):
        h = self.feat(x)
        z = self.to_vox(h).view(-1, self.C, self.D, self.D, self.D)
        return self.cls(h), z

    def render(self, z, R):
        return self.dec(rotate3d(z, R))
