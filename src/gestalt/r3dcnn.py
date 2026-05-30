"""A faithful miniature of 3D-RCNN (Kundu et al., CVPR 2018) on synthetic objects.

3D-RCNN's recipe, reproduced in small:
  - a low-dimensional SHAPE BASIS (PCA over a collection of 3D models) -> a shape
    code; the object is mean + sum_i code_i * basis_i (the "shape modes");
  - per-instance prediction of (shape code, pose) from the image;
  - a RENDER-AND-COMPARE loss: render the predicted shape at the predicted pose to
    a silhouette and compare to the 2D GT mask -> 2D supervision trains 3D.

Modernised vs the paper: a DIFFERENTIABLE voxel renderer (true backprop) instead
of finite-difference gradients through an OpenGL renderer; single object, no
detection backbone (RoI = whole image).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae3d import rotate3d, _down


# ---------------- shape basis (PCA over object voxel grids) ----------------

def voxelize(points, D):
    idx = np.clip(((points + 1) / 2 * D).astype(int), 0, D - 1)
    occ = np.zeros((D, D, D), np.float32)
    np.add.at(occ, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return (occ > 0).astype(np.float32)            # solid-ish occupancy


def build_shape_basis(object_points, D=24, n_comp=10):
    """PCA over the objects' occupancy grids -> (mean, basis, per-object codes)."""
    occ = np.stack([voxelize(p, D).ravel() for p in object_points])   # (O, D^3)
    mean = occ.mean(0)
    U, S, Vt = np.linalg.svd(occ - mean, full_matrices=False)
    basis = Vt[:n_comp]                                               # (n_comp, D^3)
    codes = (occ - mean) @ basis.T                                    # (O, n_comp)
    return mean.astype(np.float32), basis.astype(np.float32), codes.astype(np.float32)


# ---------------- differentiable renderer (shape code + pose -> silhouette) ----

def quat_to_R(q):
    q = F.normalize(q, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    return R.view(-1, 3, 3)


def render_silhouette(code, R, mean_t, basis_t, D, H):
    """code (N,n_comp), R (N,3,3) -> soft silhouette (N,H,H) in [0,1].
    occupancy = mean + code . basis ; rotate ; soft alpha-composite along depth."""
    occ = (mean_t + code @ basis_t).clamp(0, 1).view(-1, 1, D, D, D)
    occ = rotate3d(occ, R).clamp(0, 1)
    sil = 1 - torch.prod(1 - occ, dim=2)            # (N,1,D,D) prob a ray hits the shape
    return F.interpolate(sil, size=(H, H), mode="bilinear", align_corners=False)[:, 0]


# ---------------- the network: image -> (shape code, pose) ----------------

class R3DCNN(nn.Module):
    def __init__(self, n_comp, H=48):
        super().__init__()
        self.conv = nn.Sequential(_down(1, 16), _down(16, 32), _down(32, 64))
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                  nn.Linear(128, n_comp + 4))   # code + quaternion
        self.n_comp = n_comp

    def forward(self, x):
        h = F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)
        out = self.head(h)
        return out[:, :self.n_comp], quat_to_R(out[:, self.n_comp:])    # code, R
