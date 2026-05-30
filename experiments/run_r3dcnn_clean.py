"""Honest re-measure of the voxel 3D-RCNN with CLEAN GT masks.

Fixes the corrupted masks found in run_r3dcnn_inspect.py:
  - SOLID voxelization (binary_closing + fill_holes) -> no hollow-shell holes;
  - GT mask = silhouette of the TRUE solid occupancy (NOT the PCA reconstruction)
    -> no ghosting, independent of the limited shape basis;
  - (the soft depth-union of a solid BINARY occupancy is clean -> no bloat.)

The model still renders its PCA-shape silhouette and render-and-compares to the
clean true mask, so IoU now honestly measures shape recovery (bounded by how well
the PCA basis can match the true silhouette), not "matches the bloated GT".

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_r3dcnn_clean.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.render3d import library
from gestalt.ae3d import rotate3d
from gestalt.r3dcnn import voxelize, render_silhouette, quat_to_R, R3DCNN

D, NC, H = 20, 8, 48
torch.manual_seed(0)


def voxelize_solid(points):
    occ = voxelize(points, D) > 0
    occ = binary_closing(occ, iterations=1)
    occ = binary_fill_holes(occ)
    return occ.astype(np.float32).ravel()


def sil_from(occ_flat, R):
    occ = occ_flat.clamp(0, 1).view(-1, 1, D, D, D)
    s = 1 - torch.prod(1 - rotate3d(occ, R), dim=2)
    return F.interpolate(s, (H, H), mode="bilinear", align_corners=False)[:, 0]


@torch.no_grad()
def gen(n, solid_t, codes_t, mean_t, basis_t, seed):
    rng = np.random.default_rng(seed); o = rng.integers(0, len(solid_t), n)
    R = quat_to_R(F.normalize(torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32), 1))
    occ = solid_t[o].view(n, 1, D, D, D); occ_r = rotate3d(occ, R)
    gt = F.interpolate(1 - torch.prod(1 - occ_r, 2), (H, H), mode="bilinear", align_corners=False)[:, 0]
    mask = (gt > 0.5).float()                                  # CLEAN true silhouette
    w = torch.linspace(1, .35, D).view(1, 1, D, 1, 1)
    shaded = F.interpolate((occ_r * w).amax(2), (H, H), mode="bilinear", align_corners=False)[:, 0]
    tex = .6 + .4 * torch.tensor(rng.random((n, H, H)), dtype=torch.float32)
    img = (shaded * tex + torch.tensor(rng.normal(0, .04, (n, H, H)), dtype=torch.float32)).clamp(0, 1)
    return img, mask, codes_t[o], R


def main():
    lib = library(); names = list(lib)
    solid = np.stack([voxelize_solid(lib[k][0]) for k in names])     # (O, D^3) clean solid
    mean = solid.mean(0); U, S, Vt = np.linalg.svd(solid - mean, full_matrices=False)
    basis = Vt[:NC]; codes = (solid - mean) @ basis.T
    solid_t = torch.tensor(solid); mean_t = torch.tensor(mean.astype(np.float32))
    basis_t = torch.tensor(basis.astype(np.float32)); codes_t = torch.tensor(codes.astype(np.float32))

    # sanity: GT mask vs true silhouette is now exact (GT IS the true silhouette)
    print("=== honest re-measure: clean GT masks (solid voxelization) ===")
    print(f"  solid-occupancy fill: hollow-shell holes removed; GT = true silhouette\n")

    Xtr, Mtr, Ctr, Rtr = gen(900, solid_t, codes_t, mean_t, basis_t, 0)
    Xte, Mte, Cte, Rte = gen(160, solid_t, codes_t, mean_t, basis_t, 1)
    net = R3DCNN(NC, H); opt = torch.optim.Adam(net.parameters(), lr=1.5e-3)
    t0 = time.time()
    for ep in range(70):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]; code, R = net(Xtr[idx].unsqueeze(1))
            sil = render_silhouette(code, R, mean_t, basis_t, D, H).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(sil, Mtr[idx]) + 0.1 * F.mse_loss(code, Ctr[idx]) + 0.1 * F.mse_loss(R, Rtr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        code, R = net(Xte.unsqueeze(1)); sil = render_silhouette(code, R, mean_t, basis_t, D, H)
        iou = (((sil > .5) & (Mte > .5)).sum((-1, -2)).float() / (((sil > .5) | (Mte > .5)).sum((-1, -2)).float() + 1e-6)).mean().item()
        # also the ceiling: best the PCA shape basis itself can do vs the clean true mask
        gt_sil = sil_from((mean_t + Cte @ basis_t), Rte)         # render the GROUND-TRUTH code+pose
        ceil = (((gt_sil > .5) & (Mte > .5)).sum((-1, -2)).float() / (((gt_sil > .5) | (Mte > .5)).sum((-1, -2)).float() + 1e-6)).mean().item()
    print(f"  render-and-compare IoU vs CLEAN true mask = {iou:.3f}   ({time.time()-t0:.0f}s)")
    print(f"  basis ceiling (true code+pose rendered vs clean mask) = {ceil:.3f}")
    print(f"  [prior, against CORRUPTED GT: ~0.77 -- flattered]")

    nv = quat_to_R(F.normalize(torch.tensor(np.random.default_rng(9).normal(size=(6, 4)), dtype=torch.float32), 1))
    with torch.no_grad(): novel = render_silhouette(code[:6], nv, mean_t, basis_t, D, H)
    rows = [(Xte, "input"), (Mte, "CLEAN GT mask"), (sil, "model render (R&C)"), (novel, "model shape, NOVEL pose")]
    fig, ax = plt.subplots(4, 6, figsize=(9, 6))
    for r, (A, lab) in enumerate(rows):
        for c in range(6): ax[r, c].imshow(A[c].numpy(), cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
        ax[r, 0].set_title(lab, fontsize=8, loc="left")
    fig.suptitle("Honest re-measure: clean GT masks (solid voxelization)"); fig.tight_layout()
    fig.savefig(ROOT / "docs" / "r3dcnn_clean_recon.png", dpi=110); plt.close(fig)
    print("  wrote docs/r3dcnn_clean_recon.png")


if __name__ == "__main__":
    main()
