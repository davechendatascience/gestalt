"""Inspect the 3D-RCNN GT masks. Are the silhouettes used as targets actually the
object's silhouette, or are they corrupted by the occupancy-PCA representation?

Renders, at a fixed pose, for several objects:
  row 1  TRUE silhouette       (binary voxelization of the real object)
  row 2  GT mask used in training = silhouette of the PCA reconstruction
                                    (mean + code . basis), thresholded
  row 3  silhouette of a HARD-thresholded PCA occupancy (sharper)
If row 2 is bloated/ghosted vs row 1, the masks are corrupted -> two causes:
PCA ghosting (small occupancy everywhere) AND the soft depth-union amplifying it.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_r3dcnn_inspect.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.render3d import library
from gestalt.ae3d import rotate3d
from gestalt.r3dcnn import build_shape_basis, voxelize, quat_to_R

D, NC, H = 20, 8, 48


def sil(occ_flat, R):
    occ = occ_flat.clamp(0, 1).view(-1, 1, D, D, D)
    occ_r = rotate3d(occ, R)
    s = 1 - torch.prod(1 - occ_r, dim=2)
    return F.interpolate(s, (H, H), mode="bilinear", align_corners=False)[:, 0]


def main():
    lib = library(); allnames = list(lib)
    mean, basis, codes = build_shape_basis([lib[k][0] for k in allnames], D, NC)
    mean_t, basis_t, codes_t = torch.tensor(mean), torch.tensor(basis), torch.tensor(codes)
    show = ["sphere", "cube", "torus", "Lshape", "cone", "rod"]
    idx = [allnames.index(n) for n in show]
    q = F.normalize(torch.tensor([[1.0, 0.3, 0.5, 0.2]]), 1)
    R = quat_to_R(q).repeat(len(show), 1, 1)

    true_occ = torch.tensor(np.stack([voxelize(lib[n][0], D).ravel() for n in show]))
    pca_occ = mean_t + codes_t[idx] @ basis_t
    true_sil = sil(true_occ, R)
    pca_sil = sil(pca_occ, R)
    sharp_sil = sil((pca_occ > 0.5).float(), R)

    # how much does the GT mask (pca, >0.3) over-cover the true silhouette?
    tm = (true_sil > 0.5); gm = (pca_sil > 0.3)
    iou = (((tm & gm).sum((-1, -2)).float()) / ((tm | gm).sum((-1, -2)).float() + 1e-6))
    bloat = gm.sum((-1, -2)).float() / (tm.sum((-1, -2)).float() + 1e-6)
    print("per-object   IoU(GTmask, true):", [f"{x:.2f}" for x in iou.tolist()])
    print("per-object   area(GTmask)/area(true):", [f"{x:.2f}" for x in bloat.tolist()])
    print(f"mean GTmask-vs-true IoU = {iou.mean():.3f}   mean bloat = {bloat.mean():.2f}")

    rows = [(true_sil, "TRUE silhouette"),
            (pca_sil, "GT mask (PCA recon) <- used in training"),
            (sharp_sil, "PCA hard-thresholded (sharper)")]
    fig, ax = plt.subplots(3, len(show), figsize=(len(show) * 1.4, 4.6))
    for r, (A, lab) in enumerate(rows):
        for c in range(len(show)):
            ax[r, c].imshow(A[c].numpy(), cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(show[c], fontsize=8)
        ax[r, 0].set_ylabel(lab, fontsize=7)
        ax[r, 0].set_title(f"{show[0]}\n{lab}" if r == 0 else lab, fontsize=7, loc="left")
    fig.suptitle("3D-RCNN GT-mask inspection: PCA-recon silhouette vs the true silhouette", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "r3dcnn_gt_inspect.png", dpi=110); plt.close(fig)
    print("wrote docs/r3dcnn_gt_inspect.png")


if __name__ == "__main__":
    main()
