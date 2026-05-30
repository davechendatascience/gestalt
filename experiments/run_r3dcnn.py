"""Train the 3D-RCNN miniature: image -> (shape code, pose), supervised by a
RENDER-AND-COMPARE loss against the 2D silhouette (+ light shape/pose aux).

Synthetic single-object scenes: an object (occupancy from the PCA shape basis) at
a random rotation, rendered to a shaded image (input) and a silhouette (the 2D
mask, the render-and-compare target). The net infers shape+pose; we render its
prediction and compare to the mask. Then we render the predicted shape from a
NOVEL pose to show full 3D was recovered from one view.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_r3dcnn.py
"""
from __future__ import annotations
import sys, time
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
from gestalt.r3dcnn import (build_shape_basis, render_silhouette, quat_to_R, R3DCNN, voxelize)

D, NC, H = 20, 10, 48
torch.manual_seed(0)


def rand_quats(n, rng):
    q = torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32)
    return F.normalize(q, dim=1)


@torch.no_grad()
def gen(n, codes, mean_t, basis_t, seed):
    rng = np.random.default_rng(seed)
    o = rng.integers(0, len(codes), n)
    gt_code = codes[o]
    R = quat_to_R(rand_quats(n, rng))
    occ = (mean_t + gt_code @ basis_t).clamp(0, 1).view(n, 1, D, D, D)
    occ_r = rotate3d(occ, R)
    w = torch.linspace(1.0, 0.35, D).view(1, 1, D, 1, 1)
    shaded = F.interpolate((occ_r * w).amax(2), size=(H, H), mode="bilinear",
                           align_corners=False)[:, 0]
    sil = F.interpolate(1 - torch.prod(1 - occ_r, dim=2), size=(H, H),
                        mode="bilinear", align_corners=False)[:, 0]
    mask = (sil > 0.3).float()
    tex = 0.6 + 0.4 * torch.tensor(rng.random((n, H, H)), dtype=torch.float32)
    img = (shaded * tex + torch.tensor(rng.normal(0, 0.04, (n, H, H)), dtype=torch.float32)).clamp(0, 1)
    return img, mask, gt_code, R


def main():
    pts = [library()[k][0] for k in library()]
    mean, basis, codes = build_shape_basis(pts, D, NC)
    mean_t = torch.tensor(mean); basis_t = torch.tensor(basis); codes_t = torch.tensor(codes)
    print(f"=== 3D-RCNN miniature: shape basis ({NC} modes) + render-and-compare ===")
    rec = (mean_t + codes_t @ basis_t).clamp(0, 1)                # basis reconstruction sanity
    true_occ = torch.tensor(np.stack([voxelize(p, D).ravel() for p in pts]))
    print(f"  shape-basis reconstruction err (occupancy): "
          f"{F.mse_loss(rec, true_occ).item():.4f}\n")

    Xtr, Mtr, Ctr, Rtr = gen(900, codes_t, mean_t, basis_t, 0)
    Xte, Mte, Cte, Rte = gen(120, codes_t, mean_t, basis_t, 1)

    net = R3DCNN(NC, H)
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3)
    n, bs = len(Xtr), 32
    t0 = time.time()
    for ep in range(70):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            code, R = net(Xtr[idx].unsqueeze(1))
            sil = render_silhouette(code, R, mean_t, basis_t, D, H).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(sil, Mtr[idx]) \
                + 0.1 * F.mse_loss(code, Ctr[idx]) \
                + 0.1 * F.mse_loss(R, Rtr[idx])           # render-and-compare + light aux
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/70  loss={loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        code, R = net(Xte.unsqueeze(1))
        sil = render_silhouette(code, R, mean_t, basis_t, D, H)
        miou = (((sil > 0.5) & (Mte > 0.5)).sum((-1, -2)).float()
                / (((sil > 0.5) | (Mte > 0.5)).sum((-1, -2)).float() + 1e-6)).mean().item()
        code_err = F.mse_loss(code, Cte).item()
    print(f"\n  render-and-compare silhouette IoU (test) = {miou:.3f}   "
          f"shape-code MSE = {code_err:.4f}   ({time.time()-t0:.0f}s)")

    # visualise: input, GT mask, predicted silhouette, predicted shape at a NOVEL pose
    with torch.no_grad():
        nv = quat_to_R(rand_quats(6, np.random.default_rng(9)))
        novel = render_silhouette(code[:6], nv, mean_t, basis_t, D, H)
    fig, ax = plt.subplots(4, 6, figsize=(9, 6))
    rows = [(Xte, "input"), (Mte, "GT mask"), (sil, "pred silhouette (R&C)"),
            (novel, "pred shape, NOVEL pose")]
    for r, (A, lab) in enumerate(rows):
        for c in range(6):
            ax[r, c].imshow(A[c].numpy(), cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
        ax[r, 0].set_ylabel(lab, fontsize=8)
        ax[r, 0].set_title(lab, fontsize=8, loc="left")
    fig.suptitle("3D-RCNN miniature: 2D render-and-compare recovers 3D shape+pose", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "r3dcnn_recon.png", dpi=110); plt.close(fig)
    print("  wrote docs/r3dcnn_recon.png")


if __name__ == "__main__":
    main()
