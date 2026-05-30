"""Multi-class 3D-RCNN over our 14 objects done two ways, with CLEAN masks + viz.

Paper (Kundu CVPR2018, papers/): per-CLASS shape space (a 10-dim PCA basis over the
*intra-class* CAD variation, e.g. many cars), class from the detector, allocentric
pose, shape+pose as binned classification+regression. Our 14 objects have NO
intra-class variation -> each is a CLASS with a FIXED shape, so the faithful form is
a class head that selects the shape + a pose head, trained by render-and-compare.

Compared:
  shared-basis : one low-dim (NC=8) PCA basis over all 14 (continuous code+pose) --
                 wrong for heterogeneous objects (lossy/ghosting), the ~0.66 case.
  per-class    : 14 classes, each a FIXED shape; class head + pose; render the
                 predicted class's exact shape -> no ghosting.

Clean GT masks (solid voxelization, true silhouettes). Reports IoU + class accuracy
and writes docs/r3dcnn_multiclass.png.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_r3dcnn_multiclass.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.render3d import library
from gestalt.ae3d import rotate3d, _down
from gestalt.r3dcnn import voxelize, render_silhouette, quat_to_R, R3DCNN

D, NCsh, H = 20, 8, 48
torch.manual_seed(0)


def voxelize_solid(points):
    occ = binary_fill_holes(binary_closing(voxelize(points, D) > 0, iterations=1))
    return occ.astype(np.float32).ravel()


def render_occ(occ_n, R):                              # occ_n (n, D^3) -> silhouette
    occ = occ_n.clamp(0, 1).view(-1, 1, D, D, D)
    s = 1 - torch.prod(1 - rotate3d(occ, R), dim=2)
    return F.interpolate(s, (H, H), mode="bilinear", align_corners=False)[:, 0]


class MCNet(nn.Module):                                # class head + pose (fixed per-class shape)
    def __init__(self, K, H):
        super().__init__()
        self.conv = nn.Sequential(_down(1, 16), _down(16, 32), _down(32, 64))
        self.cls = nn.Linear(64, K); self.pose = nn.Linear(64, 4)
    def forward(self, x):
        h = F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)
        return self.cls(h), quat_to_R(self.pose(h))


def iou(a, b):
    return (((a > .5) & (b > .5)).sum((-1, -2)).float() / (((a > .5) | (b > .5)).sum((-1, -2)).float() + 1e-6))


def main():
    lib = library(); names = list(lib); K = len(names)
    occ = np.stack([voxelize_solid(lib[n][0]) for n in names])          # (14, D^3) fixed shapes
    occ_t = torch.tensor(occ)
    mean = occ.mean(0); _, _, Vt = np.linalg.svd(occ - mean, full_matrices=False)
    bsh = Vt[:NCsh].astype(np.float32); csh = ((occ - mean) @ bsh.T).astype(np.float32)
    msh_t, bsh_t, csh_t = torch.tensor(mean.astype(np.float32)), torch.tensor(bsh), torch.tensor(csh)

    @torch.no_grad()
    def gen(n, seed):
        r = np.random.default_rng(seed); o = r.integers(0, K, n)
        R = quat_to_R(F.normalize(torch.tensor(r.normal(size=(n, 4)), dtype=torch.float32), 1))
        mask = (render_occ(occ_t[o], R) > .5).float()                   # CLEAN true silhouette
        occ_r = rotate3d(occ_t[o].view(n, 1, D, D, D), R)
        w = torch.linspace(1, .35, D).view(1, 1, D, 1, 1)
        shaded = F.interpolate((occ_r * w).amax(2), (H, H), mode="bilinear", align_corners=False)[:, 0]
        tex = .6 + .4 * torch.tensor(r.random((n, H, H)), dtype=torch.float32)
        img = (shaded * tex + torch.tensor(r.normal(0, .04, (n, H, H)), dtype=torch.float32)).clamp(0, 1)
        return img, mask, torch.tensor(o), csh_t[o], R

    Xtr, Mtr, Ltr, Ctr, Rtr = gen(1200, 0)
    Xte, Mte, Lte, Cte, Rte = gen(240, 1)
    print(f"=== multi-class 3D-RCNN over {K} classes: shared basis vs per-class fixed shapes ===\n")

    # shared low-dim basis (continuous code + pose)
    sh = R3DCNN(NCsh, H); opt = torch.optim.Adam(sh.parameters(), 1.5e-3); t0 = time.time()
    for ep in range(60):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(p), 32):
            j = p[i:i + 32]; code, R = sh(Xtr[j].unsqueeze(1))
            sil = render_silhouette(code, R, msh_t, bsh_t, D, H).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(sil, Mtr[j]) + .1 * F.mse_loss(code, Ctr[j]) + .1 * F.mse_loss(R, Rtr[j])
            opt.zero_grad(); loss.backward(); opt.step()

    # per-class: class head + pose, decode the (true at train / predicted at eval) class's fixed shape
    mc = MCNet(K, H); opt = torch.optim.Adam(mc.parameters(), 1.5e-3)
    for ep in range(60):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(p), 32):
            j = p[i:i + 32]; lg, R = mc(Xtr[j].unsqueeze(1))
            sil = render_occ(occ_t[Ltr[j]], R).clamp(1e-4, 1 - 1e-4)
            loss = F.cross_entropy(lg, Ltr[j]) + F.binary_cross_entropy(sil, Mtr[j]) + .1 * F.mse_loss(R, Rtr[j])
            opt.zero_grad(); loss.backward(); opt.step()

    sh.eval(); mc.eval()
    with torch.no_grad():
        c, R = sh(Xte.unsqueeze(1)); sil_sh = render_silhouette(c, R, msh_t, bsh_t, D, H)
        lg, Rp = mc(Xte.unsqueeze(1)); pred = lg.argmax(1); sil_mc = render_occ(occ_t[pred], Rp)
        iou_sh, iou_mc = iou(sil_sh, Mte).mean().item(), iou(sil_mc, Mte).mean().item()
        acc = (pred == Lte).float().mean().item()
    print(f"  shared 8-dim basis (continuous code) : IoU {iou_sh:.3f}")
    print(f"  per-class 14 fixed shapes + class head: IoU {iou_mc:.3f}   class-acc {acc:.3f}")
    print(f"  ({time.time()-t0:.0f}s)")

    # viz: 6 classes
    show = [names.index(n) for n in ["sphere", "cube", "torus", "Lshape", "cone", "rod"]]
    with torch.no_grad():
        idx = [int((Lte == s).nonzero()[0]) for s in show]
        rows = [(Xte[idx], "input"), (Mte[idx], "CLEAN GT"),
                (sil_sh[idx], "shared-basis render"), (sil_mc[idx], "per-class render")]
    fig, ax = plt.subplots(4, 6, figsize=(9, 6))
    for r, (A, lab) in enumerate(rows):
        for cc in range(6): ax[r, cc].imshow(A[cc].numpy(), cmap="gray", vmin=0, vmax=1); ax[r, cc].axis("off")
        ax[r, 0].set_title(lab, fontsize=8, loc="left")
    fig.suptitle(f"Multi-class 3D-RCNN ({K} classes): shared basis (ghosts) vs per-class fixed shapes")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "r3dcnn_multiclass.png", dpi=110); plt.close(fig)
    print("  wrote docs/r3dcnn_multiclass.png")


if __name__ == "__main__":
    main()
