"""Is the voxel-PCA shape representation the problem? Same 3D-RCNN pipeline on
(A) the 14 HETEROGENEOUS objects vs (B) a SMOOTH superquadric family.

If a smooth single-family shape set (where linear occupancy-PCA is a valid shape
space) trains markedly better, the heterogeneous cross-category occupancy PCA was
the culprit -- exactly why the paper uses a per-category deformable CAD basis.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_r3dcnn_repr.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from gestalt.render3d import library, rand_rotation
from gestalt.ae3d import rotate3d
from gestalt.r3dcnn import build_shape_basis, render_silhouette, quat_to_R, R3DCNN, voxelize

D, NC, H = 20, 8, 48
torch.manual_seed(0)


def superquadric(a, b, c, e1, e2, n=36):
    th = np.linspace(-np.pi / 2, np.pi / 2, n); ph = np.linspace(-np.pi, np.pi, 2 * n)
    TH, PH = np.meshgrid(th, ph)
    f = lambda w, e: np.sign(np.cos(w)) * np.abs(np.cos(w)) ** e
    g = lambda w, e: np.sign(np.sin(w)) * np.abs(np.sin(w)) ** e
    p = np.stack([a * f(TH, e1) * f(PH, e2), b * g(TH, e1), c * f(TH, e1) * g(PH, e2)], -1).reshape(-1, 3)
    return p / (np.linalg.norm(p, axis=1).max() + 1e-9)


def smooth_family(k=40, seed=3):
    rng = np.random.default_rng(seed)
    return [superquadric(*rng.uniform([.6, .6, .6, .3, .3], [1.2, 1.2, 1.2, 1.7, 1.7]))
            for _ in range(k)]


def variance_explained(points):
    occ = np.stack([voxelize(p, D).ravel() for p in points])
    s = np.linalg.svd(occ - occ.mean(0), compute_uv=False)
    return float((s[:NC] ** 2).sum() / (s ** 2).sum())


@torch.no_grad()
def gen(n, codes_t, mean_t, basis_t, seed):
    rng = np.random.default_rng(seed); o = rng.integers(0, len(codes_t), n)
    gt = codes_t[o]; q = F.normalize(torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32), 1)
    R = quat_to_R(q)
    occ = (mean_t + gt @ basis_t).clamp(0, 1).view(n, 1, D, D, D); occ_r = rotate3d(occ, R)
    w = torch.linspace(1, .35, D).view(1, 1, D, 1, 1)
    shaded = F.interpolate((occ_r * w).amax(2), (H, H), mode="bilinear", align_corners=False)[:, 0]
    sil = F.interpolate(1 - torch.prod(1 - occ_r, 2), (H, H), mode="bilinear", align_corners=False)[:, 0]
    mask = (sil > .3).float()
    tex = .6 + .4 * torch.tensor(rng.random((n, H, H)), dtype=torch.float32)
    img = (shaded * tex + torch.tensor(rng.normal(0, .04, (n, H, H)), dtype=torch.float32)).clamp(0, 1)
    return img, mask, gt, R


def run(points, label):
    mean, basis, codes = build_shape_basis(points, D, NC)
    mean_t, basis_t, codes_t = (torch.tensor(mean), torch.tensor(basis), torch.tensor(codes))
    ve = variance_explained(points)
    Xtr, Mtr, Ctr, Rtr = gen(900, codes_t, mean_t, basis_t, 0)
    Xte, Mte, Cte, Rte = gen(160, codes_t, mean_t, basis_t, 1)
    net = R3DCNN(NC, H); opt = torch.optim.Adam(net.parameters(), lr=1.5e-3)
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
        # normalised shape-code error (relative to code variance) -> comparable across sets
        ncode = (F.mse_loss(code, Cte) / Cte.var()).item()
    print(f"  {label:24s} | {len(points):3d} objs | basis var explained={ve:.3f} | "
          f"IoU={iou:.3f} | norm shape-code err={ncode:.3f}")


def main():
    print(f"=== is voxel-PCA the bottleneck? heterogeneous vs smooth shape family (NC={NC}) ===")
    print(f"  {'shape set':24s} |       | basis quality   | recon quality")
    t0 = time.time()
    run([library()[k][0] for k in library()], "heterogeneous (library)")
    run(smooth_family(40), "smooth superquadrics")
    print(f"\n({time.time()-t0:.0f}s) Higher basis-variance-explained + higher IoU + lower code")
    print("err for the smooth family => the linear occupancy-PCA basis is the issue across")
    print("heterogeneous objects; a per-category (smooth) shape basis -- as the paper uses -- fixes it.")


if __name__ == "__main__":
    main()
