"""Verify the 3D multi-view renders are geometrically correct.

Two checks:
  (1) VISUAL: azimuth sweep (el=0) of ASYMMETRIC objects. A correct camera orbit
      must rotate them predictably (slab gets wider/narrower; the L-shape spins;
      the cube shows 1 face head-on, 2-3 faces obliquely). A wrong az/el/flip
      convention shows up instantly here. -> docs/multiview3d_verify.png

  (2) NUMERICAL: (a) cube head-on (0,0) is a square; (b) the stored poses in the
      .npz equal rotation(az,el); (c) projecting the 3D points with the stored
      pose lands inside the rendered silhouette (poses <-> images are consistent,
      which downstream factorization relies on); (d) the orbit is smooth and has
      the symmetry it should.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_verify_multiview.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.render3d import library, render, rotation
from gestalt.synth import load_dataset  # not used; kept import surface minimal

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
MV = ROOT / "data" / "multiview3d"
H, SCALE = 64, 0.40


def silhouette(img):
    return img > 0.01


def project_visible(pts, nrm, R):
    P = pts @ R.T
    Ncz = (nrm @ R.T)[:, 2]
    vis = Ncz > 0.02
    u = np.clip(((P[vis, 0] * SCALE + 0.5) * H).astype(int), 0, H - 1)
    v = np.clip(((0.5 - P[vis, 1] * SCALE) * H).astype(int), 0, H - 1)
    return u, v


def main():
    lib = library()
    diag = ["slab", "Lshape", "cube", "ellipsoid_tall", "cone", "disc"]

    # (1) visual diagnostic sheet
    fig, axes = plt.subplots(len(diag), len(AZ), figsize=(len(AZ) * 1.1, len(diag) * 1.1))
    for r, name in enumerate(diag):
        pts, nrm = lib[name]
        for c, az in enumerate(AZ):
            axes[r, c].imshow(render(pts, nrm, rotation(az, 0)), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"az {az}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("azimuth sweep (el=0) of asymmetric objects -- must rotate predictably", fontsize=10)
    fig.tight_layout()
    out = ROOT / "docs" / "multiview3d_verify.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}\n")

    # (2) numerical checks
    print("--- numerical sanity ---")
    # (a) cube head-on is a square (single face, aspect ~1)
    pts, nrm = lib["cube"]
    img = render(pts, nrm, rotation(0, 0))
    ys, xs = np.nonzero(silhouette(img))
    ar = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
    print(f"cube @(0,0) aspect ratio = {ar:.3f}  (expect ~1.00 -> a square)")
    img35 = render(pts, nrm, rotation(35, 20))
    print(f"cube @(0,0) distinct shades = {np.unique(np.round(img[img>0.01],2)).size} "
          f"(flat, 1 face);  @(35,20) = {np.unique(np.round(img35[img35>0.01],2)).size} (3 faces)")

    # (b) stored npz poses equal rotation(az,el); (c) reprojection containment
    print("\n--- stored-pose integrity + reprojection containment ---")
    for d in sorted(MV.glob("obj_0*"))[:4]:
        z = np.load(d / "views.npz")
        name = str(z["name"]); pts, nrm = lib[name]
        pose_ok = all(np.allclose(z["R"][k], rotation(z["az"][k], z["el"][k]))
                      for k in range(len(z["az"])))
        cont = []
        for k in range(len(z["az"])):
            u, v = project_visible(pts, nrm, z["R"][k])
            inside = silhouette(z["imgs"][k])[v, u]
            cont.append(inside.mean())
        print(f"  {d.name:22s} poses==rotation(az,el): {pose_ok}   "
              f"reproj-in-silhouette: {np.mean(cont):.3f}")

    # (d) orbit symmetry/smoothness on a chiral object (Lshape)
    pts, nrm = lib["Lshape"]
    orbit = np.stack([render(pts, nrm, rotation(az, 0)) for az in range(0, 360, 15)])
    nxt = np.array([np.corrcoef(orbit[i].ravel(), orbit[(i + 1) % len(orbit)].ravel())[0, 1]
                    for i in range(len(orbit))])
    half = np.corrcoef(orbit[0].ravel(), orbit[len(orbit)//2].ravel())[0, 1]
    print(f"\nLshape orbit: consecutive-view corr min={nxt.min():.3f} (smooth, no jumps); "
          f"\n  corr(0deg, 180deg)={half:.3f} (low => 180deg-distinct, i.e. genuinely chiral/asymmetric)")


if __name__ == "__main__":
    main()
