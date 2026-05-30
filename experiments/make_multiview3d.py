"""Materialise the 3D multi-view dataset spanning ALL generative DOF.

Per view DOF: orientation ~ SO(3) (3, incl. roll), distance ~[3,7] (1, scale +
perspective), image pan (2), pinhole PERSPECTIVE projection. Same object_id ==
same object, many views, full poses stored.

  data/multiview3d/obj_<id>_<name>/views.npz  {imgs (K,H,W), R, dist, pan, f, name}
  data/multiview3d/obj_<id>_<name>/contact.png
  docs/multiview3d_objects.png   library at a canonical view
  docs/multiview3d_dof.png       single-DOF showcase (roll / distance / pan / random)

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/make_multiview3d.py
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.render3d import library, sample_views, render_camera, rotation, render

OUT = ROOT / "data" / "multiview3d"
H, K = 64, 36


def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def sheet(imgs, path, titles=None, ncol=9):
    n = len(imgs); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow * 1.05))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i], fontsize=6)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lib = library()
    t0 = time.time(); manifest = []
    for i, (name, po) in enumerate(lib.items()):
        imgs, pose = sample_views(po, K, np.random.default_rng(100 + i), H)
        d = OUT / f"obj_{i:03d}_{name}"; d.mkdir(exist_ok=True)
        np.savez_compressed(d / "views.npz", imgs=imgs, name=name, **pose)
        sheet(imgs, d / "contact.png")
        manifest.append({"id": i, "name": name, "n_views": K})
        print(f"  obj_{i:03d}_{name}: {K} full-DOF views")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # library at a canonical view (orthographic, clean)
    Rc = rotation(35, 20); names = list(lib)
    fig, axes = plt.subplots(2, 7, figsize=(11, 3.4))
    for ax, (nm, po) in zip(np.array(axes).ravel(), lib.items()):
        ax.imshow(render(po[0], po[1], Rc, H), cmap="gray", vmin=0, vmax=1)
        ax.set_title(nm, fontsize=7); ax.axis("off")
    fig.suptitle("multiview3d library (canonical view)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "multiview3d_objects.png", dpi=100)
    plt.close(fig)

    # single-DOF showcase on the cube (each row isolates one DOF)
    pts, nrm = lib["cube"]; base = rotation(35, 20)
    rows, titles = [], []
    for r in [0, 20, 40, 60, 80, 100, 120, 140, 160]:                 # ROLL
        rows.append(render_camera(pts, nrm, _Rz(r) @ base, 4.5, 2.0, (0, 0), H))
        titles.append(f"roll {r}")
    for dval in np.linspace(3, 7.5, 9):                               # DISTANCE (scale+persp)
        rows.append(render_camera(pts, nrm, base, dval, 2.0, (0, 0), H))
        titles.append(f"dist {dval:.1f}")
    for px in np.linspace(-0.22, 0.22, 9):                            # PAN
        rows.append(render_camera(pts, nrm, base, 4.5, 2.0, (px, 0.08), H))
        titles.append(f"pan {px:+.2f}")
    rimg, _ = sample_views((pts, nrm), 9, np.random.default_rng(0), H)   # all DOF random
    for k in range(9):
        rows.append(rimg[k]); titles.append("random")
    sheet(np.stack(rows), ROOT / "docs" / "multiview3d_dof.png", titles, ncol=9)

    print(f"\n{len(lib)} objects x {K} full-DOF views -> {OUT.relative_to(ROOT)}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
