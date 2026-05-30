"""Materialise the 3D multi-view dataset.

For each object in the procedural library we render K = |az| x |el| viewpoints
and store them together under one object folder (same object_id == same object,
different views), with the camera poses:

  data/multiview3d/obj_<id>_<name>/views.npz   {imgs (K,H,W), az, el, R (K,3,3), name}
  data/multiview3d/obj_<id>_<name>/contact.png  contact sheet of the K views

Plus two tracked preview grids:
  docs/multiview3d_objects.png   every object at a canonical view (the library)
  docs/multiview3d_views.png     one object across all viewpoints (the orbit)

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

from gestalt.render3d import library, render_views, rotation, render

OUT = ROOT / "data" / "multiview3d"
H = 64
AZ = [0, 45, 90, 135, 180, 225, 270, 315]      # 8 azimuths
EL = [-30, 0, 30]                               # 3 elevations  -> 24 views/object


def contact_sheet(imgs, az, el, path, ncol=8):
    n = len(imgs); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{int(az[i])},{int(el[i])}", fontsize=6)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lib = library()
    t0 = time.time()
    manifest = []
    for i, (name, po) in enumerate(lib.items()):
        imgs, az, el, R = render_views(po, AZ, EL, H)
        d = OUT / f"obj_{i:03d}_{name}"
        d.mkdir(exist_ok=True)
        np.savez_compressed(d / "views.npz", imgs=imgs.astype(np.float32),
                            az=az, el=el, R=R, name=name)
        contact_sheet(imgs, az, el, d / "contact.png")
        manifest.append({"id": i, "name": name, "n_views": len(imgs),
                         "n_points": int(len(po[0]))})
        print(f"  obj_{i:03d}_{name}: {len(imgs)} views, {len(po[0])} pts")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # preview 1: the library at a canonical view (az=35, el=20)
    Rc = rotation(35, 20)
    names = list(lib)
    fig, axes = plt.subplots(2, 7, figsize=(11, 3.4))
    for ax, (name, po) in zip(np.array(axes).ravel(), lib.items()):
        ax.imshow(render(po[0], po[1], Rc, H), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=7); ax.axis("off")
    fig.suptitle("multiview3d asset library (canonical view)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "multiview3d_objects.png", dpi=100)
    plt.close(fig)

    # preview 2: one object across all viewpoints (the orbit)
    po = lib["dumbbell"]
    imgs, az, el, _ = render_views(po, AZ, EL, H)
    contact_sheet(imgs, az, el, ROOT / "docs" / "multiview3d_views.png")

    print(f"\n{len(lib)} objects x {len(AZ)*len(EL)} views -> {OUT.relative_to(ROOT)}"
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
