"""Visualise the geometry the model infers, against its input.

Two views of "rendering the learned geometry through inference":

  (1) STN canonicalisation (the viewpoint module): the same digit at several
      rotations is pushed through the learned canonicaliser. If the model has
      encoded viewpoint-invariant geometry, the canonicalised images across
      angles should COLLAPSE TO A COMMON POSE -> invariance made visible.
      -> docs/viz_canonicalization.png

  (2) Analysis-by-synthesis reconstruction: for each input we render the MAP
      explanation g(z*) = the best-aligned prototype. Input vs synthesis shows
      what geometry the energy model "imagined" to explain the pixels.
      -> docs/viz_absyn_reconstruction.png

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_visualize_geometry.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gestalt.models import GestaltNet, rotate_batch
from gestalt.inference import aligned_energy
from run_viewpoint_mnist import load, train
from run_mnist_absyn import build_prototypes

DOCS = ROOT / "docs"
ANGLES = [0, 30, 60, 90]


def pick_one_per_class(X, y, k=6):
    idx = []
    for c in range(k):
        idx.append(int(np.where(y.numpy() == c)[0][0]))
    return X[idx], y[idx]


def viz_canonicalization(model, Xs):
    model.eval()
    n = len(Xs)
    cols = len(ANGLES) * 2
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.0, n * 1.0))
    for r in range(n):
        for ci, ang in enumerate(ANGLES):
            xb = rotate_batch(Xs[r:r + 1], torch.tensor([float(ang)]))
            with torch.no_grad():
                xc, _ = model.stn(xb)
            a_in = axes[r, ci * 2]; a_out = axes[r, ci * 2 + 1]
            a_in.imshow(xb[0, 0].numpy(), cmap="gray"); a_in.axis("off")
            a_out.imshow(xc[0, 0].numpy(), cmap="gray"); a_out.axis("off")
            if r == 0:
                a_in.set_title(f"in {ang}", fontsize=8)
                a_out.set_title("canon", fontsize=8)
    fig.suptitle("STN canonicalisation: rotated input (in) -> learned canonical pose (canon)."
                 "\nIf geometry is encoded viewpoint-invariantly, the 'canon' columns match across angles.",
                 fontsize=9)
    fig.tight_layout()
    out = DOCS / "viz_canonicalization.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def viz_absyn(protos, Xs):
    n = len(Xs)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.1, 2.4))
    for j in range(n):
        obs = Xs[j, 0].numpy()
        best_e, best = np.inf, None
        for p in protos:
            e, dy, dx = aligned_energy(obs, p, max_shift=3)
            if e < best_e:
                best_e, best = e, np.roll(np.roll(p, dy, 0), dx, 1)
        axes[0, j].imshow(obs, cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(best, cmap="gray"); axes[1, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("input"); axes[1, j].set_ylabel("synthesis")
    axes[0, 0].set_title("input", fontsize=9, loc="left")
    axes[1, 0].set_title("g(z*) rendered explanation", fontsize=9, loc="left")
    fig.suptitle("Analysis-by-synthesis: input vs the geometry the energy model rendered to explain it",
                 fontsize=9)
    fig.tight_layout()
    out = DOCS / "viz_absyn_reconstruction.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    Xtr, ytr, Xte, yte = load(8000, 2000)
    print("training GestaltNet for visualisation ...")
    model = train(GestaltNet(), Xtr, ytr, epochs=8, augment=True, invariance=True)
    Xs, ys = pick_one_per_class(Xte, yte, k=6)
    p1 = viz_canonicalization(model, Xs)
    print(f"  wrote {p1.relative_to(ROOT)}")

    protos, _ = build_prototypes(Xtr, ytr, k=16)
    Xs2, _ = pick_one_per_class(Xte, yte, k=8)
    p2 = viz_absyn(protos, Xs2)
    print(f"  wrote {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
