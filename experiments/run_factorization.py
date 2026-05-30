"""Tomasi-Kanade factorization: the exact linear-algebra core of "compress the
3D construct, project to each view" -- no training.

We track P surface points of one object across F views (correspondence is free:
same point cloud). Under ORTHOGRAPHIC/affine projection the centred measurement
matrix W_hat (2F x P) has RANK 3, and SVD factors it:

    W_hat = M . S,   M (2F x 3) = the per-view projections,   S (3 x P) = the
                     shared 3D structure (the compressed construct).

This is exactly the network's bottleneck made linear: one structure S explains
every view; each view is a linear projection M_f of it. We then show:
  - the rank-3 singular-value cliff (and that PERSPECTIVE inflates it -> needs
    projective factorization);
  - reprojection error ~ 0;
  - S recovers the true 3D shape up to the AFFINE GAUGE (the invariance is
    "modulo a group", as discussed).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_factorization.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from gestalt.render3d import library, rand_rotation

F, P = 16, 200


def measurement_matrix(X, rng, perspective, dist=4.0, f=2.0):
    """Project P points X (P,3) into F views; return W (2F,P) and the cameras."""
    W = np.zeros((2 * F, X.shape[0]))
    for k in range(F):
        R = rand_rotation(rng); s = rng.uniform(0.8, 1.2)
        Xr = (X @ R.T)
        t = rng.uniform(-0.4, 0.4, 2)
        if perspective:
            depth = Xr[:, 2] + dist
            w = f * Xr[:, :2] / depth[:, None] + t
        else:
            w = s * Xr[:, :2] + t                  # scaled orthographic
        W[2 * k] = w[:, 0]; W[2 * k + 1] = w[:, 1]
    return W


def main():
    rng = np.random.default_rng(0)
    X = library()["Lshape"][0]
    X = X[rng.choice(len(X), P, replace=False)]
    X = X - X.mean(0)                               # centre the structure

    print("=== Tomasi-Kanade factorization: image = projection x structure ===")
    print(f"object=Lshape, F={F} views, P={P} tracked points\n")

    # ---- orthographic / affine: exact rank 3 ----
    W = measurement_matrix(X, np.random.default_rng(1), perspective=False)
    Wc = W - W.mean(1, keepdims=True)              # register: remove per-view translation
    sv = np.linalg.svd(Wc, compute_uv=False)
    print("ORTHOGRAPHIC centred measurement matrix W_hat (2F x P):")
    print("  top singular values: " + "  ".join(f"{s:.2e}" for s in sv[:6]))
    print(f"  energy in first 3 / total = {(sv[:3]**2).sum()/(sv**2).sum():.6f}  "
          f"(==1 -> exactly rank 3)")

    U, s, Vt = np.linalg.svd(Wc, full_matrices=False)
    M = U[:, :3] * s[:3]                            # (2F,3) per-view projections
    S = Vt[:3, :]                                   # (3,P) recovered structure
    rms = np.sqrt(np.mean((Wc - M @ S) ** 2))
    print(f"  rank-3 reprojection RMSE = {rms:.2e}  (structure x projection reproduces every view)")

    # recovered structure == true shape up to a 3x3 AFFINE gauge A (S ~ A X^T)
    A, *_ = np.linalg.lstsq(S.T, X, rcond=None)     # S^T A ~ X  ->  A maps S->X
    resid = np.sqrt(np.mean((S.T @ A - X) ** 2)) / (np.abs(X).mean())
    print(f"  S matches true 3D shape up to an affine map: rel-residual = {resid:.2e}")
    print("  (so the recovered 'construct' is the shape MODULO the affine gauge)\n")

    # ---- perspective: rank inflates ----
    Wp = measurement_matrix(X, np.random.default_rng(1), perspective=True)
    Wpc = Wp - Wp.mean(1, keepdims=True)
    svp = np.linalg.svd(Wpc, compute_uv=False)
    print("PERSPECTIVE centred measurement matrix:")
    print("  top singular values: " + "  ".join(f"{s:.2e}" for s in svp[:6]))
    print(f"  energy in first 3 / total = {(svp[:3]**2).sum()/(svp**2).sum():.6f}  "
          f"(<1 -> rank > 3; needs PROJECTIVE factorization / Sturm-Triggs)")

    print("\nReading: the rigid multi-view orbit IS low-rank bilinear "
          "(structure x projection).\nUnder ortho it is exactly rank 3 and SVD "
          "recovers both factors; perspective\ninflates the rank -- which is "
          "precisely why the neural decoder must divide by depth.")


if __name__ == "__main__":
    main()
