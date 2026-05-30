"""Amortised generative classifier: 14 LEARNED object latents + pose-aware
matchability, image in, top score out -- no stored pose bank.

  LatentMatch : score(x,o) = max_g <phi(x), rho(g) z_o>, z_o learned, end-to-end.
  CNN         : vanilla classifier (same encoder backbone), for reference.

Single-view ortho SO(3); test on held-out poses. The latent matcher replaces the
80-image bank with one learned volume per object and folds the pose-max into a
(sampled) group correlation.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_latent_classifier.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from gestalt.render3d import library, render_camera, rand_rotation, rand_rotation_cone
from gestalt.models import VanillaCNN
from gestalt.ae3d import LatentMatchNet

H, NTR, NTE = 48, 40, 20
torch.manual_seed(0)


def build(ptr, pte, seed=0):
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(seed)
    Xtr = np.zeros((O, NTR, H, H), np.float32); Xte = np.zeros((O, NTE, H, H), np.float32)
    for o, nm in enumerate(names):
        pts, nrm = lib[nm]
        for v in range(NTR):
            Xtr[o, v] = render_camera(pts, nrm, ptr(rng), 4, 2, (0, 0), H, False)
        for v in range(NTE):
            Xte[o, v] = render_camera(pts, nrm, pte(rng), 4, 2, (0, 0), H, False)
    return torch.tensor(Xtr), torch.tensor(Xte), O


def fit(net, X, y, epochs, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    n, bs = len(y), 32
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = net(X[idx]); out = out[0] if isinstance(out, tuple) else out
            opt.zero_grad(); F.cross_entropy(out, y[idx]).backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def acc(net, Xte, O):
    X = Xte.reshape(-1, 1, H, H); y = torch.repeat_interleave(torch.arange(O), NTE)
    out = net(X); out = out[0] if isinstance(out, tuple) else out
    return float((out.argmax(1) == y).float().mean())


def main():
    regimes = {
        "full  ": (lambda r: rand_rotation(r), lambda r: rand_rotation(r)),
        "extrap": (lambda r: rand_rotation_cone(r, 0, 60), lambda r: rand_rotation_cone(r, 90, 180)),
    }
    print("=== learned-latent pose-aware matcher vs CNN (single view, 14 objects) ===")
    print(f"{'regime':8} | {'CNN':>6} | {'LatentMatch':>12} | {'delta':>6}")
    print("-" * 42)
    for rn, (ptr, pte) in regimes.items():
        Xtr, Xte, O = build(ptr, pte)
        y = torch.repeat_interleave(torch.arange(O), NTR); Xf = Xtr.reshape(-1, 1, H, H)
        a_cnn = acc(fit(VanillaCNN(n_classes=O), Xf, y, 60), Xte, O)
        a_lm = acc(fit(LatentMatchNet(O, C=4, D=10, M=12, H=H), Xf, y, 60), Xte, O)
        print(f"{rn:8} | {a_cnn:>6.3f} | {a_lm:>12.3f} | {a_lm-a_cnn:>+6.3f}")
    print(f"\nchance={1/O:.3f}. LatentMatch = 14 learned object volumes (not an image bank),")
    print("matchability = pose-max correlation. Watch 'extrap' -- the structure should pay")
    print("on unseen poses the CNN can't memorise.")


if __name__ == "__main__":
    main()
