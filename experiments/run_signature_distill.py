"""Single-view recognition by DISTILLING the rotation-invariant 3D signature.

The analytic D2 signature is a POSE-FREE target (same for every view of an
object, 100% separable in 3D). We train a single-view CNN two ways, same encoder:

  classifier : view -> class  (cross-entropy; learn invariance from labels)
  distill    : view -> the object's invariant SIGNATURE (MSE); classify a test
               view by nearest signature. The target itself is pose-invariant and
               geometry-grounded, so the net is explicitly pulled toward a
               canonical, shape-structured code -- richer supervision than a
               one-hot label.

Single view -> full-shape signature is ill-posed (hidden back), so the net learns
the best guess; the question is whether that geometry-grounded target helps
single-view identity, especially under pose EXTRAPOLATION.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_signature_distill.py
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
from gestalt.invariants import d2_descriptor
from gestalt.models import VanillaCNN

H, NTR, NTE = 48, 20, 20
torch.manual_seed(0)


def build(ptr, pte, seed=0):
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(seed)
    sig = np.stack([d2_descriptor(lib[nm][0], rng) for nm in names]).astype(np.float32)  # (O,48)
    Xtr = np.zeros((O, NTR, H, H), np.float32); Xte = np.zeros((O, NTE, H, H), np.float32)
    for o, nm in enumerate(names):
        pts, nrm = lib[nm]
        for v in range(NTR):
            Xtr[o, v] = render_camera(pts, nrm, ptr(rng), 4, 2, (0, 0), H, False)
        for v in range(NTE):
            Xte[o, v] = render_camera(pts, nrm, pte(rng), 4, 2, (0, 0), H, False)
    return torch.tensor(Xtr), torch.tensor(Xte), torch.tensor(sig), O


def fit(X, target, out_dim, loss_kind, epochs=70, seed=0):
    torch.manual_seed(seed)
    net = VanillaCNN(n_classes=out_dim)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = len(target), 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = net(X[idx])[0]
            loss = F.cross_entropy(pred, target[idx]) if loss_kind == "ce" \
                else F.mse_loss(pred, target[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def acc_classifier(net, Xte, O):
    X = Xte.reshape(-1, 1, H, H); y = torch.repeat_interleave(torch.arange(O), NTE)
    return float((net(X)[0].argmax(1) == y).float().mean())


@torch.no_grad()
def acc_distill(net, Xte, sig, O):
    X = Xte.reshape(-1, 1, H, H); y = torch.repeat_interleave(torch.arange(O), NTE)
    pred = F.normalize(net(X)[0], dim=1)
    ref = F.normalize(sig, dim=1)
    return float(((pred @ ref.T).argmax(1) == y).float().mean())


def main():
    regimes = {
        "full  ": (lambda r: rand_rotation(r), lambda r: rand_rotation(r)),
        "extrap": (lambda r: rand_rotation_cone(r, 0, 60), lambda r: rand_rotation_cone(r, 90, 180)),
    }
    print("=== single-view recognition: CNN classifier vs invariant-signature distillation ===")
    print(f"{'regime':8} | {'CNN cls':>8} | {'distill':>8} | {'delta':>6}")
    print("-" * 42)
    for rn, (ptr, pte) in regimes.items():
        Xte_each = NTE
        Xtr, Xte, sig, O = build(ptr, pte)
        ytr = torch.repeat_interleave(torch.arange(O), NTR)
        Xtr_f = Xtr.reshape(-1, 1, H, H)
        sig_tr = sig[ytr]                                   # per-sample signature target
        t0 = time.time()
        cls = fit(Xtr_f, ytr, O, "ce")
        dst = fit(Xtr_f, sig_tr, sig.shape[1], "mse")
        a_cls = acc_classifier(cls, Xte, O)
        a_dst = acc_distill(dst, Xte, sig, O)
        print(f"{rn:8} | {a_cls:>8.3f} | {a_dst:>8.3f} | {a_dst-a_cls:>+6.3f}   ({time.time()-t0:.0f}s)")
    print(f"\nchance={1/O:.3f}; analytic 3D invariant (upper bound) = 1.000.")
    print("distill > classifier => predicting the pose-free geometry signature is")
    print("better single-view supervision than one-hot labels (esp. extrapolation).")


if __name__ == "__main__":
    main()
