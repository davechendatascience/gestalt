"""Does equipping the recogniser with the EXTRA information (pose + 3D structure)
help recognition? Same CNN encoder, ablate the 3D render-and-compare auxiliary.

  cnn      : classification only (CE).            (the bare view->identity learner)
  cnn+3d   : CE + render-and-compare auxiliary.   (same encoder, ALSO told to
             explain the 3D from a rotated voxel -- uses the pose labels sim gives)

Two regimes:
  full     : train & test poses both span random SO(3) (CNN can memorise angles).
  extrap   : train poses in a CONE (<=60 deg), test poses FAR outside (90-180 deg).
             A single view is lossy and unseen poses can't be memorised -- the
             regime where 3D structure should pay.

Test = classification accuracy on held-out poses (encode image -> logits; no pose
used at test time).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_multitask_compare.py
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
from gestalt.ae3d import MultiTaskNet

H, NTR, NTE = 48, 16, 24
torch.manual_seed(0)


def build(ptr, pte, seed):
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(seed)
    Xtr = np.zeros((O, NTR, H, H), np.float32); Rtr = np.zeros((O, NTR, 3, 3), np.float32)
    Xte = np.zeros((O, NTE, H, H), np.float32)
    for o, name in enumerate(names):
        pts, nrm = lib[name]
        for v in range(NTR):
            R = ptr(rng); Xtr[o, v] = render_camera(pts, nrm, R, 4, 2, (0, 0), H, False); Rtr[o, v] = R
        for v in range(NTE):
            R = pte(rng); Xte[o, v] = render_camera(pts, nrm, R, 4, 2, (0, 0), H, False)
    return (torch.tensor(Xtr), torch.tensor(Rtr), torch.tensor(Xte), O)


def fit(Xtr, Rtr, O, lam, steps=700, seed=0):
    torch.manual_seed(seed)
    net = MultiTaskNet(O, 8, 16, H)
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); bs = 24
    for s in range(steps):
        o = torch.tensor(rng.integers(0, O, bs))
        i = torch.tensor(rng.integers(0, NTR, bs)); j = torch.tensor(rng.integers(0, NTR, bs))
        xi = Xtr[o, i].unsqueeze(1); xj = Xtr[o, j].unsqueeze(1)
        li, zi = net(xi); lj, _ = net(xj)
        loss = F.cross_entropy(li, o) + F.cross_entropy(lj, o)
        if lam > 0:
            loss = loss + lam * (F.mse_loss(net.render(zi, Rtr[o, i]), xi)
                                 + F.mse_loss(net.render(zi, Rtr[o, j]), xj))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def test_acc(net, Xte, O):
    X = Xte.reshape(-1, 1, H, H); y = torch.repeat_interleave(torch.arange(O), NTE)
    return float((net(X)[0].argmax(1) == y).float().mean())


def main():
    regimes = {
        "full   (train SO3 / test SO3)": (lambda r: rand_rotation(r), lambda r: rand_rotation(r)),
        "extrap (train<=60 / test 90-180)": (lambda r: rand_rotation_cone(r, 0, 60),
                                             lambda r: rand_rotation_cone(r, 90, 180)),
    }
    print("=== does the 3D auxiliary help view->identity? (14-class, held-out poses) ===")
    print(f"{'regime':36} | {'cnn':>6} | {'cnn+3d':>7} | {'delta':>6}")
    print("-" * 64)
    for name, (ptr, pte) in regimes.items():
        Xtr, Rtr, Xte, O = build(ptr, pte, seed=0)
        t0 = time.time()
        a_cnn = test_acc(fit(Xtr, Rtr, O, lam=0.0), Xte, O)
        a_aux = test_acc(fit(Xtr, Rtr, O, lam=1.0), Xte, O)
        print(f"{name:36} | {a_cnn:>6.3f} | {a_aux:>7.3f} | {a_aux-a_cnn:>+6.3f}   ({time.time()-t0:.0f}s)")
    print(f"\nchance = {1/O:.3f}.  Same encoder/capacity/labels; the only difference is")
    print("whether it is ALSO told to explain the 3D (pose + render-and-compare). Positive")
    print("delta = the extra information helps recognition -- most expected in 'extrap'.")


if __name__ == "__main__":
    main()
