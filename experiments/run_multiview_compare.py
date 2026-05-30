"""Does MORE INPUT information help recognition? Single view (lossy) vs K-view
fusion, with and without the equivariant 3D structure.

  single : 1 view -> classify          (the lossy baseline; what the CNN had)
  pool   : K views -> mean-pool -> classify        (more info, no 3D)
  fuse3d : K views -> rotate each to canonical by its pose -> aggregate -> classify
           (more info + equivariant 3D fusion: stitch lossy views into one shape)

Regimes: full (train/test span SO(3)) and extrap (train poses <=60 deg, test
90-180). Test = classification over held-out poses.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_multiview_compare.py
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
from gestalt.ae3d import MVNet

H, NTR, NTE, K = 48, 20, 20, 4
torch.manual_seed(0)


def build(ptr, pte, seed=0):
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(seed)
    Xtr = np.zeros((O, NTR, H, H), np.float32); Rtr = np.zeros((O, NTR, 3, 3), np.float32)
    Xte = np.zeros((O, NTE, H, H), np.float32); Rte = np.zeros((O, NTE, 3, 3), np.float32)
    for o, name in enumerate(names):
        pts, nrm = lib[name]
        for v in range(NTR):
            R = ptr(rng); Xtr[o, v] = render_camera(pts, nrm, R, 4, 2, (0, 0), H, False); Rtr[o, v] = R
        for v in range(NTE):
            R = pte(rng); Xte[o, v] = render_camera(pts, nrm, R, 4, 2, (0, 0), H, False); Rte[o, v] = R
    return tuple(torch.tensor(a) for a in (Xtr, Rtr, Xte, Rte)) + (O,)


def fit(mode, Xtr, Rtr, O, steps=500, seed=0):
    torch.manual_seed(seed)
    k = 1 if mode == "single" else K
    net = MVNet(O, mode, 8, 16, H)
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); bs = 16
    for s in range(steps):
        o = rng.integers(0, O, bs)
        idx = rng.integers(0, NTR, (bs, k))
        xs = Xtr[o[:, None], idx].unsqueeze(2)              # (bs,k,1,H,W)
        Rs = Rtr[o[:, None], idx]                           # (bs,k,3,3)
        logits = net(xs, Rs)
        loss = F.cross_entropy(logits, torch.tensor(o))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def test_acc(net, mode, Xte, Rte, O, groups=10, seed=1):
    k = 1 if mode == "single" else K
    rng = np.random.default_rng(seed)
    correct = tot = 0
    for o in range(O):
        for _ in range(groups):
            idx = rng.integers(0, NTE, k)
            xs = Xte[o, idx].unsqueeze(0).unsqueeze(2)      # (1,k,1,H,W)
            Rs = Rte[o, idx].unsqueeze(0)
            correct += int(net(xs, Rs).argmax(1).item() == o); tot += 1
    return correct / tot


def main():
    regimes = {
        "full  ": (lambda r: rand_rotation(r), lambda r: rand_rotation(r)),
        "extrap": (lambda r: rand_rotation_cone(r, 0, 60), lambda r: rand_rotation_cone(r, 90, 180)),
    }
    print(f"=== more INPUT info? single vs {K}-view pool vs {K}-view 3D-fusion (14-class) ===\n")
    print(f"{'regime':8} | {'single':>7} | {'pool':>6} | {'fuse3d':>7}")
    print("-" * 40)
    for rn, (ptr, pte) in regimes.items():
        Xtr, Rtr, Xte, Rte, O = build(ptr, pte)
        t0 = time.time()
        accs = {}
        for mode in ["single", "pool", "fuse3d"]:
            accs[mode] = test_acc(fit(mode, Xtr, Rtr, O), mode, Xte, Rte, O)
        print(f"{rn:8} | {accs['single']:>7.3f} | {accs['pool']:>6.3f} | {accs['fuse3d']:>7.3f}"
              f"    ({time.time()-t0:.0f}s)")
    print(f"\nchance={1/O:.3f}. single=1 lossy view; pool/fuse3d={K} views (more info).")
    print("If pool/fuse3d > single, more input information helps recognition (your point).")
    print("If fuse3d > pool, the equivariant 3D fusion beats naive averaging.")


if __name__ == "__main__":
    main()
