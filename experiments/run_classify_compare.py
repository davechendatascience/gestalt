"""14-class object classification under random 3D rotation: equivariant-AE code
vs a vanilla CNN, swept over training-set size.

Random orthographic SO(3) views of the 14 objects (this isolates the rotation
DOF the AE's depth-collapse decoder models exactly; scale/pan/perspective need a
projective decoder -- a documented extension). Test poses are DISJOINT random
rotations, so accuracy measures generalisation to UNSEEN viewpoints.

  CNN     : VanillaCNN, supervised end-to-end on (image, class).
  Equiv   : train the equivariant AE render-and-compare on (image, POSE) -- no
            class labels -- freeze, then nearest-centroid on the frozen code.

The equivariant side gets the free POSE labels a sim provides; the CNN gets
class labels + the same images. The question: does the 3D inductive bias win,
especially with FEW views (where memorising augmentation can't)?

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_classify_compare.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from gestalt.render3d import library, render_camera, rand_rotation
from gestalt.models import VanillaCNN
from gestalt.ae3d import EquivAE

H, POOL, N_TE = 48, 64, 24
TR_SIZES = [4, 12, 40]
torch.manual_seed(0)


def build_pool():
    lib = library(); names = list(lib); O = len(names)
    imgs = np.zeros((O, POOL, H, H), np.float32); Rs = np.zeros((O, POOL, 3, 3), np.float32)
    rng = np.random.default_rng(0)
    for o, name in enumerate(names):
        pts, nrm = lib[name]
        for v in range(POOL):
            R = rand_rotation(rng)
            imgs[o, v] = render_camera(pts, nrm, R, 4.0, 2.0, (0, 0), H, perspective=False)
            Rs[o, v] = R
    return torch.tensor(imgs), torch.tensor(Rs)


def train_cnn(imgs, labels, epochs=40, seed=0):
    torch.manual_seed(seed)
    net = VanillaCNN(n_classes=imgs.shape[0] if imgs.dim() == 4 else 14)
    return net


def fit_cnn(X, y, n_classes, epochs=60, seed=0):
    torch.manual_seed(seed)
    net = VanillaCNN(n_classes=n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = len(y), 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(net(X[idx])[0], y[idx])
            loss.backward(); opt.step()
    net.eval()
    return net


def fit_ae(X, R, steps=700, seed=0):
    torch.manual_seed(seed)
    ae = EquivAE(8, 16, H)
    opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
    n, bs = len(X), 24
    rng = np.random.default_rng(seed)
    for s in range(steps):
        i = torch.tensor(rng.integers(0, n, bs)); j = torch.tensor(rng.integers(0, n, bs))
        z = ae.encode(X[i])
        loss = F.mse_loss(ae.render(z, R[i]), X[i]) + F.mse_loss(ae.render(z, R[j]), X[j])
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    return ae


@torch.no_grad()
def ae_centroid_acc(ae, Xtr, ytr, Xte, yte, O):
    ctr = ae.code(Xtr).numpy(); cte = ae.code(Xte).numpy()
    mu = np.concatenate([ctr, cte]).mean(0, keepdims=True)
    ctr, cte = ctr - mu, cte - mu
    ctr /= np.linalg.norm(ctr, axis=1, keepdims=True) + 1e-9
    cte /= np.linalg.norm(cte, axis=1, keepdims=True) + 1e-9
    cent = np.stack([ctr[ytr.numpy() == o].mean(0) for o in range(O)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    pred = (cte @ cent.T).argmax(1)
    return float((pred == yte.numpy()).mean())


def main():
    imgs, Rs = build_pool()
    O = imgs.shape[0]
    # held-out test poses = last N_TE views; train pool = first POOL-N_TE
    teI = imgs[:, -N_TE:].reshape(-1, 1, H, H)
    teY = torch.repeat_interleave(torch.arange(O), N_TE)
    print(f"=== 14-class object classification under random SO(3) (test = {N_TE} unseen poses/class) ===")
    print(f"{'train views/cls':>16} | {'CNN acc':>8} | {'Equiv acc':>10}")
    print("-" * 42)
    for ntr in TR_SIZES:
        sel = imgs[:, :ntr]
        Xtr = sel.reshape(-1, 1, H, H); yTr = torch.repeat_interleave(torch.arange(O), ntr)
        Rtr = Rs[:, :ntr].reshape(-1, 3, 3)

        t0 = time.time()
        cnn = fit_cnn(Xtr, yTr, O)
        cnn_acc = float((cnn(teI)[0].argmax(1) == teY).float().mean())

        ae = fit_ae(Xtr, Rtr)
        equiv_acc = ae_centroid_acc(ae, Xtr, yTr, teI, teY, O)
        print(f"{ntr:>16} | {cnn_acc:>8.3f} | {equiv_acc:>10.3f}    ({time.time()-t0:.0f}s)")

    print(f"\nchance = {1/O:.3f}.  Reading: the equivariant code bakes rotation-")
    print("invariance into the architecture (and uses sim's free pose labels); the CNN")
    print("must learn it from the views it sees. Watch the low-data column.")


if __name__ == "__main__":
    main()
