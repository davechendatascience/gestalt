"""Viewpoint extrapolation on MNIST: vanilla CNN vs an STN+invariance "gestalt"
module. Viewpoint = in-plane rotation (2D proxy for 3D camera viewpoint).

Protocol: all trainable models see rotations only within +-TRAIN_RANGE during
training. We then test at a sweep of FIXED rotation magnitudes, including angles
well beyond the training range (extrapolation). The thesis: a model that infers
the transform (STN) and is trained for cross-view invariance degrades more
gracefully on unseen viewpoints than a vanilla CNN that just saw augmentation.

Models:
  cnn_plain : vanilla CNN, NO rotation augmentation (shows the raw fragility)
  cnn_aug   : vanilla CNN, rotation augmentation in +-TRAIN_RANGE (fair baseline)
  gestalt   : STN canonicaliser + encoder, same augmentation + multi-view
              invariance loss

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_viewpoint_mnist.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from gestalt.models import VanillaCNN, GestaltNet, rotate_batch

DATA = ROOT / "data" / "gestalt"
MN = DATA / "mnist.npz"
TRAIN_RANGE = 30.0          # train rotations in +-30 deg
TEST_ANGLES = [0, 15, 30, 45, 60, 75, 90]
torch.manual_seed(0)


def load(n_train, n_test, seed=0):
    if MN.exists():
        d = np.load(MN); X, y = d["X"], d["y"].astype(int)
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="liac-arff")
        y = y.astype(int); DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MN, X=X.astype(np.float32), y=y)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
    f = lambda a: torch.tensor((a / 255.0).reshape(-1, 1, 28, 28), dtype=torch.float32)
    return f(X[tr]), torch.tensor(y[tr]), f(X[te]), torch.tensor(y[te])


def rand_angles(n, rng):
    return torch.tensor(rng.uniform(-TRAIN_RANGE, TRAIN_RANGE, n), dtype=torch.float32)


def train(model, Xtr, ytr, epochs, augment, invariance, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = len(ytr), 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb = Xtr[idx]; yb = ytr[idx]
            if augment:
                xb1 = rotate_batch(xb, rand_angles(len(idx), rng))
            else:
                xb1 = xb
            opt.zero_grad()
            if invariance:
                xb2 = rotate_batch(xb, rand_angles(len(idx), rng))
                o1, h1, _ = model(xb1); o2, h2, _ = model(xb2)
                loss = (F.cross_entropy(o1, yb) + F.cross_entropy(o2, yb)) / 2
                loss = loss + 1.0 * F.mse_loss(h1, h2)          # cross-view invariance
            else:
                out = model(xb1); out = out[0]
                loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_at_angles(model, Xte, yte):
    model.eval(); accs = {}
    for ang in TEST_ANGLES:
        xb = rotate_batch(Xte, torch.full((len(Xte),), float(ang)))
        out = model(xb); out = out[0] if isinstance(out, tuple) else out
        accs[ang] = float((out.argmax(1) == yte).float().mean())
    return accs


def main(n_train=8000, n_test=2000, epochs=10):
    Xtr, ytr, Xte, yte = load(n_train, n_test)
    print("=== viewpoint (rotation) extrapolation: vanilla CNN vs gestalt STN+invariance ===")
    print(f"train rotations +-{TRAIN_RANGE:.0f} deg; test at {TEST_ANGLES} deg "
          f"(>{TRAIN_RANGE:.0f} = extrapolation)\n")

    specs = [
        ("cnn_plain", VanillaCNN(), dict(augment=False, invariance=False)),
        ("cnn_aug",   VanillaCNN(), dict(augment=True,  invariance=False)),
        ("gestalt",   GestaltNet(), dict(augment=True,  invariance=True)),
    ]
    results = {}
    for name, model, kw in specs:
        t0 = time.time()
        train(model, Xtr, ytr, epochs, **kw)
        results[name] = eval_at_angles(model, Xte, yte)
        print(f"  trained {name:9s} ({time.time()-t0:.0f}s)")

    print("\nangle:     " + "".join(f"{a:>7d}" for a in TEST_ANGLES))
    for name in results:
        row = "".join(f"{results[name][a]:7.3f}" for a in TEST_ANGLES)
        print(f"  {name:9s}" + row)

    inr = [a for a in TEST_ANGLES if a <= TRAIN_RANGE]
    ood = [a for a in TEST_ANGLES if a > TRAIN_RANGE]
    print("\nmean in-range vs extrapolation:")
    for name in results:
        mi = np.mean([results[name][a] for a in inr])
        mo = np.mean([results[name][a] for a in ood])
        print(f"  {name:9s} in-range={mi:.3f}  extrapolation={mo:.3f}  drop={mi-mo:+.3f}")
    print("\nReading: all three should be close in-range; the question is who holds up")
    print("on extrapolated viewpoints. STN canonicalisation + cross-view invariance is")
    print("the 2D proxy of equivariant-rendering; the 3D version trains it on multi-view")
    print("renders (Isaac) with a latent that transforms like the scene.")


if __name__ == "__main__":
    main()
