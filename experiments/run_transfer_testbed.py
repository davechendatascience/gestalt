"""Transfer testbed: does appearance-invariant STRUCTURE survive a sim->real
appearance shift that a region-local CNN does not?

Reproduces the sim2real collapse in miniature:
  - faithful geometry, two disjoint appearance regimes (sim, real);
  - CNN on RGB:        trained on sim -> high sim-val, (expected) low real-test;
  - descriptors + readout: appearance-blind shape features -> transfer.

North-star metric = the GAP (sim-val acc - real-test acc), plus absolute
real-test acc (the shape-only ceiling).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_transfer_testbed.py
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gestalt.synth import build, sim_regime, real_regime, CLASSES
from gestalt.descriptors import descriptor_matrix, FEATURE_NAMES


def acc(pred, y):
    return float((pred == y).mean())


# ---------------- CNN on RGB (region-local baseline) ----------------

def run_cnn(tr, va, te, size, epochs, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cpu"

    def to_x(imgs):
        return torch.tensor(imgs.transpose(0, 3, 1, 2), dtype=torch.float32)

    Xtr, ytr = to_x(tr[0]), torch.tensor(tr[2])
    Xva, yva = to_x(va[0]), va[2]
    Xte, yte = to_x(te[0]), te[2]

    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, len(CLASSES)),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(ytr); bs = 128
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pva = net(Xva).argmax(1).numpy()
        pte = net(Xte).argmax(1).numpy()
    return acc(pva, yva), acc(pte, yte)


# ---------------- Descriptors + readout (appearance-blind) ----------------

def run_descriptors(tr, va, te, use_csp=False):
    Dtr = descriptor_matrix(tr[1]); ytr = tr[2]
    Dva = descriptor_matrix(va[1]); yva = va[2]
    Dte = descriptor_matrix(te[1]); yte = te[2]
    sc = StandardScaler().fit(Dtr)
    Ztr, Zva, Zte = sc.transform(Dtr), sc.transform(Dva), sc.transform(Dte)

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    res = {"logistic": (acc(clf.predict(Zva), yva), acc(clf.predict(Zte), yte))}

    if use_csp:
        res["csp"] = _csp_ovr(Ztr, ytr, Zva, yva, Zte, yte)
    return res, (Dtr, ytr)


def _csp_ovr(Ztr, ytr, Zva, yva, Zte, yte):
    """One-vs-rest csp: a symbolic score per class, argmax to predict."""
    from tessera.search.csp import discover, CSPSRConfig
    from tessera.expression.tree import evaluate
    names = [f"f{i}" for i in range(Ztr.shape[1])]
    envtr = {names[i]: Ztr[:, i] for i in range(len(names))}
    envva = {names[i]: Zva[:, i] for i in range(len(names))}
    envte = {names[i]: Zte[:, i] for i in range(len(names))}
    sva = np.zeros((len(yva), len(CLASSES)))
    ste = np.zeros((len(yte), len(CLASSES)))
    for c in range(len(CLASSES)):
        t = (ytr == c).astype(float)
        r = discover(envtr, t, CSPSRConfig(poly_degree=2, max_terms=8, stlsq_threshold=0.03))
        sva[:, c] = evaluate(r.expr, envva)
        ste[:, c] = evaluate(r.expr, envte)
    return acc(sva.argmax(1), yva), acc(ste.argmax(1), yte)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--n-train", type=int, default=600, help="per class")
    p.add_argument("--n-eval", type=int, default=200, help="per class (val and test)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csp", action="store_true", help="also run csp symbolic readout")
    args = p.parse_args()

    print("=== gestalt transfer testbed ===")
    print(f"classes={CLASSES} size={args.size}")
    t0 = time.time()
    tr = build(args.n_train, args.size, sim_regime, seed=args.seed)        # sim train
    va = build(args.n_eval, args.size, sim_regime, seed=args.seed + 1)     # sim val
    te = build(args.n_eval, args.size, real_regime, seed=args.seed + 2)    # REAL test
    print(f"built data in {time.time()-t0:.0f}s "
          f"(train {len(tr[2])}, val {len(va[2])}, real-test {len(te[2])})\n")

    print("--- CNN on RGB (region-local) ---")
    t0 = time.time()
    cnn_va, cnn_te = run_cnn(tr, va, te, args.size, args.epochs, args.seed)
    print(f"  sim-val={cnn_va:.3f}  real-test={cnn_te:.3f}  gap={cnn_va-cnn_te:+.3f}  "
          f"({time.time()-t0:.0f}s)\n")

    print("--- Descriptors on mask (appearance-invariant) ---")
    t0 = time.time()
    res, _ = run_descriptors(tr, va, te, use_csp=args.csp)
    for k, (a_va, a_te) in res.items():
        print(f"  [{k:8s}] sim-val={a_va:.3f}  real-test={a_te:.3f}  gap={a_va-a_te:+.3f}")
    print(f"  ({time.time()-t0:.0f}s)\n")

    print("=== summary ===")
    print(f"  CNN-RGB        gap={cnn_va-cnn_te:+.3f}  (real-test {cnn_te:.3f})")
    for k, (a_va, a_te) in res.items():
        print(f"  descr-{k:8s} gap={a_va-a_te:+.3f}  (real-test {a_te:.3f})")
    print("\nReading: a large CNN gap with a small descriptor gap = the triangle's"
          "\npremise holds (structure transfers where region-local appearance does not)."
          "\nThe descriptor real-test acc is the shape-only ceiling.")


if __name__ == "__main__":
    main()
