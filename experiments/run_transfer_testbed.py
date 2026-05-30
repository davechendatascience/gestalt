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
from gestalt.descriptors import descriptor_matrix, relational_matrix


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

def _readout(Ftr, ytr, Fva, yva, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Ftr), ytr)
    return (acc(clf.predict(sc.transform(Fva)), yva),
            acc(clf.predict(sc.transform(Fte)), yte))


def run_descriptors(tr, va, te):
    """Three readouts: shape (invariant), relational (fg-vs-bg appearance),
    and shape+relational. Logistic in all cases to isolate the REPRESENTATION
    effect from the readout."""
    S = [descriptor_matrix(x[1]) for x in (tr, va, te)]        # shape
    R = [relational_matrix(x[0], x[1]) for x in (tr, va, te)]  # relational
    ytr, yva, yte = tr[2], va[2], te[2]
    reps = {
        "shape": S,
        "relational": R,
        "shape+rel": [np.hstack([s, r]) for s, r in zip(S, R)],
    }
    return {k: _readout(F[0], ytr, F[1], yva, F[2], yte) for k, F in reps.items()}


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


def run_condition(cue_mode, args):
    tr = build(args.n_train, args.size, sim_regime, seed=args.seed, cue_mode=cue_mode)
    va = build(args.n_eval, args.size, sim_regime, seed=args.seed + 1, cue_mode=cue_mode)
    te = build(args.n_eval, args.size, real_regime, seed=args.seed + 2, cue_mode=cue_mode)
    cnn_va, cnn_te = run_cnn(tr, va, te, args.size, args.epochs, args.seed)
    desc = run_descriptors(tr, va, te)
    return {"cnn": (cnn_va, cnn_te), **desc}


def _row(name, va, te):
    return f"  {name:14s} sim-val={va:.3f}  real-test={te:.3f}  gap={va-te:+.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--n-train", type=int, default=600, help="per class")
    p.add_argument("--n-eval", type=int, default=200, help="per class (val and test)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=== gestalt transfer testbed (ambiguous-pair conditions) ===")
    print(f"classes={CLASSES}\nchance={1/len(CLASSES):.3f}  "
          f"shape-only ceiling ~ {(len(CLASSES)-1)/len(CLASSES):.3f} "
          f"(disc pair collapses to a coin-flip)\n")

    results = {}
    for mode in ["transferable", "spurious"]:
        t0 = time.time()
        print(f"### condition: cue = {mode.upper()} "
              f"({'cue survives sim->real' if mode=='transferable' else 'cue is sim-only'})")
        res = run_condition(mode, args)
        for name, (va, te) in res.items():
            tag = {"cnn": "CNN-RGB", "shape": "shape(inv)",
                   "relational": "relational", "shape+rel": "shape+rel"}[name]
            print(_row(tag, va, te))
        print(f"  ({time.time()-t0:.0f}s)\n")
        results[mode] = res

    print("=== what to read ===")
    print("shape(inv): zero gap, capped at the ceiling (blind to the disc cue) in BOTH modes.")
    print("CNN-RGB:    big gap in both modes (texture reliance).")
    print("shape+rel TRANSFERABLE: should BEAT the shape ceiling on real-test")
    print("           (relational fg-vs-bg cue survives the shift) -> the band is fillable.")
    print("shape+rel SPURIOUS: should NOT beat shape on real-test (cue is sim-only);")
    print("           may even dip below as the readout trusts a cue that vanishes -> the floor.")


if __name__ == "__main__":
    main()
