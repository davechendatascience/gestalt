"""Treat the module as a BACKBONE and probe its features for viewpoint quality.

Widened viewpoint group: rotation + scale + translation + shear (severity in
[0,1]). Two backbones, trained then FROZEN:
  vanilla : VanillaCNN encoder, trained with the same wide augmentation (CE)
  gestalt : STN canonicaliser + encoder, trained with CE + cross-view invariance

Two feature-quality probes (the El-Banani-style "is the representation viewpoint
robust" question, done with frozen features):

  A. Cross-view embedding consistency: two random transforms of the same image
     -> cosine similarity of the frozen embeddings. Higher = more invariant.

  B. Linear-probe invariance curve: freeze the backbone, fit ONE linear probe on
     low-severity embeddings, then test accuracy across a severity sweep. The
     flatter the curve, the more viewpoint-invariant the FEATURES (not just the
     end-to-end classifier).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_backbone_probe.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from gestalt.models import VanillaCNN, GestaltNet, random_affine_batch
from run_viewpoint_mnist import load

S_TRAIN = 0.4
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
torch.manual_seed(0)


def train_backbone(model, Xtr, ytr, epochs, mode, seed=0):
    """mode: 'plain'      - CE only (vanilla CNN + augmentation)
             'invariance' - STN + CE + cross-view EMBEDDING invariance (implicit;
                            no canonical GT)
             'canon'      - STN + CE + reconstruct the UNTRANSFORMED image
                            ||STN(transform(x)) - x||^2 (GT = the unrolled image,
                            the strong signal a sim/Isaac provides for free)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = len(ytr), 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx], ytr[idx]
            x1 = random_affine_batch(xb, S_TRAIN, rng)
            opt.zero_grad()
            if mode == "plain":
                loss = F.cross_entropy(model(x1)[0], yb)
            elif mode == "invariance":
                x2 = random_affine_batch(xb, S_TRAIN, rng)
                o1, h1, _ = model(x1); o2, h2, _ = model(x2)
                loss = (F.cross_entropy(o1, yb) + F.cross_entropy(o2, yb)) / 2 \
                    + 1.0 * F.mse_loss(h1, h2)
            elif mode == "canon":
                o1, _, xc1 = model(x1)
                loss = F.cross_entropy(o1, yb) + 1.0 * F.mse_loss(xc1, xb)  # GT = unrolled
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model, x):
    h = model.embed(x)
    return F.normalize(h, dim=1)


@torch.no_grad()
def consistency(model, Xte, rng, severity):
    x1 = random_affine_batch(Xte, severity, rng)
    x2 = random_affine_batch(Xte, severity, rng)
    h1, h2 = embed(model, x1), embed(model, x2)
    return float((h1 * h2).sum(1).mean())


@torch.no_grad()
def probe_curve(model, Xtr, ytr, Xte, yte, rng):
    Ztr = embed(model, random_affine_batch(Xtr, 0.2, rng)).numpy()
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Ztr, ytr.numpy())
    accs = {}
    for s in SEVERITIES:
        Zte = embed(model, random_affine_batch(Xte, s, rng)).numpy()
        accs[s] = float((clf.predict(Zte) == yte.numpy()).mean())
    return accs


def main(n_train=8000, n_test=2000, epochs=10):
    Xtr, ytr, Xte, yte = load(n_train, n_test)
    print("=== backbone feature-quality probe (widened viewpoint group) ===")
    print(f"group: rotation+scale+translation+shear; train severity {S_TRAIN}; "
          f"sweep {SEVERITIES}\n")

    backbones = {}
    for name, model, mode in [("vanilla", VanillaCNN(), "plain"),
                              ("gestalt-inv", GestaltNet(), "invariance"),
                              ("gestalt-canon", GestaltNet(), "canon")]:
        t0 = time.time()
        backbones[name] = train_backbone(model, Xtr, ytr, epochs, mode)
        print(f"  trained {name} ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(1)
    print("\n--- Probe A: cross-view embedding cosine consistency (higher=better) ---")
    print("severity:  " + "".join(f"{s:>7.1f}" for s in [0.2, 0.5, 0.8]))
    for name, m in backbones.items():
        row = "".join(f"{consistency(m, Xte, np.random.default_rng(7), s):7.3f}"
                      for s in [0.2, 0.5, 0.8])
        print(f"  {name:8s}" + row)

    print("\n--- Probe B: frozen linear-probe accuracy vs viewpoint severity ---")
    print("severity:  " + "".join(f"{s:>7.1f}" for s in SEVERITIES))
    curves = {}
    for name, m in backbones.items():
        c = probe_curve(m, Xtr, ytr, Xte, yte, np.random.default_rng(3))
        curves[name] = c
        print(f"  {name:8s}" + "".join(f"{c[s]:7.3f}" for s in SEVERITIES))

    print("\nmean linear-probe acc over severity sweep (feature-quality score):")
    for name, c in curves.items():
        print(f"  {name:8s} {np.mean([c[s] for s in SEVERITIES]):.3f}  "
              f"(drop 0->1.0: {c[0.0]-c[1.0]:+.3f})")
    print("\nReading: a flatter Probe-B curve + higher Probe-A cosine = the FROZEN")
    print("features are viewpoint-invariant, i.e. a better geometry-encoding backbone")
    print("(not just a better end-to-end classifier).")


if __name__ == "__main__":
    main()
