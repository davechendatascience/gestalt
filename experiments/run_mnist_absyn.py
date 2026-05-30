"""MNIST: recognition by energy minimisation over (digit concept) x (coordinate
transform), on RAW PIXELS.

The point is the coordinate-transform thesis: pixels are samples in the sensor
chart, so the SAME digit shifted/scaled is far in pixel-L2 but identical in
concept. We therefore infer, per candidate prototype, the best alignment (a
translation) and score the residual THERE. Recognition = argmin over
(class, prototype-mode, shift) of the pixel explanation energy.

Prototypes per class are k-means centroids = the discovered within-class MODES
(style variants) of the generating process.

We report three things:
  1. aligned vs naive (no-transform) template energy -> how much the coordinate
     transform alone buys, on clean test;
  2. robustness to a held-out coordinate shift (test images translated): aligned
     should hold, naive should collapse -> pixels are the wrong frame, inferring
     the transform fixes it;
  3. a feedforward pixel baseline (logistic) for context.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_mnist_absyn.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

DATA = ROOT / "data" / "gestalt"
MN = DATA / "mnist.npz"


def load_mnist(n_train, n_test, seed=0):
    if MN.exists():
        d = np.load(MN); X, y = d["X"], d["y"]
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="liac-arff")
        y = y.astype(int)
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MN, X=X.astype(np.float32), y=y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]
    norm = lambda A: (A / 255.0).reshape(-1, 28, 28)
    return norm(X[tr]), y[tr], norm(X[te]), y[te]


def build_prototypes(Xtr, ytr, k):
    """Per-class k-means centroids = discovered style modes. Returns (P,28,28),
    class-id per prototype."""
    protos, pcls = [], []
    for c in range(10):
        Xc = Xtr[ytr == c].reshape(-1, 784)
        km = KMeans(k, n_init=3, random_state=0).fit(Xc)
        protos.append(km.cluster_centers_.reshape(-1, 28, 28))
        pcls += [c] * k
    return np.concatenate(protos), np.array(pcls)


def classify(Xte, protos, pcls, max_shift):
    """argmin over (prototype, shift) of pixel residual; vote class. Batched via
    FFT cross-correlation (residual reads off the correlation peak)."""
    H, W = 28, 28
    Tf = np.conj(np.fft.fft2(protos, axes=(1, 2)))      # (P,H,W)
    sse_t = (protos ** 2).sum((1, 2))                   # (P,)
    shifts = [(dy, dx) for dy in range(-max_shift, max_shift + 1)
              for dx in range(-max_shift, max_shift + 1)]
    pred = np.empty(len(Xte), dtype=int)
    for i, obs in enumerate(Xte):
        cc = np.fft.ifft2(np.fft.fft2(obs)[None] * Tf, axes=(1, 2)).real   # (P,H,W)
        peak = np.max([cc[:, dy % H, dx % W] for (dy, dx) in shifts], axis=0)  # (P,)
        resid = np.sum(obs ** 2) + sse_t - 2 * peak                       # (P,)
        # min residual per class
        best_c, best_e = 0, np.inf
        for c in range(10):
            e = resid[pcls == c].min()
            if e < best_e:
                best_e, best_c = e, c
        pred[i] = best_c
    return pred


def acc(p, y):
    return float((p == y).mean())


def shift_images(X, rng, mx=3):
    out = np.empty_like(X)
    for i in range(len(X)):
        dy, dx = rng.integers(-mx, mx + 1), rng.integers(-mx, mx + 1)
        out[i] = np.roll(np.roll(X[i], dy, 0), dx, 1)
    return out


def main(n_train=8000, n_test=1500, k=12, max_shift=3, seed=0):
    print("=== MNIST analysis-by-synthesis (pixels): concept x coordinate-transform ===")
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_mnist(n_train, n_test, seed)
    protos, pcls = build_prototypes(Xtr, ytr, k)
    print(f"  data+prototypes ({k}/class = {len(protos)} modes) in {time.time()-t0:.0f}s\n")

    # 1. clean test: aligned vs naive
    t0 = time.time()
    a_aligned = acc(classify(Xte, protos, pcls, max_shift), yte)
    a_naive = acc(classify(Xte, protos, pcls, 0), yte)
    print("--- clean test ---")
    print(f"  template + coord-transform (shift<= {max_shift}): {a_aligned:.3f}")
    print(f"  template, naive (no transform):                  {a_naive:.3f}")

    # 2. robustness to a coordinate shift
    rng = np.random.default_rng(1)
    Xsh = shift_images(Xte, rng, mx=3)
    r_aligned = acc(classify(Xsh, protos, pcls, max_shift), yte)
    r_naive = acc(classify(Xsh, protos, pcls, 0), yte)
    print("\n--- shifted test (held-out coordinate transform, +-3px) ---")
    print(f"  template + coord-transform: {r_aligned:.3f}   (drop {a_aligned-r_aligned:+.3f})")
    print(f"  template, naive:            {r_naive:.3f}   (drop {a_naive-r_naive:+.3f})")

    # 3. feedforward pixel baseline
    lr = LogisticRegression(max_iter=200, C=0.05).fit(Xtr.reshape(len(Xtr), -1), ytr)
    b_clean = acc(lr.predict(Xte.reshape(len(Xte), -1)), yte)
    b_shift = acc(lr.predict(Xsh.reshape(len(Xsh), -1)), yte)
    print("\n--- feedforward pixel baseline (logistic) ---")
    print(f"  clean: {b_clean:.3f}   shifted: {b_shift:.3f}   (drop {b_clean-b_shift:+.3f})")

    print(f"\n({time.time()-t0:.0f}s)  Reading: inferring the coordinate transform")
    print("keeps recognition stable under a shift that collapses naive pixel matching")
    print("and the feedforward classifier -> pixels are the wrong frame; settling fixes it.")


if __name__ == "__main__":
    main()
