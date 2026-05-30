"""Single-view recognition as BAYESIAN generative inference (analysis-by-synthesis),
the toolbox the invariant approach was missing.

Posterior  P(o|x) ~ P(o) * max_g sim(x, render(o,g))  -- we marginalise the unknown
pose by searching it (max = MAP-over-pose; softmax-sum approximates the integral),
and the match is on the rendered VIEW, so self-occlusion is handled naturally (we
only compare what is visible). This is Grenander pattern theory / inverse graphics:
identity inferred by which object, at SOME pose, best EXPLAINS the partial view.

Compared to a CNN trained on the same rendered views, swept over how many poses K
each method gets (template bank size = CNN training size).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_generative_recognition.py
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

H, NTE = 48, 20
KS = [8, 32, 128]
torch.manual_seed(0)


def _norm_rows(A):
    A = A - A.mean(1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def build(Kmax, seed=0):
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(seed)
    Tmpl = np.zeros((O, Kmax, H * H), np.float32)
    Xte = np.zeros((O, NTE, H, H), np.float32)
    for o, nm in enumerate(names):
        pts, nrm = lib[nm]
        for k in range(Kmax):
            Tmpl[o, k] = render_camera(pts, nrm, rand_rotation(rng), 4, 2, (0, 0), H, False).ravel()
        for v in range(NTE):
            Xte[o, v] = render_camera(pts, nrm, rand_rotation(rng), 4, 2, (0, 0), H, False)
    return Tmpl, Xte, O


def generative_acc(Tmpl, Xte, K, O):
    T = _norm_rows(Tmpl[:, :K].reshape(O * K, H * H))            # (O*K, HW)
    Xt = _norm_rows(Xte.reshape(O * NTE, H * H))                 # (M, HW)
    corr = Xt @ T.T                                             # (M, O*K) match scores
    score = corr.reshape(O * NTE, O, K).max(2)                  # max over pose per object
    pred = score.argmax(1)
    y = np.repeat(np.arange(O), NTE)
    return float((pred == y).mean())


def cnn_acc(Tmpl, Xte, K, O, epochs=50, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(Tmpl[:, :K].reshape(O * K, 1, H, H))
    y = torch.repeat_interleave(torch.arange(O), K)
    net = VanillaCNN(n_classes=O)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = len(y), 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); F.cross_entropy(net(X[idx])[0], y[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xte.reshape(O * NTE, 1, H, H))
        yt = torch.repeat_interleave(torch.arange(O), NTE)
        return float((net(Xt)[0].argmax(1) == yt).float().mean())


def main():
    Tmpl, Xte, O = build(max(KS))
    print("=== single-view recognition: generative (pose-marginalised) vs CNN ===")
    print(f"{'poses/obj K':>12} | {'generative':>10} | {'CNN':>6}")
    print("-" * 36)
    for K in KS:
        t0 = time.time()
        g = generative_acc(Tmpl, Xte, K, O)
        c = cnn_acc(Tmpl, Xte, K, O)
        print(f"{K:>12} | {g:>10.3f} | {c:>6.3f}    ({time.time()-t0:.0f}s)")
    print(f"\nchance={1/O:.3f}. generative = which object at SOME pose best EXPLAINS the")
    print("view (search pose, compare visible projection). It uses the exact 3D model")
    print("at test time; the CNN must learn invariance from K views. Watch low K.")


if __name__ == "__main__":
    main()
