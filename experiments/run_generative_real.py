"""Generative recognition across an APPEARANCE shift -- the honest 'actual dataset'
test. Templates are clean SIM renders (pose-covering bank, no pose labels needed);
queries are the SAME geometry with a different appearance ("real": fg texture +
noise), to mimic sim->real. We match the query to the bank two ways:

  pixel      : correlate raw images        -> should DIE under the appearance shift
  silhouette : correlate binarised masks   -> appearance-blind, should SURVIVE

This is the recipe for a real dataset: pose handled by SEARCH over a sim-rendered
bank (no pose labels), and an appearance-robust match (silhouette / later depth or
learned features) to cross the sim->real gap.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_generative_real.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from gestalt.render3d import library, render_camera, rand_rotation

H, K, NTE = 48, 128, 20


def appearance_shift(img, rng):
    """Same silhouette, different look: replace smooth shading with a random
    texture (+ noise) inside the object; background stays black."""
    mask = img > 0.03
    tex = 0.35 + 0.65 * rng.random(img.shape)
    out = np.where(mask, tex, 0.0) + rng.normal(0, 0.04, img.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def _norm(A):
    A = A - A.mean(1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def match_acc(Tmpl, Q, O):
    T = _norm(Tmpl.reshape(O * K, -1)); X = _norm(Q.reshape(O * NTE, -1))
    score = (X @ T.T).reshape(O * NTE, O, K).max(2)
    y = np.repeat(np.arange(O), NTE)
    return float((score.argmax(1) == y).mean())


def main():
    lib = library(); names = list(lib); O = len(names)
    rng = np.random.default_rng(0)
    Tmpl = np.zeros((O, K, H, H), np.float32)
    Qclean = np.zeros((O, NTE, H, H), np.float32); Qreal = np.zeros_like(Qclean)
    for o, nm in enumerate(names):
        pts, nrm = lib[nm]
        for k in range(K):
            Tmpl[o, k] = render_camera(pts, nrm, rand_rotation(rng), 4, 2, (0, 0), H, False)
        for v in range(NTE):
            c = render_camera(pts, nrm, rand_rotation(rng), 4, 2, (0, 0), H, False)
            Qclean[o, v] = c; Qreal[o, v] = appearance_shift(c, rng)

    sil = lambda A: (A > 0.05).astype(np.float32)
    print("=== generative match across a sim->real APPEARANCE shift (no pose labels) ===")
    print(f"templates = clean sim renders (K={K} poses);  query = appearance-shifted\n")
    print(f"{'match on':>12} | {'clean query':>12} | {'shifted query':>14}")
    print("-" * 46)
    t0 = time.time()
    print(f"{'pixels':>12} | {match_acc(Tmpl, Qclean, O):>12.3f} | "
          f"{match_acc(Tmpl, Qreal, O):>14.3f}")
    print(f"{'silhouette':>12} | {match_acc(sil(Tmpl), sil(Qclean), O):>12.3f} | "
          f"{match_acc(sil(Tmpl), sil(Qreal), O):>14.3f}")
    print(f"\n({time.time()-t0:.0f}s) chance={1/O:.3f}. Pixel match collapses under the")
    print("appearance shift; silhouette match survives -> on real data, search the pose")
    print("over a sim bank (free, no labels) and match on STRUCTURE, not pixels.")


if __name__ == "__main__":
    main()
