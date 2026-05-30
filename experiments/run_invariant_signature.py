"""Rotation-invariant 3D signature: classify the 14 objects under ARBITRARY
rotation with no training and no pose inference -- the efficient quotient-by-SO(3)
path, contrasted with the equivariant AE (0.23) and the CNN (0.75).

For each invariant (D2 distance distribution, SH power spectrum):
  - invariance: signature of a randomly-rotated object vs the canonical one
    (cosine -> 1 means rotation-blind by construction);
  - identity: nearest-centroid over the 14 canonical signatures classifies
    randomly-rotated instances (no training, no pose search).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_invariant_signature.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from gestalt.render3d import library, rand_rotation
from gestalt.invariants import d2_descriptor, sh_power_descriptor


def evaluate(name, sig_fn, lib, n_rot=12, seed=0):
    rng = np.random.default_rng(seed)
    names = list(lib); O = len(names)
    # canonical library signatures (reference, computed once per object)
    ref = np.stack([sig_fn(lib[nm][0], rng) for nm in names])
    refn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)

    inv_cos, correct, tot = [], 0, 0
    for o, nm in enumerate(names):
        P = lib[nm][0]
        for _ in range(n_rot):
            Pr = P @ rand_rotation(rng).T                     # arbitrary rotation
            s = sig_fn(Pr, rng); s = s / (np.linalg.norm(s) + 1e-9)
            inv_cos.append(float(s @ refn[o]))                # invariance to its own canonical
            pred = int((refn @ s).argmax())                   # nearest-centroid identity
            correct += int(pred == o); tot += 1
    print(f"  [{name:14s}] invariance cosine={np.mean(inv_cos):.4f}   "
          f"identity acc (rotated, no training)={correct/tot:.3f}")
    return correct / tot


def main():
    lib = library()
    # sig_fn signature: (points, rng) -> vector
    sig_d2 = lambda P, rng: d2_descriptor(P, rng)
    sig_sh = lambda P, rng: sh_power_descriptor(P)
    print(f"=== rotation-invariant 3D signature: 14 objects, arbitrary rotation, no training ===")
    print(f"chance = {1/len(lib):.3f}   (compare: equivariant AE 0.23, CNN 0.75)\n")
    t0 = time.time()
    evaluate("D2 distances", sig_d2, lib)
    evaluate("SH power spec", sig_sh, lib)
    print(f"\n({time.time()-t0:.0f}s)  Invariant computed in closed form -- pose is")
    print("quotiented out analytically (Peter-Weyl / isometry invariants), not inferred.")
    print("Caveat: this uses the 3D point cloud, where SO(3) acts cleanly. A single 2D")
    print("view breaks that (projection) -- the spherical-CNN lift is the bridge.")


if __name__ == "__main__":
    main()
