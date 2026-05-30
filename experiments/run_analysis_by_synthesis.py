"""Analysis-by-synthesis: recognise shapes by ENERGY MINIMISATION over a
generative concept space, not by a feedforward classifier.

For each observed mask we infer z* = argmin_z E(x, z) (inference.py). We report:
  - concept (silhouette) accuracy, sim vs real  -> does the inferred concept
    TRANSFER? (it should: the energy is anchored to shared geometry, blind to
    appearance);
  - the energy LANDSCAPE structure: which concepts form near-degenerate basins
    (small margin) -> the emergent "modes" geometry;
  - the built-in multimodality: discA/discB share the 'disc' silhouette, so the
    shape decoder identifies 'disc' confidently but CANNOT resolve A-vs-B -> a
    genuine 2-mode posterior that only an appearance/energy term could break.

Uses the cached testbed (run make_data.py first), cue_mode 'none'.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_analysis_by_synthesis.py
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from gestalt.synth import load_dataset, CLASSES
from gestalt.descriptors import radial_signature
from gestalt.generative import SignatureGenerator
from gestalt.inference import infer

DATA = ROOT / "data" / "gestalt"

# map the 8 testbed classes onto the 7 silhouette families the decoder knows
SILH_OF = {c: c for c in CLASSES}
SILH_OF["discA"] = SILH_OF["discB"] = "disc"


def evaluate(split):
    _, masks, labels = load_dataset(DATA / f"none__{split}.npz")
    gen = SignatureGenerator()
    correct = 0
    margins, conf = [], Counter()
    disc_idx, disc_resolved_disc = 0, 0
    for m, y in zip(masks, labels):
        sig = radial_signature(m)
        inf = infer(sig, gen)
        truth = SILH_OF[CLASSES[y]]
        correct += int(inf.type == truth)
        margins.append(inf.margin)
        if inf.type != truth:
            conf[f"{truth}->{inf.type}"] += 1
        if truth == "disc":
            disc_idx += 1
            disc_resolved_disc += int(inf.type == "disc")
    n = len(labels)
    return {
        "n": n, "acc": correct / n, "median_margin": float(np.median(margins)),
        "confusions": conf.most_common(4),
        "disc_as_disc": disc_resolved_disc / max(1, disc_idx),
    }


def main():
    if not (DATA / "none__sim_val.npz").exists():
        print("run experiments/make_data.py first"); return
    print("=== analysis-by-synthesis: recognition as energy minimisation ===")
    print("concept = silhouette family; z = (type, scale, rotation); "
          "z* = argmin_z E(x,z)\n")
    for split in ["sim_val", "real_test"]:
        r = evaluate(split)
        dom = "SIM " if "sim" in split else "REAL"
        print(f"[{dom}] concept-acc={r['acc']:.3f}  median margin={r['median_margin']:.2f}  "
              f"(disc identified as disc: {r['disc_as_disc']:.3f})")
        if r["confusions"]:
            print(f"       nearest-basin confusions: {r['confusions']}")
    print("\nReading:")
    print(" - SIM vs REAL concept-acc ~ equal  => the inferred concept TRANSFERS")
    print("   (energy is anchored to shared geometry, blind to appearance).")
    print(" - 'disc identified as disc' high, but A-vs-B is unresolvable from shape:")
    print("   the residual 2-mode posterior that an appearance/energy term must break.")
    print(" - small-margin confusions = concepts that form near-degenerate energy")
    print("   basins (e.g. hexagon vs disc) = the emergent mode geometry.")


if __name__ == "__main__":
    main()
