"""Materialise the synthetic testbed to disk for reuse + visual inspection.

Writes:
  data/gestalt/<mode>_<split>.npz   — cached datasets (gitignored, regenerable)
  docs/testbed_preview.png          — a grid you can eyeball (tracked)

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/make_data.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gestalt.synth import build_cached, sim_regime, real_regime, save_preview

DATA = ROOT / "data" / "gestalt"


def main(size=48, n_train=500, n_eval=200, seed=0):
    t0 = time.time()
    splits = [("sim_train", sim_regime, n_train, seed),
              ("sim_val", sim_regime, n_eval, seed + 1),
              ("real_test", real_regime, n_eval, seed + 2)]
    for mode in ["transferable", "spurious", "none"]:
        for name, regime_fn, n, sd in splits:
            path = DATA / f"{mode}__{name}.npz"
            imgs, masks, labels = build_cached(n, size, regime_fn, sd, mode, path)
            print(f"  {path.relative_to(ROOT)}  imgs{imgs.shape} labels{labels.shape}")
    prev = ROOT / "docs" / "testbed_preview.png"
    save_preview(prev, size=size)
    print(f"  preview -> {prev.relative_to(ROOT)}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
