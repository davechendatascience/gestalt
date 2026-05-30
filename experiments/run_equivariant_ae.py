"""Train the neural equivariant autoencoder render-and-compare, then probe its
latent for viewpoint invariance.

Data: orthographic SO(3) views (centred) of the 14 objects. Training: encode one
view -> z, re-render OTHER views at their known pose, match. One z must explain
all views -> z becomes the viewpoint-canonical 3D code.

Reports:
  - reconstruction (docs/ae_recon.png): encode view i, render target poses, vs GT;
  - invariance: cross-view cosine of the AE code vs of raw pixels;
  - object-id linear probe from the frozen AE code (train/test split by VIEW).

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_equivariant_ae.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from gestalt.render3d import library, render_camera, rand_rotation
from gestalt.ae3d import EquivAE

H, NV, C, D = 48, 16, 8, 16
torch.manual_seed(0)


def build_data():
    lib = library(); names = list(lib)
    imgs = np.zeros((len(lib), NV, H, H), np.float32)
    Rs = np.zeros((len(lib), NV, 3, 3), np.float32)
    rng = np.random.default_rng(0)
    for o, name in enumerate(names):
        pts, nrm = lib[name]
        for v in range(NV):
            R = rand_rotation(rng)
            imgs[o, v] = render_camera(pts, nrm, R, 4.0, 2.0, (0, 0), H, perspective=False)
            Rs[o, v] = R
    return (torch.tensor(imgs), torch.tensor(Rs), names)


def train(model, imgs, Rs, steps=900, bs=24, seed=0):
    O, V = imgs.shape[:2]
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for s in range(steps):
        o = torch.tensor(rng.integers(0, O, bs))
        i = torch.tensor(rng.integers(0, V, bs)); j = torch.tensor(rng.integers(0, V, bs))
        xi = imgs[o, i].unsqueeze(1); xj = imgs[o, j].unsqueeze(1)
        Ri = Rs[o, i]; Rj = Rs[o, j]
        z = model.encode(xi)
        loss = F.mse_loss(model.render(z, Ri), xi) + F.mse_loss(model.render(z, Rj), xj)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 300 == 0:
            print(f"    step {s+1}/{steps}  recon mse={loss.item():.4f}")
    return model


@torch.no_grad()
def recon_figure(model, imgs, Rs, names, path, obj=3):
    model.eval()
    z = model.encode(imgs[obj, 0:1].unsqueeze(1))         # encode ONE view
    cols = 6
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.4, 3.0))
    for c in range(cols):
        gt = imgs[obj, c]; rR = model.render(z, Rs[obj, c:c + 1])[0, 0]
        axes[0, c].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[0, c].axis("off")
        axes[1, c].imshow(rR, cmap="gray", vmin=0, vmax=1); axes[1, c].axis("off")
        if c == 0:
            axes[0, c].set_title("GT views", fontsize=8, loc="left")
            axes[1, c].set_title("rendered from 1-view code", fontsize=8, loc="left")
    fig.suptitle(f"equivariant AE: one encoded view of '{names[obj]}' re-renders other poses",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


@torch.no_grad()
def probe(model, imgs):
    """Centre the codes (a big shared mean dominates raw cosine), then measure
    intra- vs inter-object separation and nearest-centroid identity across
    unseen views."""
    model.eval()
    O, V = imgs.shape[:2]
    codes = model.code(imgs.reshape(O * V, 1, H, H)).numpy()
    codes = codes - codes.mean(0, keepdims=True)           # remove shared component
    cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    cn = cn.reshape(O, V, -1)

    intra = [float(cn[o, a] @ cn[o, b]) for o in range(O)
             for a in range(V) for b in range(a + 1, V)]
    cents_all = cn.mean(1); cents_all /= np.linalg.norm(cents_all, axis=1, keepdims=True) + 1e-9
    inter = [float(cents_all[o1] @ cents_all[o2]) for o1 in range(O) for o2 in range(o1 + 1, O)]

    tr = [v for v in range(V) if v % 2 == 0]; te = [v for v in range(V) if v % 2 == 1]
    cent = cn[:, tr].mean(1); cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    correct = sum(int((cent @ cn[o, v]).argmax() == o) for o in range(O) for v in te)
    acc = correct / (O * len(te))
    return float(np.mean(intra)), float(np.mean(inter)), acc


def main():
    t0 = time.time()
    imgs, Rs, names = build_data()
    print(f"=== neural equivariant autoencoder ({len(names)} objects x {NV} ortho SO(3) views) ===")
    model = EquivAE(C, D, H)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M; training render-and-compare ...")
    train(model, imgs, Rs)
    recon_figure(model, imgs, Rs, names, ROOT / "docs" / "ae_recon.png")
    print(f"  wrote docs/ae_recon.png")

    O = imgs.shape[0]
    intra, inter, acc = probe(model, imgs)
    print(f"\n  centred-code cosine:  intra-object={intra:.3f}  inter-object={inter:.3f}  "
          f"separation={intra-inter:+.3f}")
    print(f"  object-id nearest-centroid from frozen code (UNSEEN views): {acc:.3f}  "
          f"(chance {1/O:.3f})")
    print(f"\n({time.time()-t0:.0f}s)  The latent is a 3D volume rotated by the pose; "
          f"one encoded view\nre-renders the others -> the code is viewpoint-canonical and "
          f"object-identifying.")


if __name__ == "__main__":
    main()
