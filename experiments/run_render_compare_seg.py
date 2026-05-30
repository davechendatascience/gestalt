"""Render-and-compare SEGMENTATION vs a learned segmenter, across an appearance
shift -- how 3D models help segmentation, and why it crosses sim->real.

Scenes: one object at random pose + small offset.
  sim     : object shaded on black background (the renderer's look).
  shifted : object filled with random texture on a textured gray background
            (same geometry, different appearance == sim->real proxy).

  learned  : a small U-Net trained on SIM scenes to predict the foreground mask
             (learns pixels->mask from sim appearance).
  render&  : fit the 3D-model templates by matching EDGES (appearance-invariant);
  compare    the predicted mask is the fitted model's SILHOUETTE at the best
             (object, pose, position). The mask comes from geometry, not appearance.

Metric = mask IoU on sim and on shifted scenes. The learned segmenter should drop
sim->shifted (it overfit appearance); render-and-compare should hold (it matched
structure) -- the 90->10 in miniature, and the fix.

Run:
  ../market-analysis/.koopman-env/Scripts/python.exe experiments/run_render_compare_seg.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import sobel, binary_erosion, gaussian_filter

from gestalt.render3d import library, render_camera, rand_rotation

H, K, OFF = 64, 24, 8
torch.manual_seed(0)


# ---------------- scene generation ----------------

def object_render(pts, nrm, R):
    img = render_camera(pts, nrm, R, 5.0, 2.0, (0, 0), H, perspective=False)
    return img, img > 0.03


def place(a, dy, dx):
    return np.roll(np.roll(a, dy, 0), dx, 1)


def make_scene(shaded, silh, rng, mode):
    dy, dx = rng.integers(-OFF, OFF + 1), rng.integers(-OFF, OFF + 1)
    sh = place(shaded, dy, dx); mask = place(silh, dy, dx)
    if mode == "sim":
        scene = sh.copy()
    else:  # shifted appearance
        tex = 0.4 + 0.6 * gaussian_filter(rng.random((H, H)), 0.8)
        bg = rng.uniform(0.3, 0.55) + 0.1 * gaussian_filter(rng.random((H, H)), 1.0)
        scene = np.where(mask, tex, bg) + rng.normal(0, 0.04, (H, H))
    return np.clip(scene, 0, 1).astype(np.float32), mask.astype(np.float32)


def dataset(n, mode, seed):
    lib = library(); names = list(lib); rng = np.random.default_rng(seed)
    S = np.zeros((n, H, H), np.float32); M = np.zeros((n, H, H), np.float32)
    for i in range(n):
        nm = names[rng.integers(len(names))]
        sh, si = object_render(*lib[nm], rand_rotation(rng))
        S[i], M[i] = make_scene(sh, si, rng, mode)
    return torch.tensor(S), torch.tensor(M)


def iou(pred, gt):
    p = pred > 0.5; g = gt > 0.5
    inter = (p & g).sum((-1, -2)).float(); union = (p | g).sum((-1, -2)).float()
    return (inter / (union + 1e-6)).mean().item()


# ---------------- learned segmenter (small U-Net) ----------------

class UNet(nn.Module):
    def __init__(s, c=16):
        super().__init__()
        s.e1 = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        s.e2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU())
        s.b = nn.Sequential(nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU())
        s.d2 = nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.ReLU())
        s.d1 = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.ReLU())
        s.out = nn.Conv2d(c, 1, 1)

    def forward(s, x):
        x1 = s.e1(x); x2 = s.e2(F.max_pool2d(x1, 2)); xb = s.b(F.max_pool2d(x2, 2))
        u2 = F.interpolate(xb, scale_factor=2, mode="nearest")
        d2 = s.d2(torch.cat([u2, x2], 1))
        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = s.d1(torch.cat([u1, x1], 1))
        return s.out(d1)


def train_unet(S, M, epochs=40, seed=0):
    torch.manual_seed(seed); net = UNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n, bs = len(S), 32
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(S[idx].unsqueeze(1))[:, 0], M[idx])
            loss.backward(); opt.step()
    net.eval(); return net


@torch.no_grad()
def unet_iou(net, S, M):
    return iou(torch.sigmoid(net(S.unsqueeze(1))[:, 0]), M)


# ---------------- render-and-compare matcher ----------------

def build_templates():
    lib = library(); names = list(lib); rng = np.random.default_rng(0)
    sils, bnds = [], []
    for nm in names:
        for _ in range(K):
            _, si = object_render(*lib[nm], rand_rotation(rng))
            sils.append(si.astype(np.float32))
            bnds.append((si & ~binary_erosion(si, iterations=1)).astype(np.float32))
    return np.stack(sils), np.stack(bnds)        # (O*K, H, H)


def edge_map(scene):
    g = np.hypot(sobel(scene, 0), sobel(scene, 1))
    return (g / (g.max() + 1e-9)).astype(np.float32)


def render_compare_iou(S, M, sils, bnds):
    Bf = np.conj(np.fft.fft2(bnds, axes=(1, 2)))          # (T,H,W)
    ious = []
    for s in range(len(S)):
        E = edge_map(S[s].numpy())
        corr = np.fft.ifft2(np.fft.fft2(E)[None] * Bf, axes=(1, 2)).real   # (T,H,W)
        flat = corr.reshape(len(bnds), -1)
        t = flat.max(1).argmax()                          # best template
        pk = flat[t].argmax(); dy, dx = pk // H, pk % H    # peak shift
        pred = place(sils[t], dy, dx)
        ious.append(iou(torch.tensor(pred)[None], M[s:s + 1]))
    return float(np.mean(ious))


def main():
    print("=== render-and-compare segmentation vs learned U-Net, across appearance shift ===")
    t0 = time.time()
    Str, Mtr = dataset(600, "sim", 0)
    Ste_s, Mte_s = dataset(120, "sim", 1)
    Ste_r, Mte_r = dataset(120, "shifted", 2)
    print(f"  scenes built ({time.time()-t0:.0f}s); training U-Net on sim ...")

    net = train_unet(Str, Mtr)
    u_sim, u_shift = unet_iou(net, Ste_s, Mte_s), unet_iou(net, Ste_r, Mte_r)

    sils, bnds = build_templates()
    r_sim = render_compare_iou(Ste_s, Mte_s, sils, bnds)
    r_shift = render_compare_iou(Ste_r, Mte_r, sils, bnds)

    print(f"\n{'method':18} | {'sim IoU':>8} | {'shifted IoU':>12} | {'drop':>6}")
    print("-" * 54)
    print(f"{'learned U-Net':18} | {u_sim:>8.3f} | {u_shift:>12.3f} | {u_sim-u_shift:>+6.3f}")
    print(f"{'render-and-compare':18} | {r_sim:>8.3f} | {r_shift:>12.3f} | {r_sim-r_shift:>+6.3f}")
    print(f"\n({time.time()-t0:.0f}s) The U-Net learns pixels->mask from sim appearance and")
    print("drops on the shift; render-and-compare derives the mask from the model's")
    print("silhouette via edge matching -> appearance-invariant, crosses sim->real.")


if __name__ == "__main__":
    main()
