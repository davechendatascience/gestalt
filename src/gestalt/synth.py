"""Synthetic shape-transfer testbed.

Faithful geometry, two DISJOINT appearance regimes. A shape's silhouette
(its mask) is drawn from the SAME distribution in both the "sim" and "real"
regimes; only appearance (fill texture, colour palette, background, noise,
blur) differs. This reproduces the sim2real setup in miniature with a knob on
the gap: geometry is shared, the domain shift lives entirely in appearance.

A region-local CNN that latches onto texture should score high on sim-val and
collapse on real-test; an appearance-invariant shape readout should transfer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

CLASSES = ["triangle", "square", "pentagon", "hexagon", "star", "cross"]


# ----------------------------------------------------------------------
# Geometry: class -> polygon vertices (unit, centered), then place into a mask
# ----------------------------------------------------------------------

def _ngon(n: int, phase=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.stack([np.cos(a), np.sin(a)], 1)


def _star(p=5, inner=0.45):
    a = np.linspace(0, 2 * np.pi, 2 * p, endpoint=False) - np.pi / 2
    r = np.where(np.arange(2 * p) % 2 == 0, 1.0, inner)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def _cross(arm=0.36):
    w = arm
    pts = [(-w, -1), (w, -1), (w, -w), (1, -w), (1, w), (w, w),
           (w, 1), (-w, 1), (-w, w), (-1, w), (-1, -w), (-w, -w)]
    return np.asarray(pts, float)


def _verts(cls: str):
    if cls == "triangle": return _ngon(3, np.pi / 2)
    if cls == "square":   return _ngon(4, np.pi / 4)
    if cls == "pentagon": return _ngon(5, np.pi / 2)
    if cls == "hexagon":  return _ngon(6)
    if cls == "star":     return _star(5)
    if cls == "cross":    return _cross()
    raise ValueError(cls)


def make_mask(cls: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Rasterise a class's silhouette with random pose + mild deformation."""
    v = _verts(cls).copy()
    v += rng.normal(0, 0.025, v.shape)                       # vertex jitter (deformation)
    th = rng.uniform(0, 2 * np.pi)                           # rotation
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    v = v @ R.T
    scale = rng.uniform(0.30, 0.42) * size                  # size
    cx = rng.uniform(0.42, 0.58) * size
    cy = rng.uniform(0.42, 0.58) * size                     # translation
    v = v * scale + np.array([cx, cy])

    ys, xs = np.mgrid[0:size, 0:size]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    inside = Path(v).contains_points(pts).reshape(size, size)
    return inside


# ----------------------------------------------------------------------
# Appearance: render a mask into RGB under a regime (sim | real)
# ----------------------------------------------------------------------

@dataclass
class Regime:
    name: str
    fg_palette: np.ndarray      # (K,3) candidate foreground colours
    bg_palette: np.ndarray      # (K,3) candidate background colours
    texture: str                # "speckle" | "stripes"
    tex_scale: float            # spatial frequency of the texture
    noise: float                # additive pixel noise std
    blur: float                 # gaussian blur sigma


def _palette(rng, base, spread=0.12, k=6):
    return np.clip(base + rng.uniform(-spread, spread, (k, 3)), 0, 1)


def sim_regime(rng):
    return Regime("sim", _palette(rng, np.array([0.85, 0.55, 0.25])),  # warm
                  _palette(rng, np.array([0.20, 0.22, 0.28])),
                  texture="speckle", tex_scale=1.0, noise=0.05, blur=0.0)


def real_regime(rng):
    return Regime("real", _palette(rng, np.array([0.25, 0.55, 0.80])),  # cool (disjoint)
                  _palette(rng, np.array([0.85, 0.85, 0.80])),
                  texture="stripes", tex_scale=6.0, noise=0.09, blur=0.7)


def _texture_field(size, reg: Regime, rng):
    if reg.texture == "speckle":
        t = rng.normal(0, 1, (size, size))
        t = gaussian_filter(t, 0.8)
    else:  # stripes at a random orientation
        th = rng.uniform(0, np.pi)
        ys, xs = np.mgrid[0:size, 0:size]
        t = np.sin((xs * np.cos(th) + ys * np.sin(th)) * reg.tex_scale * 2 * np.pi / size)
    t = (t - t.min()) / (np.ptp(t) + 1e-9)
    return 0.65 + 0.35 * t                                  # brightness modulation in [0.65,1]


def render(mask: np.ndarray, reg: Regime, rng) -> np.ndarray:
    """mask (H,W bool) -> RGB (H,W,3) in [0,1] under appearance regime `reg`."""
    size = mask.shape[0]
    fg = reg.fg_palette[rng.integers(len(reg.fg_palette))]
    bg = reg.bg_palette[rng.integers(len(reg.bg_palette))]
    tex = _texture_field(size, reg, rng)[..., None]
    img = np.where(mask[..., None], fg * tex, bg * tex)
    img = img + rng.normal(0, reg.noise, img.shape)
    if reg.blur > 0:
        for c in range(3):
            img[..., c] = gaussian_filter(img[..., c], reg.blur)
    return np.clip(img, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------

def build(n_per_class: int, size: int, regime_fn, seed: int):
    """Return (imgs (N,H,W,3), masks (N,H,W) bool, labels (N,)) for a regime."""
    rng = np.random.default_rng(seed)
    reg = regime_fn(rng)
    imgs, masks, labels = [], [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            m = make_mask(cls, size, rng)
            imgs.append(render(m, reg, rng))
            masks.append(m)
            labels.append(ci)
    idx = rng.permutation(len(labels))
    return (np.stack(imgs)[idx], np.stack(masks)[idx],
            np.asarray(labels)[idx])
