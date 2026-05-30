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

# Six silhouette-separable classes, plus an AMBIGUOUS PAIR (discA/discB) that
# share an identical disc silhouette and can ONLY be told apart by appearance.
# The pair drops the shape-only ceiling below 100% and opens the band where a
# transfer-robust appearance reader (direction A) could beat pure shape.
CLASSES = ["triangle", "square", "pentagon", "hexagon", "star", "cross",
           "discA", "discB"]

# Per-class disambiguating "cue" scalar in [0,1]. Neutral (0.5) for the
# shape-separable classes; the disc pair is split to the extremes so only the
# cue separates them.
CUE = {c: 0.5 for c in CLASSES}
CUE["discA"], CUE["discB"] = 0.12, 0.88


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
    if cls in ("discA", "discB"):  return _ngon(24)   # identical disc silhouette
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


# The two regimes share BACKGROUND LUMINANCE (~0.5) so a relative-luminance cue
# can transfer; they differ in HUE (warm vs cool) and TEXTURE (speckle vs
# stripes), which is what a region-local CNN latches onto and fails to transfer.
def sim_regime(rng):
    return Regime("sim", _palette(rng, np.array([0.72, 0.52, 0.34])),   # warm fg, lum~0.53
                  _palette(rng, np.array([0.56, 0.50, 0.44])),          # warm-gray bg, lum~0.50
                  texture="speckle", tex_scale=1.0, noise=0.05, blur=0.0)


def real_regime(rng):
    return Regime("real", _palette(rng, np.array([0.34, 0.52, 0.72])),  # cool fg, lum~0.53
                  _palette(rng, np.array([0.44, 0.50, 0.56])),          # cool-gray bg, lum~0.50
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


def _apply_cue(fg, reg, rng, cue, cue_mode):
    """Encode the disambiguating cue into the foreground appearance.

    "transferable": the cue sets the foreground's brightness RELATIVE to its
        base colour, applied IDENTICALLY in sim and real. Because it rides on a
        relational property (fg-vs-bg contrast) rather than an absolute palette,
        it survives the sim->real shift -> a model that reads it generalises.
    "spurious": the cue tints the foreground (red channel) in SIM only; in REAL
        the tint is randomised. The cue is predictive in sim but carries no
        transferable information -> shape is the honest ceiling on real.
    "none": no cue (the ambiguous pair is then unsolvable by anyone).
    """
    fg = fg.astype(float).copy()
    if cue_mode in ("transferable", "spurious"):
        # Both modes encode the cue in foreground LUMINANCE (rides on the matched
        # background baseline, so it is readable the same way in both regimes).
        # The ONLY difference: spurious randomises the cue in the real regime, so
        # it predicts class in sim but carries no transferable information.
        eff = cue
        if cue_mode == "spurious" and reg.name == "real":
            eff = rng.random()
        fg *= (0.55 + 0.9 * eff)
    return np.clip(fg, 0, 1)


def render(mask: np.ndarray, reg: Regime, rng, cue=0.5, cue_mode="none") -> np.ndarray:
    """mask (H,W bool) -> RGB (H,W,3) in [0,1] under appearance regime `reg`."""
    size = mask.shape[0]
    fg = reg.fg_palette[rng.integers(len(reg.fg_palette))]
    bg = reg.bg_palette[rng.integers(len(reg.bg_palette))]
    fg = _apply_cue(fg, reg, rng, cue, cue_mode)
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

def build(n_per_class: int, size: int, regime_fn, seed: int, cue_mode="none"):
    """Return (imgs (N,H,W,3), masks (N,H,W) bool, labels (N,)) for a regime.

    `cue_mode` controls how the ambiguous disc pair is disambiguated:
    "transferable" | "spurious" | "none" (see `_apply_cue`).
    """
    rng = np.random.default_rng(seed)
    reg = regime_fn(rng)
    imgs, masks, labels = [], [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            m = make_mask(cls, size, rng)
            imgs.append(render(m, reg, rng, cue=CUE[cls], cue_mode=cue_mode))
            masks.append(m)
            labels.append(ci)
    idx = rng.permutation(len(labels))
    return (np.stack(imgs)[idx], np.stack(masks)[idx],
            np.asarray(labels)[idx])


# ----------------------------------------------------------------------
# Persistence + visual inspection
# ----------------------------------------------------------------------

def save_dataset(path, imgs, masks, labels):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, imgs=imgs.astype(np.float32),
                        masks=masks.astype(bool), labels=labels.astype(np.int64))


def load_dataset(path):
    d = np.load(path)
    return d["imgs"], d["masks"], d["labels"]


def build_cached(n_per_class, size, regime_fn, seed, cue_mode, path):
    """build(), but cached to `path` (.npz). Reuses the file if it exists."""
    from pathlib import Path
    if Path(path).exists():
        return load_dataset(path)
    data = build(n_per_class, size, regime_fn, seed, cue_mode=cue_mode)
    save_dataset(path, *data)
    return data


def save_preview(path, size=48, seed=7):
    """Grid PNG: one example per class, sim row vs real row, per cue mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    modes = ["transferable", "spurious", "none"]
    fig, axes = plt.subplots(len(modes) * 2, len(CLASSES),
                             figsize=(1.4 * len(CLASSES), 1.4 * len(modes) * 2))
    for mi, mode in enumerate(modes):
        for ri, regime_fn in enumerate((sim_regime, real_regime)):
            rng = np.random.default_rng(seed + ri)
            reg = regime_fn(rng)
            for ci, cls in enumerate(CLASSES):
                ax = axes[mi * 2 + ri, ci]
                m = make_mask(cls, size, rng)
                ax.imshow(render(m, reg, rng, cue=CUE[cls], cue_mode=mode))
                ax.set_xticks([]); ax.set_yticks([])
                if mi == 0 and ri == 0:
                    ax.set_title(cls, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"{mode[:5]}/{reg.name}", fontsize=7)
    fig.suptitle("gestalt testbed: rows = (cue-mode / regime), cols = classes", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
