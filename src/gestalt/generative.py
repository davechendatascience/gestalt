"""The generative model: g(z) -> observation.

This is the "symbolic generative process" the rest of the framework inverts.
A concept has a discrete TYPE (a silhouette family) and continuous pose params
(scale, rotation). Each type owns a canonical radial-signature template; g(z)
transforms it. Recognition (inference.py) inverts g by energy minimisation:
"which z best explains x?".

Deliberately small and swappable: today the decoder is a fixed template bank in
signature space (no training needed). The interface (`types`, `synthesize`) is
what a learned or RGB-rendering decoder would also expose, so heavier generative
models drop in without touching energy.py / inference.py.
"""
from __future__ import annotations
import numpy as np
from matplotlib.path import Path
from .synth import _verts

# Silhouette families. The testbed's discA/discB BOTH render as "disc" -> they
# are indistinguishable to a shape-only decoder: the built-in multimodality.
SILHOUETTES = ["triangle", "square", "pentagon", "hexagon", "star", "cross", "disc"]
N_ANGLES = 64


def _canonical_signature(name, n_angles=N_ANGLES, res=96):
    v = _verts("discA") if name == "disc" else _verts(name)
    v = v - v.mean(0)
    v = v / (np.abs(v).max() + 1e-9)
    poly = v * 0.42 * res + res / 2
    ys, xs = np.mgrid[0:res, 0:res]
    P = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    mask = Path(poly).contains_points(P).reshape(res, res)
    yy, xx = np.nonzero(mask)
    cx, cy = xx.mean(), yy.mean()
    ang = np.arctan2(yy - cy, xx - cx) % (2 * np.pi)
    rad = np.hypot(xx - cx, yy - cy)
    b = np.floor(ang / (2 * np.pi) * n_angles).astype(int) % n_angles
    sig = np.array([rad[b == k].max() if (b == k).any() else 0.0 for k in range(n_angles)])
    return sig / (sig.mean() + 1e-9)                 # scale-normalised template


class SignatureGenerator:
    """g(z): z = (type, scale, phase) -> radial signature in R^N_ANGLES."""

    def __init__(self):
        self.types = SILHOUETTES
        self.templates = {t: _canonical_signature(t) for t in SILHOUETTES}

    def synthesize(self, type: str, scale: float = 1.0, phase: float = 0.0):
        t = self.templates[type]
        k = int(round(phase / (2 * np.pi) * len(t))) % len(t)
        return scale * np.roll(t, k)
