"""Rotation-invariant 3D shape signatures -- compute the invariant directly,
never inferring the pose (the efficient path: quotient SO(3) analytically).

  d2_descriptor          : histogram of pairwise surface-point distances
                           (Osada et al.). Invariant to rotation AND translation
                           by construction -- distances are isometry invariants.

  sh_power_descriptor    : spherical-harmonic POWER SPECTRUM of the object's
                           occupancy on radial shells (Kazhdan et al.). The
                           per-degree energy sum_m |a_lm|^2 is SO(3)-invariant
                           because each degree-l subspace is rotation-invariant
                           (Peter-Weyl) -- the harmonic-analysis invariant.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates

try:                                          # scipy >= 1.15
    from scipy.special import sph_harm_y
    def _Ylm(l, m, polar, azim):
        return sph_harm_y(l, m, polar, azim)
except ImportError:                           # older scipy
    from scipy.special import sph_harm
    def _Ylm(l, m, polar, azim):
        return sph_harm(m, l, azim, polar)


def d2_descriptor(points, rng, n_pairs=40000, bins=48, dmax=2.0):
    n = len(points)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    d = np.linalg.norm(points[i] - points[j], axis=1)
    h, _ = np.histogram(d, bins=bins, range=(0.0, dmax), density=True)
    return h / (np.linalg.norm(h) + 1e-9)


def _voxelize(points, D=32):
    idx = np.clip(((points + 1) / 2 * D).astype(int), 0, D - 1)
    occ = np.zeros((D, D, D), np.float32)
    np.add.at(occ, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return occ


def sh_power_descriptor(points, D=32, n_shell=8, Lmax=10, ntheta=30, nphi=60):
    occ = _voxelize(points, D)
    polar = np.linspace(0.03, np.pi - 0.03, ntheta)
    azim = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(polar, azim, indexing="ij")          # (ntheta, nphi)
    w = np.sin(TH) * (polar[1] - polar[0]) * (azim[1] - azim[0])
    Ys = {(l, m): _Ylm(l, m, TH, PH)
          for l in range(Lmax + 1) for m in range(-l, l + 1)}
    desc = []
    for r in np.linspace(0.12, 0.95, n_shell):
        x = r * np.sin(TH) * np.cos(PH); y = r * np.sin(TH) * np.sin(PH); z = r * np.cos(TH)
        c = [((x + 1) / 2 * D).ravel(), ((y + 1) / 2 * D).ravel(), ((z + 1) / 2 * D).ravel()]
        f = map_coordinates(occ, c, order=1, mode="constant").reshape(TH.shape)
        for l in range(Lmax + 1):
            P = sum(abs(np.sum(f * np.conj(Ys[(l, m)]) * w)) ** 2 for m in range(-l, l + 1))
            desc.append(P)
    d = np.asarray(desc, float)
    return d / (np.linalg.norm(d) + 1e-9)
