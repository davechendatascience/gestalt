"""Appearance-invariant shape descriptors computed on a binary mask.

Pure numpy/scipy (no skimage/cv2). Every descriptor here is a function of the
SILHOUETTE only — by construction it cannot see texture, colour, lighting, or
noise, so it is identical for a shape rendered in the sim or real regime. This
is the "control C" representation: how much of the task is solvable from
appearance-blind structure, and (trivially) it transfers with zero gap.

Descriptors:
  - Hu moment invariants (7, log-scaled): translation/scale/rotation invariant.
  - circularity (4*pi*A / P^2), eccentricity, extent (A / bbox), rel. area.
  - centroid-distance Fourier signature magnitudes (rotation/scale invariant):
    all CLASSES here are star-shaped about their centroid, so r(theta) is well
    defined and its FFT magnitude spectrum cleanly encodes #sides / symmetry.
"""
from __future__ import annotations
import numpy as np

N_ANGLES = 64
N_HARM = 12


def _moments(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x = xs.astype(float); y = ys.astype(float)
    m00 = len(xs)
    xc = x.mean(); yc = y.mean()
    xm = x - xc; ym = y - yc

    def mu(p, q):
        return np.sum((xm ** p) * (ym ** q))

    def nu(p, q):
        return mu(p, q) / (m00 ** (1 + (p + q) / 2.0))

    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n12, n21, n03 = nu(3, 0), nu(1, 2), nu(2, 1), nu(0, 3)
    h = np.empty(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03))
    h[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu = -np.sign(h) * np.log10(np.abs(h) + 1e-30)         # log-scaled, sign-preserving
    # second-moment eccentricity
    cov = np.array([[mu(2, 0), mu(1, 1)], [mu(1, 1), mu(0, 2)]]) / m00
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-9, None)
    ecc = np.sqrt(1 - ev.min() / ev.max())
    return hu, m00, (xc, yc), ecc


def _radial_signature(mask, xc, yc):
    """r(theta): max in-mask radius along each of N_ANGLES directions."""
    ys, xs = np.nonzero(mask)
    dx = xs - xc; dy = ys - yc
    ang = np.arctan2(dy, dx) % (2 * np.pi)
    rad = np.hypot(dx, dy)
    bins = np.floor(ang / (2 * np.pi) * N_ANGLES).astype(int) % N_ANGLES
    sig = np.zeros(N_ANGLES)
    for b in range(N_ANGLES):
        r = rad[bins == b]
        sig[b] = r.max() if len(r) else 0.0
    # fill empty angular bins by interpolation (robustness)
    if (sig == 0).any():
        good = sig > 0
        if good.sum() >= 2:
            idx = np.arange(N_ANGLES)
            sig[~good] = np.interp(idx[~good], idx[good], sig[good], period=N_ANGLES)
    return sig


def radial_signature(mask: np.ndarray) -> np.ndarray:
    """Public: centroid-distance signature r(theta) of a mask (N_ANGLES,).

    The observation fed to analysis-by-synthesis inference. Appearance-blind
    (a function of the silhouette only)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(N_ANGLES)
    return _radial_signature(mask, xs.mean(), ys.mean())


def _perimeter(mask):
    # boundary pixels: in-mask pixels adjacent to a background pixel (4-neigh)
    m = mask
    b = np.zeros_like(m)
    b[:-1] |= m[:-1] & ~m[1:]
    b[1:] |= m[1:] & ~m[:-1]
    b[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    b[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    return float(b.sum())


def descriptor(mask: np.ndarray) -> np.ndarray:
    """Mask -> 1D appearance-invariant feature vector."""
    mo = _moments(mask)
    size = mask.shape[0]
    if mo is None:
        return np.zeros(7 + 4 + N_HARM, dtype=np.float64)
    hu, area, (xc, yc), ecc = mo
    perim = _perimeter(mask) + 1e-9
    circ = 4 * np.pi * area / (perim ** 2)
    ys, xs = np.nonzero(mask)
    bbox = (np.ptp(xs) + 1) * (np.ptp(ys) + 1)
    extent = area / (bbox + 1e-9)
    rel_area = area / (size * size)

    sig = _radial_signature(mask, xc, yc)
    mag = np.abs(np.fft.rfft(sig))
    mag = mag[1:1 + N_HARM] / (mag[0] + 1e-9)              # scale-invariant, drop DC

    return np.concatenate([hu, [circ, ecc, extent, rel_area], mag]).astype(np.float64)


def descriptor_matrix(masks: np.ndarray) -> np.ndarray:
    return np.stack([descriptor(m) for m in masks])


def relational_descriptor(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Appearance read RELATIONALLY: foreground statistics measured RELATIVE to
    the background, not in absolute palette terms.

    This is the simplest stand-in for the "relation" leg of the triangle. A
    transferable cue (fg-vs-bg contrast) shows up here and survives the regime
    shift; an absolute-palette cue does not. Compare against a region-local CNN
    that keys on absolute texture and fails to isolate the relational signal.
    """
    fg = img[mask]
    bg = img[~mask]
    if len(fg) == 0 or len(bg) == 0:
        return np.zeros(5, dtype=np.float64)
    contrast = float(fg.mean() - bg.mean())            # relative luminance (transferable cue)
    color_rel = (fg.mean(0) - bg.mean(0)).astype(float)  # per-channel relative (3,)
    fg_spread = float(fg.std())                        # internal texture energy
    return np.concatenate([[contrast], color_rel, [fg_spread]]).astype(np.float64)


def relational_matrix(imgs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.stack([relational_descriptor(i, m) for i, m in zip(imgs, masks)])


RELATIONAL_NAMES = ["contrast", "rel_R", "rel_G", "rel_B", "fg_spread"]


FEATURE_NAMES = (
    [f"hu{i}" for i in range(7)]
    + ["circularity", "eccentricity", "extent", "rel_area"]
    + [f"fft{k}" for k in range(1, N_HARM + 1)]
)
