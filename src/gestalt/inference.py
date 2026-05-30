"""Recognition = energy minimisation over the structured latent z = (type, scale, phase).

This is the "first-class energy at inference" step: instead of a feedforward
classifier, we SEARCH the concept space for the explanation that minimises
E(x, z). Discrete part (type) by enumeration; continuous part (scale, rotation)
in closed form per type — rotation by circular cross-correlation (so recognition
is rotation-invariant by construction), scale by least squares.

Crucially it returns the FULL energy landscape, so multimodality is first-class:
when several concepts explain x nearly equally well (small `margin`, several
`modes`), the model reports the ambiguity instead of hiding it behind an argmax.
That is the "different modes able to explain the data" made operational.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .energy import prior_energy


@dataclass
class Inference:
    type: str                  # MAP concept (lowest-energy explanation)
    scale: float
    phase: float               # inferred rotation (radians)
    energy: float              # explanation residual of the MAP concept
    landscape: dict            # type -> energy (the full posterior shape)
    margin: float              # (E_runnerup - E_best)/E_best ; small => ambiguous
    modes: list                # all types within `tol` of the best (the modes)


def _fit_type(obs: np.ndarray, templ: np.ndarray):
    """Closed-form best (scale, rotation) of one template to the observation."""
    n = len(templ)
    # rotation: argmax circular cross-correlation (via FFT)
    cc = np.fft.irfft(np.fft.rfft(obs) * np.conj(np.fft.rfft(templ)), n)
    k = int(np.argmax(cc))
    t = np.roll(templ, k)
    a = float(np.dot(obs, t) / (np.dot(t, t) + 1e-9))      # least-squares scale
    e = float(np.mean((obs - a * t) ** 2))
    return e, a, k * 2 * np.pi / n


def infer(obs: np.ndarray, generator, lam: float = 0.0, tol: float = 0.15) -> Inference:
    """z* = argmin_z E(obs, z), plus the full energy landscape over concepts."""
    land, fits = {}, {}
    for ty, templ in generator.templates.items():
        e, a, ph = _fit_type(obs, templ)
        land[ty] = e + lam * prior_energy(a)
        fits[ty] = (a, ph)
    order = sorted(land, key=land.get)
    best, e0 = order[0], land[order[0]]
    margin = (land[order[1]] - e0) / (e0 + 1e-12)
    modes = [t for t in order if (land[t] - e0) / (e0 + 1e-12) <= tol]
    a, ph = fits[best]
    return Inference(best, a, ph, e0, land, margin, modes)


def aligned_energy(obs: np.ndarray, templ: np.ndarray, max_shift: int = 3):
    """Pixel-space explanation residual AFTER inferring the coordinate transform.

    The transform here is a 2D translation (the minimal "change of coordinates"
    a raster image needs); recognition in pixel space is brittle without it. We
    find the best small shift by circular cross-correlation (FFT) and return the
    residual there. Using ||obs - shift(templ)||^2 = ||obs||^2 + ||templ||^2 -
    2<obs, shift(templ)>, the residual reads straight off the correlation peak.
    """
    H, W = obs.shape
    cc = np.fft.ifft2(np.fft.fft2(obs) * np.conj(np.fft.fft2(templ))).real
    best, bd = -np.inf, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            v = cc[dy % H, dx % W]
            if v > best:
                best, bd = v, (dy, dx)
    e = float(np.sum(obs ** 2) + np.sum(templ ** 2) - 2 * best) / obs.size
    return e, bd[0], bd[1]


def synthesize(generator, inf: Inference) -> np.ndarray:
    """The generative direction: render the MAP concept back into a signature."""
    return generator.synthesize(inf.type, inf.scale, inf.phase)
