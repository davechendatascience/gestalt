"""The energy E(x, z) = how badly concept z explains observation x.

E = reconstruction(x, g(z)) + prior(z). Recognition minimises it; training (a
future step) shapes it so true (x, z) pairs sit in deep, well-separated basins.
Kept as plain functions so the same energy serves both inference and (later)
learning, and so the reconstruction / prior terms are independently swappable.
"""
from __future__ import annotations
import numpy as np


def reconstruction_energy(obs: np.ndarray, gen: np.ndarray) -> float:
    """Mean-squared explanation residual between observation and synthesis."""
    return float(np.mean((obs - gen) ** 2))


def prior_energy(scale: float) -> float:
    """Penalise implausible poses (log-scale away from 1). Cheap MDL-ish prior."""
    return float(np.log(max(scale, 1e-6)) ** 2)


def energy(obs: np.ndarray, gen: np.ndarray, scale: float = 1.0, lam: float = 0.0) -> float:
    return reconstruction_energy(obs, gen) + lam * prior_energy(scale)
