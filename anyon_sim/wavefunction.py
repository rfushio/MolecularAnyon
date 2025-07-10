"""Wavefunction definitions for composite fermion (CF) states on a sphere.

Note
----
This module provides *placeholder* implementations that reproduce only the
qualitative structure of CF states. Implementing fully quantitative Jain–Kamilla
projected wavefunctions requires substantial additional code that depends on
high-precision special functions and is beyond the scope of this scaffold.
Nevertheless, the interfaces defined here should cover most use-cases; the
*evaluate* methods can later be replaced by more sophisticated versions (see
Refs. in the accompanying README).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import Protocol

import numpy as np

from .geometry import distance_on_sphere

__all__ = [
    "Wavefunction",
    "LaughlinWavefunction",
    "CompositeFermionWavefunction",
]


class Wavefunction(ABC):
    """Abstract base class for many-body wavefunctions."""

    N: int  # Number of particles in the state

    def __init__(self, N: int):
        self.N = N

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @abstractmethod
    def amplitude(self, positions: np.ndarray) -> complex:
        """Return Ψ(r₁,…,r_N) for a given configuration.

        Parameters
        ----------
        positions
            Shape ``(N, 3)`` Cartesian coordinates on the unit sphere.
        """

    # Convenience wrappers -------------------------------------------------

    def probability(self, positions: np.ndarray) -> float:
        """Return |Ψ|², i.e. the probability density up to normalization."""

        amp = self.amplitude(positions)
        return float(np.abs(amp) ** 2)


# ---------------------------------------------------------------------------
# Laughlin state as a reference implementation.
# ---------------------------------------------------------------------------


class LaughlinWavefunction(Wavefunction):
    """Laughlin state at filling ν = 1/m in spherical coordinates.

    The simplest version uses the form

    .. math::
        Ψ_M = \prod_{i<j}(u_i v_j - u_j v_i)^m

    where :math:`u = \cos(θ/2) \, e^{i φ/2}`, :math:`v = \sin(θ/2) \, e^{-i φ/2}`
    are spinor coordinates on the sphere. Implementing this exactly is involved;
    here we adopt a crude placeholder using inter-particle chord distances to
    mimic Jastrow correlations.
    """

    def __init__(self, N: int, m: int):
        super().__init__(N)
        self.m = m

    def amplitude(self, positions: np.ndarray) -> complex:  # noqa: D401
        if positions.shape != (self.N, 3):
            raise ValueError("positions must have shape (N,3)")
        amp_log = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Chord distance as a proxy for interparticle separation
                r = distance_on_sphere(positions[i], positions[j])
                amp_log += self.m * np.log(np.sin(r / 2) + 1e-12)
        return np.exp(amp_log)


# ---------------------------------------------------------------------------
# Composite fermion (CF) wavefunction (stub)
# ---------------------------------------------------------------------------


class CompositeFermionWavefunction(Wavefunction):
    """Placeholder for Jain composite-fermion wavefunctions.

    Actual evaluation of CF wavefunctions with Jain–Kamilla projection is
    considerably more complex; the present class only returns a Laughlin-like
    proxy that preserves the nodal structure relevant for Monte-Carlo estimates.
    """

    def __init__(
        self,
        N: int,
        n: int,
        p: int,
        q: int = 0,
        *,
        seed: int | None = None,
    ):
        super().__init__(N)
        self.n = n
        self.p = p
        self.q = q
        self._rng = np.random.default_rng(seed)

        # Effective exponent controlling Jastrow factor strength
        self._m_eff = 2 * p + 1

    # ------------------------------------------------------------------
    # Naive amplitude implementation
    # ------------------------------------------------------------------

    @cache
    def amplitude(self, positions: np.ndarray) -> complex:  # noqa: D401
        """Return ψ_CF using a simplified Jastrow-only model.

        This is **not** quantitatively correct but is sufficient for code flow
        development and testing. Replace with a full implementation for
        production runs.
        """

        if positions.shape != (self.N, 3):
            raise ValueError("positions must have shape (N,3)")
        amp_log = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = distance_on_sphere(positions[i], positions[j])
                amp_log += self._m_eff * np.log(np.sin(r / 2) + 1e-12)
        return np.exp(amp_log) 