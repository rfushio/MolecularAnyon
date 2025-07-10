"""Matrix element evaluation for CF basis states using Monte Carlo sampling."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .monte_carlo import MetropolisSampler, estimate_expectation
from .wavefunction import Wavefunction

__all__ = [
    "evaluate_overlap_matrix",
    "evaluate_coulomb_matrix",
]


# ---------------------------------------------------------------------------
# Helper: Coulomb operator on the sphere
# ---------------------------------------------------------------------------


def coulomb_potential(positions: np.ndarray) -> float:
    """Return Coulomb energy for a configuration on unit sphere.

    For simplicity we use the pairwise inverse chord distance. The physical unit
    prefactor is absorbed elsewhere.
    """

    N = positions.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            energy += 1.0 / (r + 1e-12)
    return float(energy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_overlap_matrix(states: Sequence[Wavefunction], num_samples: int = 5_000) -> np.ndarray:
    """Return the Hermitian overlap matrix O_{αβ}."""

    n = len(states)
    O = np.empty((n, n), dtype=np.complex128)

    # Reuse samplers keyed by state to exploit caching of wavefunction amplitude
    samplers = [MetropolisSampler(psi=s) for s in states]

    for i in range(n):
        # Diagonal element ≡ 1 by definition if states are normalised, but we estimate for consistency
        O[i, i] = 1.0
        for j in range(i + 1, n):
            psi_i, psi_j = states[i], states[j]

            def integrand(cfg: np.ndarray, psi_i=psi_i, psi_j=psi_j):  # noqa: B023
                return np.conjugate(psi_i.amplitude(cfg)) * psi_j.amplitude(cfg)

            est, err = estimate_expectation(samplers[i], integrand, num_samples)
            O[i, j] = est
            O[j, i] = np.conjugate(est)
    return O


def evaluate_coulomb_matrix(states: Sequence[Wavefunction], num_samples: int = 5_000) -> np.ndarray:
    """Return Coulomb interaction matrix V_{αβ}."""

    n = len(states)
    V = np.empty((n, n), dtype=np.complex128)

    samplers = [MetropolisSampler(psi=s) for s in states]

    for i in range(n):
        for j in range(i, n):
            psi_i, psi_j = states[i], states[j]

            def integrand(cfg: np.ndarray, psi_i=psi_i, psi_j=psi_j):  # noqa: B023
                return (
                    np.conjugate(psi_i.amplitude(cfg))
                    * psi_j.amplitude(cfg)
                    * coulomb_potential(cfg)
                )

            est, err = estimate_expectation(samplers[i], integrand, num_samples)
            V[i, j] = est
            V[j, i] = np.conjugate(est)
    return V 