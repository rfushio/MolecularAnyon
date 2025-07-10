"""Generalised eigenvalue problem solver for CF diagonalisation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import linalg as sla

__all__ = [
    "solve_generalised_eigen",
]


def solve_generalised_eigen(V: np.ndarray, O: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Vc x = E O x and return eigenvalues and eigenvectors sorted ascending."""

    # Ensure Hermiticity symmetry numerically
    V = 0.5 * (V + V.conj().T)
    O = 0.5 * (O + O.conj().T)

    eigvals, eigvecs = sla.eigh(V, O, check_finite=False)
    sort_idx = np.argsort(eigvals.real)
    return eigvals[sort_idx], eigvecs[:, sort_idx] 