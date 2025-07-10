"""Analysis utilities: finite-size extrapolation and binding energy computation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "finite_size_extrapolation",
    "binding_energy",
]


def finite_size_extrapolation(
    Ns: Sequence[int],
    energies: Sequence[float],
) -> Tuple[float, float, float]:
    """Linear fit of energy vs 1/N to extrapolate to N→∞.

    Returns intercept (thermodynamic limit), slope, and R².
    """

    x = 1.0 / np.asarray(list(Ns), dtype=float)  # type: ignore[arg-type]
    y = np.asarray(list(energies), dtype=float)  # type: ignore[arg-type]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = float(r_value) ** 2
    return float(intercept), float(slope), r2


def binding_energy(E_q: float, E_q_minus_1: float, E_1: float) -> float:
    """Return ∆_q = E(q) - (E(q-1) + E(1))."""

    return E_q - (E_q_minus_1 + E_1) 