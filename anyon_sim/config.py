"""Global configuration and physical constants used across the anyon_sim package."""

from __future__ import annotations

import dataclasses as _dc
import math
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Physical constants (in cgs unless otherwise noted)
# -----------------------------------------------------------------------------

#: Elementary charge (statC)
e = 4.80320427e-10  # noqa: E741 naming e for charge

#: Reduced Planck constant (erg*s)
hbar = 1.054571817e-27

#: Coulomb constant (erg*cm/e^2)
k_e = 1.0  # In Gaussian units 1 / (4*pi*epsilon_0) = 1

#: Magnetic flux quantum (Gauss*cm^2)
phi_0 = 2.067833848e-7

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

@_dc.dataclass
class SimulationParameters:
    """Container for global simulation parameters.

    These parameters can be serialised to/from JSON for convenience.
    """

    N: int  # Number of electrons
    n: int  # Jain integer n (for filling factor)
    p: int  # Jain integer p (for flux attachment)
    q: int  # Number of quasiparticles / quasiholes

    # Monte Carlo settings
    mc_samples: int = 10_000
    mc_burn_in: int = 1_000
    mc_thin: int = 10
    step_size: float = 0.2  # Move size in angular coordinates (rad)

    seed: int | None = None

    @property
    def filling_factor(self) -> float:
        """Return the Jain filling factor ν = n /(2 p n ± 1).

        The sign convention follows the positive *p* branch (even-denominator
        states). Use negative *p* for the other branch.
        """

        return self.n / (2 * self.p * self.n + 1)

    @property
    def flux_quanta(self) -> int:
        """Return the number of flux quanta 2S through the sphere.

        For CF construction on a sphere, the flux 2S is related to system size
        via the relation specific to the filling factor and quasiparticle
        number. Here we expose it as a configurable attribute; the user is
        expected to supply consistent values when building wavefunctions.
        """

        # Placeholder; the precise relation depends on q and shift.
        return int(math.ceil(1 / self.filling_factor * self.N))


# Convenience factory for default parameters for common filling factors

def make_default_params(N: int, filling: Tuple[int, int], q: int, *, mc_samples: int = 20_000) -> SimulationParameters:
    """Helper to quickly build :class:`SimulationParameters` from ν = num/den."""

    num, den = filling
    # Solve Jain relation den = 2pn ± 1; we assume positive branch
    n = num
    p = (den - 1) // (2 * n)
    return SimulationParameters(N=N, n=n, p=p, q=q, mc_samples=mc_samples) 