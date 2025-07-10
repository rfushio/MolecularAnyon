"""Metropolis–Hastings Monte Carlo integrator for many-body wavefunctions."""

from __future__ import annotations

import dataclasses as _dc
import math
from typing import Callable, Iterable, Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

from .geometry import move_on_sphere, random_points_on_sphere
from .wavefunction import Wavefunction

__all__ = [
    "MetropolisSampler",
    "estimate_expectation",
]


@_dc.dataclass
class MetropolisSampler:
    """Simple Metropolis–Hastings sampler in configuration space."""

    psi: Wavefunction  # Wavefunction providing amplitude evaluations
    step_size: float = 0.2  # Angular step size in radians
    rng: np.random.Generator = _dc.field(default_factory=np.random.default_rng)

    # Internal state -------------------------------------------------------
    _positions: NDArray | None = None
    _prob_current: float | None = None
    _accepted: int = 0
    _attempted: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def equilibrate(self, burn_in: int = 1_000) -> None:
        """Run burn-in sweeps to reach equilibrium."""

        if self._positions is None:
            # Initialise positions uniformly on sphere
            self._positions = random_points_on_sphere(self.psi.N, rng=self.rng)
            self._prob_current = self.psi.probability(self._positions)

        for _ in range(burn_in):
            self._sweep()

    def sample(self, num_samples: int, thin: int = 10) -> NDArray:
        """Return an array of sampled configurations.

        Parameters
        ----------
        num_samples
            Number of Monte-Carlo samples to draw.
        thin
            Number of sweeps between recorded samples.
        """

        if self._positions is None:
            self.equilibrate(1000)

        assert self._positions is not None  # For type checker
        samples = []
        for _ in range(num_samples):
            for _ in range(thin):
                self._sweep()
            samples.append(self._positions.copy())
        return np.array(samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sweep(self) -> None:
        """Attempt one global move of all electrons."""

        assert self._positions is not None and self._prob_current is not None
        new_pos = move_on_sphere(self._positions, self.step_size, rng=self.rng)
        prob_new = self.psi.probability(new_pos)
        accept_ratio = prob_new / self._prob_current

        self._attempted += 1
        if accept_ratio >= 1 or self.rng.random() < accept_ratio:
            self._positions = new_pos
            self._prob_current = prob_new
            self._accepted += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self._attempted == 0 else self._accepted / self._attempted


# ---------------------------------------------------------------------------
# Expectation value estimation utilities
# ---------------------------------------------------------------------------

def estimate_expectation(
    sampler: MetropolisSampler,
    operator: Callable[[NDArray], float | complex],
    num_samples: int,
    thin: int = 10,
) -> Tuple[complex, float]:
    """Estimate ⟨Ψ|Ô|Ψ⟩ / ⟨Ψ|Ψ⟩ via Monte Carlo.

    Returns the estimate and its standard error using simple re-sampling.
    """

    configs = sampler.sample(num_samples, thin=thin)
    values = np.empty(num_samples, dtype=np.complex128)
    for i, cfg in enumerate(configs):
        values[i] = operator(cfg)
    mean = values.mean()
    stderr = values.std(ddof=1) / math.sqrt(len(values))
    return mean, stderr 