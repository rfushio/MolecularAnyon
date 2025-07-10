"""Plotting helpers using matplotlib."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analysis import finite_size_extrapolation

__all__ = [
    "plot_energy_vs_invN",
    "plot_binding_energies",
]



def plot_energy_vs_invN(Ns: Sequence[int], energies: Sequence[float], *, ax=None, label: str | None = None):
    """Scatter energy vs 1/N with linear fit extrapolation."""

    x = 1.0 / np.asarray(list(Ns), dtype=float)
    y = np.asarray(list(energies), dtype=float)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(x, y, marker="o", label=label or "data")

    intercept, slope, r2 = finite_size_extrapolation(Ns, energies)
    x_fit = np.linspace(0, max(x) * 1.05, 100)
    y_fit = intercept + slope * x_fit
    ax.plot(x_fit, y_fit, "--", label=f"fit (r^2={r2:.3f})")

    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel("Energy")
    ax.legend()
    return fig, ax


def plot_binding_energies(qs: Sequence[int], deltas: Sequence[float], *, ax=None):
    """Bar plot of binding energies ∆_q."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.bar(qs, deltas, width=0.6)
    ax.set_xlabel("q (number of quasiparticles)")
    ax.set_ylabel(r"$\Delta_q$")
    ax.axhline(0, color="k", lw=0.8)
    return fig, ax 