"""Example workflow for running a small molecular anyon simulation."""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from .analysis import binding_energy, finite_size_extrapolation
from .config import SimulationParameters, make_default_params
from .diagonalization import solve_generalised_eigen
from .matrix_elements import evaluate_coulomb_matrix, evaluate_overlap_matrix
from .plotting import plot_binding_energies, plot_energy_vs_invN
from .wavefunction import CompositeFermionWavefunction

__all__ = [
    "run_sample",
]


def run_sample():
    """Run a minimal demonstration for ν=2/5 and ν=3/7 with q=2..4."""

    # Simulation settings --------------------------------------------------
    fillings = [(2, 5), (3, 7)]
    qs = [2, 3, 4]
    sizes = [6, 8, 10]  # Electron numbers for finite-size scaling

    energies_infinite: Dict[Tuple[int, int], Dict[int, float]] = {}

    for filling in fillings:
        FF_label = f"{filling[0]}/{filling[1]}"
        energies_infinite[filling] = {}
        for q in qs:
            E_vs_N = []
            for N in sizes:
                params = make_default_params(N, filling, q, mc_samples=5_000)
                # Build a very small CF basis consisting only of one state (stub)
                state = CompositeFermionWavefunction(N, params.n, params.p, q)
                states = [state]
                O = evaluate_overlap_matrix(states, num_samples=params.mc_samples)
                V = evaluate_coulomb_matrix(states, num_samples=params.mc_samples)
                eigvals, eigvecs = solve_generalised_eigen(V, O)
                E0 = float(np.real(eigvals[0]))
                E_vs_N.append(E0)
            E_inf, slope, r2 = finite_size_extrapolation(sizes, E_vs_N)
            energies_infinite[filling][q] = E_inf
            print(f"ν={FF_label}, q={q}: E_inf={E_inf:.6f} (r2={r2:.3f})")

    # Binding energies ----------------------------------------------------
    for filling in fillings:
        FF_label = f"{filling[0]}/{filling[1]}"
        deltas = []
        for q in qs:
            E_q = energies_infinite[filling][q]
            E_qm1 = energies_infinite[filling].get(q - 1, 0.0)
            E_1 = energies_infinite[filling].get(1, 0.0)
            deltas.append(binding_energy(E_q, E_qm1, E_1))
        plot_binding_energies(qs, deltas)
        import matplotlib.pyplot as plt

        plt.title(f"Binding energies Δ_q for ν={FF_label}")
    plt.show() 