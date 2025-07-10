"""Geometry utilities for electrons on a sphere."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "random_points_on_sphere",
    "move_on_sphere",
    "distance_on_sphere",
]

_rng = np.random.default_rng()


def random_points_on_sphere(num_points: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate *num_points* random points uniformly on the unit sphere.

    Returns
    -------
    np.ndarray
        Shape ``(num_points, 3)`` array with Cartesian coordinates.
    """

    rng = rng or _rng
    u = rng.random(num_points)
    v = rng.random(num_points)
    theta = 2 * math.pi * u  # azimuthal angle
    phi = np.arccos(2 * v - 1)  # polar angle
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T


def move_on_sphere(positions: np.ndarray, step_size: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Propose a random move for each electron by a small angular step.

    Parameters
    ----------
    positions
        Array of shape ``(N, 3)`` of current Cartesian coordinates on the unit sphere.
    step_size
        Angular step size in radians.
    rng
        Random generator to use.
    """

    rng = rng or _rng
    N = positions.shape[0]

    # Sample random small rotations via axis-angle formulation
    axes = random_points_on_sphere(N, rng)
    angles = rng.normal(scale=step_size, size=N)

    new_positions = np.empty_like(positions)
    for i, (axis, angle) in enumerate(zip(axes, angles)):
        axis = axis / np.linalg.norm(axis)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        ux, uy, uz = axis
        R = np.array(
            [
                [cos_a + ux * ux * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
                [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy * uy * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
                [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz * uz * (1 - cos_a)],
            ]
        )
        new_positions[i] = R @ positions[i]
        new_positions[i] /= np.linalg.norm(new_positions[i])  # renormalise
    return new_positions


def distance_on_sphere(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle distance between two points on unit sphere."""

    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.acos(dot) 