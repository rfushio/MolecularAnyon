"""Anyon simulation package for fractional quantum Hall molecular anyons.

This package provides modules for constructing composite-fermion (CF) wavefunctions,
performing Monte-Carlo evaluation of matrix elements, solving the resulting
generalised eigenvalue problems, and analysing binding energies of molecular
anyons on a spherical geometry.

Typical usage (see notebooks folder)::

    from anyon_sim.examples import run_sample
    run_sample()
"""

from importlib.metadata import version as _version

try:
    __version__: str = _version(__name__)
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Re-export most used high-level helpers
try:
    from .examples import run_sample  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    # The submodule may not yet be built when linting.
    def run_sample(*args, **kwargs):  # type: ignore
        raise RuntimeError("examples module not available yet") 