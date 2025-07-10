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

# Lazy re-export of high-level helper -------------------------------------------------

# Importing `anyon_sim.examples` at *package import time* triggers a Python runtime
# warning (module already in sys.modules) when the same submodule is executed via
# `python -m anyon_sim.examples`.  To avoid this, we provide a **lazy loader** that
# resolves the implementation only when the helper is first called.

from types import ModuleType as _ModuleType
from typing import Any as _Any


def _lazy_run_sample(*args: _Any, **kwargs: _Any):  # type: ignore
    from importlib import import_module as _import_module

    _examples: _ModuleType = _import_module(".examples", package=__name__)
    _actual = getattr(_examples, "run_sample")
    globals()["run_sample"] = _actual  # Cache for subsequent calls
    return _actual(*args, **kwargs)


# Expose under the expected name
run_sample = _lazy_run_sample  # type: ignore 