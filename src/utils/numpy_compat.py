# src/utils/numpy_compat.py

"""
Lets pickles/cloudpickles saved under NumPy>=2.0 (which introduced the
internal `numpy._core` package, renamed from `numpy.core`) load correctly
under the project's pinned NumPy 1.26.4. Import before unpickling any
model artifact (PPO/joblib/torch) that may have been trained elsewhere.
"""

import sys
import importlib

import numpy as np

_SUBMODULES = [
    "numeric",
    "multiarray",
    "_multiarray_umath",
    "umath",
    "numerictypes",
    "fromnumeric",
    "_exceptions",
    "overrides",
    "_methods",
    "records",
    "arrayprint",
    "shape_base",
]


def patch() -> None:
    """Re-register the numpy._core alias. Idempotent; call right before
    unpickling anything that might reference numpy._core, since other
    imports (torch/sklearn/etc.) can reset sys.modules state in between."""
    if hasattr(np, "_core"):
        return
    sys.modules["numpy._core"] = np.core
    for sub in _SUBMODULES:
        try:
            sys.modules[f"numpy._core.{sub}"] = importlib.import_module(
                f"numpy.core.{sub}"
            )
        except ImportError:
            pass


patch()
