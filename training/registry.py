# training/registry.py

"""
Model registry — tracks which stocks have fully trained model files.

Per-stock models live under:
    models/{SYMBOL_DIR}/lstm.pt
    models/{SYMBOL_DIR}/tcn.pt
    models/{SYMBOL_DIR}/ppo.zip
    models/{SYMBOL_DIR}/ml.joblib

where SYMBOL_DIR is the Yahoo Finance symbol with '.' replaced by '_'
(e.g. HDFCBANK.NS → HDFCBANK_NS).

Generic fallback models (trained on HDFCBANK.NS) exist at:
    models/lstm_HDFCBANK_NS.pt
    models/tcn_HDFCBANK_NS.pt
    models/ppo_hdfc.zip
    models/ml_return_model.joblib
"""

import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")


class ModelPaths(NamedTuple):
    """File paths for the four model types associated with a symbol."""

    lstm: str
    tcn: str
    ppo: str
    ml: str
    is_stock_specific: bool  # True if per-stock model, False if generic fallback


# ---------------------------------------------------------------------------
# Generic fallback paths (always present after first-run training)
# ---------------------------------------------------------------------------

GENERIC_PATHS = ModelPaths(
    lstm="models/lstm_HDFCBANK_NS.pt",
    tcn="models/tcn_HDFCBANK_NS.pt",
    ppo="models/ppo_hdfc.zip",
    ml="models/ml_return_model.joblib",
    is_stock_specific=False,
)

# Expected file names within each per-stock directory
_PER_STOCK_FILES = {
    "lstm": "lstm.pt",
    "tcn": "tcn.pt",
    "ppo": "ppo.zip",
    "ml": "ml.joblib",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def symbol_to_dir(symbol: str) -> str:
    """
    Convert a Yahoo Finance symbol to a directory-safe name.

    Examples:
        ``HDFCBANK.NS``  →  ``HDFCBANK_NS``
        ``BAJAJ-AUTO.NS`` →  ``BAJAJ-AUTO_NS``
    """
    return symbol.replace(".", "_")


def _stock_dir(symbol: str) -> Path:
    """Return the Path of the per-stock model directory."""
    return MODELS_DIR / symbol_to_dir(symbol)


def _per_stock_paths(symbol: str) -> ModelPaths:
    """Build ModelPaths pointing to per-stock files (without checking existence)."""
    d = str(_stock_dir(symbol))
    return ModelPaths(
        lstm=os.path.join(d, _PER_STOCK_FILES["lstm"]),
        tcn=os.path.join(d, _PER_STOCK_FILES["tcn"]),
        ppo=os.path.join(d, _PER_STOCK_FILES["ppo"]),
        ml=os.path.join(d, _PER_STOCK_FILES["ml"]),
        is_stock_specific=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_paths(symbol: str) -> ModelPaths:
    """
    Return the appropriate :class:`ModelPaths` for a symbol.

    If all four per-stock model files exist, returns stock-specific paths.
    Otherwise returns :data:`GENERIC_PATHS`.

    Args:
        symbol: Yahoo Finance ticker, e.g. ``"HDFCBANK.NS"``.

    Returns:
        :class:`ModelPaths` — either stock-specific or the generic fallback.
    """
    paths = _per_stock_paths(symbol)
    if all(Path(p).exists() for p in [paths.lstm, paths.tcn, paths.ppo, paths.ml]):
        return paths
    return GENERIC_PATHS


def is_trained(symbol: str) -> bool:
    """
    Return ``True`` if all four model files exist for the given symbol.

    Args:
        symbol: Yahoo Finance ticker, e.g. ``"RELIANCE.NS"``.
    """
    paths = _per_stock_paths(symbol)
    return all(
        Path(p).exists()
        for p in [paths.lstm, paths.tcn, paths.ppo, paths.ml]
    )


def list_trained_symbols() -> List[str]:
    """
    Scan ``models/`` and return every symbol with a complete set of models.

    Returns:
        Sorted list of symbol strings (dot notation, e.g. ``"HDFCBANK.NS"``).
    """
    trained: List[str] = []

    if not MODELS_DIR.exists():
        return trained

    for entry in MODELS_DIR.iterdir():
        if not entry.is_dir():
            continue
        # Reconstruct symbol from directory name
        dir_name = entry.name
        symbol = dir_name.replace("_NS", ".NS")  # reverse the underscore swap

        # Verify all four files exist
        all_present = all(
            (entry / fname).exists()
            for fname in _PER_STOCK_FILES.values()
        )
        if all_present:
            trained.append(symbol)

    return sorted(trained)


def get_registry_status() -> Dict[str, dict]:
    """
    Return a status dictionary for all trained symbols plus the generic fallback.

    Returns:
        ``{symbol: {"is_trained": bool, "paths": {...}}}``

    Example::

        {
            "HDFCBANK.NS": {
                "is_trained": True,
                "paths": {
                    "lstm": "models/HDFCBANK_NS/lstm.pt",
                    ...
                    "is_stock_specific": True,
                }
            },
            "__generic__": {
                "is_trained": True,
                "paths": {
                    "lstm": "models/lstm_HDFCBANK_NS.pt",
                    ...
                    "is_stock_specific": False,
                }
            }
        }
    """
    status: Dict[str, dict] = {}

    # Per-stock entries
    for symbol in list_trained_symbols():
        paths = _per_stock_paths(symbol)
        status[symbol] = {
            "is_trained": True,
            "paths": paths._asdict(),
        }

    # Generic fallback
    generic_trained = all(
        Path(p).exists()
        for p in [
            GENERIC_PATHS.lstm,
            GENERIC_PATHS.tcn,
            GENERIC_PATHS.ppo,
            GENERIC_PATHS.ml,
        ]
    )
    status["__generic__"] = {
        "is_trained": generic_trained,
        "paths": GENERIC_PATHS._asdict(),
    }

    return status
