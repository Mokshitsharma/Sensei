# src/domain/export.py

import json
from pathlib import Path
from typing import Dict

import pandas as pd


EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


def export_equity_curve(
    equity: pd.Series,
    filename: str,
) -> Path:
    """
    Export equity curve to CSV.
    """
    path = EXPORT_DIR / f"{filename}_equity.csv"
    equity.rename("equity").to_csv(path)
    return path


def export_metrics(
    metrics: Dict[str, float],
    filename: str,
) -> Path:
    """
    Export backtest metrics to JSON.
    """
    path = EXPORT_DIR / f"{filename}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def export_signal_snapshot(
    signal_data: Dict[str, object],
    filename: str,
) -> Path:
    """
    Export model + decision snapshot for a single date.
    """
    path = EXPORT_DIR / f"{filename}_signal.json"
    with open(path, "w") as f:
        json.dump(signal_data, f, indent=2, default=str)
    return path


def export_full_run(
    equity: pd.Series,
    metrics: Dict[str, float],
    signal_data: Dict[str, object],
    run_name: str,
) -> Dict[str, Path]:
    """
    Export everything from a single experiment run.
    """
    return {
        "equity": export_equity_curve(equity, run_name),
        "metrics": export_metrics(metrics, run_name),
        "signal": export_signal_snapshot(signal_data, run_name),
    }