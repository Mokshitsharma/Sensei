# src/domain/export.py
import pandas as pd
from typing import Dict


def export_backtest(stats: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "Total Trades",
                "Win Rate (%)",
                "Return (%)",
                "Max Drawdown (%)",
            ],
            "Value": [
                stats["trades"],
                stats["win_rate"],
                stats["return_pct"],
                stats["max_drawdown"],
            ],
        }
    )
