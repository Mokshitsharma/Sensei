# src/data/providers/nse.py

import pandas as pd
from nsepython import equity_history

from src.data.providers.base import PriceProvider
from src.utils.data import sanitize_ohlcv


class NSEProvider(PriceProvider):
    """
    NSE data provider using nsepython.
    Supports daily historical data.
    """

    def fetch_daily_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Fetch last 1 year daily OHLCV from NSE.
        """

        df = equity_history(
            symbol=symbol.replace(".NS", ""),
            series="EQ",
            start_date=None,
            end_date=None,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(
            columns={
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "TOTTRDQTY": "volume",
                "CH_TIMESTAMP": "date",
            }
        )

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])

        return sanitize_ohlcv(df)

    def fetch_intraday_ohlcv(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Intraday not reliably supported for NSE (free).
        """
        return pd.DataFrame()
