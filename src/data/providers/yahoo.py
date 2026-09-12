# src/data/providers/yahoo.py

import yfinance as yf
import pandas as pd

from src.data.providers.base import PriceProvider
from src.utils.data import sanitize_ohlcv


class YahooProvider(PriceProvider):

    def fetch_daily_ohlcv(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        df = yf.download(
            symbol,
            period=period,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        return sanitize_ohlcv(df)

    def fetch_intraday_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        if lookback_days <= 5:
            period = "5d"
        elif lookback_days <= 30:
            period = "1mo"
        elif lookback_days <= 60:
            period = "2mo"
        else:
            period = "5d"

        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = sanitize_ohlcv(df)
        return df.rename(columns={"datetime": "date"})
