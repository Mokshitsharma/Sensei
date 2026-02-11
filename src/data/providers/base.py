# src/data/providers/base.py

import pandas as pd
from abc import ABC, abstractmethod


class PriceProvider(ABC):
    """
    Abstract base class for price data providers.
    """

    @abstractmethod
    def fetch_daily_ohlcv(self, symbol: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_intraday_ohlcv(self, symbol: str, interval: str) -> pd.DataFrame:
        pass
