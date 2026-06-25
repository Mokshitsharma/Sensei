# src/data/providers/mock.py
"""
Synthetic OHLCV data provider for offline/testing environments
where Yahoo Finance is unavailable (e.g. restricted cloud environments).

Generates a realistic random-walk price series seeded on the ticker symbol
so results are deterministic and repeatable per symbol.
"""

import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.providers.base import PriceProvider


def _seed_for(symbol: str) -> int:
    return int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)


class MockProvider(PriceProvider):
    """Returns synthetic but statistically realistic OHLCV data."""

    # Approximate starting prices for well-known tickers
    _BASE_PRICES = {
        "HDFCBANK.NS": 1680, "RELIANCE.NS": 2890, "TCS.NS": 3950,
        "INFY.NS": 1780, "ICICIBANK.NS": 1230, "SBIN.NS": 820,
        "BHARTIARTL.NS": 1740, "ITC.NS": 470, "KOTAKBANK.NS": 1880,
        "LT.NS": 3700, "WIPRO.NS": 560, "HCLTECH.NS": 1640,
        "AXISBANK.NS": 1240, "BAJFINANCE.NS": 7100, "SUNPHARMA.NS": 1780,
        "TITAN.NS": 3500, "MARUTI.NS": 13200, "NESTLEIND.NS": 2380,
        "ULTRACEMCO.NS": 11500, "NTPC.NS": 370,
    }

    def _generate(self, symbol: str, n_days: int, base_price: float) -> pd.DataFrame:
        rng = np.random.default_rng(_seed_for(symbol))

        # Daily log returns: slight upward drift, realistic volatility
        annual_vol = rng.uniform(0.18, 0.35)
        daily_vol = annual_vol / np.sqrt(252)
        daily_drift = rng.uniform(-0.0002, 0.0008)

        log_returns = rng.normal(daily_drift, daily_vol, n_days)
        closes = base_price * np.exp(np.cumsum(log_returns))

        # OHLC from close
        intraday_vol = daily_vol * 0.6
        opens  = closes * np.exp(rng.normal(0, intraday_vol * 0.3, n_days))
        highs  = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, intraday_vol, n_days)))
        lows   = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, intraday_vol, n_days)))

        # Volume: log-normal, correlated with volatility
        avg_vol = rng.uniform(1_000_000, 50_000_000)
        volumes = rng.lognormal(np.log(avg_vol), 0.5, n_days).astype(int)

        # Build date index (business days)
        end = datetime.today()
        dates = pd.bdate_range(end=end, periods=n_days)

        df = pd.DataFrame({
            "date": dates,
            "open":   np.round(opens, 2),
            "high":   np.round(highs, 2),
            "low":    np.round(lows, 2),
            "close":  np.round(closes, 2),
            "volume": volumes,
        })
        df = df.set_index("date")
        df.index.name = "date"
        return df

    def fetch_daily_ohlcv(self, symbol: str) -> pd.DataFrame:
        base = self._BASE_PRICES.get(symbol, 1000.0)
        return self._generate(symbol, n_days=365, base_price=base)

    def fetch_intraday_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        base = self._BASE_PRICES.get(symbol, 1000.0)
        n_bars = lookback_days * 26  # ~26 bars per trading day for 15m
        return self._generate(symbol, n_days=n_bars, base_price=base)
