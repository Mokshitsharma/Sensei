# src/data/prices.py

import pandas as pd

from src.data.providers.nse import NSEProvider
from src.data.providers.yahoo import YahooProvider
from src.utils.logger import get_logger

logger = get_logger("prices")

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_prices(
    ticker: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Load OHLCV prices using provider abstraction.
    """

    nse = NSEProvider()
    yahoo = YahooProvider()

    # -----------------------------
    # Intraday → fallback only
    # -----------------------------
    if timeframe.endswith(("m", "h")):
        logger.warning("Intraday NSE data not supported reliably. Using Yahoo.")
        df = yahoo.fetch_intraday_ohlcv(ticker, timeframe)

    # -----------------------------
    # Daily → NSE primary
    # -----------------------------
    else:
        df = nse.fetch_daily_ohlcv(ticker)

        if df.empty:
            logger.warning("NSE failed, falling back to Yahoo")
            df = yahoo.fetch_daily_ohlcv(ticker)

    if df.empty:
        raise ValueError(f"No price data available for {ticker}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    logger.info(f"Loaded {len(df)} rows for {ticker}")
    return df
