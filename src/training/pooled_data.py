# src/training/pooled_data.py
"""Shared per-ticker data fetching for pooled (all-NIFTY-50) training. Used
by src/ml/train.py, src/dl/train.py, and src/rl/train.py so the three model
families train on identically-built features from the same tickers.

Returns (ticker, df) pairs rather than one concatenated frame — LSTM/TCN
sequence windows and PPO episodes must never cross from one stock's price
history into another's, so callers pool at the dataset/episode level, not
by concatenating raw rows.
"""

import time
from typing import Iterable, List, Tuple

import pandas as pd

from src.data.prices import load_prices
from src.domain.indicators import add_indicators
from src.ml.features import build_features
from src.utils.config import FEATURE_COLUMNS, TARGET_RETURN


def build_ticker_feature_df(ticker: str, timeframe: str) -> pd.DataFrame:
    df = load_prices(ticker, timeframe)
    df = add_indicators(df)
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_RETURN]).reset_index(drop=True)
    return df


def build_pooled_dataset(
    tickers: Iterable[str],
    timeframe: str = "2y",
    min_rows: int = 60,
    pause_s: float = 0.25,
) -> List[Tuple[str, pd.DataFrame]]:
    """Fetches + features every ticker, skipping ones that fail or come
    back too small to be useful (delisted symbols, thin history, transient
    yfinance errors). A short pause between requests avoids tripping
    Yahoo's rate limiting across 50 sequential calls."""
    out: List[Tuple[str, pd.DataFrame]] = []
    for ticker in tickers:
        try:
            df = build_ticker_feature_df(ticker, timeframe)
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            continue
        if len(df) < min_rows:
            print(f"  skip {ticker}: only {len(df)} usable rows")
            continue
        out.append((ticker, df))
        time.sleep(pause_s)
    return out
