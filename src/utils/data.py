# src/utils/data.py

import pandas as pd


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce flat, lowercase OHLCV schema.
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).lower() for c in df.columns]
    return df
