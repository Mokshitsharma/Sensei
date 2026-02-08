# src/data/prices.py
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def load_prices(
    ticker: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Loads price data for Indian stocks with daily or intraday support.
    """

    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = f"{ticker}.NS"

    if timeframe in ("5m", "15m"):
        df = yf.download(
            ticker,
            interval=timeframe,
            period="60d",
            auto_adjust=False,
            progress=False,
        )
    else:
        df = yf.download(
            ticker,
            period=timeframe,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

    if df.empty:
        raise ValueError("No price data returned")

    df = _flatten_columns(df)
    df = df.reset_index()

    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "datetime": "date",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    df = df.rename(columns=rename_map)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
