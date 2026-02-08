# src/domain/fundamentals.py
from typing import Dict, Any
import yfinance as yf


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental metrics for Indian stocks.

    Returns:
    {
        "pe": float | None,
        "roe": float | None,
        "eps": float | None,
    }
    """

    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = f"{ticker}.NS"

    stock = yf.Ticker(ticker)
    info = stock.info or {}

    pe = info.get("trailingPE")
    eps = info.get("trailingEps")

    roe = info.get("returnOnEquity")
    if roe is not None:
        roe = roe * 100  # convert to %

    return {
        "pe": round(pe, 2) if isinstance(pe, (int, float)) else None,
        "eps": round(eps, 2) if isinstance(eps, (int, float)) else None,
        "roe": round(roe, 2) if isinstance(roe, (int, float)) else None,
    }