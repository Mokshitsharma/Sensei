# src/domain/fundamentals.py

import time
from typing import Dict
import yfinance as yf


def _fetch_info(ticker: str, attempts: int = 3) -> dict:
    """yfinance occasionally returns an empty/stale info dict for a ticker
    within a long-running process even though a fresh call succeeds — retry
    with a fresh Ticker object before giving up."""
    last_info: dict = {}
    for attempt in range(attempts):
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        if info.get("currentPrice") or info.get("regularMarketPrice"):
            return info
        last_info = info
        if attempt < attempts - 1:
            time.sleep(0.5)
    return last_info


def load_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Load key fundamental metrics for a stock.

    Returns ML-friendly numeric dictionary.
    """

    info = _fetch_info(ticker)

    def _safe(key: str, default: float = 0.0) -> float:
        val = info.get(key, default)
        return float(val) if val is not None else default

    def _first_nonzero(*keys: str, default: float = 0.0) -> float:
        for key in keys:
            val = info.get(key)
            if val:
                return float(val)
        return default

    fundamentals = {
        # --- Price (currentPrice is occasionally missing/stale from yfinance;
        # fall back to the other price fields it usually does populate) ---
        "current_price": _first_nonzero(
            "currentPrice", "regularMarketPrice", "previousClose"
        ),

        # --- Valuation ---
        "market_cap": _safe("marketCap"),
        "book_value": _safe("bookValue"),

        # --- Balance Sheet ---
        "debt_to_equity": _safe("debtToEquity"),

        # --- Profitability ---
        "roe": _safe("returnOnEquity"),

        # --- Risk / Range ---
        "52_week_high": _safe("fiftyTwoWeekHigh"),
        "52_week_low": _safe("fiftyTwoWeekLow"),
    }

    return fundamentals