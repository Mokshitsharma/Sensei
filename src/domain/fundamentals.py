# src/domain/fundamentals.py

from typing import Dict
import yfinance as yf


def load_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Load key fundamental metrics for a stock.

    Returns ML-friendly numeric dictionary.
    """

    stock = yf.Ticker(ticker)
    info = stock.info

    def _safe(key: str, default: float = 0.0) -> float:
        val = info.get(key, default)
        return float(val) if val is not None else default

    fundamentals = {
        # --- Price ---
        "current_price": _safe("currentPrice"),

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