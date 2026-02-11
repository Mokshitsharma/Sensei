# src/charts/lightweight.py

import pandas as pd
from typing import List, Dict
from streamlit_lightweight_charts import renderLightweightCharts


def _format_time(ts: pd.Timestamp) -> int | str:
    """
    Daily data  -> YYYY-MM-DD
    Intraday    -> UNIX timestamp (seconds)
    """
    if ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%Y-%m-%d")
    return int(ts.timestamp())


def _ohlc_series(df: pd.DataFrame) -> List[Dict]:
    return [
        {
            "time": _format_time(d),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
        }
        for d, o, h, l, c in zip(
            df["date"],
            df["open"],
            df["high"],
            df["low"],
            df["close"],
        )
    ]


def _ema_series(df: pd.DataFrame, span: int) -> List[Dict]:
    ema = df["close"].ewm(span=span, adjust=False).mean()
    return [
        {"time": _format_time(d), "value": float(v)}
        for d, v in zip(df["date"], ema)
        if pd.notna(v)
    ]


def _pattern_markers(patterns: List[Dict]) -> List[Dict]:
    markers = []

    for p in patterns:
        color = "#4caf50" if p["type"] in ("golden_cross", "breakout") else "#f44336"
        shape = "arrowUp" if p["type"] in ("golden_cross", "breakout") else "arrowDown"

        markers.append(
            {
                "time": _format_time(p["date"]),
                "position": "aboveBar" if shape == "arrowUp" else "belowBar",
                "color": color,
                "shape": shape,
                "text": p["type"].replace("_", " ").title(),
            }
        )

    return markers


def render_price_chart(
    df: pd.DataFrame,
    patterns: List[Dict] | None = None,
    ema_periods: tuple[int, int] = (20, 50),
) -> None:
    """
    Render interactive candlestick chart with EMAs and patterns.
    """

    candle = {
        "type": "Candlestick",
        "data": _ohlc_series(df),
    }

    series = [candle]

    for p in ema_periods:
        series.append(
            {
                "type": "Line",
                "data": _ema_series(df, p),
                "options": {
                    "lineWidth": 2,
                },
            }
        )

    if patterns:
        candle["markers"] = _pattern_markers(patterns)

    chart = {
        "width": 0,
        "height": 500,
        "layout": {
            "background": {"type": "solid", "color": "#0e1117"},
            "textColor": "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "#1e222d"},
            "horzLines": {"color": "#1e222d"},
        },
        "crosshair": {"mode": 1},
        "rightPriceScale": {"borderColor": "#2a2e39"},
        "timeScale": {
            "borderColor": "#2a2e39",
            "timeVisible": True,
            "secondsVisible": True,
        },
        "series": series,
    }

    renderLightweightCharts([chart])