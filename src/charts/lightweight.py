# src/charts/lightweight.py
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts


def _format_time(ts: pd.Timestamp) -> int | str:
    """
    Daily -> YYYY-MM-DD
    Intraday -> UNIX timestamp (seconds)
    """
    if ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%Y-%m-%d")
    return int(ts.timestamp())


def _ohlc_series(df: pd.DataFrame) -> list[dict]:
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


def _ema_series(df: pd.DataFrame, span: int) -> list[dict]:
    ema = df["close"].ewm(span=span, adjust=False).mean()
    return [
        {
            "time": _format_time(d),
            "value": float(v),
        }
        for d, v in zip(df["date"], ema)
        if pd.notna(v)
    ]


def render_price_chart(
    df: pd.DataFrame,
    patterns: list[dict],
    ema_periods: tuple[int, int] = (20, 50),
) -> None:
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
                "options": {"lineWidth": 2},
            }
        )

    chart = {
        "width": 0,
        "height": 480,
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
