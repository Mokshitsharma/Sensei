from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from api import compute
from api.cache import ttl_cache
from api.routes.meta import CAP_SEGMENTS, _MOVERS_UNIVERSE
from src.domain.indicators import add_indicators

router = APIRouter()

# Same 35/65 thresholds src/domain/setup_engine.py already uses for its
# narrative "RSI oversold/overbought" callouts — keep the screener
# consistent with what the AI report tells users elsewhere.
_OVERSOLD_MAX = 35
_OVERBOUGHT_MIN = 65

# Within this % of a 52-week extreme counts as "near" it for the
# breakout/breakdown screens.
_NEAR_EXTREME_PCT = 3.0

SCREENS = {
    "rsi_oversold": {"label": "RSI Oversold", "bias": "Bullish"},
    "rsi_overbought": {"label": "RSI Overbought", "bias": "Bearish"},
    "macd_bullish": {"label": "MACD Above Signal Line", "bias": "Bullish"},
    "macd_bearish": {"label": "MACD Below Signal Line", "bias": "Bearish"},
    "near_52w_high": {"label": "Near 52-Week High", "bias": "Bullish"},
    "near_52w_low": {"label": "Near 52-Week Low", "bias": "Bearish"},
}


def _row(ticker: str) -> dict | None:
    price_df = compute.prices(ticker, "1y")
    if price_df.empty or len(price_df) < 30:
        return None
    ind = add_indicators(price_df)
    latest = ind.iloc[-1]
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    if rsi != rsi or macd != macd:  # NaN check
        return None
    price = float(price_df["close"].iloc[-1])
    high_52w = float(price_df["close"].max())
    low_52w = float(price_df["close"].min())
    return {
        "ticker": ticker,
        "company": compute.company_for(ticker),
        "price": price,
        "rsi": round(rsi, 1),
        "macd_bullish": macd > macd_signal,
        "pct_from_52w_high": round((high_52w - price) / high_52w * 100, 2) if high_52w else None,
        "pct_from_52w_low": round((price - low_52w) / low_52w * 100, 2) if low_52w else None,
    }


@ttl_cache(300)
def _screener_universe(segment: str) -> list[dict]:
    universe = _MOVERS_UNIVERSE if segment == "all" else CAP_SEGMENTS.get(segment, {})
    tickers = list(universe.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_row, tickers))
    return [r for r in results if r is not None]


@router.get("/screener/screens")
def list_screens():
    return [{"id": k, **v} for k, v in SCREENS.items()]


@router.get("/screener")
def get_screener(filter: str = "rsi_oversold", segment: str = "all"):
    if filter not in SCREENS:
        raise HTTPException(status_code=400, detail=f"filter must be one of {sorted(SCREENS)}")
    if segment not in ("all", "large", "mid", "small"):
        raise HTTPException(status_code=400, detail="segment must be all, large, mid, or small")

    rows = _screener_universe(segment)
    if filter == "rsi_oversold":
        matches = [r for r in rows if r["rsi"] < _OVERSOLD_MAX]
        matches.sort(key=lambda r: r["rsi"])
    elif filter == "rsi_overbought":
        matches = [r for r in rows if r["rsi"] > _OVERBOUGHT_MIN]
        matches.sort(key=lambda r: r["rsi"], reverse=True)
    elif filter == "macd_bullish":
        matches = [r for r in rows if r["macd_bullish"]]
    elif filter == "macd_bearish":
        matches = [r for r in rows if not r["macd_bullish"]]
    elif filter == "near_52w_high":
        matches = [r for r in rows if r["pct_from_52w_high"] is not None and r["pct_from_52w_high"] <= _NEAR_EXTREME_PCT]
        matches.sort(key=lambda r: r["pct_from_52w_high"])
    else:  # near_52w_low
        matches = [r for r in rows if r["pct_from_52w_low"] is not None and r["pct_from_52w_low"] <= _NEAR_EXTREME_PCT]
        matches.sort(key=lambda r: r["pct_from_52w_low"])
    return matches
