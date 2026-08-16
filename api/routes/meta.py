from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from src.data.nifty50 import NIFTY_50
from src.data.nse_stocks import ALL_NSE_STOCKS
from src.data.market_cap import MID_CAP, SMALL_CAP
from api import compute
from api.serializers import to_jsonable

router = APIRouter()

CAP_SEGMENTS = {
    "large": dict(list(NIFTY_50.items())[:16]),
    "mid": MID_CAP,
    "small": SMALL_CAP,
}

_MOVERS_UNIVERSE: dict[str, str] = {}
for _segment in CAP_SEGMENTS.values():
    _MOVERS_UNIVERSE.update(_segment)


def _predict_many(universe: dict[str, str]) -> list[dict]:
    """Run compute.prediction_7d across a curated ticker set in parallel —
    each call is I/O-bound (news fetch), so a small thread pool keeps a
    cold (uncached) request from taking dozens of seconds sequentially."""
    tickers = list(universe.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(compute.prediction_7d, tickers))
    return [to_jsonable(r) for r in results if r.get("current_price", 0) > 0]


INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
}

# Curated large-cap set for the home page cards + background pre-warming —
# deliberately NOT the full ~2,400-stock universe (see NEXT_STEPS.md on
# why pre-warming everything isn't feasible on a single free instance).
POPULAR = list(NIFTY_50.items())[:8]

# Full searchable universe: every NSE-listed stock, deduped by ticker (not
# name) so a stock never appears twice under two different display names —
# NIFTY_50's curated names take precedence where they overlap.
_TICKER_TO_NAME = {ticker: name for name, ticker in ALL_NSE_STOCKS.items()}
_TICKER_TO_NAME.update({ticker: name for name, ticker in NIFTY_50.items()})
ALL_STOCKS = [{"name": name, "ticker": ticker} for ticker, name in _TICKER_TO_NAME.items()]


@router.get("/indices")
def get_indices():
    return {name: compute.index_quote(symbol) for name, symbol in INDICES.items()}


@router.get("/stocks")
def list_stocks():
    return ALL_STOCKS


@router.get("/stocks/popular")
def popular_stocks():
    out = []
    for name, ticker in POPULAR:
        quote = compute.stock_quote(ticker)
        out.append({"name": name, "ticker": ticker, "quote": to_jsonable(quote)})
    return out


@router.get("/stocks/movers")
def stock_movers():
    """Top 5 predicted gainers and losers over the next 7 days, from the
    curated large/mid/small-cap universe (see market_cap.py)."""
    results = _predict_many(_MOVERS_UNIVERSE)
    ranked = sorted(results, key=lambda r: r["expected_move_pct"], reverse=True)
    losers = [r for r in reversed(ranked) if r["expected_move_pct"] < 0][:5]
    gainers = [r for r in ranked if r["expected_move_pct"] > 0][:5]
    return {"gainers": gainers, "losers": losers}


@router.get("/stocks/by-cap")
def stocks_by_cap(segment: str):
    universe = CAP_SEGMENTS.get(segment)
    if universe is None:
        raise HTTPException(status_code=400, detail="segment must be large, mid, or small")
    return _predict_many(universe)
