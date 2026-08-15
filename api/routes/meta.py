from fastapi import APIRouter

from src.data.nifty50 import NIFTY_50
from api import compute
from api.serializers import to_jsonable

router = APIRouter()

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
}

POPULAR = list(NIFTY_50.items())[:8]


@router.get("/indices")
def get_indices():
    return {name: compute.index_quote(symbol) for name, symbol in INDICES.items()}


@router.get("/stocks")
def list_stocks():
    return [{"name": name, "ticker": ticker} for name, ticker in NIFTY_50.items()]


@router.get("/stocks/popular")
def popular_stocks():
    out = []
    for name, ticker in POPULAR:
        quote = compute.stock_quote(ticker)
        out.append({"name": name, "ticker": ticker, "quote": to_jsonable(quote)})
    return out
