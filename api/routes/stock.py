from fastapi import APIRouter, HTTPException

from api import compute
from api.serializers import df_to_records, to_jsonable

router = APIRouter(prefix="/stocks/{ticker}")


def _validate(ticker: str) -> None:
    if ticker not in compute.TICKER_TO_COMPANY:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")


@router.get("/quote")
def get_quote(ticker: str):
    _validate(ticker)
    return to_jsonable(compute.stock_quote(ticker))


@router.get("/price")
def get_price(ticker: str, timeframe: str = "1y"):
    _validate(ticker)
    df = compute.prices(ticker, timeframe)
    if df.empty:
        raise HTTPException(status_code=502, detail="No price data available")
    return {
        "ticker": ticker,
        "company": compute.company_for(ticker),
        "timeframe": timeframe,
        "records": df_to_records(df),
    }


@router.get("/fundamentals")
def get_fundamentals(ticker: str):
    _validate(ticker)
    return to_jsonable(compute.fundamentals(ticker))


@router.get("/news")
def get_news(ticker: str):
    _validate(ticker)
    return to_jsonable(compute.news(ticker))


@router.get("/support-resistance")
def get_support_resistance_route(ticker: str, timeframe: str = "1y"):
    _validate(ticker)
    return to_jsonable(compute.support_resistance(ticker, timeframe))


@router.get("/setup")
def get_setup(ticker: str, mode: str = "swing", timeframe: str = "1y"):
    _validate(ticker)
    if mode not in ("intraday", "swing"):
        raise HTTPException(status_code=400, detail="mode must be 'intraday' or 'swing'")
    return to_jsonable(compute.trade_setup(ticker, timeframe, mode))


@router.get("/backtest")
def get_backtest(ticker: str, timeframe: str = "1y"):
    _validate(ticker)
    return to_jsonable(compute.backtest(ticker, timeframe))


@router.get("/analysis")
def get_analysis(ticker: str, timeframe: str = "1y"):
    """Combined endpoint mirroring what the Streamlit detail view renders:
    signals, AI decision, news forecast, support/resistance, both trade
    setups. Heaviest endpoint — hits the ML pipeline (cached per ticker)."""
    _validate(ticker)

    signals = compute.signal_pipeline(ticker, timeframe)
    dec = compute.decision(ticker, timeframe)
    forecast = compute.news_price_forecast(ticker, timeframe)
    sr = compute.support_resistance(ticker, timeframe)
    intraday_setup = compute.trade_setup(ticker, timeframe, "intraday")
    swing_setup = compute.trade_setup(ticker, timeframe, "swing")

    return to_jsonable({
        "ticker": ticker,
        "company": compute.company_for(ticker),
        "signals": signals,
        "decision": dec,
        "news_price_forecast": forecast,
        "support_resistance": sr,
        "setup": {
            "intraday": intraday_setup,
            "swing": swing_setup,
        },
    })
