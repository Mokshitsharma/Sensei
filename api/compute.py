"""Cached wrappers around the existing ML/data pipeline in src/. This is a
near 1:1 port of the caching layer that lived in app.py — same functions,
same TTLs, just keyed by plain args instead of st.cache_data."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import src.utils.numpy_compat  # noqa: F401  (must run before any model unpickling)

import pandas as pd

from src.data.nifty50 import NIFTY_50
from src.data.nse_stocks import ALL_NSE_STOCKS
from src.data.prices import load_prices
from src.domain.fundamentals import load_fundamentals
from src.data.news import get_news_signal
from src.data.providers.yahoo import YahooProvider
from src.pipeline.signal_pipeline import run_signal_pipeline
from src.pipeline.decision_engine import make_final_decision
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_metrics
from src.domain.setup_engine import build_intraday_setup, build_swing_setup, _daily_atr
from src.domain.support_resistance import get_support_resistance
from src.domain.news_price_model import predict_news_price_impact

from api.cache import ttl_cache

# Full NSE universe first, then overlay NIFTY_50's curated display names
# (e.g. "HDFC Bank" instead of NSE's official "HDFC Bank Limited") so the
# well-known large-caps keep their nicer names while every other listed
# stock is still valid/searchable.
TICKER_TO_COMPANY = {v: k for k, v in ALL_NSE_STOCKS.items()}
TICKER_TO_COMPANY.update({v: k for k, v in NIFTY_50.items()})


def company_for(ticker: str) -> str:
    return TICKER_TO_COMPANY.get(ticker, ticker)


@ttl_cache(300)
def prices(ticker: str, timeframe: str) -> pd.DataFrame:
    return load_prices(ticker, timeframe)


@ttl_cache(300, should_cache=lambda v: v.get("current_price", 0) > 0)
def fundamentals(ticker: str) -> dict:
    return load_fundamentals(ticker)


@ttl_cache(300)
def news(ticker: str) -> dict:
    return get_news_signal(company_for(ticker), ticker=ticker, max_items=10)


@ttl_cache(300)
def intraday(ticker: str) -> pd.DataFrame:
    try:
        return YahooProvider().fetch_intraday_ohlcv(ticker, interval="15m", lookback_days=5)
    except Exception:
        return pd.DataFrame()


@ttl_cache(300)
def support_resistance(ticker: str, timeframe: str) -> dict:
    return get_support_resistance(prices(ticker, timeframe))


@ttl_cache(300)
def signal_pipeline(ticker: str, timeframe: str) -> dict:
    return run_signal_pipeline(
        price_df=prices(ticker, timeframe),
        fundamentals=fundamentals(ticker),
        company=company_for(ticker),
        lstm_model_path="models/lstm_HDFCBANK_NS.pt",
        tcn_model_path="models/tcn_HDFCBANK_NS.pt",
        ppo_model_path="models/ppo_hdfc.zip",
    )


@ttl_cache(300)
def decision(ticker: str, timeframe: str) -> dict:
    signals = signal_pipeline(ticker, timeframe)
    news_result = news(ticker)
    return make_final_decision(
        signals=signals,
        news_sentiment=news_result["sentiment_score"],
        shap_values=signals.get("shap_values"),
        feature_values=signals.get("feature_values"),
        company=company_for(ticker),
    )


@ttl_cache(1800, should_cache=lambda v: v.get("current_price", 0) > 0)
def prediction_7d(ticker: str) -> dict:
    """Lightweight 7-day price prediction used by the home page's
    gainers/losers and market-cap sections. Reuses the news+ATR heuristic
    forecast (cheap: no LSTM/TCN/PPO inference) so it's feasible to compute
    across a few dozen curated tickers per request. Cached for 30 minutes —
    this is a directional estimate, not something that needs second-level
    freshness."""
    price_df = prices(ticker, "1y")
    if price_df.empty:
        return {"ticker": ticker, "company": company_for(ticker), "current_price": 0}
    try:
        atr_val = _daily_atr(price_df)
    except Exception:
        atr_val = float(price_df["close"].std()) * 0.1
    fund = fundamentals(ticker)
    current_price = fund.get("current_price") or float(price_df["close"].iloc[-1])
    forecast = predict_news_price_impact(
        current_price=current_price,
        news_result=news(ticker),
        atr=atr_val,
        horizon="7d",
    )
    return {
        "ticker": ticker,
        "company": company_for(ticker),
        "current_price": current_price,
        "predicted_price": forecast["predicted_price"],
        "expected_move_pct": forecast["expected_move_pct"],
        "confidence": forecast["confidence"],
        "direction": forecast["direction"],
    }


@ttl_cache(300)
def news_price_forecast(ticker: str, timeframe: str) -> dict:
    price_df = prices(ticker, timeframe)
    try:
        atr_val = _daily_atr(price_df)
    except Exception:
        atr_val = float(price_df["close"].std()) * 0.1
    fund = fundamentals(ticker)
    return predict_news_price_impact(
        current_price=fund.get("current_price", float(price_df["close"].iloc[-1])),
        news_result=news(ticker),
        atr=atr_val,
        horizon="3d",
    )


@ttl_cache(300)
def trade_setup(ticker: str, timeframe: str, mode: str) -> dict:
    price_df = prices(ticker, timeframe)
    if mode == "intraday":
        intraday_df = intraday(ticker)
        if intraday_df.empty:
            return {
                "error": "Intraday data unavailable", "mode": "Intraday",
                "bias": "NEUTRAL", "entry_zone": (0, 0), "stop_loss": 0,
                "target_1": 0, "target_2": 0, "risk_reward": 0,
                "pattern": "—", "key_levels": {}, "validity": "—",
                "plan": "Intraday data could not be loaded.",
            }
        return build_intraday_setup(intraday_df, price_df, ticker)
    return build_swing_setup(price_df, ticker)


@ttl_cache(300)
def backtest(ticker: str, timeframe: str) -> dict:
    price_df = prices(ticker, timeframe)
    signals_series = pd.Series(
        ["BUY" if x > 0 else "SELL" for x in price_df["close"].pct_change().fillna(0)]
    )
    backtest_df = run_backtest(price_df, signals_series)
    metrics = calculate_metrics(backtest_df["equity"])
    return {
        "equity_curve": backtest_df["equity"].tolist(),
        "metrics": metrics,
    }


@ttl_cache(120)
def index_quote(symbol: str) -> dict | None:
    try:
        df = YahooProvider().fetch_daily_ohlcv(symbol)
        if df.empty or len(df) < 2:
            return None
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        change = last - prev
        pct = (change / prev * 100) if prev else 0.0
        return {"value": last, "change": change, "pct": pct}
    except Exception:
        return None


stock_quote = index_quote  # same shape — last close vs prior close
