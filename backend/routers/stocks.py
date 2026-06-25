from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backend.schemas.stocks import (
    StockListItem, AnalysisRequest, AnalysisResult,
    SignalBreakdown, ShapFeature, Fundamentals, BacktestMetrics,
    SupportResistance, CandleBar, NewsItem,
)

router = APIRouter(prefix="/stocks", tags=["stocks"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(val, default: float = 0.0) -> float:
    """Return a JSON-safe float, replacing NaN/Inf with default."""
    try:
        f = float(val)
        return default if (f != f or f == float("inf") or f == float("-inf")) else f
    except Exception:
        return default


def _get_nifty500():
    """Lazy-import NIFTY_500 so startup is fast even if src is not on path yet."""
    from src.data.nifty500 import NIFTY_500
    return NIFTY_500


def _symbol_to_company(symbol: str, nifty500: dict) -> Optional[str]:
    """Reverse-lookup: given a ticker symbol, return the company name key."""
    for name, meta in nifty500.items():
        if isinstance(meta, dict):
            if meta.get("symbol", "").upper() == symbol.upper():
                return name
        elif isinstance(meta, str):
            # Support the flat {name: symbol} variant
            if meta.upper() == symbol.upper():
                return name
    return None


def _build_stock_list_item(name: str, meta) -> StockListItem:
    if isinstance(meta, dict):
        return StockListItem(
            symbol=meta.get("symbol", ""),
            name=name,
            sector=meta.get("sector", "Unknown"),
            tier=meta.get("tier", "large"),
        )
    # Flat {name: "TICKER.NS"} format
    return StockListItem(
        symbol=str(meta),
        name=name,
        sector="Unknown",
        tier="large",
    )


# ---------------------------------------------------------------------------
# List / search endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=List[StockListItem])
async def list_stocks(
    tier: Optional[str] = Query(None, description="large|mid|small"),
    sector: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Search query"),
    limit: int = Query(500, le=500),
):
    """List all available stocks with optional filtering."""
    nifty500 = _get_nifty500()
    items: List[StockListItem] = []

    for name, meta in nifty500.items():
        item = _build_stock_list_item(name, meta)

        if tier and item.tier.lower() != tier.lower():
            continue
        if sector and item.sector.lower() != sector.lower():
            continue
        if q:
            q_lower = q.lower()
            if q_lower not in item.name.lower() and q_lower not in item.symbol.lower():
                continue

        items.append(item)
        if len(items) >= limit:
            break

    return items


@router.get("/search", response_model=List[StockListItem])
async def search_stocks(q: str = Query(..., min_length=1)):
    """Search stocks by name or ticker."""
    nifty500 = _get_nifty500()
    q_lower = q.lower()
    results: List[StockListItem] = []

    for name, meta in nifty500.items():
        item = _build_stock_list_item(name, meta)
        if q_lower in item.name.lower() or q_lower in item.symbol.lower():
            results.append(item)

    return results


@router.get("/sectors")
async def list_sectors():
    """List all available sectors."""
    nifty500 = _get_nifty500()
    sectors = set()

    for meta in nifty500.values():
        if isinstance(meta, dict):
            s = meta.get("sector")
            if s:
                sectors.add(s)

    return {"sectors": sorted(sectors)}


# ---------------------------------------------------------------------------
# Chart data endpoint
# ---------------------------------------------------------------------------

@router.get("/{symbol}/chart")
async def get_chart_data(
    symbol: str,
    timeframe: str = Query("1y", pattern="^(1y|2y|5y)$"),
):
    """Get OHLCV data for charting."""
    from src.data.prices import load_prices

    try:
        price_df = load_prices(symbol, timeframe)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Price data unavailable: {exc}")

    candles = []
    for _, row in price_df.iterrows():
        try:
            ts = int(pd.Timestamp(row.name).timestamp() * 1000)
        except Exception:
            continue
        candles.append({
            "time":   ts,
            "open":   safe_float(row.get("open", 0)),
            "high":   safe_float(row.get("high", 0)),
            "low":    safe_float(row.get("low", 0)),
            "close":  safe_float(row.get("close", 0)),
            "volume": safe_float(row.get("volume", 0)),
        })

    return {"symbol": symbol, "timeframe": timeframe, "data": candles}


# ---------------------------------------------------------------------------
# Full analysis endpoint
# ---------------------------------------------------------------------------

@router.post("/{symbol}/analyze", response_model=AnalysisResult)
async def analyze_stock(symbol: str, request: AnalysisRequest):
    """Run full AI analysis pipeline for a stock."""
    timeframe = request.timeframe

    # ── 1. Company name lookup ──────────────────────────────────────────────
    nifty500 = _get_nifty500()
    company_name = _symbol_to_company(symbol, nifty500)
    if company_name is None:
        # Accept unknown symbols but use symbol as name
        company_name = symbol.replace(".NS", "").replace(".BO", "")

    # ── 2. Load prices ──────────────────────────────────────────────────────
    from src.data.prices import load_prices

    try:
        price_df = load_prices(symbol, timeframe)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Price data unavailable for {symbol}: {exc}")

    # ── 3. Load fundamentals (best-effort) ──────────────────────────────────
    from src.domain.fundamentals import load_fundamentals

    try:
        fundamentals_raw = load_fundamentals(symbol)
    except Exception:
        fundamentals_raw = {}

    # ── 4. Fetch news (best-effort) ─────────────────────────────────────────
    from src.data.news import get_news_signal

    try:
        news_result = get_news_signal(company_name, ticker=symbol, max_items=10)
    except Exception:
        news_result = {
            "sentiment_score": 0.0,
            "weighted_score": 0.0,
            "bull_count": 0,
            "bear_count": 0,
            "neutral_count": 0,
            "details": [],
            "summary": "News unavailable.",
        }

    news_sentiment = safe_float(news_result.get("sentiment_score", 0.0))

    # ── 5. Model paths from registry ────────────────────────────────────────
    model_source = "generic"
    try:
        from training.registry import get_model_paths
        model_paths = get_model_paths(symbol)
        lstm_path = model_paths.lstm
        tcn_path = model_paths.tcn
        ppo_path = model_paths.ppo
        model_source = "trained"
    except Exception:
        # Fall back to generic model paths
        lstm_path = "models/lstm_generic.pt"
        tcn_path = "models/tcn_generic.pt"
        ppo_path = "models/ppo_generic.zip"

    # ── 6. Run signal pipeline ──────────────────────────────────────────────
    from src.pipeline.signal_pipeline import run_signal_pipeline

    signals = run_signal_pipeline(
        price_df=price_df,
        fundamentals=fundamentals_raw,
        company=company_name,
        lstm_model_path=lstm_path,
        tcn_model_path=tcn_path,
        ppo_model_path=ppo_path,
    )

    # ── 7. Decision engine ──────────────────────────────────────────────────
    from src.pipeline.decision_engine import make_final_decision

    decision = make_final_decision(
        signals=signals,
        news_sentiment=news_sentiment,
        shap_values=signals.get("shap_values"),
        feature_values=signals.get("feature_values"),
        company=company_name,
    )

    # ── 8. Backtest ─────────────────────────────────────────────────────────
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import calculate_metrics

    try:
        signals_series = pd.Series(
            ["BUY" if x > 0 else "SELL"
             for x in price_df["close"].pct_change().fillna(0)],
            index=price_df.index,
        )
        backtest_df = run_backtest(price_df, signals_series)
        metrics_raw = calculate_metrics(backtest_df["equity"])
        backtest_metrics = BacktestMetrics(
            total_return=safe_float(metrics_raw.get("total_return", 0.0)),
            sharpe_ratio=safe_float(metrics_raw.get("sharpe_ratio", 0.0)),
            max_drawdown=safe_float(metrics_raw.get("max_drawdown", 0.0)),
        )
    except Exception:
        backtest_metrics = BacktestMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
        )

    # ── 9. Support & Resistance ─────────────────────────────────────────────
    from src.domain.support_resistance import get_support_resistance

    try:
        sr_data = get_support_resistance(price_df)
        # Each level is a dict {price, strength, methods, touches}; extract price floats
        support_prices = [
            safe_float(lv["price"]) for lv in sr_data.get("supports", [])
        ]
        resistance_prices = [
            safe_float(lv["price"]) for lv in sr_data.get("resistances", [])
        ]
    except Exception:
        support_prices = []
        resistance_prices = []

    sr = SupportResistance(supports=support_prices, resistances=resistance_prices)

    # ── 10. Trade setups (best-effort) ──────────────────────────────────────
    from src.domain.setup_engine import build_swing_setup

    trade_setups: dict = {}

    try:
        swing_setup = build_swing_setup(price_df, ticker=symbol)
        trade_setups["swing"] = swing_setup
    except Exception:
        trade_setups["swing"] = {"error": "Swing setup unavailable"}

    try:
        from src.data.providers.yahoo import YahooProvider
        from src.domain.setup_engine import build_intraday_setup
        _yahoo = YahooProvider()
        intraday_df = _yahoo.fetch_intraday_ohlcv(symbol, interval="15m", lookback_days=5)
        if not intraday_df.empty:
            intra_setup = build_intraday_setup(intraday_df, price_df, ticker=symbol)
            trade_setups["intraday"] = intra_setup
        else:
            trade_setups["intraday"] = {"error": "Intraday data unavailable"}
    except Exception:
        trade_setups["intraday"] = {"error": "Intraday setup unavailable"}

    # ── 11. Chart data ──────────────────────────────────────────────────────
    chart_data: List[CandleBar] = []
    for _, row in price_df.iterrows():
        try:
            ts = int(pd.Timestamp(row.name).timestamp() * 1000)
        except Exception:
            continue
        chart_data.append(CandleBar(
            time=ts,
            open=safe_float(row.get("open", 0)),
            high=safe_float(row.get("high", 0)),
            low=safe_float(row.get("low", 0)),
            close=safe_float(row.get("close", 0)),
            volume=safe_float(row.get("volume", 0)),
        ))

    # ── 12. News items ──────────────────────────────────────────────────────
    news_items: List[NewsItem] = []
    for detail in news_result.get("details", []):
        try:
            news_items.append(NewsItem(
                headline=str(detail.get("headline", "")),
                source=str(detail.get("source", "")),
                published=str(detail.get("published", "")),
                url=str(detail.get("url", "")),
                label=str(detail.get("label", "NEUTRAL")),
                confidence=safe_float(detail.get("confidence", 0.5)),
                impact_type=str(detail.get("impact_type", "General")),
            ))
        except Exception:
            continue

    # ── 13. SHAP features ───────────────────────────────────────────────────
    shap_feats: List[ShapFeature] = [
        ShapFeature(
            feature=str(item.get("feature", "")),
            value=safe_float(item.get("shap_value", item.get("shap", 0))),
            direction="bullish" if safe_float(item.get("shap_value", item.get("shap", 0))) > 0 else "bearish",
        )
        for item in (signals.get("shap_ranked") or decision.get("shap_ranked") or [])
    ]

    # ── 14. Signals breakdown ───────────────────────────────────────────────
    signal_breakdown = SignalBreakdown(
        rule_signal=str(signals.get("rule_signal", "HOLD")),
        ml_prob_up=safe_float(signals.get("ml_prob_up", 0.5)),
        lstm_return=safe_float(signals.get("lstm_return", 0.0)),
        tcn_return=safe_float(signals.get("tcn_return", 0.0)),
        regime=str(signals.get("regime", "NEUTRAL")),
        ppo_action=str(signals.get("ppo_action", "HOLD")),
    )

    # ── 15. Fundamentals schema ─────────────────────────────────────────────
    fundamentals_schema = Fundamentals(
        current_price=safe_float(fundamentals_raw.get("current_price")) or None,
        market_cap=safe_float(fundamentals_raw.get("market_cap")) or None,
        pe_ratio=safe_float(fundamentals_raw.get("pe_ratio")) or None,
        roe=safe_float(fundamentals_raw.get("roe")) or None,
        week_52_high=safe_float(fundamentals_raw.get("52_week_high")) or None,
        week_52_low=safe_float(fundamentals_raw.get("52_week_low")) or None,
        debt_to_equity=safe_float(fundamentals_raw.get("debt_to_equity")) or None,
        book_value=safe_float(fundamentals_raw.get("book_value")) or None,
    )

    # ── 16. Narrative ────────────────────────────────────────────────────────
    narrative_raw = decision.get("narrative", {})
    if not isinstance(narrative_raw, dict):
        narrative_raw = {}
    # Ensure all values are strings
    narrative: dict = {str(k): str(v) for k, v in narrative_raw.items()}

    # ── 17. Assemble result ─────────────────────────────────────────────────
    return AnalysisResult(
        symbol=symbol,
        name=company_name,
        timeframe=timeframe,
        action=str(decision.get("action", "HOLD")),
        confidence=safe_float(decision.get("confidence", 0.0)),
        score=safe_float(decision.get("score", 0.0)),
        explanation=str(decision.get("explanation", "")),
        signals=signal_breakdown,
        news_sentiment=news_sentiment,
        shap_ranked=shap_feats,
        narrative=narrative,
        fundamentals=fundamentals_schema,
        backtest=backtest_metrics,
        support_resistance=sr,
        trade_setups=trade_setups,
        chart_data=chart_data,
        news=news_items,
        model_source=model_source,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
