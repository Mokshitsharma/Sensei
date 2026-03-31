# src/domain/setup_engine.py
"""
Trade Setup Engine — generates actionable Intraday and Swing trade setups.

Intraday setup  → uses 15-min candles (last 5 trading days)
Swing setup     → uses daily candles (last 6 months)

Both setups produce:
    entry_zone   : (low, high) price range to enter
    stop_loss    : hard SL price
    target_1     : first profit target
    target_2     : extended profit target
    risk_reward  : R:R ratio at target_1
    bias         : BULLISH | BEARISH | NEUTRAL
    pattern      : what triggered the setup
    plan         : plain-English trade description
    key_levels   : dict of named price levels (support, resistance, vwap, …)
    validity     : how long the setup is valid ("Today" / "3–5 days" / etc.)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.domain.indicators import add_indicators
from src.domain.intraday import add_intraday_features, intraday_bias


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_intraday_setup(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ticker: str = "",
) -> Dict:
    """
    Generate an intraday trade setup from 15-min candles.

    Args:
        intraday_df : 15-min OHLCV (at least 1 day)
        daily_df    : daily OHLCV for context (S/R levels)
        ticker      : symbol string (used in plan text)

    Returns:
        Setup dict (see module docstring for keys).
    """
    try:
        intra = add_intraday_features(intraday_df.copy())
        daily = add_indicators(daily_df.copy())

        latest     = intra.iloc[-1]
        price      = float(latest["close"])
        vwap       = float(latest["vwap"])
        ema_9      = float(latest["ema_9"])
        ema_21     = float(latest["ema_21"])

        atr_daily  = _daily_atr(daily)
        atr_intra  = _intraday_atr(intra)
        atr        = atr_intra if atr_intra > 0 else atr_daily * 0.4  # scale daily → intraday

        bias       = intraday_bias(intra)
        support, resistance = _intraday_levels(intra)
        patterns   = _intraday_patterns(intra)

        if bias == "BULLISH":
            entry_low  = max(support, price - atr * 0.3)
            entry_high = price + atr * 0.1
            stop_loss  = support - atr * 0.25
            target_1   = resistance
            target_2   = resistance + atr * 0.8
        elif bias == "BEARISH":
            entry_low  = price - atr * 0.1
            entry_high = min(resistance, price + atr * 0.3)
            stop_loss  = resistance + atr * 0.25
            target_1   = support
            target_2   = support - atr * 0.8
        else:
            mid        = (support + resistance) / 2
            entry_low  = mid - atr * 0.15
            entry_high = mid + atr * 0.15
            stop_loss  = support - atr * 0.2
            target_1   = resistance
            target_2   = resistance + atr * 0.5

        rr = _risk_reward(price, stop_loss, target_1)

        key_levels = {
            "VWAP":       round(vwap, 2),
            "EMA 9":      round(ema_9, 2),
            "EMA 21":     round(ema_21, 2),
            "Support":    round(support, 2),
            "Resistance": round(resistance, 2),
        }

        plan = _intraday_plan(
            bias, price, vwap, ema_9, ema_21,
            entry_low, entry_high, stop_loss, target_1, target_2,
            rr, patterns, atr, ticker,
        )

        return {
            "mode":       "Intraday",
            "bias":       bias,
            "price":      round(price, 2),
            "entry_zone": (round(entry_low, 2), round(entry_high, 2)),
            "stop_loss":  round(stop_loss, 2),
            "target_1":   round(target_1, 2),
            "target_2":   round(target_2, 2),
            "risk_reward": round(rr, 2),
            "pattern":    patterns[0] if patterns else "No clear pattern",
            "key_levels": key_levels,
            "validity":   "Today's session only",
            "plan":       plan,
            "error":      None,
        }

    except Exception as exc:
        return _error_setup("Intraday", str(exc))


def build_swing_setup(
    daily_df: pd.DataFrame,
    ticker: str = "",
) -> Dict:
    """
    Generate a swing trade setup from daily candles.

    Args:
        daily_df : daily OHLCV (at least 60 rows recommended)
        ticker   : symbol string

    Returns:
        Setup dict (see module docstring for keys).
    """
    try:
        df      = add_indicators(daily_df.copy())
        latest  = df.iloc[-1]
        price   = float(latest["close"])

        atr     = _daily_atr(df)
        bias    = _swing_bias(df)
        support, resistance = _swing_levels(df)
        patterns            = _swing_patterns(df)
        ema20   = float(latest["ema_20"])
        ema50   = float(latest["ema_50"])
        rsi     = float(latest["rsi"])
        macd    = float(latest["macd"])
        macd_sig= float(latest["macd_signal"])

        if bias == "BULLISH":
            entry_low  = max(support, ema20 * 0.995)
            entry_high = price + atr * 0.1
            stop_loss  = support - atr * 0.5
            target_1   = resistance
            target_2   = resistance + atr * 1.5
            holding    = "3–7 trading days"
        elif bias == "BEARISH":
            entry_low  = price - atr * 0.1
            entry_high = min(resistance, ema20 * 1.005)
            stop_loss  = resistance + atr * 0.5
            target_1   = support
            target_2   = support - atr * 1.5
            holding    = "3–7 trading days"
        else:
            entry_low  = support + atr * 0.1
            entry_high = resistance - atr * 0.1
            stop_loss  = support - atr * 0.4
            target_1   = resistance
            target_2   = resistance + atr * 1.0
            holding    = "5–10 trading days"

        rr = _risk_reward(price, stop_loss, target_1)

        # Bollinger Bands for key levels
        bb_mid, bb_upper, bb_lower = _bollinger(df)

        key_levels = {
            "EMA 20":      round(ema20, 2),
            "EMA 50":      round(ema50, 2),
            "BB Upper":    round(bb_upper, 2),
            "BB Lower":    round(bb_lower, 2),
            "Support":     round(support, 2),
            "Resistance":  round(resistance, 2),
            "52W High":    round(float(df["high"].tail(252).max()), 2),
            "52W Low":     round(float(df["low"].tail(252).min()), 2),
        }

        plan = _swing_plan(
            bias, price, ema20, ema50, rsi, macd, macd_sig,
            entry_low, entry_high, stop_loss, target_1, target_2,
            rr, patterns, atr, holding, ticker,
        )

        return {
            "mode":       "Swing",
            "bias":       bias,
            "price":      round(price, 2),
            "entry_zone": (round(entry_low, 2), round(entry_high, 2)),
            "stop_loss":  round(stop_loss, 2),
            "target_1":   round(target_1, 2),
            "target_2":   round(target_2, 2),
            "risk_reward": round(rr, 2),
            "pattern":    patterns[0] if patterns else "No clear pattern",
            "key_levels": key_levels,
            "validity":   holding,
            "plan":       plan,
            "error":      None,
        }

    except Exception as exc:
        return _error_setup("Swing", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Level detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _intraday_levels(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Find intraday support and resistance from recent session pivots.
    Uses the last 20 bars (≈ 5 hours of 15-min data).
    """
    window = df.tail(20)
    support    = float(window["low"].min())
    resistance = float(window["high"].max())

    # Try to find pivot clusters if we have enough data
    if len(df) >= 40:
        pivots = _find_pivots(df.tail(40))
        if pivots["support"]:
            support    = max(pivots["support"])   # nearest support below price
        if pivots["resistance"]:
            resistance = min(pivots["resistance"])  # nearest resistance above price

    return support, resistance


def _swing_levels(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Find swing support and resistance using 3-month pivot highs/lows.
    """
    window = df.tail(65)  # ~3 months
    price  = float(df.iloc[-1]["close"])

    pivots = _find_pivots(window, lookback=5)

    support_candidates    = [p for p in pivots["support"]    if p < price]
    resistance_candidates = [p for p in pivots["resistance"] if p > price]

    support    = max(support_candidates)    if support_candidates    else float(window["low"].min())
    resistance = min(resistance_candidates) if resistance_candidates else float(window["high"].max())

    return support, resistance


def _find_pivots(
    df: pd.DataFrame,
    lookback: int = 3,
) -> Dict[str, List[float]]:
    """
    Find local pivot highs and lows using a rolling window.
    """
    support_levels    = []
    resistance_levels = []

    highs = df["high"].values
    lows  = df["low"].values

    for i in range(lookback, len(df) - lookback):
        # Pivot high
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            resistance_levels.append(float(highs[i]))
        # Pivot low
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            support_levels.append(float(lows[i]))

    return {
        "support":    sorted(set(round(s, 2) for s in support_levels)),
        "resistance": sorted(set(round(r, 2) for r in resistance_levels)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bias helpers
# ─────────────────────────────────────────────────────────────────────────────

def _swing_bias(df: pd.DataFrame) -> str:
    """
    Determine swing bias from daily indicators.
    """
    latest   = df.iloc[-1]
    price    = latest["close"]
    ema20    = latest["ema_20"]
    ema50    = latest["ema_50"]
    rsi      = latest["rsi"]
    macd     = latest["macd"]
    macd_sig = latest["macd_signal"]

    bull_score = 0
    bear_score = 0

    if price > ema20 > ema50:
        bull_score += 2
    elif price < ema20 < ema50:
        bear_score += 2
    elif price > ema20:
        bull_score += 1
    elif price < ema20:
        bear_score += 1

    if macd > macd_sig:
        bull_score += 1
    else:
        bear_score += 1

    if rsi > 55:
        bull_score += 1
    elif rsi < 45:
        bear_score += 1

    if bull_score >= 3:
        return "BULLISH"
    if bear_score >= 3:
        return "BEARISH"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _intraday_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    # VWAP reclaim
    if prev["close"] < prev["vwap"] and latest["close"] > latest["vwap"]:
        patterns.append("VWAP reclaim (bullish)")
    elif prev["close"] > prev["vwap"] and latest["close"] < latest["vwap"]:
        patterns.append("VWAP breakdown (bearish)")

    # EMA 9/21 cross
    if prev["ema_9"] < prev["ema_21"] and latest["ema_9"] > latest["ema_21"]:
        patterns.append("EMA 9/21 bullish cross")
    elif prev["ema_9"] > prev["ema_21"] and latest["ema_9"] < latest["ema_21"]:
        patterns.append("EMA 9/21 bearish cross")

    # Price above VWAP and accelerating
    if latest["close"] > latest["vwap"] and latest["ema_9"] > latest["ema_21"]:
        patterns.append("Price above VWAP with EMA alignment")

    # Inside bar (consolidation)
    if latest["high"] < prev["high"] and latest["low"] > prev["low"]:
        patterns.append("Inside bar (consolidation)")

    return patterns if patterns else ["No clear intraday pattern"]


def _swing_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    # EMA Golden/Death cross
    if prev["ema_20"] < prev["ema_50"] and latest["ema_20"] > latest["ema_50"]:
        patterns.append("Golden cross (EMA 20 > EMA 50)")
    elif prev["ema_20"] > prev["ema_50"] and latest["ema_20"] < latest["ema_50"]:
        patterns.append("Death cross (EMA 20 < EMA 50)")

    # MACD crossover
    if prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
        patterns.append("MACD bullish crossover")
    elif prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
        patterns.append("MACD bearish crossover")

    # RSI zones
    if latest["rsi"] < 35:
        patterns.append(f"RSI oversold ({latest['rsi']:.1f})")
    elif latest["rsi"] > 65:
        patterns.append(f"RSI overbought ({latest['rsi']:.1f})")

    # 20-day breakout
    high_20 = df["high"].tail(21).iloc[:-1].max()
    low_20  = df["low"].tail(21).iloc[:-1].min()
    if latest["close"] > high_20:
        patterns.append("20-day high breakout")
    elif latest["close"] < low_20:
        patterns.append("20-day low breakdown")

    # Bollinger band squeeze / expansion
    _, bb_upper, bb_lower = _bollinger(df)
    bb_width = (bb_upper - bb_lower) / ((bb_upper + bb_lower) / 2)
    if bb_width < 0.04:
        patterns.append("Bollinger band squeeze (volatility contraction)")
    elif latest["close"] > bb_upper:
        patterns.append("Bollinger upper band breakout")
    elif latest["close"] < bb_lower:
        patterns.append("Bollinger lower band breakdown")

    return patterns if patterns else ["No clear swing pattern"]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _daily_atr(df: pd.DataFrame, period: int = 14) -> float:
    if "atr" in df.columns:
        val = df["atr"].dropna().iloc[-1] if not df["atr"].dropna().empty else 0
        return float(val)
    high_low  = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def _intraday_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def _bollinger(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[float, float, float]:
    close = df["close"]
    mid   = close.rolling(period).mean().dropna()
    std   = close.rolling(period).std().dropna()
    if mid.empty:
        p = float(close.iloc[-1])
        return p, p, p
    m = float(mid.iloc[-1])
    s = float(std.iloc[-1])
    return m, m + num_std * s, m - num_std * s


def _risk_reward(
    entry: float,
    stop: float,
    target: float,
) -> float:
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Plain-English plan builders
# ─────────────────────────────────────────────────────────────────────────────

def _intraday_plan(
    bias, price, vwap, ema9, ema21,
    entry_low, entry_high, stop_loss, target_1, target_2,
    rr, patterns, atr, ticker,
) -> str:
    name = ticker or "the stock"
    bias_word = {"BULLISH": "bullish", "BEARISH": "bearish"}.get(bias, "neutral")

    vwap_rel = "above" if price > vwap else "below"
    ema_cross = "aligned bullishly (EMA 9 > EMA 21)" if ema9 > ema21 else "aligned bearishly (EMA 9 < EMA 21)"

    pattern_line = f"Key intraday trigger: {patterns[0]}." if patterns else ""

    if bias == "BULLISH":
        action_desc = (
            f"Look to go LONG on a dip into the ₹{entry_low:.2f}–₹{entry_high:.2f} zone, "
            f"ideally on a candle close above VWAP (₹{vwap:.2f})."
        )
        sl_desc = (
            f"Place stop loss below ₹{stop_loss:.2f} "
            f"(below session support, ~{abs(price - stop_loss) / price * 100:.1f}% risk)."
        )
    elif bias == "BEARISH":
        action_desc = (
            f"Look to go SHORT on a bounce into the ₹{entry_low:.2f}–₹{entry_high:.2f} zone, "
            f"ideally on a candle rejection below VWAP (₹{vwap:.2f})."
        )
        sl_desc = (
            f"Place stop loss above ₹{stop_loss:.2f} "
            f"(above session resistance, ~{abs(price - stop_loss) / price * 100:.1f}% risk)."
        )
    else:
        action_desc = (
            f"No directional bias. Wait for price to break either side of the "
            f"₹{entry_low:.2f}–₹{entry_high:.2f} range with volume confirmation."
        )
        sl_desc = f"Keep stop tight at ₹{stop_loss:.2f} given the neutral setup."

    return (
        f"{name} is {bias_word} intraday. "
        f"Price (₹{price:.2f}) is {vwap_rel} VWAP (₹{vwap:.2f}) with EMAs {ema_cross}. "
        f"{pattern_line} "
        f"{action_desc} "
        f"{sl_desc} "
        f"Target 1: ₹{target_1:.2f} | Target 2: ₹{target_2:.2f}. "
        f"Risk/Reward at T1: {rr:.1f}x. "
        f"This setup is valid for today's session only — exit by market close."
    )


def _swing_plan(
    bias, price, ema20, ema50, rsi, macd, macd_sig,
    entry_low, entry_high, stop_loss, target_1, target_2,
    rr, patterns, atr, holding, ticker,
) -> str:
    name = ticker or "the stock"
    bias_word = {"BULLISH": "bullish", "BEARISH": "bearish"}.get(bias, "neutral")

    ema_desc = (
        "Price is above both EMA 20 and EMA 50, confirming uptrend structure."
        if price > ema20 > ema50
        else "Price is below both EMA 20 and EMA 50, confirming downtrend structure."
        if price < ema20 < ema50
        else f"Price is between EMA 20 (₹{ema20:.2f}) and EMA 50 (₹{ema50:.2f}) — mixed structure."
    )

    rsi_desc = (
        f"RSI at {rsi:.1f} is oversold — potential mean-reversion bounce." if rsi < 35
        else f"RSI at {rsi:.1f} is overbought — momentum may exhaust soon." if rsi > 65
        else f"RSI at {rsi:.1f} is neutral."
    )

    macd_desc = (
        "MACD is above its signal line (bullish momentum)."
        if macd > macd_sig
        else "MACD is below its signal line (bearish momentum)."
    )

    pattern_line = f"Key swing trigger: {patterns[0]}." if patterns else ""

    if bias == "BULLISH":
        action_desc = (
            f"Consider entering LONG in the ₹{entry_low:.2f}–₹{entry_high:.2f} zone "
            f"on a daily candle close above EMA 20 (₹{ema20:.2f}) or on a pullback to support."
        )
        sl_desc = (
            f"Stop loss at ₹{stop_loss:.2f} "
            f"(below key support, ~{abs(price - stop_loss) / price * 100:.1f}% from current price)."
        )
    elif bias == "BEARISH":
        action_desc = (
            f"Consider entering SHORT in the ₹{entry_low:.2f}–₹{entry_high:.2f} zone "
            f"on a daily candle rejection below EMA 20 (₹{ema20:.2f}) or a failed bounce."
        )
        sl_desc = (
            f"Stop loss at ₹{stop_loss:.2f} "
            f"(above key resistance, ~{abs(price - stop_loss) / price * 100:.1f}% from current price)."
        )
    else:
        action_desc = (
            f"No clear directional edge. Wait for a confirmed daily close above resistance "
            f"(₹{target_1:.2f}) or below support (₹{stop_loss:.2f}) before committing."
        )
        sl_desc = f"Stop loss at ₹{stop_loss:.2f} if a breakout trade is taken."

    return (
        f"{name} shows a {bias_word} swing structure. "
        f"{ema_desc} {rsi_desc} {macd_desc} "
        f"{pattern_line} "
        f"{action_desc} "
        f"{sl_desc} "
        f"Target 1: ₹{target_1:.2f} | Target 2: ₹{target_2:.2f}. "
        f"Risk/Reward at T1: {rr:.1f}x. "
        f"Expected holding period: {holding}. "
        f"Re-evaluate if price closes beyond stop loss on a daily basis."
    )


def _error_setup(mode: str, error: str) -> Dict:
    return {
        "mode":        mode,
        "bias":        "NEUTRAL",
        "price":       0.0,
        "entry_zone":  (0.0, 0.0),
        "stop_loss":   0.0,
        "target_1":    0.0,
        "target_2":    0.0,
        "risk_reward": 0.0,
        "pattern":     "—",
        "key_levels":  {},
        "validity":    "—",
        "plan":        f"Setup unavailable: {error}",
        "error":       error,
    }
