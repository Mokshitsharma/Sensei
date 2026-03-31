# src/domain/support_resistance.py
"""
Support & Resistance Engine.

Methods used (all combined, deduplicated, strength-scored):
  1. Classic Pivot Points   (previous day's H/L/C)
  2. Fibonacci Retracements (recent swing high → swing low)
  3. Camarilla Pivots       (tighter intraday S/R)
  4. Swing Highs/Lows       (rolling pivot detection on daily data)
  5. Volume Profile Peaks   (price levels with above-average volume)

Returns exactly:
    supports    : 3 levels, sorted descending (nearest first)
    resistances : 3 levels, sorted ascending  (nearest first)

Each level:
    price    : float
    strength : "Strong" | "Moderate" | "Weak"
    methods  : list[str]  — which methods confirmed this level
    touches  : int        — how many times price reacted from this zone
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_support_resistance(
    df: pd.DataFrame,
    price: Optional[float] = None,
    n_levels: int = 3,
    zone_tolerance: float = 0.008,   # 0.8 % — levels within this are merged
) -> Dict:
    """
    Compute top-3 support and top-3 resistance levels.

    Args:
        df              : daily OHLCV dataframe (needs at least 30 rows)
        price           : current price (defaults to latest close)
        n_levels        : number of levels to return per side (default 3)
        zone_tolerance  : % tolerance for merging nearby levels

    Returns dict:
        supports    : List[Dict]   (nearest first)
        resistances : List[Dict]   (nearest first)
        pivot_data  : Dict         (classic pivot point values for reference)
    """
    if df is None or len(df) < 20:
        return {"supports": [], "resistances": [], "pivot_data": {}}

    df = df.copy()
    if price is None:
        price = float(df["close"].iloc[-1])

    # ── Collect all candidate levels ────────────────────────────────────────
    candidates: List[Dict] = []

    candidates.extend(_classic_pivots(df))
    candidates.extend(_camarilla_pivots(df))
    candidates.extend(_fibonacci_levels(df))
    candidates.extend(_swing_levels(df))
    candidates.extend(_volume_profile_levels(df))

    if not candidates:
        return {"supports": [], "resistances": [], "pivot_data": {}}

    # ── Merge nearby levels ──────────────────────────────────────────────────
    merged = _merge_levels(candidates, zone_tolerance)

    # ── Count historical touches ─────────────────────────────────────────────
    merged = _count_touches(merged, df)

    # ── Score each level ─────────────────────────────────────────────────────
    for lv in merged:
        lv["strength"] = _score_strength(lv)

    # ── Split into supports / resistances ────────────────────────────────────
    supports    = sorted(
        [lv for lv in merged if lv["price"] < price],
        key=lambda x: x["price"], reverse=True
    )[:n_levels]

    resistances = sorted(
        [lv for lv in merged if lv["price"] > price],
        key=lambda x: x["price"]
    )[:n_levels]

    # ── Pad if fewer than n_levels found ────────────────────────────────────
    while len(supports) < n_levels:
        last = supports[-1]["price"] if supports else price
        supports.append(_dummy_level(last * (1 - 0.015 * (len(supports) + 1)), "Weak"))

    while len(resistances) < n_levels:
        last = resistances[-1]["price"] if resistances else price
        resistances.append(_dummy_level(last * (1 + 0.015 * (len(resistances) + 1)), "Weak"))

    # Classic pivot for reference
    pivot_data = _classic_pivot_raw(df)

    return {
        "supports":    [_format_level(lv) for lv in supports],
        "resistances": [_format_level(lv) for lv in resistances],
        "pivot_data":  pivot_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method implementations
# ─────────────────────────────────────────────────────────────────────────────

def _classic_pivots(df: pd.DataFrame) -> List[Dict]:
    """Classic pivot points from previous session's H/L/C."""
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])

    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    R2 = PP + (H - L)
    R3 = H + 2 * (PP - L)
    S1 = 2 * PP - H
    S2 = PP - (H - L)
    S3 = L - 2 * (H - PP)

    levels = []
    for price, label in [(R1, "Pivot R1"), (R2, "Pivot R2"), (R3, "Pivot R3"),
                         (S1, "Pivot S1"), (S2, "Pivot S2"), (S3, "Pivot S3"),
                         (PP, "Pivot Point")]:
        if price > 0:
            levels.append({"price": round(price, 2), "methods": [label], "touches": 0})
    return levels


def _classic_pivot_raw(df: pd.DataFrame) -> Dict:
    """Return named pivot values for display."""
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
    PP = (H + L + C) / 3
    return {
        "PP": round(PP, 2),
        "R1": round(2 * PP - L, 2),
        "R2": round(PP + (H - L), 2),
        "R3": round(H + 2 * (PP - L), 2),
        "S1": round(2 * PP - H, 2),
        "S2": round(PP - (H - L), 2),
        "S3": round(L - 2 * (H - PP), 2),
    }


def _camarilla_pivots(df: pd.DataFrame) -> List[Dict]:
    """Camarilla pivot points — tighter levels."""
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
    diff = H - L

    levels = []
    for mult, label in [
        (1.1 / 12, "Cam R1"), (1.1 / 6, "Cam R2"), (1.1 / 4, "Cam R3"),
        (-1.1 / 12, "Cam S1"), (-1.1 / 6, "Cam S2"), (-1.1 / 4, "Cam S3"),
    ]:
        price = round(C + mult * diff, 2)
        if price > 0:
            levels.append({"price": price, "methods": [label], "touches": 0})
    return levels


def _fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> List[Dict]:
    """Fibonacci retracement from recent swing high/low."""
    window = df.tail(lookback)
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return []

    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    levels = []
    for ratio in fib_ratios:
        price = round(swing_high - ratio * diff, 2)
        if price > 0:
            levels.append({
                "price":   price,
                "methods": [f"Fib {int(ratio*100)}%"],
                "touches": 0,
            })
    return levels


def _swing_levels(df: pd.DataFrame, lookback: int = 90, n: int = 3) -> List[Dict]:
    """Rolling pivot highs and lows."""
    window = df.tail(lookback)
    highs  = window["high"].values
    lows   = window["low"].values
    step   = 3   # lookback each side

    levels = []
    for i in range(step, len(window) - step):
        if highs[i] == max(highs[i - step: i + step + 1]):
            levels.append({
                "price":   round(float(highs[i]), 2),
                "methods": ["Swing High"],
                "touches": 0,
            })
        if lows[i] == min(lows[i - step: i + step + 1]):
            levels.append({
                "price":   round(float(lows[i]), 2),
                "methods": ["Swing Low"],
                "touches": 0,
            })
    return levels


def _volume_profile_levels(
    df: pd.DataFrame,
    lookback: int = 60,
    bins: int = 30,
) -> List[Dict]:
    """Price levels with above-average volume (volume profile peaks)."""
    window = df.tail(lookback).copy()
    if "volume" not in window.columns or window["volume"].sum() == 0:
        return []

    prices = (window["high"] + window["low"]) / 2
    volumes = window["volume"]

    price_min, price_max = prices.min(), prices.max()
    if price_max <= price_min:
        return []

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_per_bin = np.zeros(bins)

    for i in range(bins):
        mask = (prices >= bin_edges[i]) & (prices < bin_edges[i + 1])
        vol_per_bin[i] = volumes[mask].sum()

    avg_vol = vol_per_bin.mean()
    levels  = []

    for i in range(bins):
        if vol_per_bin[i] > avg_vol * 1.5:
            mid_price = round((bin_edges[i] + bin_edges[i + 1]) / 2, 2)
            levels.append({
                "price":   mid_price,
                "methods": ["Volume Profile"],
                "touches": 0,
            })

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# Merging and scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_levels(
    candidates: List[Dict],
    tolerance: float,
) -> List[Dict]:
    """Cluster levels within tolerance% of each other."""
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda x: x["price"])
    merged = []
    current_group = [sorted_candidates[0]]

    for item in sorted_candidates[1:]:
        ref_price = current_group[0]["price"]
        if ref_price > 0 and abs(item["price"] - ref_price) / ref_price <= tolerance:
            current_group.append(item)
        else:
            merged.append(_consolidate_group(current_group))
            current_group = [item]

    merged.append(_consolidate_group(current_group))
    return merged


def _consolidate_group(group: List[Dict]) -> Dict:
    prices  = [g["price"] for g in group]
    methods = []
    for g in group:
        methods.extend(g.get("methods", []))
    return {
        "price":   round(sum(prices) / len(prices), 2),
        "methods": list(set(methods)),
        "touches": max(g.get("touches", 0) for g in group),
    }


def _count_touches(
    levels: List[Dict],
    df: pd.DataFrame,
    tolerance: float = 0.008,
    window: int = 90,
) -> List[Dict]:
    """Count how many candles came within tolerance% of each level."""
    recent = df.tail(window)
    for lv in levels:
        p = lv["price"]
        low_bound  = p * (1 - tolerance)
        high_bound = p * (1 + tolerance)
        touches = int(((recent["low"] <= high_bound) & (recent["high"] >= low_bound)).sum())
        lv["touches"] = touches
    return levels


def _score_strength(lv: Dict) -> str:
    method_count = len(lv.get("methods", []))
    touches      = lv.get("touches", 0)

    score = method_count + (touches // 3)

    high_quality_methods = {"Pivot R1", "Pivot S1", "Fib 61.8%", "Fib 38.2%",
                             "Swing High", "Swing Low", "Volume Profile"}
    if any(m in high_quality_methods for m in lv.get("methods", [])):
        score += 1

    if score >= 4:
        return "Strong"
    if score >= 2:
        return "Moderate"
    return "Weak"


def _dummy_level(price: float, strength: str) -> Dict:
    return {
        "price":   round(price, 2),
        "methods": ["Estimated"],
        "touches": 0,
        "strength": strength,
    }


def _format_level(lv: Dict) -> Dict:
    return {
        "price":    lv["price"],
        "strength": lv.get("strength", "Weak"),
        "methods":  lv.get("methods", []),
        "touches":  lv.get("touches", 0),
    }


# Optional import guard
from typing import Optional
