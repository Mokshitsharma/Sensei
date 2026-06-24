# src/domain/ipo_analyzer.py

"""
IPO scoring and recommendation engine.

Scores each IPO across five dimensions:
    1. GMP (grey market premium)
    2. Subscription demand
    3. Valuation vs industry
    4. Financial quality (PAT, ROE, D/E)
    5. Promoter governance

Produces a 0-100 composite score, a recommendation (INVEST / NEUTRAL / AVOID),
a risk level, pros/cons bullet-points, and a plain-English summary.
"""

from typing import Dict, List, Optional, Tuple
from src.data.ipo import IPOData


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# GMP score buckets (0-25 pts)
_GMP_THRESHOLDS: List[Tuple[float, float]] = [
    (30.0, 25.0),
    (15.0, 18.0),
    (5.0,  12.0),
    (0.0,   6.0),
]

# Subscription score buckets (0-25 pts)
_SUB_THRESHOLDS: List[Tuple[float, float]] = [
    (100.0, 25.0),
    (50.0,  20.0),
    (20.0,  15.0),
    (10.0,  10.0),
    (5.0,    5.0),
]

_QIB_CONFIDENCE_BONUS = 3.0   # extra pts when QIB > 50x

# Valuation score (0-20 pts)
_VALUATION_EXACT_SCORE = 10.0
_VALUATION_BELOW_SCORE = 20.0
_VALUATION_ABOVE_MILD_SCORE = 5.0   # PE within 20% above industry
_VALUATION_ABOVE_HIGH_SCORE = 0.0   # PE > 20% above industry
_VALUATION_ABOVE_MILD_THRESHOLD = 0.20  # 20%

# Financial quality (0-20 pts)
_PAT_POSITIVE_PTS = 8.0
_ROE_THRESHOLD = 15.0
_ROE_PTS = 6.0
_DE_THRESHOLD = 1.0
_DE_PTS = 6.0

# Promoter governance (0-10 pts)
_PROMOTER_HIGH_THRESHOLD = 50.0
_PROMOTER_MID_THRESHOLD = 40.0
_PROMOTER_HIGH_PTS = 10.0
_PROMOTER_MID_PTS = 7.0
_PROMOTER_LOW_PTS = 3.0

# Recommendation thresholds
_INVEST_THRESHOLD = 65.0
_AVOID_THRESHOLD = 35.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gmp_score(ipo: IPOData) -> Tuple[float, List[str], List[str]]:
    """Return (score, pros, cons) for the GMP dimension."""
    pros: List[str] = []
    cons: List[str] = []
    gmp_pct = ipo.gmp_pct

    score = 0.0
    if gmp_pct < 0:
        cons.append(
            f"Negative GMP of {gmp_pct:.1f}% signals weak grey-market sentiment."
        )
    else:
        for threshold, pts in _GMP_THRESHOLDS:
            if gmp_pct >= threshold:
                score = pts
                break

    if gmp_pct >= 30:
        pros.append(
            f"Strong GMP of {gmp_pct:.1f}% reflects high unlisted demand."
        )
    elif gmp_pct >= 15:
        pros.append(f"Healthy GMP of {gmp_pct:.1f}% indicates positive market interest.")
    elif gmp_pct >= 5:
        pros.append(f"Moderate GMP of {gmp_pct:.1f}%.")
    elif 0 <= gmp_pct < 5:
        cons.append(f"Low GMP of {gmp_pct:.1f}% — limited listing pop expected.")

    return score, pros, cons


def _subscription_score(ipo: IPOData) -> Tuple[float, List[str], List[str]]:
    """Return (score, pros, cons) for the subscription dimension."""
    pros: List[str] = []
    cons: List[str] = []
    total = ipo.subscription_total
    qib = ipo.subscription_qib

    score = 0.0
    for threshold, pts in _SUB_THRESHOLDS:
        if total >= threshold:
            score = pts
            break

    if qib >= 50.0:
        score = min(score + _QIB_CONFIDENCE_BONUS, 25.0)
        pros.append(
            f"Strong institutional (QIB) demand of {qib:.1f}x signals quality validation."
        )

    if total >= 100:
        pros.append(f"Oversubscribed {total:.0f}x — intense investor demand.")
    elif total >= 20:
        pros.append(f"Well-subscribed at {total:.1f}x.")
    elif total >= 5:
        pros.append(f"Moderately subscribed at {total:.1f}x.")
    else:
        cons.append(f"Weak subscription of {total:.1f}x — insufficient demand.")

    nii = ipo.subscription_nii
    rii = ipo.subscription_rii
    if nii < 1.0 and ipo.status in ("listed",):
        cons.append("HNI/NII category barely subscribed — HNI interest was low.")
    if rii < 1.0 and ipo.status in ("listed",):
        cons.append("Retail portion under-subscribed — weak retail participation.")

    return score, pros, cons


def _valuation_score(ipo: IPOData) -> Tuple[float, List[str], List[str]]:
    """Return (score, pros, cons) for the valuation dimension."""
    pros: List[str] = []
    cons: List[str] = []

    pe = ipo.pe_ratio
    ind_pe = ipo.industry_pe

    if pe is None or ind_pe is None or ind_pe <= 0:
        # No data — give neutral half-score
        return _VALUATION_EXACT_SCORE, ["Valuation data unavailable for comparison."], []

    if pe < ind_pe:
        score = _VALUATION_BELOW_SCORE
        discount = (ind_pe - pe) / ind_pe * 100
        pros.append(
            f"Attractively priced at P/E {pe:.1f}x vs industry {ind_pe:.1f}x "
            f"({discount:.0f}% discount)."
        )
    elif pe == ind_pe:
        score = _VALUATION_EXACT_SCORE
        pros.append(f"Fairly valued at P/E {pe:.1f}x in-line with industry.")
    else:
        premium = (pe - ind_pe) / ind_pe
        if premium <= _VALUATION_ABOVE_MILD_THRESHOLD:
            score = _VALUATION_ABOVE_MILD_SCORE
            cons.append(
                f"P/E {pe:.1f}x is a small premium over industry {ind_pe:.1f}x."
            )
        else:
            score = _VALUATION_ABOVE_HIGH_SCORE
            cons.append(
                f"P/E {pe:.1f}x is significantly above industry {ind_pe:.1f}x "
                f"({premium * 100:.0f}% premium) — stretched valuation."
            )

    return score, pros, cons


def _financial_score(ipo: IPOData) -> Tuple[float, List[str], List[str]]:
    """Return (score, pros, cons) for the financial quality dimension."""
    pros: List[str] = []
    cons: List[str] = []
    score = 0.0

    # PAT
    if ipo.pat_cr is not None:
        if ipo.pat_cr > 0:
            score += _PAT_POSITIVE_PTS
            pros.append(f"Profitable — PAT of ₹{ipo.pat_cr:.0f} Cr.")
        else:
            cons.append(f"Loss-making — PAT of ₹{ipo.pat_cr:.0f} Cr.")
    else:
        cons.append("Profitability data not available.")

    # ROE
    if ipo.roe is not None:
        if ipo.roe >= _ROE_THRESHOLD:
            score += _ROE_PTS
            pros.append(f"High ROE of {ipo.roe:.1f}% — strong capital efficiency.")
        else:
            cons.append(f"Low ROE of {ipo.roe:.1f}% (below 15% benchmark).")
    else:
        cons.append("ROE data not available.")

    # D/E
    if ipo.debt_equity is not None:
        if ipo.debt_equity < _DE_THRESHOLD:
            score += _DE_PTS
            pros.append(
                f"Conservative debt — D/E of {ipo.debt_equity:.2f}x."
            )
        else:
            cons.append(
                f"Elevated leverage — D/E of {ipo.debt_equity:.2f}x (above 1.0)."
            )
    else:
        cons.append("Debt/equity data not available.")

    return score, pros, cons


def _promoter_score(ipo: IPOData) -> Tuple[float, List[str], List[str]]:
    """Return (score, pros, cons) for the promoter/governance dimension."""
    pros: List[str] = []
    cons: List[str] = []

    post = ipo.promoter_holding_post

    if post is None:
        return _PROMOTER_MID_PTS, ["Promoter holding post-IPO not disclosed."], []

    if post >= _PROMOTER_HIGH_THRESHOLD:
        score = _PROMOTER_HIGH_PTS
        pros.append(
            f"Strong promoter commitment — {post:.1f}% stake retained post-IPO."
        )
    elif post >= _PROMOTER_MID_THRESHOLD:
        score = _PROMOTER_MID_PTS
        pros.append(f"Adequate promoter holding of {post:.1f}% post-IPO.")
    else:
        score = _PROMOTER_LOW_PTS
        cons.append(
            f"Low promoter stake post-IPO ({post:.1f}%) — significant dilution."
        )

    # Flag if pre→post drop is very large
    pre = ipo.promoter_holding_pre
    if pre is not None and (pre - post) > 30:
        cons.append(
            f"Promoters diluting significantly from {pre:.1f}% to {post:.1f}%."
        )

    return score, pros, cons


def _risk_level(
    ipo: IPOData,
    total_score: float,
    subscription_score: float,
) -> str:
    """
    Determine risk level: "HIGH", "MEDIUM", or "LOW".

    Logic:
        - Subscription < 5x or no QIB → HIGH risk
        - Negative GMP → bump risk up one level
        - HIGH subscription + HIGH GMP → LOW risk
    """
    risk = "MEDIUM"

    if ipo.subscription_total < 5.0 or ipo.subscription_qib < 1.0:
        risk = "HIGH"
    elif ipo.subscription_total >= 50.0 and ipo.gmp_pct >= 15.0:
        risk = "LOW"
    elif total_score >= 70:
        risk = "LOW"
    elif total_score <= 30:
        risk = "HIGH"

    # Negative GMP bumps risk up one level
    if ipo.gmp_pct < 0:
        if risk == "LOW":
            risk = "MEDIUM"
        elif risk == "MEDIUM":
            risk = "HIGH"

    return risk


def _expected_listing_gain(ipo: IPOData) -> Optional[float]:
    """
    Estimate expected listing gain as a percentage.

    Uses GMP as the primary signal when available, blended with subscription.
    Returns None if there is no reliable data.
    """
    if ipo.issue_price_max <= 0:
        return None

    # GMP-based estimate
    if ipo.gmp_pct != 0:
        # Assume ~70% of GMP is realised on listing day
        estimate = ipo.gmp_pct * 0.70
        return round(estimate, 2)

    # Subscription-based heuristic when GMP is unknown
    if ipo.subscription_total > 0:
        if ipo.subscription_total >= 100:
            return 12.0
        elif ipo.subscription_total >= 50:
            return 8.0
        elif ipo.subscription_total >= 20:
            return 5.0
        elif ipo.subscription_total >= 10:
            return 2.0

    return None


def _generate_summary(
    ipo: IPOData,
    recommendation: str,
    score: float,
    risk: str,
    listing_gain: Optional[float],
) -> str:
    """Generate a 2-3 sentence plain-English analysis summary."""
    status_phrase = {
        "active": "is currently open for subscription",
        "upcoming": "is opening soon for subscription",
        "listed": "recently listed on NSE",
    }.get(ipo.status, "is an IPO")

    lines = [
        f"{ipo.company_name} {status_phrase} in the {ipo.sector} sector "
        f"and scores {score:.0f}/100 on our composite model."
    ]

    if recommendation == "INVEST":
        lines.append(
            "The IPO presents an attractive opportunity driven by strong subscription "
            "demand, favourable valuation, and healthy financials."
        )
    elif recommendation == "AVOID":
        lines.append(
            "The IPO shows concerning signals including weak demand, stretched valuation, "
            "or poor financials — investors should exercise caution."
        )
    else:
        lines.append(
            "The IPO has mixed signals and may suit investors with moderate risk "
            "tolerance who are comfortable with the sector."
        )

    if listing_gain is not None and ipo.status != "listed":
        direction = "gain" if listing_gain >= 0 else "loss"
        lines.append(
            f"Based on current GMP, an estimated listing {direction} of "
            f"{abs(listing_gain):.1f}% is expected (risk level: {risk})."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_ipo(ipo: IPOData) -> Dict:
    """
    Score an IPO across five dimensions and return a recommendation dict.

    Args:
        ipo: An :class:`~src.data.ipo.IPOData` instance.

    Returns:
        Dictionary with keys:
            - ``recommendation`` — "INVEST" | "NEUTRAL" | "AVOID"
            - ``score``          — composite score 0-100
            - ``risk_level``     — "HIGH" | "MEDIUM" | "LOW"
            - ``expected_listing_gain_pct`` — float or None
            - ``pros``           — list of positive bullet strings
            - ``cons``           — list of negative bullet strings
            - ``analysis_summary`` — plain-English 2-3 sentence analysis
            - ``dimension_scores`` — individual dimension scores dict
    """
    all_pros: List[str] = []
    all_cons: List[str] = []

    # --- Dimension scores ---
    g_score, g_pros, g_cons = _gmp_score(ipo)
    s_score, s_pros, s_cons = _subscription_score(ipo)
    v_score, v_pros, v_cons = _valuation_score(ipo)
    f_score, f_pros, f_cons = _financial_score(ipo)
    p_score, p_pros, p_cons = _promoter_score(ipo)

    all_pros.extend(g_pros + s_pros + v_pros + f_pros + p_pros)
    all_cons.extend(g_cons + s_cons + v_cons + f_cons + p_cons)

    total = g_score + s_score + v_score + f_score + p_score
    total = min(total, 100.0)

    # --- Recommendation ---
    if total >= _INVEST_THRESHOLD:
        recommendation = "INVEST"
    elif total <= _AVOID_THRESHOLD:
        recommendation = "AVOID"
    else:
        recommendation = "NEUTRAL"

    # --- Risk ---
    risk = _risk_level(ipo, total, s_score)

    # --- Expected listing gain ---
    listing_gain = _expected_listing_gain(ipo)

    # --- Summary ---
    summary = _generate_summary(ipo, recommendation, total, risk, listing_gain)

    return {
        "recommendation": recommendation,
        "score": round(total, 2),
        "risk_level": risk,
        "expected_listing_gain_pct": listing_gain,
        "pros": all_pros,
        "cons": all_cons,
        "analysis_summary": summary,
        "dimension_scores": {
            "gmp_score": round(g_score, 2),
            "subscription_score": round(s_score, 2),
            "valuation_score": round(v_score, 2),
            "financial_score": round(f_score, 2),
            "fundamentals_score": round(p_score, 2),
        },
    }


def enrich_ipo(ipo: IPOData) -> IPOData:
    """
    Compute and attach the IPO score/recommendation to the IPO object in-place.

    The result dict is stored as ``ipo.score_result`` (a plain attribute added
    dynamically).  The original dataclass fields are not modified.

    Args:
        ipo: An :class:`~src.data.ipo.IPOData` instance.

    Returns:
        The same ``ipo`` object with a ``score_result`` attribute attached.
    """
    result = score_ipo(ipo)
    # Attach as a dynamic attribute so callers can access ipo.score_result
    object.__setattr__(ipo, "score_result", result) if hasattr(ipo, "__dataclass_fields__") else setattr(ipo, "score_result", result)
    return ipo
