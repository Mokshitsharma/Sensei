from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from dataclasses import asdict

from backend.schemas.ipo import IPOListItem, IPODetail, IPOFinancials

router = APIRouter(prefix="/ipo", tags=["ipo"])


def _safe_float(val, default=None):
    """Return a JSON-safe float or None."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f or f == float("inf") or f == float("-inf"):
            return default
        return f
    except Exception:
        return default


def _safe_str(val, default="") -> str:
    if val is None:
        return default
    return str(val)


def _ipo_to_list_item(ipo, score_data: dict) -> IPOListItem:
    """Convert an IPOData dataclass + score_data dict into an IPOListItem."""
    d = asdict(ipo) if hasattr(ipo, "__dataclass_fields__") else dict(ipo)

    return IPOListItem(
        id=_safe_str(d.get("id")),
        company_name=_safe_str(d.get("company_name")),
        symbol=d.get("symbol") or None,
        status=_safe_str(d.get("status", "upcoming")),
        open_date=_safe_str(d.get("open_date", "")),
        close_date=_safe_str(d.get("close_date", "")),
        listing_date=d.get("listing_date") or None,
        issue_price_min=_safe_float(d.get("issue_price_min"), 0.0),
        issue_price_max=_safe_float(d.get("issue_price_max"), 0.0),
        lot_size=int(d.get("lot_size") or 0),
        issue_size_cr=_safe_float(d.get("issue_size_cr"), 0.0),
        gmp=_safe_float(d.get("gmp"), 0.0),
        gmp_pct=_safe_float(d.get("gmp_pct"), 0.0),
        subscription_total=_safe_float(d.get("subscription_total"), 0.0),
        subscription_qib=_safe_float(d.get("subscription_qib"), 0.0),
        subscription_nii=_safe_float(d.get("subscription_nii"), 0.0),
        subscription_rii=_safe_float(d.get("subscription_rii"), 0.0),
        recommendation=_safe_str(score_data.get("recommendation", "NEUTRAL")),
        recommendation_score=_safe_float(score_data.get("recommendation_score", score_data.get("score")), 0.0),
        expected_listing_gain_pct=_safe_float(score_data.get("expected_listing_gain_pct")),
        risk_level=_safe_str(score_data.get("risk_level", "MEDIUM")),
        sector=_safe_str(d.get("sector", "Unknown")),
        registrar=_safe_str(d.get("registrar", "")),
    )


def _ipo_to_detail(ipo, score_data: dict) -> IPODetail:
    """Convert an IPOData dataclass + score_data dict into an IPODetail."""
    d = asdict(ipo) if hasattr(ipo, "__dataclass_fields__") else dict(ipo)

    financials = IPOFinancials(
        revenue_cr=_safe_float(d.get("revenue_cr")),
        ebitda_cr=_safe_float(d.get("ebitda_cr")),
        pat_cr=_safe_float(d.get("pat_cr")),
        eps=_safe_float(d.get("eps")),
        pe_ratio=_safe_float(d.get("pe_ratio")),
        roe=_safe_float(d.get("roe")),
        debt_equity=_safe_float(d.get("debt_equity")),
    )

    lead_managers = d.get("lead_managers") or []
    if isinstance(lead_managers, str):
        lead_managers = [lead_managers] if lead_managers else []

    objects_of_issue = d.get("objects_of_issue") or []
    if isinstance(objects_of_issue, str):
        objects_of_issue = [objects_of_issue] if objects_of_issue else []

    return IPODetail(
        # IPOListItem fields
        id=_safe_str(d.get("id")),
        company_name=_safe_str(d.get("company_name")),
        symbol=d.get("symbol") or None,
        status=_safe_str(d.get("status", "upcoming")),
        open_date=_safe_str(d.get("open_date", "")),
        close_date=_safe_str(d.get("close_date", "")),
        listing_date=d.get("listing_date") or None,
        issue_price_min=_safe_float(d.get("issue_price_min"), 0.0),
        issue_price_max=_safe_float(d.get("issue_price_max"), 0.0),
        lot_size=int(d.get("lot_size") or 0),
        issue_size_cr=_safe_float(d.get("issue_size_cr"), 0.0),
        gmp=_safe_float(d.get("gmp"), 0.0),
        gmp_pct=_safe_float(d.get("gmp_pct"), 0.0),
        subscription_total=_safe_float(d.get("subscription_total"), 0.0),
        subscription_qib=_safe_float(d.get("subscription_qib"), 0.0),
        subscription_nii=_safe_float(d.get("subscription_nii"), 0.0),
        subscription_rii=_safe_float(d.get("subscription_rii"), 0.0),
        recommendation=_safe_str(score_data.get("recommendation", "NEUTRAL")),
        recommendation_score=_safe_float(score_data.get("recommendation_score", score_data.get("score")), 0.0),
        expected_listing_gain_pct=_safe_float(score_data.get("expected_listing_gain_pct")),
        risk_level=_safe_str(score_data.get("risk_level", "MEDIUM")),
        sector=_safe_str(d.get("sector", "Unknown")),
        registrar=_safe_str(d.get("registrar", "")),
        # IPODetail-specific fields
        company_description=_safe_str(d.get("company_description", "")),
        financials=financials,
        promoter_holding_pre=_safe_float(d.get("promoter_holding_pre")),
        promoter_holding_post=_safe_float(d.get("promoter_holding_post")),
        industry_pe=_safe_float(d.get("industry_pe")),
        objects_of_issue=list(objects_of_issue),
        pros=list(score_data.get("pros") or []),
        cons=list(score_data.get("cons") or []),
        analysis_summary=_safe_str(score_data.get("analysis_summary", "")),
        lead_managers=list(lead_managers),
        dimension_scores=dict(score_data.get("dimension_scores") or {}),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=List[IPOListItem])
async def list_ipos(
    status: Optional[str] = Query(None, description="active|upcoming|listed|all"),
):
    """List IPOs with optional status filter."""
    from src.data.ipo import fetch_ipo_list
    from src.domain.ipo_analyzer import score_ipo

    fetch_status = status if status else "all"
    ipos = fetch_ipo_list(fetch_status)

    result: List[IPOListItem] = []
    for ipo in ipos:
        try:
            score_data = score_ipo(ipo)
        except Exception:
            score_data = {"recommendation": "NEUTRAL", "score": 0.0, "risk_level": "MEDIUM"}
        try:
            result.append(_ipo_to_list_item(ipo, score_data))
        except Exception:
            continue

    return result


@router.get("/{ipo_id}", response_model=IPODetail)
async def get_ipo(ipo_id: str):
    """Get detailed IPO info with recommendation."""
    from src.data.ipo import fetch_ipo_list
    from src.domain.ipo_analyzer import score_ipo

    all_ipos = fetch_ipo_list("all")
    ipo = None
    for item in all_ipos:
        item_id = (
            item.id
            if hasattr(item, "id")
            else (asdict(item).get("id") if hasattr(item, "__dataclass_fields__") else None)
        )
        if str(item_id) == str(ipo_id):
            ipo = item
            break

    if ipo is None:
        raise HTTPException(status_code=404, detail=f"IPO '{ipo_id}' not found")

    try:
        score_data = score_ipo(ipo)
    except Exception:
        score_data = {"recommendation": "NEUTRAL", "score": 0.0, "risk_level": "MEDIUM"}

    return _ipo_to_detail(ipo, score_data)
