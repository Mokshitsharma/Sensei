from pydantic import BaseModel
from typing import Optional, List, Dict


class IPOListItem(BaseModel):
    id: str
    company_name: str
    symbol: Optional[str] = None
    status: str  # "active" | "upcoming" | "listed"
    open_date: str
    close_date: str
    listing_date: Optional[str] = None
    issue_price_min: float
    issue_price_max: float
    lot_size: int
    issue_size_cr: float
    gmp: float
    gmp_pct: float
    subscription_total: float
    subscription_qib: float
    subscription_nii: float
    subscription_rii: float
    recommendation: str       # "INVEST" | "AVOID" | "NEUTRAL"
    recommendation_score: float
    expected_listing_gain_pct: Optional[float] = None
    risk_level: str           # "HIGH" | "MEDIUM" | "LOW"
    sector: str
    registrar: str


class IPOFinancials(BaseModel):
    revenue_cr: Optional[float] = None
    ebitda_cr: Optional[float] = None
    pat_cr: Optional[float] = None
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_equity: Optional[float] = None


class IPODetail(IPOListItem):
    company_description: str = ""
    financials: IPOFinancials = IPOFinancials()
    promoter_holding_pre: Optional[float] = None
    promoter_holding_post: Optional[float] = None
    industry_pe: Optional[float] = None
    objects_of_issue: List[str] = []
    pros: List[str] = []
    cons: List[str] = []
    analysis_summary: str = ""
    lead_managers: List[str] = []
    dimension_scores: Dict[str, float] = {}
