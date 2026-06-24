# src/data/ipo.py

"""
IPO data fetcher for NSE-listed Indian IPOs.

Attempts to scrape live data from Chittorgarh and NSE India.
Falls back to realistic sample data if scraping fails.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import re
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP headers — mimic a real browser to reduce bot-detection
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

REQUEST_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IPOData:
    """Unified representation of a single IPO."""

    id: str                           # slug / identifier used in URLs
    company_name: str
    symbol: Optional[str]             # NSE symbol after listing
    status: str                       # "active" | "upcoming" | "listed"
    open_date: str                    # ISO date string YYYY-MM-DD
    close_date: str
    listing_date: Optional[str]
    issue_price_min: float
    issue_price_max: float
    lot_size: int
    issue_size_cr: float              # issue size in crores (₹)
    gmp: float                        # grey market premium (absolute ₹)
    gmp_pct: float                    # GMP as % of issue price max
    subscription_total: float         # times subscribed
    subscription_qib: float           # QIB subscription (times)
    subscription_nii: float           # NII / HNI subscription (times)
    subscription_rii: float           # Retail subscription (times)
    sector: str
    registrar: str
    lead_managers: List[str] = field(default_factory=list)

    # ---- Detail fields (populated by fetch_ipo_detail) ----
    company_description: str = ""
    revenue_cr: Optional[float] = None
    ebitda_cr: Optional[float] = None
    pat_cr: Optional[float] = None
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_equity: Optional[float] = None
    promoter_holding_pre: Optional[float] = None
    promoter_holding_post: Optional[float] = None
    industry_pe: Optional[float] = None
    objects_of_issue: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Return a plain dictionary (JSON-serialisable)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Sample / fallback IPO data
# ---------------------------------------------------------------------------

_FALLBACK_IPOS: List[IPOData] = [
    IPOData(
        id="infratech-solutions-ipo",
        company_name="InfraTech Solutions Ltd",
        symbol="INFRATECH",
        status="active",
        open_date="2025-06-20",
        close_date="2025-06-24",
        listing_date="2025-07-02",
        issue_price_min=420.0,
        issue_price_max=440.0,
        lot_size=34,
        issue_size_cr=1850.0,
        gmp=88.0,
        gmp_pct=20.0,
        subscription_total=48.5,
        subscription_qib=72.3,
        subscription_nii=55.1,
        subscription_rii=22.4,
        sector="IT & Technology",
        registrar="Link Intime India Pvt Ltd",
        lead_managers=["ICICI Securities", "Kotak Mahindra Capital"],
        company_description=(
            "InfraTech Solutions provides cloud-managed infrastructure "
            "services to enterprise clients across India and Southeast Asia."
        ),
        revenue_cr=680.0,
        ebitda_cr=142.0,
        pat_cr=89.0,
        eps=24.5,
        pe_ratio=17.9,
        roe=18.2,
        debt_equity=0.4,
        promoter_holding_pre=74.5,
        promoter_holding_post=58.2,
        industry_pe=22.0,
        objects_of_issue=[
            "Expansion of data centre capacity",
            "Repayment of borrowings",
            "General corporate purposes",
        ],
    ),
    IPOData(
        id="greenpower-energy-ipo",
        company_name="GreenPower Energy Ltd",
        symbol="GRPWR",
        status="upcoming",
        open_date="2025-07-05",
        close_date="2025-07-09",
        listing_date=None,
        issue_price_min=0.0,
        issue_price_max=280.0,
        lot_size=53,
        issue_size_cr=2400.0,
        gmp=0.0,
        gmp_pct=0.0,
        subscription_total=0.0,
        subscription_qib=0.0,
        subscription_nii=0.0,
        subscription_rii=0.0,
        sector="Energy & Oil",
        registrar="KFin Technologies Ltd",
        lead_managers=["Axis Capital", "SBI Capital Markets", "HDFC Bank"],
        company_description=(
            "GreenPower Energy develops and operates utility-scale solar and "
            "wind energy projects with an installed capacity of 3.2 GW."
        ),
        revenue_cr=1120.0,
        pat_cr=210.0,
        pe_ratio=21.0,
        industry_pe=25.0,
        promoter_holding_pre=68.0,
        promoter_holding_post=55.0,
        objects_of_issue=[
            "Funding capital expenditure for renewable projects",
            "Prepayment of outstanding debt",
        ],
    ),
    IPOData(
        id="swiftmed-pharma-ipo",
        company_name="SwiftMed Pharma Ltd",
        symbol="SWIFTMED",
        status="listed",
        open_date="2025-05-12",
        close_date="2025-05-15",
        listing_date="2025-05-22",
        issue_price_min=310.0,
        issue_price_max=326.0,
        lot_size=46,
        issue_size_cr=960.0,
        gmp=52.0,
        gmp_pct=15.95,
        subscription_total=112.4,
        subscription_qib=180.3,
        subscription_nii=98.7,
        subscription_rii=60.2,
        sector="Pharma & Healthcare",
        registrar="Bigshare Services Pvt Ltd",
        lead_managers=["JM Financial", "Emkay Global"],
        company_description=(
            "SwiftMed Pharma manufactures generic APIs and formulations "
            "exported to 40+ countries under USFDA-approved facilities."
        ),
        revenue_cr=540.0,
        ebitda_cr=108.0,
        pat_cr=74.0,
        eps=19.2,
        pe_ratio=16.9,
        roe=22.5,
        debt_equity=0.3,
        promoter_holding_pre=69.8,
        promoter_holding_post=54.1,
        industry_pe=28.0,
        objects_of_issue=[
            "Capacity expansion of API manufacturing",
            "Working capital requirements",
            "General corporate purposes",
        ],
    ),
    IPOData(
        id="bharat-logistics-ipo",
        company_name="Bharat Logistics Network Ltd",
        symbol="BHLNET",
        status="listed",
        open_date="2025-04-03",
        close_date="2025-04-07",
        listing_date="2025-04-16",
        issue_price_min=180.0,
        issue_price_max=195.0,
        lot_size=77,
        issue_size_cr=1450.0,
        gmp=-9.75,
        gmp_pct=-5.0,
        subscription_total=3.8,
        subscription_qib=5.1,
        subscription_nii=2.4,
        subscription_rii=3.2,
        sector="Aviation & Logistics",
        registrar="Link Intime India Pvt Ltd",
        lead_managers=["Elara Capital", "Choice Capital"],
        company_description=(
            "Bharat Logistics Network operates a pan-India last-mile delivery "
            "and cold-chain warehousing platform serving FMCG brands."
        ),
        revenue_cr=820.0,
        pat_cr=32.0,
        pe_ratio=42.0,
        roe=8.4,
        debt_equity=1.8,
        promoter_holding_pre=62.0,
        promoter_holding_post=48.0,
        industry_pe=30.0,
        objects_of_issue=[
            "Acquisition of fleet vehicles",
            "Expansion of warehousing network",
        ],
    ),
    IPOData(
        id="nexgen-consumer-ipo",
        company_name="NexGen Consumer Brands Ltd",
        symbol="NEXGCB",
        status="upcoming",
        open_date="2025-07-18",
        close_date="2025-07-22",
        listing_date=None,
        issue_price_min=0.0,
        issue_price_max=155.0,
        lot_size=96,
        issue_size_cr=720.0,
        gmp=0.0,
        gmp_pct=0.0,
        subscription_total=0.0,
        subscription_qib=0.0,
        subscription_nii=0.0,
        subscription_rii=0.0,
        sector="FMCG & Consumer",
        registrar="Purva Sharegistry India Pvt Ltd",
        lead_managers=["Nuvama Wealth", "Systematix Securities"],
        company_description=(
            "NexGen Consumer Brands is a D2C personal care and homecare "
            "company selling through digital channels and quick-commerce."
        ),
        revenue_cr=240.0,
        pat_cr=18.0,
        pe_ratio=35.0,
        roe=14.0,
        debt_equity=0.6,
        promoter_holding_pre=78.0,
        promoter_holding_post=61.0,
        industry_pe=40.0,
        objects_of_issue=[
            "Brand building and marketing",
            "Expansion into new product categories",
            "General corporate purposes",
        ],
    ),
    IPOData(
        id="stellar-defence-ipo",
        company_name="Stellar Defence & Aerospace Ltd",
        symbol="STELDEF",
        status="active",
        open_date="2025-06-18",
        close_date="2025-06-21",
        listing_date="2025-07-01",
        issue_price_min=590.0,
        issue_price_max=620.0,
        lot_size=24,
        issue_size_cr=3100.0,
        gmp=155.0,
        gmp_pct=25.0,
        subscription_total=89.2,
        subscription_qib=145.6,
        subscription_nii=78.3,
        subscription_rii=42.1,
        sector="Defence & Aerospace",
        registrar="KFin Technologies Ltd",
        lead_managers=["ICICI Securities", "Axis Capital", "BOB Capital Markets"],
        company_description=(
            "Stellar Defence manufactures precision-guided munitions, drone "
            "subsystems, and radar electronics for Indian defence forces."
        ),
        revenue_cr=1240.0,
        ebitda_cr=298.0,
        pat_cr=182.0,
        eps=35.8,
        pe_ratio=17.3,
        roe=26.4,
        debt_equity=0.2,
        promoter_holding_pre=80.0,
        promoter_holding_post=63.5,
        industry_pe=35.0,
        objects_of_issue=[
            "Capacity expansion of manufacturing facilities",
            "R&D for next-generation defence systems",
            "Repayment of working capital loans",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(text: str) -> Optional[float]:
    """Parse a numeric string, stripping commas/₹/% and returning None on failure."""
    if not text:
        return None
    cleaned = re.sub(r"[₹,%\s]", "", str(text).replace(",", "").strip())
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _slugify(name: str) -> str:
    """Convert a company name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug + "-ipo"


def _get_session() -> requests.Session:
    """Return a requests session with preset headers."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


# ---------------------------------------------------------------------------
# Chittorgarh scraper
# ---------------------------------------------------------------------------

def _fetch_chittorgarh_list() -> List[IPOData]:
    """
    Scrape IPO list from https://www.chittorgarh.com/ipo/ipo_dashboard.asp

    Returns an empty list if the request fails or the page structure changes.
    """
    url = "https://www.chittorgarh.com/ipo/ipo_dashboard.asp"
    session = _get_session()
    ipos: List[IPOData] = []

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Chittorgarh request failed: %s", exc)
        return ipos

    soup = BeautifulSoup(resp.text, "html.parser")

    # Chittorgarh uses multiple tables — one per status bucket
    status_map = {
        "open": "active",
        "upcoming": "upcoming",
        "recently listed": "listed",
    }

    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = heading.get_text(strip=True).lower()
        matched_status = None
        for key, val in status_map.items():
            if key in heading_text:
                matched_status = val
                break

        if matched_status is None:
            continue

        table = heading.find_next("table")
        if not table:
            continue

        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 5:
                continue

            company_name = cols[0]
            if not company_name:
                continue

            # Parse dates — format varies but common: "Jun 20, 2025"
            open_date = cols[1] if len(cols) > 1 else ""
            close_date = cols[2] if len(cols) > 2 else ""
            price_text = cols[3] if len(cols) > 3 else ""
            gmp_text = cols[4] if len(cols) > 4 else "0"

            price_min, price_max = 0.0, 0.0
            if "-" in price_text:
                parts = price_text.split("-")
                price_min = _safe_float(parts[0]) or 0.0
                price_max = _safe_float(parts[1]) or 0.0
            else:
                price_max = _safe_float(price_text) or 0.0
                price_min = price_max

            gmp_val = _safe_float(gmp_text) or 0.0
            gmp_pct = (gmp_val / price_max * 100) if price_max > 0 else 0.0

            ipo = IPOData(
                id=_slugify(company_name),
                company_name=company_name,
                symbol=None,
                status=matched_status,
                open_date=open_date,
                close_date=close_date,
                listing_date=None,
                issue_price_min=price_min,
                issue_price_max=price_max,
                lot_size=0,
                issue_size_cr=0.0,
                gmp=gmp_val,
                gmp_pct=round(gmp_pct, 2),
                subscription_total=0.0,
                subscription_qib=0.0,
                subscription_nii=0.0,
                subscription_rii=0.0,
                sector="Unknown",
                registrar="",
            )
            ipos.append(ipo)

    logger.info("Chittorgarh returned %d IPOs", len(ipos))
    return ipos


# ---------------------------------------------------------------------------
# NSE India scraper
# ---------------------------------------------------------------------------

def _fetch_nse_list() -> List[IPOData]:
    """
    Fetch upcoming IPO data from NSE India API.

    NSE offers a JSON endpoint; falls back gracefully on errors.
    """
    url = "https://www.nseindia.com/api/ipo-current-issue"
    session = _get_session()
    # NSE requires a cookie from the main page first
    try:
        session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("NSE seed request failed: %s", exc)
        return []

    ipos: List[IPOData] = []
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("NSE IPO API failed: %s", exc)
        return ipos

    for item in data if isinstance(data, list) else []:
        try:
            company_name = item.get("companyName", "")
            symbol = item.get("symbol") or None
            open_date = item.get("openDate", "")
            close_date = item.get("closeDate", "")
            listing_date = item.get("listingDate") or None
            price_min = float(item.get("minPrice", 0) or 0)
            price_max = float(item.get("maxPrice", 0) or 0)
            lot_size = int(item.get("lotSize", 0) or 0)
            issue_size = float(item.get("issueSize", 0) or 0)

            status = "upcoming"
            if item.get("status", "").lower() == "open":
                status = "active"
            elif listing_date:
                status = "listed"

            ipo = IPOData(
                id=_slugify(company_name),
                company_name=company_name,
                symbol=symbol,
                status=status,
                open_date=open_date,
                close_date=close_date,
                listing_date=listing_date,
                issue_price_min=price_min,
                issue_price_max=price_max,
                lot_size=lot_size,
                issue_size_cr=issue_size,
                gmp=0.0,
                gmp_pct=0.0,
                subscription_total=0.0,
                subscription_qib=0.0,
                subscription_nii=0.0,
                subscription_rii=0.0,
                sector="Unknown",
                registrar="",
            )
            ipos.append(ipo)
        except Exception as exc:
            logger.debug("Skipping NSE IPO row: %s", exc)
            continue

    logger.info("NSE returned %d IPOs", len(ipos))
    return ipos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ipo_list(status: str = "all") -> List[IPOData]:
    """
    Fetch IPO list from live sources with fallback to sample data.

    Args:
        status: Filter by status. One of ``"all"``, ``"active"``,
                ``"upcoming"``, or ``"listed"``.

    Returns:
        List of :class:`IPOData` objects ordered by open_date descending.
        Returns sample data when all live sources fail.
    """
    ipos: List[IPOData] = []

    # --- Try Chittorgarh first ---
    try:
        ipos = _fetch_chittorgarh_list()
    except Exception as exc:
        logger.warning("Chittorgarh scraper raised unexpectedly: %s", exc)

    # --- If Chittorgarh empty, try NSE ---
    if not ipos:
        try:
            ipos = _fetch_nse_list()
        except Exception as exc:
            logger.warning("NSE scraper raised unexpectedly: %s", exc)

    # --- Fall back to sample data ---
    if not ipos:
        logger.info("All live sources failed — returning fallback sample IPOs.")
        ipos = list(_FALLBACK_IPOS)

    # Apply status filter
    if status != "all":
        ipos = [ipo for ipo in ipos if ipo.status == status]

    # Sort: active first, then upcoming, then listed; within group by open_date desc
    _order = {"active": 0, "upcoming": 1, "listed": 2}
    ipos.sort(key=lambda x: (_order.get(x.status, 9), x.open_date), reverse=False)

    return ipos


def fetch_ipo_detail(ipo_id: str) -> Optional[IPOData]:
    """
    Fetch detailed IPO data by its ID/slug.

    Tries to scrape a Chittorgarh detail page; if that fails, searches the
    fallback list. Returns ``None`` if no matching IPO is found.

    Args:
        ipo_id: The IPO slug identifier (e.g. ``"infratech-solutions-ipo"``).

    Returns:
        :class:`IPOData` with detail fields populated, or ``None``.
    """
    # First check the fallback list — useful in offline / test scenarios
    for sample in _FALLBACK_IPOS:
        if sample.id == ipo_id:
            return sample

    # Try fetching a detail page from Chittorgarh
    # URL pattern: https://www.chittorgarh.com/ipo/{slug}/
    slug = ipo_id.replace("-ipo", "")
    url = f"https://www.chittorgarh.com/ipo/{slug}/"
    session = _get_session()

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("IPO detail fetch failed for %s: %s", ipo_id, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract key-value pairs from detail tables
    kv: Dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True)
            kv[key] = val

    def _kv(keys: List[str]) -> Optional[float]:
        for k in keys:
            for actual_key, val in kv.items():
                if k in actual_key:
                    return _safe_float(val)
        return None

    # Try to find the company name from the page <h1>
    h1 = soup.find("h1")
    company_name = h1.get_text(strip=True) if h1 else ipo_id

    # Derive price band
    price_text = kv.get("price band", kv.get("issue price", ""))
    price_min, price_max = 0.0, 0.0
    if "-" in price_text:
        parts = price_text.split("-")
        price_min = _safe_float(parts[0]) or 0.0
        price_max = _safe_float(parts[1]) or 0.0
    else:
        price_max = _safe_float(price_text) or 0.0
        price_min = price_max

    ipo = IPOData(
        id=ipo_id,
        company_name=company_name,
        symbol=None,
        status="listed",
        open_date=kv.get("open date", kv.get("ipo open date", "")),
        close_date=kv.get("close date", kv.get("ipo close date", "")),
        listing_date=kv.get("listing date") or None,
        issue_price_min=price_min,
        issue_price_max=price_max,
        lot_size=int(_kv(["lot size", "market lot"]) or 0),
        issue_size_cr=_kv(["issue size"]) or 0.0,
        gmp=_kv(["grey market", "gmp"]) or 0.0,
        gmp_pct=0.0,
        subscription_total=_kv(["total subscription", "overall subscription"]) or 0.0,
        subscription_qib=_kv(["qib"]) or 0.0,
        subscription_nii=_kv(["nii", "hni"]) or 0.0,
        subscription_rii=_kv(["rii", "retail"]) or 0.0,
        sector=kv.get("industry", kv.get("sector", "Unknown")),
        registrar=kv.get("registrar", ""),
        revenue_cr=_kv(["revenue", "total income"]),
        pat_cr=_kv(["pat", "profit after tax", "net profit"]),
        eps=_kv(["eps", "earnings per share"]),
        pe_ratio=_kv(["p/e", "pe ratio", "price/earnings"]),
        roe=_kv(["roe", "return on equity"]),
        debt_equity=_kv(["debt/equity", "d/e ratio"]),
        promoter_holding_pre=_kv(["promoter.*pre", "pre-ipo promoter"]),
        promoter_holding_post=_kv(["promoter.*post", "post-ipo promoter"]),
        industry_pe=_kv(["industry pe", "peer pe"]),
    )

    # Compute GMP %
    if ipo.issue_price_max > 0:
        ipo.gmp_pct = round(ipo.gmp / ipo.issue_price_max * 100, 2)

    return ipo
