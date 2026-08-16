# src/data/nse_stocks.py

"""
Full NSE-listed equity universe (~2,400 companies), used for search /
"all stocks" coverage. Snapshotted from NSE's public equity list
(archives.nseindia.com/content/equities/EQUITY_L.csv) rather than
fetched live at startup — that endpoint is unofficial/scrape-only and
the list changes rarely, so a periodically-refreshed static file is
more reliable than a live call on every server boot.

To refresh: re-run the generation snippet in NEXT_STEPS.md under
"Refreshing the NSE stock list".

Key:   Human-readable company name
Value: Yahoo Finance ticker symbol
"""

import json
from pathlib import Path
from typing import Dict

_DATA_FILE = Path(__file__).parent / "nse_equity_list.json"


def _load() -> Dict[str, str]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


ALL_NSE_STOCKS: Dict[str, str] = _load()
