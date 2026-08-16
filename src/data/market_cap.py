# src/data/market_cap.py
"""Curated mid-cap / small-cap NSE constituents, in the same style as
nifty50.py (NIFTY_50 already serves as the large-cap list). Computing a
live 7-day prediction for the full ~2,400-stock NSE universe on every
request isn't feasible on a single instance, so the "gainers/losers" and
"by market cap" home-page sections are scoped to these curated, well-known
names — the same tradeoff nifty50.py already makes for large caps.
"""

MID_CAP = {
    "Federal Bank": "FEDERALBNK.NS",
    "AU Small Finance Bank": "AUBANK.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Coforge": "COFORGE.NS",
    "Astral": "ASTRAL.NS",
    "Page Industries": "PAGEIND.NS",
    "Voltas": "VOLTAS.NS",
    "Indian Hotels": "INDHOTEL.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Polycab India": "POLYCAB.NS",
    "PI Industries": "PIIND.NS",
    "Balkrishna Industries": "BALKRISIND.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Jubilant Foodworks": "JUBLFOOD.NS",
}

SMALL_CAP = {
    "Blue Star": "BLUESTARCO.NS",
    "KEI Industries": "KEI.NS",
    "Sonata Software": "SONATSOFTW.NS",
    "Aarti Industries": "AARTIIND.NS",
    "Ratnamani Metals": "RATNAMANI.NS",
    "CDSL": "CDSL.NS",
    "Radico Khaitan": "RADICO.NS",
    "Redington": "REDINGTON.NS",
    "Grindwell Norton": "GRINDWELL.NS",
    "TTK Prestige": "TTKPRESTIG.NS",
    "VIP Industries": "VIPIND.NS",
    "Sundram Fasteners": "SUNDRMFAST.NS",
    "Craftsman Automation": "CRAFTSMAN.NS",
    "Zee Entertainment": "ZEEL.NS",
    "MCX": "MCX.NS",
}
