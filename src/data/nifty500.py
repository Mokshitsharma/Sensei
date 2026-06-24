# src/data/nifty500.py

"""
NIFTY 500 constituents mapping — ~500 major NSE-listed Indian stocks.

Structure:
    Company Name -> {symbol, sector, tier}

tier:
    "large"  — NIFTY 100 / large cap (market cap > ~20,000 Cr)
    "mid"    — NIFTY Midcap 150 (market cap ~5,000–20,000 Cr)
    "small"  — NIFTY Smallcap 250 (market cap < ~5,000 Cr)

symbol:
    Yahoo Finance ticker (suffix ".NS" for NSE)
"""

from typing import TypedDict, Dict


class StockInfo(TypedDict):
    symbol: str
    sector: str
    tier: str


NIFTY_500: Dict[str, StockInfo] = {

    # =========================================================================
    # LARGE CAP — NIFTY 100
    # =========================================================================

    # --- IT & Technology ---
    "TCS": {
        "symbol": "TCS.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "Infosys": {
        "symbol": "INFY.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "HCL Technologies": {
        "symbol": "HCLTECH.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "Wipro": {
        "symbol": "WIPRO.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "Tech Mahindra": {
        "symbol": "TECHM.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "LTIMindtree": {
        "symbol": "LTIM.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },

    # --- Banking & Finance ---
    "HDFC Bank": {
        "symbol": "HDFCBANK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "ICICI Bank": {
        "symbol": "ICICIBANK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Kotak Mahindra Bank": {
        "symbol": "KOTAKBANK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "SBI": {
        "symbol": "SBIN.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Axis Bank": {
        "symbol": "AXISBANK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "IndusInd Bank": {
        "symbol": "INDUSINDBK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Bajaj Finance": {
        "symbol": "BAJFINANCE.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Bajaj Finserv": {
        "symbol": "BAJAJFINSV.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "SBI Life": {
        "symbol": "SBILIFE.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "HDFC Life": {
        "symbol": "HDFCLIFE.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "AU Small Finance Bank": {
        "symbol": "AUBANK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Cholamandalam Investment": {
        "symbol": "CHOLAFIN.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "ICICI Prudential Life": {
        "symbol": "ICICIPRULI.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Bank of Baroda": {
        "symbol": "BANKBARODA.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Canara Bank": {
        "symbol": "CANBK.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Muthoot Finance": {
        "symbol": "MUTHOOTFIN.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "SBI Cards": {
        "symbol": "SBICARD.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },

    # --- Energy & Oil ---
    "Reliance Industries": {
        "symbol": "RELIANCE.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "ONGC": {
        "symbol": "ONGC.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "BPCL": {
        "symbol": "BPCL.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "NTPC": {
        "symbol": "NTPC.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "Power Grid": {
        "symbol": "POWERGRID.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "Coal India": {
        "symbol": "COALINDIA.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "GAIL India": {
        "symbol": "GAIL.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "Indian Oil": {
        "symbol": "IOC.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "Adani Green Energy": {
        "symbol": "ADANIGREEN.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "Adani Total Gas": {
        "symbol": "ATGL.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },
    "JSW Energy": {
        "symbol": "JSWENERGY.NS",
        "sector": "Energy & Oil",
        "tier": "large",
    },

    # --- Conglomerates & Industrials ---
    "Larsen & Toubro": {
        "symbol": "LT.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },
    "Adani Enterprises": {
        "symbol": "ADANIENT.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },
    "Adani Ports": {
        "symbol": "ADANIPORTS.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },
    "Grasim": {
        "symbol": "GRASIM.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },
    "Siemens India": {
        "symbol": "SIEMENS.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },
    "ABB India": {
        "symbol": "ABB.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "large",
    },

    # --- Pharma & Healthcare ---
    "Sun Pharma": {
        "symbol": "SUNPHARMA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Divi's Laboratories": {
        "symbol": "DIVISLAB.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Cipla": {
        "symbol": "CIPLA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Dr Reddy's Labs": {
        "symbol": "DRREDDY.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Apollo Hospitals": {
        "symbol": "APOLLOHOSP.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Aurobindo Pharma": {
        "symbol": "AUROPHARMA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Lupin": {
        "symbol": "LUPIN.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Max Healthcare": {
        "symbol": "MAXHEALTH.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },
    "Torrent Pharma": {
        "symbol": "TORNTPHARM.NS",
        "sector": "Pharma & Healthcare",
        "tier": "large",
    },

    # --- FMCG & Consumer ---
    "Hindustan Unilever": {
        "symbol": "HINDUNILVR.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "ITC": {
        "symbol": "ITC.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Nestle India": {
        "symbol": "NESTLEIND.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Britannia": {
        "symbol": "BRITANNIA.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Asian Paints": {
        "symbol": "ASIANPAINT.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Dabur India": {
        "symbol": "DABUR.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Godrej Consumer Products": {
        "symbol": "GODREJCP.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Marico": {
        "symbol": "MARICO.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Colgate-Palmolive India": {
        "symbol": "COLPAL.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Tata Consumer": {
        "symbol": "TATACONSUM.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Varun Beverages": {
        "symbol": "VBL.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "United Breweries": {
        "symbol": "UBL.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },

    # --- Automobiles ---
    "Maruti Suzuki": {
        "symbol": "MARUTI.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Tata Motors": {
        "symbol": "TATAMOTORS.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Bajaj Auto": {
        "symbol": "BAJAJ-AUTO.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Hero MotoCorp": {
        "symbol": "HEROMOTOCO.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Eicher Motors": {
        "symbol": "EICHERMOT.NS",
        "sector": "Automobiles",
        "tier": "large",
    },

    # --- Metals & Mining ---
    "Tata Steel": {
        "symbol": "TATASTEEL.NS",
        "sector": "Metals & Mining",
        "tier": "large",
    },
    "JSW Steel": {
        "symbol": "JSWSTEEL.NS",
        "sector": "Metals & Mining",
        "tier": "large",
    },
    "Hindalco": {
        "symbol": "HINDALCO.NS",
        "sector": "Metals & Mining",
        "tier": "large",
    },
    "Vedanta": {
        "symbol": "VEDL.NS",
        "sector": "Metals & Mining",
        "tier": "large",
    },
    "Jindal Steel": {
        "symbol": "JINDALSTEL.NS",
        "sector": "Metals & Mining",
        "tier": "large",
    },

    # --- Cement ---
    "UltraTech Cement": {
        "symbol": "ULTRACEMCO.NS",
        "sector": "Cement",
        "tier": "large",
    },
    "Shree Cement": {
        "symbol": "SHREECEM.NS",
        "sector": "Cement",
        "tier": "large",
    },
    "Ambuja Cements": {
        "symbol": "AMBUJACEM.NS",
        "sector": "Cement",
        "tier": "large",
    },

    # --- Retail & E-Commerce ---
    "Avenue Supermarts": {
        "symbol": "DMART.NS",
        "sector": "Retail & E-Commerce",
        "tier": "large",
    },
    "Titan": {
        "symbol": "TITAN.NS",
        "sector": "Retail & E-Commerce",
        "tier": "large",
    },
    "Trent": {
        "symbol": "TRENT.NS",
        "sector": "Retail & E-Commerce",
        "tier": "large",
    },
    "Nykaa": {
        "symbol": "NYKAA.NS",
        "sector": "Retail & E-Commerce",
        "tier": "large",
    },
    "Zomato": {
        "symbol": "ZOMATO.NS",
        "sector": "Retail & E-Commerce",
        "tier": "large",
    },

    # --- Telecom ---
    "Bharti Airtel": {
        "symbol": "BHARTIARTL.NS",
        "sector": "Telecom",
        "tier": "large",
    },
    "Tata Communications": {
        "symbol": "TATACOMM.NS",
        "sector": "Telecom",
        "tier": "large",
    },

    # --- Real Estate ---
    "DLF": {
        "symbol": "DLF.NS",
        "sector": "Real Estate",
        "tier": "large",
    },
    "Godrej Properties": {
        "symbol": "GODREJPROP.NS",
        "sector": "Real Estate",
        "tier": "large",
    },
    "Macrotech Developers": {
        "symbol": "LODHA.NS",
        "sector": "Real Estate",
        "tier": "large",
    },

    # --- Diversified ---
    "Pidilite Industries": {
        "symbol": "PIDILITIND.NS",
        "sector": "Chemicals & Specialty",
        "tier": "large",
    },
    "Havells India": {
        "symbol": "HAVELLS.NS",
        "sector": "Electricals & Electronics",
        "tier": "large",
    },
    "Berger Paints": {
        "symbol": "BERGEPAINT.NS",
        "sector": "FMCG & Consumer",
        "tier": "large",
    },
    "Info Edge India": {
        "symbol": "NAUKRI.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "IndiGo": {
        "symbol": "INDIGO.NS",
        "sector": "Aviation & Logistics",
        "tier": "large",
    },
    "Indian Hotels": {
        "symbol": "INDHOTEL.NS",
        "sector": "Hospitality & Tourism",
        "tier": "large",
    },
    "Tata Elxsi": {
        "symbol": "TATAELXSI.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "Persistent Systems": {
        "symbol": "PERSISTENT.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "MRF": {
        "symbol": "MRF.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Bosch": {
        "symbol": "BOSCHLTD.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "UPL": {
        "symbol": "UPL.NS",
        "sector": "Chemicals & Specialty",
        "tier": "large",
    },
    "REC Limited": {
        "symbol": "RECLTD.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },
    "Punjab National Bank": {
        "symbol": "PNB.NS",
        "sector": "Banking & Finance",
        "tier": "large",
    },

    # =========================================================================
    # MID CAP — NIFTY Midcap 150
    # =========================================================================

    # --- IT & Technology ---
    "Coforge": {
        "symbol": "COFORGE.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Mphasis": {
        "symbol": "MPHASIS.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Zensar Technologies": {
        "symbol": "ZENSARTECH.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Firstsource Solutions": {
        "symbol": "FSL.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Route Mobile": {
        "symbol": "ROUTE.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "HFCL": {
        "symbol": "HFCL.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Sterlite Technologies": {
        "symbol": "STLTECH.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },

    # --- Banking & Finance ---
    "IDFC FIRST Bank": {
        "symbol": "IDFCFIRSTB.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Federal Bank": {
        "symbol": "FEDERALBNK.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "RBL Bank": {
        "symbol": "RBLBANK.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "LIC Housing Finance": {
        "symbol": "LICHSGFIN.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Manappuram Finance": {
        "symbol": "MANAPPURAM.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Can Fin Homes": {
        "symbol": "CANFINHOME.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Max Financial Services": {
        "symbol": "MFSL.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Star Health Insurance": {
        "symbol": "STARHEALTH.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "HDFC AMC": {
        "symbol": "HDFCAMC.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Nippon India AMC": {
        "symbol": "NAM-INDIA.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "UTI AMC": {
        "symbol": "UTIAMC.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "MCX India": {
        "symbol": "MCX.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Indian Energy Exchange": {
        "symbol": "IEX.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Union Bank": {
        "symbol": "UNIONBANK.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Indian Bank": {
        "symbol": "INDIANB.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "PB Fintech": {
        "symbol": "POLICYBZR.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Sundaram Finance": {
        "symbol": "SUNDARMFIN.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },

    # --- Pharma & Healthcare ---
    "Alkem Laboratories": {
        "symbol": "ALKEM.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Mankind Pharma": {
        "symbol": "MANKIND.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Laurus Labs": {
        "symbol": "LAURUSLABS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Zydus Lifesciences": {
        "symbol": "ZYDUSLIFE.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Syngene International": {
        "symbol": "SYNGENE.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Metropolis Healthcare": {
        "symbol": "METROPOLIS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Glenmark Pharma": {
        "symbol": "GLENMARK.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Fortis Healthcare": {
        "symbol": "FORTIS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },

    # --- Chemicals & Specialty ---
    "Aarti Industries": {
        "symbol": "AARTIIND.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Deepak Nitrite": {
        "symbol": "DEEPAKNTR.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Navin Fluorine": {
        "symbol": "NAVINFLUOR.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Vinati Organics": {
        "symbol": "VINATIORGA.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Galaxy Surfactants": {
        "symbol": "GALAXYSURF.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },

    # --- Industrials & Capital Goods ---
    "Hindustan Aeronautics": {
        "symbol": "HAL.NS",
        "sector": "Defence & Aerospace",
        "tier": "mid",
    },
    "Bharat Dynamics": {
        "symbol": "BDL.NS",
        "sector": "Defence & Aerospace",
        "tier": "mid",
    },
    "Bharat Forge": {
        "symbol": "BHARATFORG.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "BHEL": {
        "symbol": "BHEL.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Cummins India": {
        "symbol": "CUMMINSIND.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Thermax": {
        "symbol": "THERMAX.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Honeywell Automation": {
        "symbol": "HONAUT.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Schaeffler India": {
        "symbol": "SCHAEFFLER.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "SKF India": {
        "symbol": "SKFINDIA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Timken India": {
        "symbol": "TIMKEN.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Sona BLW Precision": {
        "symbol": "SONACOMS.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Engineers India": {
        "symbol": "ENGINERSIN.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "KEC International": {
        "symbol": "KEC.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Kalpataru Projects": {
        "symbol": "KPIL.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Escorts Kubota": {
        "symbol": "ESCORTS.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Tube Investments": {
        "symbol": "TIINDIA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Minda Industries": {
        "symbol": "UNOMINDA.NS",
        "sector": "Automobiles",
        "tier": "mid",
    },
    "3M India": {
        "symbol": "3MINDIA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Carborundum Universal": {
        "symbol": "CARBORUNIV.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Sundram Fasteners": {
        "symbol": "SUNDRMFAST.NS",
        "sector": "Automobiles",
        "tier": "mid",
    },
    "CG Power": {
        "symbol": "CGPOWER.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },

    # --- Cement & Construction ---
    "Dalmia Bharat": {
        "symbol": "DALBHARAT.NS",
        "sector": "Cement",
        "tier": "mid",
    },
    "Ramco Cements": {
        "symbol": "RAMCOCEM.NS",
        "sector": "Cement",
        "tier": "mid",
    },
    "JK Cement": {
        "symbol": "JKCEMENT.NS",
        "sector": "Cement",
        "tier": "mid",
    },
    "NCC": {
        "symbol": "NCC.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "mid",
    },

    # --- FMCG / Consumer ---
    "Emami": {
        "symbol": "EMAMILTD.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Jubilant Foodworks": {
        "symbol": "JUBLFOOD.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Jyothy Labs": {
        "symbol": "JYOTHYLAB.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Radico Khaitan": {
        "symbol": "RADICO.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },

    # --- Retail & Real Estate ---
    "Phoenix Mills": {
        "symbol": "PHOENIXLTD.NS",
        "sector": "Real Estate",
        "tier": "mid",
    },
    "Prestige Estates": {
        "symbol": "PRESTIGE.NS",
        "sector": "Real Estate",
        "tier": "mid",
    },
    "Godrej Agrovet": {
        "symbol": "GODREJAGRO.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Kajaria Ceramics": {
        "symbol": "KAJARIACER.NS",
        "sector": "Retail & E-Commerce",
        "tier": "mid",
    },
    "Kansai Nerolac": {
        "symbol": "KANSAINER.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Century Textiles": {
        "symbol": "CENTURYTEX.NS",
        "sector": "Textiles & Apparel",
        "tier": "mid",
    },
    "PVR INOX": {
        "symbol": "PVRINOX.NS",
        "sector": "Media & Entertainment",
        "tier": "mid",
    },
    "Zee Entertainment": {
        "symbol": "ZEEL.NS",
        "sector": "Media & Entertainment",
        "tier": "mid",
    },

    # --- Metals ---
    "Hindustan Copper": {
        "symbol": "HINDCOPPER.NS",
        "sector": "Metals & Mining",
        "tier": "mid",
    },
    "NMDC": {
        "symbol": "NMDC.NS",
        "sector": "Metals & Mining",
        "tier": "mid",
    },
    "Tata Chemicals": {
        "symbol": "TATACHEM.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Welspun Corp": {
        "symbol": "WELCORP.NS",
        "sector": "Metals & Mining",
        "tier": "mid",
    },

    # --- Energy / Oil ---
    "Hindustan Petroleum": {
        "symbol": "HINDPETRO.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "Oil India": {
        "symbol": "OIL.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "Indraprastha Gas": {
        "symbol": "IGL.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "Mahanagar Gas": {
        "symbol": "MGL.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "CESC": {
        "symbol": "CESC.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "Torrent Power": {
        "symbol": "TORNTPOWER.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },

    # --- Logistics & Transport ---
    "Blue Dart Express": {
        "symbol": "BLUEDART.NS",
        "sector": "Aviation & Logistics",
        "tier": "mid",
    },
    "Container Corp": {
        "symbol": "CONCOR.NS",
        "sector": "Aviation & Logistics",
        "tier": "mid",
    },
    "IRCTC": {
        "symbol": "IRCTC.NS",
        "sector": "Aviation & Logistics",
        "tier": "mid",
    },
    "IRFC": {
        "symbol": "IRFC.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "RITES": {
        "symbol": "RITES.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },

    # --- Agri / Fertilizers ---
    "Chambal Fertilizers": {
        "symbol": "CHAMBLFERT.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },

    # --- Electronics / Infra ---
    "Dixon Technologies": {
        "symbol": "DIXON.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },
    "Polycab India": {
        "symbol": "POLYCAB.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },
    "KEI Industries": {
        "symbol": "KEI.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },
    "V-Guard Industries": {
        "symbol": "VGUARD.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },

    # --- Textiles ---
    "KPR Mill": {
        "symbol": "KPRMILL.NS",
        "sector": "Textiles & Apparel",
        "tier": "mid",
    },
    "Vardhman Textiles": {
        "symbol": "VTL.NS",
        "sector": "Textiles & Apparel",
        "tier": "mid",
    },

    # --- BSE/Exchange ---
    "BSE": {
        "symbol": "BSE.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "CDSL": {
        "symbol": "CDSL.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },

    # --- APL ---
    "APL Apollo Tubes": {
        "symbol": "APLAPOLLO.NS",
        "sector": "Metals & Mining",
        "tier": "mid",
    },
    "Astral": {
        "symbol": "ASTRAL.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "mid",
    },
    "Apollo Tyres": {
        "symbol": "APOLLOTYRE.NS",
        "sector": "Automobiles",
        "tier": "mid",
    },
    "Balkrishna Industries": {
        "symbol": "BALKRISIND.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Castrol India": {
        "symbol": "CASTROLIND.NS",
        "sector": "Energy & Oil",
        "tier": "mid",
    },
    "Solar Industries": {
        "symbol": "SOLARINDS.NS",
        "sector": "Defence & Aerospace",
        "tier": "mid",
    },
    "Vaibhav Global": {
        "symbol": "VAIBHAVGBL.NS",
        "sector": "Retail & E-Commerce",
        "tier": "mid",
    },

    # =========================================================================
    # SMALL CAP — NIFTY Smallcap 250
    # =========================================================================

    # --- Banking & Finance ---
    "Aavas Financiers": {
        "symbol": "AAVAS.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "City Union Bank": {
        "symbol": "CUB.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Karur Vysya Bank": {
        "symbol": "KARURVYSYA.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Karnataka Bank": {
        "symbol": "KTKBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "CSB Bank": {
        "symbol": "CSBBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Dhanlaxmi Bank": {
        "symbol": "DHANBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "JM Financial": {
        "symbol": "JMFINANCIL.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "IIFL Finance": {
        "symbol": "IIFL.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "IIFL Securities": {
        "symbol": "IIFLSEC.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Spandana Sphoorty": {
        "symbol": "SPANDANA.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Satin Creditcare": {
        "symbol": "SATIN.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "CreditAccess Grameen": {
        "symbol": "CREDITACC.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Fusion Micro Finance": {
        "symbol": "FUSIONMICR.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Home First Finance": {
        "symbol": "HOMEFIRST.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Capri Global": {
        "symbol": "CGCL.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "KFintech": {
        "symbol": "KFINTECH.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "CAMS": {
        "symbol": "CAMS.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Nuvama Wealth": {
        "symbol": "NUVAMA.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Bajaj Holdings": {
        "symbol": "BAJAJHLDNG.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Emkay Global": {
        "symbol": "EMKAY.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Choice International": {
        "symbol": "CHOICEIN.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "UCO Bank": {
        "symbol": "UCOBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Yes Bank": {
        "symbol": "YESBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Jammu & Kashmir Bank": {
        "symbol": "J&KBANK.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },

    # --- IT & Technology ---
    "IndiaMART Intermesh": {
        "symbol": "INDIAMART.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Just Dial": {
        "symbol": "JUSTDIAL.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Easy Trip Planners": {
        "symbol": "EASEMYTRIP.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Birlasoft": {
        "symbol": "BSOFT.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Nucleus Software": {
        "symbol": "NUCLEUS.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "MapmyIndia": {
        "symbol": "MAPMYINDIA.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Rategain Travel": {
        "symbol": "RATEGAIN.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "D-Link India": {
        "symbol": "DLINKINDIA.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },

    # --- Pharma & Healthcare ---
    "Abbott India": {
        "symbol": "ABBOTINDIA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Alembic Pharma": {
        "symbol": "APLLTD.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Ajanta Pharma": {
        "symbol": "AJANTPHARM.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "JB Chemicals": {
        "symbol": "JBCHEPHARM.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Gland Pharma": {
        "symbol": "GLAND.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Sequent Scientific": {
        "symbol": "SEQUENT.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Suven Pharma": {
        "symbol": "SUVENPHAR.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Thyrocare Technologies": {
        "symbol": "THYROCARE.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Pfizer India": {
        "symbol": "PFIZER.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Wockhardt": {
        "symbol": "WOCKPHARMA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Jubilant Pharmova": {
        "symbol": "JUBLPHARMA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Aarti Drugs": {
        "symbol": "AARTIDRUGS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "KIMS Hospitals": {
        "symbol": "KIMS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Global Health (Medanta)": {
        "symbol": "MEDANTA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },

    # --- Chemicals & Specialty ---
    "Archean Chemical": {
        "symbol": "ARCHEAN.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Laxmi Organic Industries": {
        "symbol": "LXCHEM.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Rossari Biotech": {
        "symbol": "ROSSARI.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Sudarshan Chemical": {
        "symbol": "SUDARSCHEM.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Deepak Fertilizers": {
        "symbol": "DEEPAKFERT.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "India Glycols": {
        "symbol": "INDIAGLYCO.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Himadri Speciality Chemical": {
        "symbol": "HSCL.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Chemplast Sanmar": {
        "symbol": "CHEMPLASTS.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Rain Industries": {
        "symbol": "RAIN.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Gujarat Fluorochemicals": {
        "symbol": "FLUOROCHEM.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Paradeep Phosphates": {
        "symbol": "PARADEEP.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Rallis India": {
        "symbol": "RALLIS.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "EID Parry India": {
        "symbol": "EIDPARRY.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Excel Industries": {
        "symbol": "EXCELINDUS.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },

    # --- Industrials & Capital Goods ---
    "AIA Engineering": {
        "symbol": "AIAENG.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Elgi Equipments": {
        "symbol": "ELGIEQUIP.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Grindwell Norton": {
        "symbol": "GRINDWELL.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "GMM Pfaudler": {
        "symbol": "GMMPFAUDLR.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Ion Exchange": {
        "symbol": "IONEXCHANG.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Triveni Turbine": {
        "symbol": "TRITURBINE.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Power Mech Projects": {
        "symbol": "POWERMECH.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "PSP Projects": {
        "symbol": "PSPPROJECT.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "PNC Infratech": {
        "symbol": "PNCINFRA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Patel Engineering": {
        "symbol": "PATELENG.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Craftsman Automation": {
        "symbol": "CRAFTSMAN.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Ramkrishna Forgings": {
        "symbol": "RKFORGE.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Ratnamani Metals": {
        "symbol": "RATNAMANI.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Shyam Metalics": {
        "symbol": "SHYAMMETL.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "ESAB India": {
        "symbol": "ESABINDIA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Interarch Building Products": {
        "symbol": "INTERARCH.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Stylam Industries": {
        "symbol": "STYLAM.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Apar Industries": {
        "symbol": "APARINDS.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Finolex Cables": {
        "symbol": "FINCABLES.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Finolex Industries": {
        "symbol": "FINPIPE.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "MTAR Technologies": {
        "symbol": "MTAR.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Midhani": {
        "symbol": "MIDHANI.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Mazagon Dock": {
        "symbol": "MAZDOCK.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Garden Reach Shipbuilders": {
        "symbol": "GRSE.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Cochin Shipyard": {
        "symbol": "COCHINSHIP.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "IKIO Lighting": {
        "symbol": "IKIO.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Kaynes Technology": {
        "symbol": "KAYNES.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "INOX Wind": {
        "symbol": "INOXWIND.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Borosil Renewables": {
        "symbol": "BORORENEW.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },

    # --- Automobiles ---
    "Ashok Leyland": {
        "symbol": "ASHOKLEY.NS",
        "sector": "Automobiles",
        "tier": "small",
    },
    "Bata India": {
        "symbol": "BATAINDIA.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Metro Brands": {
        "symbol": "METROBRAND.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "GE Shipping": {
        "symbol": "GESHIP.NS",
        "sector": "Aviation & Logistics",
        "tier": "small",
    },
    "SCI": {
        "symbol": "SCI.NS",
        "sector": "Aviation & Logistics",
        "tier": "small",
    },

    # --- FMCG & Consumer ---
    "TTK Prestige": {
        "symbol": "TTKPRESTIG.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Hawkins Cookers": {
        "symbol": "HAWKINCOOK.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Symphony": {
        "symbol": "SYMPHONY.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Parag Milk Foods": {
        "symbol": "PARAGMILK.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Avanti Feeds": {
        "symbol": "AVANTIFEED.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Godfrey Phillips": {
        "symbol": "GODFRYPHLP.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "VST Industries": {
        "symbol": "VSTIND.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Balrampur Chini": {
        "symbol": "BALRAMCHIN.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "DCM Shriram": {
        "symbol": "DCMSHRIRAM.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Dalmia Bharat Sugar": {
        "symbol": "DALMIASUG.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Gulf Oil Lubricants": {
        "symbol": "GULFOILLUB.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },

    # --- Retail & E-Commerce ---
    "Go Fashion India": {
        "symbol": "GOCOLORS.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Vedant Fashions": {
        "symbol": "MANYAVAR.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "ABFRL": {
        "symbol": "ABFRL.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Arvind Fashions": {
        "symbol": "ARVIND.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "VIP Industries": {
        "symbol": "VIPIND.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Kalyan Jewellers": {
        "symbol": "KALYANKJIL.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Saregama India": {
        "symbol": "SAREGAMA.NS",
        "sector": "Media & Entertainment",
        "tier": "small",
    },
    "Restaurant Brands Asia": {
        "symbol": "RBA.NS",
        "sector": "Hospitality & Tourism",
        "tier": "small",
    },
    "Indigo Paints": {
        "symbol": "INDIGOPNTS.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Akzo Nobel India": {
        "symbol": "AKZOINDIA.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },

    # --- Real Estate ---
    "Brigade Enterprises": {
        "symbol": "BRIGADE.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Sobha": {
        "symbol": "SOBHA.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Kolte-Patil Developers": {
        "symbol": "KOLTEPATIL.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Ajmera Realty": {
        "symbol": "AJMERA.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Signature Global": {
        "symbol": "SIGNATURE.NS",
        "sector": "Real Estate",
        "tier": "small",
    },

    # --- Hospitality & Tourism ---
    "Lemon Tree Hotels": {
        "symbol": "LEMONTREE.NS",
        "sector": "Hospitality & Tourism",
        "tier": "small",
    },
    "Landmark Cars": {
        "symbol": "LANDMARK.NS",
        "sector": "Automobiles",
        "tier": "small",
    },

    # --- Metals & Mining ---
    "National Aluminium": {
        "symbol": "NATIONALUM.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Hindustan Zinc": {
        "symbol": "HINDZINC.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Tata Metaliks": {
        "symbol": "TATAMETALI.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Electrosteel Castings": {
        "symbol": "ELECTCAST.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Nava Limited": {
        "symbol": "NAVA.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },

    # --- Energy ---
    "NHPC": {
        "symbol": "NHPC.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Tata Power": {
        "symbol": "TATAPOWER.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Power Finance Corporation": {
        "symbol": "PFC.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Chennai Petroleum": {
        "symbol": "CHENNPETRO.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },

    # --- Misc / Diversified ---
    "Asahi India Glass": {
        "symbol": "ASAHIINDIA.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Orient Cement": {
        "symbol": "ORIENTCEM.NS",
        "sector": "Cement",
        "tier": "small",
    },
    "Raymond": {
        "symbol": "RAYMOND.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Welspun India": {
        "symbol": "WELSPUNIND.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Banswara Syntex": {
        "symbol": "BANSWARAS.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "SIS": {
        "symbol": "SIS.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "small",
    },
    "Tata Investment Corp": {
        "symbol": "TATAINVEST.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Teamlease Services": {
        "symbol": "TEAMLEASE.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "small",
    },
    "Gati": {
        "symbol": "GATI.NS",
        "sector": "Aviation & Logistics",
        "tier": "small",
    },
    "Redington India": {
        "symbol": "REDINGTON.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "CMS Info Systems": {
        "symbol": "CMSINFO.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Tata Technologies": {
        "symbol": "TATATECH.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Embassy Office Parks REIT": {
        "symbol": "EMBASSY.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Bajel Projects": {
        "symbol": "BAJEL.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "LIC India": {
        "symbol": "LICI.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Himachal Futuristic": {
        "symbol": "HFCL.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "MMTC": {
        "symbol": "MMTC.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "small",
    },
    "P&G Hygiene": {
        "symbol": "PGHH.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "India Cements": {
        "symbol": "INDIACEM.NS",
        "sector": "Cement",
        "tier": "mid",
    },
    "West Coast Paper": {
        "symbol": "WESTCOAST.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Time Technoplast": {
        "symbol": "TIMETECHNO.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Delta Corp": {
        "symbol": "DELTACORP.NS",
        "sector": "Hospitality & Tourism",
        "tier": "small",
    },
    "Den Networks": {
        "symbol": "DEN.NS",
        "sector": "Telecom",
        "tier": "small",
    },
    "United Spirits": {
        "symbol": "MCDOWELL-N.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Hind Rectifiers": {
        "symbol": "HREKTIFIER.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Donear Industries": {
        "symbol": "DONEAR.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "SH Kelkar": {
        "symbol": "SHK.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Cera Sanitaryware": {
        "symbol": "CERA.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Orient Electric": {
        "symbol": "ORIENTELEC.NS",
        "sector": "Electricals & Electronics",
        "tier": "mid",
    },
    "Blue Star": {
        "symbol": "BLUESTARCO.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "INOX Leisure": {
        "symbol": "INOXLEISUR.NS",
        "sector": "Media & Entertainment",
        "tier": "small",
    },

    # -------------------------------------------------------------------------
    # Additional stocks to fill ~500 target
    # -------------------------------------------------------------------------
    "Vodafone Idea": {
        "symbol": "IDEA.NS",
        "sector": "Telecom",
        "tier": "small",
    },
    "IREDA": {
        "symbol": "IREDA.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Suzlon Energy": {
        "symbol": "SUZLON.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Waaree Energies": {
        "symbol": "WAAREEENER.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Premier Energies": {
        "symbol": "PREMIERENE.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "One97 Communications (Paytm)": {
        "symbol": "PAYTM.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Swiggy": {
        "symbol": "SWIGGY.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Hyundai Motor India": {
        "symbol": "HYUNDAI.NS",
        "sector": "Automobiles",
        "tier": "large",
    },
    "Ola Electric": {
        "symbol": "OLAELECTRIC.NS",
        "sector": "Automobiles",
        "tier": "small",
    },
    "JSW Infrastructure": {
        "symbol": "JSWINFRA.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "mid",
    },
    "Ixigo": {
        "symbol": "IXIGO.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Awfis Space Solutions": {
        "symbol": "AWFIS.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "TBO Tek": {
        "symbol": "TBOTEK.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Go Digit Insurance": {
        "symbol": "GODIGIT.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Aadhar Housing Finance": {
        "symbol": "AADHARHFC.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "India Shelter Finance": {
        "symbol": "INDIASHLTR.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Updater Services": {
        "symbol": "UDS.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "small",
    },
    "Senco Gold": {
        "symbol": "SENCO.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "RHI Magnesita India": {
        "symbol": "RHIM.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Chalet Hotels": {
        "symbol": "CHALET.NS",
        "sector": "Hospitality & Tourism",
        "tier": "small",
    },
    "EMS": {
        "symbol": "EMSLTD.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Akums Drugs": {
        "symbol": "AKUMS.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Brainbees Solutions (FirstCry)": {
        "symbol": "FIRSTCRY.NS",
        "sector": "Retail & E-Commerce",
        "tier": "small",
    },
    "Unimech Aerospace": {
        "symbol": "UNIMECH.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Sai Life Sciences": {
        "symbol": "SAILIFE.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Northern Arc Capital": {
        "symbol": "NORTHARC.NS",
        "sector": "Banking & Finance",
        "tier": "small",
    },
    "Doms Industries": {
        "symbol": "DOMS.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Sai Silks (Kalamandir)": {
        "symbol": "KALAMANDIR.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Zaggle Prepaid": {
        "symbol": "ZAGGLE.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Yatra Online": {
        "symbol": "YATRA.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Gandhar Oil Refinery": {
        "symbol": "GANDHAR.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Bansal Wire Industries": {
        "symbol": "BANSAL.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Jay Bharat Maruti": {
        "symbol": "JAYBARMARU.NS",
        "sector": "Automobiles",
        "tier": "small",
    },
    "Kirloskar Brothers": {
        "symbol": "KIRLOSBROS.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Bharat Gears": {
        "symbol": "BHARATGEAR.NS",
        "sector": "Automobiles",
        "tier": "small",
    },
    "Adani Wilmar": {
        "symbol": "AWL.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Zydus Wellness": {
        "symbol": "ZYDUSWELL.NS",
        "sector": "FMCG & Consumer",
        "tier": "mid",
    },
    "Jubilant Ingrevia": {
        "symbol": "JUBLINGREA.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Clean Science": {
        "symbol": "CLEANSCI.NS",
        "sector": "Chemicals & Specialty",
        "tier": "mid",
    },
    "Tatva Chintan Pharma": {
        "symbol": "TATVA.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Ami Organics": {
        "symbol": "AMIORG.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Anupam Rasayan": {
        "symbol": "ANURAS.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Valiant Organics": {
        "symbol": "VALIANTORG.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Shivalik Bimetal": {
        "symbol": "SHIVALIKBI.NS",
        "sector": "Metals & Mining",
        "tier": "small",
    },
    "Mold-Tek Packaging": {
        "symbol": "MOLDTKPAC.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Wonderla Holidays": {
        "symbol": "WONDERLA.NS",
        "sector": "Hospitality & Tourism",
        "tier": "small",
    },
    "Bikaji Foods": {
        "symbol": "BIKAJI.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Droneacharya Aerial": {
        "symbol": "DRONEACHARYA.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Ideaforge Technology": {
        "symbol": "IDEAFORGE.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Data Patterns India": {
        "symbol": "DATAPATTNS.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Paras Defence": {
        "symbol": "PARASDEF.NS",
        "sector": "Defence & Aerospace",
        "tier": "small",
    },
    "Syrma SGS Technology": {
        "symbol": "SYRMA.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Avalon Technologies": {
        "symbol": "AVALON.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Cyient DLM": {
        "symbol": "CYIENTDLM.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Rashi Peripherals": {
        "symbol": "RPTECH.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Netweb Technologies": {
        "symbol": "NETWEB.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Sanathan Textiles": {
        "symbol": "SANATHAN.NS",
        "sector": "Textiles & Apparel",
        "tier": "small",
    },
    "Cello World": {
        "symbol": "CELLO.NS",
        "sector": "FMCG & Consumer",
        "tier": "small",
    },
    "Uniparts India": {
        "symbol": "UNIPARTS.NS",
        "sector": "Industrials & Capital Goods",
        "tier": "small",
    },
    "Techno Electric": {
        "symbol": "TECHNOEI.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Genus Power Infrastructure": {
        "symbol": "GENUSPOWER.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Voltamp Transformers": {
        "symbol": "VOLTAMP.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "Kirloskar Electric": {
        "symbol": "KIRLOSELE.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "TD Power Systems": {
        "symbol": "TDPOWERSYS.NS",
        "sector": "Electricals & Electronics",
        "tier": "small",
    },
    "PTC India": {
        "symbol": "PTC.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "India Power Corporation": {
        "symbol": "INDIAPOWER.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Windworld Energy": {
        "symbol": "WINDWORLD.NS",
        "sector": "Energy & Oil",
        "tier": "small",
    },
    "Suraj Estate Developers": {
        "symbol": "SURAJ.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Mahindra Lifespace": {
        "symbol": "MAHLIFE.NS",
        "sector": "Real Estate",
        "tier": "small",
    },
    "Godrej Industries": {
        "symbol": "GODREJIND.NS",
        "sector": "Conglomerates & Industrials",
        "tier": "mid",
    },
    "Piramal Enterprises": {
        "symbol": "PEL.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
    "Piramal Pharma": {
        "symbol": "PPLPHARMA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Gufic Biosciences": {
        "symbol": "GUFICBIO.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Biocon": {
        "symbol": "BIOCON.NS",
        "sector": "Pharma & Healthcare",
        "tier": "mid",
    },
    "Strides Pharma": {
        "symbol": "STAR.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Solara Active Pharma": {
        "symbol": "SOLARA.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Neuland Laboratories": {
        "symbol": "NEULANDLAB.NS",
        "sector": "Pharma & Healthcare",
        "tier": "small",
    },
    "Hikal": {
        "symbol": "HIKAL.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Fineotex Chemical": {
        "symbol": "FCL.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Aether Industries": {
        "symbol": "AETHER.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Transpek Industry": {
        "symbol": "TRANSPEK.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "Choksi Imaging": {
        "symbol": "CHOKSI.NS",
        "sector": "Chemicals & Specialty",
        "tier": "small",
    },
    "NIIT": {
        "symbol": "NIIT.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Mastek": {
        "symbol": "MASTEK.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Cyient": {
        "symbol": "CYIENT.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "KPIT Technologies": {
        "symbol": "KPITTECH.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Hexaware Technologies": {
        "symbol": "HEXAWARE.NS",
        "sector": "IT & Technology",
        "tier": "large",
    },
    "Intellect Design Arena": {
        "symbol": "INTELLECT.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Newgen Software": {
        "symbol": "NEWGEN.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Happiest Minds": {
        "symbol": "HAPPSTMNDS.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Tanla Platforms": {
        "symbol": "TANLA.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Sonata Software": {
        "symbol": "SONATSOFTW.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Ramco Systems": {
        "symbol": "RAMCOSYS.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "3i Infotech": {
        "symbol": "3IINFOLTD.NS",
        "sector": "IT & Technology",
        "tier": "small",
    },
    "Affle India": {
        "symbol": "AFFLE.NS",
        "sector": "IT & Technology",
        "tier": "mid",
    },
    "Policybazaar": {
        "symbol": "POLICYBZR.NS",
        "sector": "Banking & Finance",
        "tier": "mid",
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible flat dict for dropdowns  {name: symbol}
# ---------------------------------------------------------------------------
NIFTY_500_SYMBOLS: Dict[str, str] = {
    name: data["symbol"] for name, data in NIFTY_500.items()
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_stocks_by_sector(sector: str) -> Dict[str, StockInfo]:
    """Return all stocks in a given sector."""
    return {
        name: data
        for name, data in NIFTY_500.items()
        if data["sector"] == sector
    }


def get_stocks_by_tier(tier: str) -> Dict[str, StockInfo]:
    """Return all stocks with a given tier ('large', 'mid', 'small')."""
    return {
        name: data
        for name, data in NIFTY_500.items()
        if data["tier"] == tier
    }


SECTORS: list = sorted(set(data["sector"] for data in NIFTY_500.values()))
