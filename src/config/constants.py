"""Constants and mappings for the geopolitical market tracker."""

# Financial instruments to track
SYMBOLS = {
    "commodities": {
        "CL=F": "Crude Oil (WTI)",
        "BZ=F": "Brent Crude",
        "GC=F": "Gold",
        "SI=F": "Silver",
        "NG=F": "Natural Gas",
        "ZW=F": "Wheat",
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
    },
    "currencies": {
        "EURUSD=X": "Euro/USD",
        "USDJPY=X": "USD/Yen",
        "GBPUSD=X": "British Pound/USD",
        "USDCNY=X": "USD/Chinese Yuan",
        "USDRUB=X": "USD/Russian Ruble",
        "USDINR=X": "USD/Indian Rupee",
        "USDBRL=X": "USD/Brazilian Real",
    },
    "etfs": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "EEM": "Emerging Markets",
        "VWO": "Emerging Markets (Vanguard)",
        "EWZ": "Brazil",
        "EWJ": "Japan",
        "FXI": "China Large Cap",
        "EWG": "Germany",
        "EWT": "Taiwan",
        "EWY": "South Korea",
        "INDA": "India",
        "XLE": "Energy Sector",
        "XLF": "Financial Sector",
        "GDX": "Gold Miners",
    },
    "volatility": {
        "^VIX": "CBOE Volatility Index",
    },
    "bonds": {
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "HYG": "High Yield Corporate",
    },
}


def get_all_symbols() -> list[str]:
    """Return flat list of all tracked symbols."""
    symbols = []
    for category in SYMBOLS.values():
        symbols.extend(category.keys())
    return symbols


def get_symbol_info(symbol: str) -> dict | None:
    """Get category and name for a symbol."""
    for category, symbols in SYMBOLS.items():
        if symbol in symbols:
            return {"symbol": symbol, "name": symbols[symbol], "category": category}
    return None


# Country to asset mappings
COUNTRY_ASSET_MAP = {
    "RUS": ["USDRUB=X", "NG=F", "CL=F"],  # Russia -> Ruble, Natural Gas, Oil
    "CHN": ["USDCNY=X", "FXI", "EEM"],  # China -> Yuan, China ETF, EM
    "SAU": ["CL=F", "BZ=F"],  # Saudi Arabia -> Oil
    "IRN": ["CL=F", "BZ=F", "GC=F"],  # Iran -> Oil, Gold (safe haven)
    "UKR": ["ZW=F", "ZC=F", "NG=F"],  # Ukraine -> Wheat, Corn, Natural Gas
    "BRA": ["USDBRL=X", "EWZ", "ZS=F"],  # Brazil -> Real, Brazil ETF, Soybeans
    "JPN": ["USDJPY=X", "EWJ"],  # Japan -> Yen, Japan ETF
    "DEU": ["EURUSD=X", "EWG"],  # Germany -> Euro, Germany ETF
    "TWN": ["EWT", "QQQ"],  # Taiwan -> Taiwan ETF, Nasdaq (semis)
    "KOR": ["EWY", "QQQ"],  # South Korea -> Korea ETF, Nasdaq
    "IND": ["USDINR=X", "INDA"],  # India -> Rupee, India ETF
    "GBR": ["GBPUSD=X"],  # UK -> Pound
    "ISR": ["GC=F", "CL=F"],  # Israel -> Gold, Oil (regional tension)
    "VEN": ["CL=F"],  # Venezuela -> Oil
    "NGA": ["CL=F"],  # Nigeria -> Oil
}

# Event type to asset mappings
EVENT_ASSET_MAP = {
    "military_action": ["GC=F", "^VIX", "CL=F"],  # Military -> Gold, VIX, Oil
    "violent_conflict": ["GC=F", "^VIX", "CL=F"],
    "sanctions": [],  # Dynamic based on target country
    "protest": [],  # Dynamic based on location
    "natural_disaster": [],  # Dynamic based on region/commodity
    "election": [],  # Dynamic based on country
    "central_bank": ["TLT", "IEF"],  # Monetary policy -> bonds
    "trade": ["EEM", "SPY"],  # Trade policy -> markets
}

# Reverse mapping: symbol to countries it's sensitive to
SYMBOL_COUNTRY_MAP = {
    "CL=F": ["RUS", "SAU", "IRN", "VEN", "NGA", "ISR"],
    "BZ=F": ["SAU", "IRN"],
    "NG=F": ["RUS", "UKR"],
    "ZW=F": ["UKR", "RUS"],
    "ZC=F": ["UKR"],
    "ZS=F": ["BRA", "USA", "ARG"],
    "GC=F": ["IRN", "ISR"],  # Safe haven during conflict
    "USDRUB=X": ["RUS"],
    "USDCNY=X": ["CHN"],
    "USDJPY=X": ["JPN"],
    "EURUSD=X": ["DEU", "FRA", "ITA", "ESP"],
    "GBPUSD=X": ["GBR"],
    "USDBRL=X": ["BRA"],
    "USDINR=X": ["IND"],
    "FXI": ["CHN"],
    "EWZ": ["BRA"],
    "EWJ": ["JPN"],
    "EWG": ["DEU"],
    "EWT": ["TWN", "CHN"],
    "EWY": ["KOR"],
    "INDA": ["IND"],
    "EEM": ["CHN", "BRA", "IND", "KOR", "TWN"],
    "^VIX": [],  # Reacts to all major negative events
    "SPY": ["USA"],
    "QQQ": ["USA", "TWN", "KOR"],  # Tech/semis exposure
}

# CAMEO event code mappings
CAMEO_CATEGORIES = {
    "01": "public_statement",
    "02": "appeal",
    "03": "intent_to_cooperate",
    "04": "consult",
    "05": "diplomatic_cooperation",
    "06": "material_cooperation",
    "07": "provide_aid",
    "08": "yield",
    "09": "investigate",
    "10": "demand",
    "11": "disapprove",
    "12": "reject",
    "13": "threaten",
    "14": "protest",
    "15": "exhibit_force",
    "16": "reduce_relations",
    "17": "coerce",
    "18": "assault",
    "19": "fight",
    "20": "mass_violence",
}

# Higher-level groupings for analysis
EVENT_GROUPS = {
    "verbal_cooperation": ["01", "02", "03", "04", "05"],
    "material_cooperation": ["06", "07", "08"],
    "verbal_conflict": ["09", "10", "11", "12", "13"],
    "material_conflict": ["14", "15", "16", "17"],
    "violent_conflict": ["18", "19", "20"],
}


def get_event_category(event_root_code: str) -> str:
    """Map CAMEO root code to category."""
    code = str(event_root_code).zfill(2)[:2]
    return CAMEO_CATEGORIES.get(code, "other")


def get_event_group(event_root_code: str) -> str:
    """Map CAMEO root code to high-level group."""
    code = str(event_root_code).zfill(2)[:2]
    for group_name, codes in EVENT_GROUPS.items():
        if code in codes:
            return group_name
    return "other"
