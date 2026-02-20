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


# Convenience alias for backward compatibility
TRACKED_SYMBOLS = get_all_symbols()


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


# FIPS 10-4 → ISO 3166-1 alpha-3 country code mapping.
# GDELT stores action_geo_country_code as FIPS; the world map TopoJSON uses ISO_A3.
FIPS_TO_ISO = {
    "AA": "ABW", "AC": "ATG", "AE": "ARE", "AF": "AFG", "AG": "DZA",
    "AJ": "AZE", "AL": "ALB", "AM": "ARM", "AN": "AND", "AO": "AGO",
    "AQ": "ASM", "AR": "ARG", "AS": "AUS", "AU": "AUT", "BA": "BHR",
    "BB": "BRB", "BC": "BWA", "BD": "BMU", "BE": "BEL", "BF": "BHS",
    "BG": "BGD", "BH": "BLZ", "BK": "BIH", "BL": "BOL", "BM": "MMR",
    "BN": "BEN", "BO": "BLR", "BP": "SLB", "BR": "BRA", "BT": "BTN",
    "BU": "BGR", "BX": "BRN", "BY": "BDI", "CA": "CAN", "CB": "KHM",
    "CD": "TCD", "CE": "LKA", "CF": "COG", "CG": "COD", "CH": "CHN",
    "CI": "CHL", "CJ": "CYM", "CM": "CMR", "CN": "COM", "CO": "COL",
    "CS": "CRI", "CT": "CAF", "CU": "CUB", "CV": "CPV", "CW": "COK",
    "CY": "CYP", "DA": "DNK", "DJ": "DJI", "DO": "DMA", "DR": "DOM",
    "EC": "ECU", "EG": "EGY", "EI": "IRL", "EK": "GNQ", "EN": "EST",
    "ER": "ERI", "ES": "SLV", "ET": "ETH", "EZ": "CZE", "FI": "FIN",
    "FJ": "FJI", "FK": "FLK", "FM": "FSM", "FO": "FRO", "FP": "PYF",
    "FR": "FRA", "GA": "GMB", "GB": "GAB", "GG": "GEO", "GH": "GHA",
    "GI": "GIB", "GJ": "GRD", "GL": "GRL", "GM": "DEU", "GP": "GLP",
    "GQ": "GUM", "GR": "GRC", "GT": "GTM", "GV": "GIN", "GY": "GUY",
    "GZ": "PSE", "HA": "HTI", "HK": "HKG", "HO": "HND", "HR": "HRV",
    "HU": "HUN", "IC": "ISL", "ID": "IDN", "IM": "IMN", "IN": "IND",
    "IO": "IOT", "IR": "IRN", "IS": "ISR", "IT": "ITA", "IV": "CIV",
    "IZ": "IRQ", "JA": "JPN", "JE": "JEY", "JM": "JAM", "JO": "JOR",
    "KE": "KEN", "KG": "KGZ", "KN": "PRK", "KR": "KIR", "KS": "KOR",
    "KU": "KWT", "KV": "XKX", "KZ": "KAZ", "LA": "LAO", "LE": "LBN",
    "LG": "LVA", "LH": "LTU", "LI": "LBR", "LO": "SVK", "LS": "LIE",
    "LT": "LSO", "LU": "LUX", "LY": "LBY", "MA": "MDG", "MB": "MTQ",
    "MC": "MAC", "MD": "MDA", "MG": "MNG", "MH": "MHL", "MI": "MWI",
    "MJ": "MNE", "MK": "MKD", "ML": "MLI", "MN": "MCO", "MO": "MAR",
    "MP": "MUS", "MR": "MRT", "MT": "MLT", "MU": "OMN", "MV": "MDV",
    "MX": "MEX", "MY": "MYS", "MZ": "MOZ", "NC": "NCL", "NG": "NER",
    "NH": "VUT", "NI": "NGA", "NL": "NLD", "NO": "NOR", "NP": "NPL",
    "NR": "NRU", "NS": "SUR", "NU": "NIC", "NZ": "NZL", "OD": "SSD",
    "PA": "PAN", "PE": "PER", "PK": "PAK", "PL": "POL", "PM": "PAN",
    "PO": "PRT", "PP": "PNG", "PU": "GNB", "QA": "QAT", "RB": "SRB",
    "RE": "REU", "RI": "SRB", "RM": "MHL", "RN": "MNE", "RO": "ROU",
    "RP": "PHL", "RQ": "PRI", "RS": "RUS", "RW": "RWA", "SA": "SAU",
    "SB": "SPM", "SC": "KNA", "SE": "SYC", "SF": "ZAF", "SG": "SEN",
    "SH": "SHN", "SI": "SVN", "SL": "SLE", "SM": "SMR", "SN": "SGP",
    "SO": "SOM", "SP": "ESP", "ST": "LCA", "SU": "SDN", "SV": "SJM",
    "SW": "SWE", "SY": "SYR", "SZ": "CHE", "TD": "TTO", "TH": "THA",
    "TI": "TJK", "TK": "TCA", "TL": "TKL", "TN": "TON", "TO": "TGO",
    "TP": "STP", "TS": "TUN", "TT": "TLS", "TU": "TUR", "TV": "TUV",
    "TW": "TWN", "TX": "TKM", "TZ": "TZA", "UG": "UGA", "UK": "GBR",
    "UP": "UKR", "US": "USA", "UV": "BFA", "UY": "URY", "UZ": "UZB",
    "VC": "VCT", "VE": "VEN", "VI": "VIR", "VM": "VNM", "VQ": "VIR",
    "VT": "VAT", "WA": "NAM", "WE": "PSE", "WI": "ESH", "WS": "WSM",
    "WZ": "SWZ", "YM": "YEM", "ZA": "ZMB", "ZI": "ZWE",
}


def fips_to_iso(fips_code: str) -> str:
    """Convert a FIPS 10-4 country code to ISO 3166-1 alpha-3."""
    return FIPS_TO_ISO.get(fips_code, fips_code)


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
