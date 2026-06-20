"""
Real Data Sources
=================
Fetches macro-financial data exclusively from free public APIs.
No synthetic or mock data is generated.

Data sources:
- World Bank WDI (free, no auth) — GDP, inflation, unemployment,
  government debt, external debt, current account, private credit
- IMF WEO (public, no auth) — fiscal balances, debt projections,
  external sector, structural indicators (covers 2025-2026 projections)

Indicators unavailable from free APIs return None; the scoring engine
handles missing data by assigning neutral midpoint scores.
"""

import logging
import subprocess
import json
from typing import Dict, List, Optional
from datetime import datetime

from src.data_ingestion.registry import DataSource

logger = logging.getLogger("debt_framework")

# 3-letter ISO → World Bank 2-letter code
WB_COUNTRY_CODES = {
    "USA": "US", "JPN": "JP", "DEU": "DE", "GBR": "GB", "FRA": "FR",
    "ITA": "IT", "CAN": "CA", "AUS": "AU", "KOR": "KR", "ESP": "ES",
    "NLD": "NL", "CHE": "CH", "SWE": "SE", "NOR": "NO", "AUT": "AT",
    "BEL": "BE", "DNK": "DK", "FIN": "FI", "PRT": "PT", "IRL": "IE",
    "GRC": "GR", "SGP": "SG", "HKG": "HK", "NZL": "NZ",
    "CHN": "CN", "IND": "IN", "BRA": "BR", "RUS": "RU", "MEX": "MX",
    "ZAF": "ZA", "TUR": "TR", "IDN": "ID", "THA": "TH", "MYS": "MY",
    "POL": "PL", "COL": "CO", "ARG": "AR", "CHL": "CL", "PER": "PE",
    "EGY": "EG", "NGA": "NG", "VNM": "VN", "PAK": "PK", "PHL": "PH",
    "HUN": "HU", "CZE": "CZ", "ROU": "RO", "KAZ": "KZ", "UKR": "UA",
    "MAR": "MA", "SAU": "SA", "ARE": "AE", "QAT": "QA", "ISR": "IL",
    "DOM": "DO", "GTM": "GT", "ECU": "EC", "BOL": "BO", "PRY": "PY",
    "URY": "UY", "TZA": "TZ", "KEN": "KE", "ETH": "ET", "GHA": "GH",
    "LUX": "LU", "SVK": "SK", "SVN": "SI", "EST": "EE", "LVA": "LV",
    "LTU": "LT", "CYP": "CY", "MLT": "MT", "ISL": "IS",
    "KWT": "KW", "IRN": "IR",
}

# World Bank indicators: (wb_code, our_field_name)
WB_INDICATORS = [
    # Macro
    ("NY.GDP.MKTP.KD.ZG", "real_gdp_growth_pct"),
    ("FP.CPI.TOTL.ZG", "inflation_pct"),
    ("SL.UEM.TOTL.ZS", "unemployment_pct"),
    # Fiscal
    ("GC.DOD.TOTL.GD.ZS", "gov_gross_debt_pct_gdp"),
    ("GC.XPN.TOTL.GD.ZS", "gov_expense_pct_gdp"),
    ("GC.REV.XPN.TOTL.GD.ZS", "gov_revenue_pct_gdp"),
    ("BN.CAB.XOKA.GD.ZS", "overall_fiscal_balance_pct_gdp"),
    # External
    ("DT.DOD.DECT.GD.ZS", "external_debt_pct_gdp"),
    ("BN.CAB.XOKA.GD.ZS", "current_account_pct_gdp"),
    ("FI.RES.TOTL.CD", "fx_reserves_usd"),
    ("FS.AST.PRVT.GD.ZS", "domestic_credit_private_pct_gdp"),
    ("FS.AST.DOMS.GD.ZS", "domestic_credit_public_pct_gdp"),
    # Trade
    ("TX.VAL.MRCH.CD.WT", "exports_usd"),
    ("TM.VAL.MRCH.CD.WT", "imports_usd"),
    # Commodity
    ("EN.ATM.CO2E.PC", "co2_per_capita"),
    # Demographics
    ("SP.POP.TOTL", "population"),
]

# IMF WEO indicators — expanded to cover all scoring dimensions
# Codes from IMF WEO database: https://www.imf.org/en/Publications/WEO/weo-database/2025/April
IMF_INDICATORS = [
    # Fiscal & debt
    ("GGXWDG_NGDP", "gov_gross_debt_pct_gdp"),
    ("GGXONLB_NGDP", "primary_balance_pct_gdp"),
    ("GGR_NGDP", "gov_revenue_pct_gdp"),
    ("GGX_NGDP", "gov_expense_pct_gdp"),
    ("GGXINT_NGDP", "interest_payment_pct_gdp"),
    ("GGXCNL_NGDP", "overall_fiscal_balance_pct_gdp"),
    ("GGXWDN_NGDP", "net_gov_debt_pct_gdp"),
    ("GGXFD_NGDP", "fiscal_deficit_pct_gdp"),
    # Debt service
    ("DSR_PT", "debt_service_ratio_pct_revenue"),
    # External sector
    ("BN_CA_GDP_NHD", "current_account_pct_gdp"),
    ("GGXWD_GDP", "external_debt_pct_gdp"),
    ("BN_GFAA_GDP_NHD", "net_fdi_pct_gdp"),
    ("FI_XPLL_GDP", "gfn_proxy"),
    # Private sector
    ("_Z_GDP_FI", "hh_plus_corp_debt_pct_gdp"),
    # Macro
    ("NGDP_RPCH", "real_gdp_growth_pct"),
    ("PCPI_IX", "inflation_pct"),
    ("PCPIEPCH", "inflation_pct"),
    ("LUR", "unemployment_pct"),
    ("INT_R", "policy_rate_pct"),
    ("GGOV_GDP", "gov_investment_pct_gdp"),
    ("NX_GDP_NHD", "trade_balance_pct_gdp"),
    # Structural
    ("GGOV_GDP", "institutional_quality"),
]


def _curl_get(url: str, timeout: int = 20) -> Optional[Dict]:
    """Fetch JSON via curl subprocess."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def _wb_fetch_indicator(wb_code: str, our_field: str, country: str,
                         start_year: int, end_year: int) -> List[Dict]:
    """Fetch one World Bank indicator for a country."""
    wb_code_2 = WB_COUNTRY_CODES.get(country.upper())
    if not wb_code_2:
        return []

    url = (
        f"https://api.worldbank.org/v2/country/{wb_code_2}/"
        f"indicator/{wb_code}?format=json&date={start_year}:{end_year}&per_page=100"
    )
    data = _curl_get(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    records = []
    for item in data[1]:
        if not isinstance(item, dict):
            continue
        year = item.get("date")
        value = item.get("value")
        if year is None:
            continue
        try:
            year = int(year)
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            continue

        rec = next((r for r in records if r["year"] == year), None)
        if rec is None:
            rec = {"country": country.upper(), "year": year,
                   "source": "world_bank_wdi"}
            records.append(rec)
        if value is not None:
            rec[our_field] = value

    return records


class WorldBankDataSource(DataSource):
    """Fetches real data from World Bank WDI API."""
    source_name = "world_bank_wdi"
    update_frequency = "annual"

    def __init__(self):
        self._cache: Dict[str, List[Dict]] = {}
        self._last_fetch: Optional[datetime] = None

    def fetch(self, country: str, start_year: int, end_year: int) -> List[Dict]:
        wb_code = WB_COUNTRY_CODES.get(country.upper())
        if not wb_code:
            return []

        cache_key = f"wb_{country}_{start_year}_{end_year}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        merged: Dict[int, Dict] = {}
        for wb_ind, our_field in WB_INDICATORS:
            for rec in _wb_fetch_indicator(wb_ind, our_field, country, start_year, end_year):
                year = rec["year"]
                if year not in merged:
                    merged[year] = {"country": country.upper(), "year": year,
                                    "source": self.source_name}
                for k, v in rec.items():
                    if k not in ("country", "year", "source"):
                        merged[year][k] = v

        result = list(merged.values())
        self._cache[cache_key] = result
        self._last_fetch = datetime.now()
        logger.info(f"WB: {len(result)} year-records for {country}")
        return result

    def get_metadata(self) -> Dict:
        return {
            "source_name": self.source_name,
            "update_frequency": self.update_frequency,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "real_data": True,
        }


def _imf_fetch_indicator(indicator_id: str, our_field: str, country: str,
                          start_year: int, end_year: int) -> List[Dict]:
    """Fetch one IMF WEO indicator for a country."""
    url = (
        f"https://dataservices.imf.org/fruapim"
        f"?indicator_id={indicator_id}&format=json"
        f"&country={country.upper()}&startYear={start_year}&endYear={end_year}"
    )
    data = _curl_get(url)
    if not data:
        return []

    items = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, list):
                items.extend(e for e in entry if isinstance(e, dict))
            elif isinstance(entry, dict):
                items.append(entry)

    records = []
    for item in items:
        year = item.get("year") or item.get("YEAR")
        value = item.get("value") or item.get("VALUE")
        if year is None:
            continue
        try:
            year = int(year)
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            continue

        rec = next((r for r in records if r["year"] == year), None)
        if rec is None:
            rec = {"country": country.upper(), "year": year,
                   "source": "imf_weo"}
            records.append(rec)
        if value is not None:
            rec[our_field] = value

    return records


class IMFWEOSource(DataSource):
    """Fetches IMF WEO data — fiscal, external, structural indicators."""
    source_name = "imf_weo"
    update_frequency = "annual"

    def __init__(self):
        self._cache: Dict[str, List[Dict]] = {}
        self._last_fetch: Optional[datetime] = None

    def fetch(self, country: str, start_year: int, end_year: int) -> List[Dict]:
        cache_key = f"imf_{country}_{start_year}_{end_year}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        merged: Dict[int, Dict] = {}
        for imf_id, our_field in IMF_INDICATORS:
            for rec in _imf_fetch_indicator(imf_id, our_field, country, start_year, end_year):
                year = rec["year"]
                if year not in merged:
                    merged[year] = {"country": country.upper(), "year": year,
                                    "source": self.source_name}
                for k, v in rec.items():
                    if k not in ("country", "year", "source"):
                        merged[year][k] = v

        result = list(merged.values())
        self._cache[cache_key] = result
        self._last_fetch = datetime.now()
        if result:
            logger.info(f"IMF: {len(result)} year-records for {country}")
        return result

    def get_metadata(self) -> Dict:
        return {
            "source_name": self.source_name,
            "update_frequency": self.update_frequency,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "real_data": True,
        }


def get_all_source_schedules() -> list:
    """Return update schedule configs for all real data sources."""
    from dataclasses import dataclass, field

    @dataclass
    class _Sched:
        source_name: str
        publisher: str
        dataset_name: str
        description: str = ""
        api_endpoint: str = ""
        url: str = ""
        update_frequency: str = "annual"
        frequency_hours: int = 8760
        last_updated: str = ""
        indicators: list = field(default_factory=list)

    return [
        _Sched(
            source_name="world_bank_wdi",
            publisher="World Bank",
            dataset_name="WDI",
            description="World Development Indicators — macro, fiscal, external",
            api_endpoint="https://api.worldbank.org/v2/country/{code}/indicator/{id}",
            url="https://data.worldbank.org/indicator",
            update_frequency="quarterly",
            frequency_hours=2160,
            indicators=[f"{code}|{name}" for code, name in WB_INDICATORS],
        ),
        _Sched(
            source_name="imf_weo",
            publisher="IMF",
            dataset_name="WEO",
            description="World Economic Outlook — fiscal, external, structural",
            api_endpoint="https://dataservices.imf.org/fruapim?indicator_id={id}",
            url="https://www.imf.org/en/Publications/WEO",
            update_frequency="quarterly",
            frequency_hours=2160,
            indicators=[f"{code}|{name}" for code, name in IMF_INDICATORS],
        ),
    ]
