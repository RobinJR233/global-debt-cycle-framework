"""
Data Ingestion Pipeline
=======================
Orchestrates data fetching from all sources for a given country/year range.
All fetched data is cached in the local SQLite database with full source
provenance tracking. The pipeline checks the DB first before hitting APIs.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

from src.config.settings import (
    ADVANCED_ECONOMIES, EMERGING_MARKETS, COMMODITY_EXPORTERS
)
from src.data_ingestion.database import DataDatabase, SourceProvenance, get_database
from src.data_ingestion.registry import registry

logger = logging.getLogger("debt_framework")


def get_country_group(country: str) -> str:
    """Determine which group a country belongs to."""
    if country in ADVANCED_ECONOMIES:
        return "AE"
    elif country in EMERGING_MARKETS:
        return "EM"
    elif country in COMMODITY_EXPORTERS:
        return "commodity_exporter"
    return "other"


def fetch_country_data(
    country: str,
    start_year: int = 2000,
    end_year: int = 2026,
) -> Dict:
    """
    Fetch all data for a country across all sources.

    1. Try to load from local DB cache first
    2. If cache is stale or missing, fetch from all sources
    3. Write results to DB with full source provenance (batched)
    4. Merge into a single panel indexed by year

    Returns a merged panel with one row per (country, year) containing
    all available indicators from all sources.
    """
    logger.info(f"Fetching data for {country} ({start_year}-{end_year})")

    db = get_database()

    # Try DB cache first
    try:
        cached = db.get_country_panel(country, start_year, end_year)
        if cached and len(cached) > 0:
            sample = cached[0]
            completeness = sum(
                1 for k in ("gov_gross_debt_pct_gdp", "real_gdp_growth_pct")
                if sample.get(k) is not None
            )
            if completeness > 0:
                logger.info(f"Loaded {len(cached)} year-rows from DB cache for {country}")
                return _build_result(country, start_year, end_year, cached)
    except Exception as e:
        logger.debug(f"DB cache miss for {country}: {e}")

    # Fetch from all sources
    raw_results = registry.fetch_all(country, start_year, end_year)

    # Build reverse index: (country, year) -> {source_rec, indicator_keys}
    # This is O(total_records) and replaces the O(records × indicators) loop
    prov_index: Dict[Tuple[str, int], Tuple[Dict, frozenset]] = {}
    _SKIP_KEYS = frozenset({
        "country", "year", "source",
        "__publisher", "__dataset", "__url",
        "__api", "__note", "__indicator_code",
    })

    for records in raw_results.values():
        for rec in records:
            cy = (rec.get("country"), rec.get("year"))
            if cy[0] is None or cy[1] is None:
                continue
            if cy not in prov_index:
                prov_index[cy] = (rec, _SKIP_KEYS)
            # We'll use the first source that provides each (country, year)

    # Merge into panel and collect DB write rows in one pass
    panel_map: Dict[str, Dict] = {}
    db_rows: List[Tuple] = []  # (country, year, indicator, value, prov)
    _now = datetime.utcnow().isoformat()

    for year in range(start_year, end_year + 1):
        key = f"{country}_{year}"
        row: Dict = {"country": country, "year": year}
        panel_map[key] = row

        for records in raw_results.values():
            for rec in records:
                if rec.get("year") != year or rec.get("country") != country:
                    continue
                for k, v in rec.items():
                    if k in _SKIP_KEYS:
                        continue
                    row[k] = v

    # Batch write to DB
    try:
        for row in panel_map.values():
            year = row["year"]
            for indicator, value in row.items():
                if indicator in ("country", "year"):
                    continue
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue

                # Get provenance from index
                cy_key = (country, year)
                if cy_key in prov_index:
                    src_rec = prov_index[cy_key][0]
                    prov = SourceProvenance(
                        source_name=src_rec.get("source", ""),
                        source_type="api",
                        publisher=src_rec.get("__publisher", ""),
                        dataset_name=src_rec.get("__dataset", ""),
                        indicator_code=src_rec.get("__indicator_code", ""),
                        api_endpoint=src_rec.get("__api", ""),
                        fetched_at=_now,
                        update_frequency="annual",
                        url=src_rec.get("__url", ""),
                    )
                else:
                    prov = SourceProvenance(
                        source_name="calculated",
                        source_type="calculated",
                        fetched_at=_now,
                    )

                db_rows.append((country, year, float(value), indicator, prov))

        if db_rows:
            db.write_records_batch(db_rows)
            logger.info(f"Wrote {len(db_rows)} indicator records to DB for {country}")
    except Exception as e:
        logger.debug(f"DB write skipped for {country}: {e}")

    return _build_result(country, start_year, end_year, list(panel_map.values()))


def _build_result(country: str, start_year: int, end_year: int,
                  panel: list) -> Dict:
    """Build the standardized result dict."""
    return {
        "country": country,
        "group": get_country_group(country),
        "start_year": start_year,
        "end_year": end_year,
        "sources": list(registry._sources.keys()),
        "data": panel,
        "fetched_at": datetime.utcnow().isoformat(),
    }


def batch_fetch(
    countries: List[str],
    start_year: int = 2000,
    end_year: int = 2026,
) -> List[Dict]:
    """Fetch data for multiple countries."""
    results = []
    for country in countries:
        try:
            result = fetch_country_data(country, start_year, end_year)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to fetch data for {country}: {e}")
    return results


def get_ingestion_summary() -> Dict:
    """Return summary of the data ingestion layer."""
    return {
        "sources": [
            {
                "name": "IMF WEO Fiscal",
                "description": "Sovereign fiscal data — debt/GDP, balances, debt service",
                "key_metrics": [
                    "gov_gross_debt_pct_gdp",
                    "primary_balance_pct_gdp",
                    "debt_service_ratio_pct_revenue",
                    "overall_fiscal_balance_pct_gdp",
                ],
                "update_freq": "quarterly",
            },
            {
                "name": "BIS International Debt Statistics",
                "description": "Private sector debt — household, corporate, credit gaps",
                "key_metrics": [
                    "hh_debt_pct_gdp",
                    "corp_debt_pct_gdp",
                    "credit_to_gdp_gap",
                    "hh_dsr_pct_income",
                    "corporate_npl_pct",
                ],
                "update_freq": "quarterly",
            },
            {
                "name": "IMF IFS / QEDS",
                "description": "External sector vulnerability data",
                "key_metrics": [
                    "current_account_pct_gdp",
                    "external_debt_pct_gdp",
                    "fx_reserves_months_imports",
                    "st_ext_debt_pct_reserves",
                    "nii_pct_gdp",
                ],
                "update_freq": "quarterly",
            },
            {
                "name": "Bloomberg / Markit / EMBI",
                "description": "Real-time market pricing of sovereign risk",
                "key_metrics": [
                    "cds_5y_bps",
                    "sovereign_yield_10y_pct",
                    "yield_spread_vs_ust_bps",
                    "credit_rating",
                ],
                "update_freq": "daily",
            },
            {
                "name": "IMF WEO Macro / OECD",
                "description": "Macro backdrop — growth, inflation, rates, unemployment",
                "key_metrics": [
                    "real_gdp_growth_pct",
                    "inflation_pct",
                    "policy_rate_pct",
                    "unemployment_pct",
                    "output_gap_pct",
                ],
                "update_freq": "quarterly",
            },
        ],
        "total_countries": len(ADVANCED_ECONOMIES) + len(EMERGING_MARKETS),
        "coverage": {
            "tier1_full": 60,
            "tier2_partial": 80,
            "tier3_flagged": "remaining",
        },
    }
