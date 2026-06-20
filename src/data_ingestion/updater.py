"""
Data Updater
============
Automated periodic refresh of the local database from all real data sources.
Sources: World Bank WDI (live API), IMF WEO (live API), and curated local
real data from official publications (BIS, IMF FSI, JP Morgan, rating agencies).

Usage:
    python src/data_ingestion/updater.py            # check and update stale sources
    python src/data_ingestion/updater.py --force     # update all sources regardless of schedule
    python src/data_ingestion/updater.py --source wb_wdi  # update a specific source only
"""

import argparse
import json
import logging
import sys
from datetime import datetime

from src.data_ingestion.database import get_database, SourceProvenance
from src.data_ingestion.real_sources import (
    WorldBankDataSource,
    IMFWEOSource,
    get_all_source_schedules,
)
from src.data_ingestion.local_data import LocalDataSource
from src.config.settings import ADVANCED_ECONOMIES, EMERGING_MARKETS

logger = logging.getLogger("debt_framework")

ALL_COUNTRIES = list(set(ADVANCED_ECONOMIES + EMERGING_MARKETS))
_YEAR_NOW = datetime.utcnow().year
_WB = WorldBankDataSource()
_IMF = IMFWEOSource()
_LOCAL = LocalDataSource()


def update_all_sources(db_path: str = "data/global_debt.db", force: bool = False) -> dict:
    """Update all registered data sources that are stale."""
    from src.data_ingestion.database import get_database

    database = get_database(db_path)
    schedules = get_all_source_schedules()
    results = {}

    for schedule in schedules:
        source_name = schedule.source_name
        needs = force or database.needs_update(source_name)
        status = "skipped"
        records = 0
        error = ""

        if not needs:
            results[source_name] = {
                "status": "skipped",
                "reason": "up_to_date",
                "last_updated": schedule.last_updated or "never",
            }
            logger.info(f"[Updater] {source_name}: up to date, skipping")
            continue

        database.register_source(schedule)
        log_id = database.log_update(source_name, schedule.publisher)

        try:
            fetched = _fetch_for_source(source_name)
            total_written = 0
            for country_code, country_data in fetched.items():
                for year, indicators in country_data.items():
                    prov = SourceProvenance(
                        source_name=source_name,
                        source_type="api",
                        publisher=schedule.publisher,
                        dataset_name=schedule.dataset_name,
                        indicator_code=indicators.get("__indicator_code", ""),
                        api_endpoint=schedule.api_endpoint,
                        fetched_at=datetime.utcnow().isoformat(),
                        update_frequency=schedule.update_frequency,
                        url=schedule.url,
                    )
                    ind = {k: v for k, v in indicators.items() if not k.startswith("__")}
                    n = database.write_records(country_code, year, ind, prov)
                    total_written += n

            status = "success"
            records = total_written
            logger.info(
                f"[Updater] {source_name}: updated {total_written} records "
                f"across {len(fetched)} countries"
            )
        except Exception as e:
            status = "error"
            error = str(e)
            logger.error(f"[Updater] {source_name}: FAILED — {e}")

        database.complete_update(log_id, records, status, error, source_name)
        results[source_name] = {
            "status": status,
            "records": records,
            "error": error,
            "updated_at": datetime.utcnow().isoformat() if status == "success" else None,
        }

    return results


def _fetch_for_source(source_name):
    """Fetch data for all countries from a named source."""
    fetched = {}
    for cc in ALL_COUNTRIES:
        try:
            if source_name == "world_bank_wdi":
                data = _WB.fetch(cc, 2000, _YEAR_NOW)
            elif source_name == "imf_weo":
                data = _IMF.fetch(cc, 2000, _YEAR_NOW)
            else:
                continue
        except Exception:
            continue

        country_map = {}
        for rec in data:
            yr = rec.get("year")
            if yr is None:
                continue
            yr = int(yr)
            inds = {k: v for k, v in rec.items()
                    if k not in ("country", "year", "source")}
            country_map[yr] = inds
        if country_map:
            fetched[cc] = country_map
    return fetched


def get_update_summary(db_path: str = "data/global_debt.db") -> dict:
    """Get a summary of all data sources and their update status."""
    database = get_database(db_path)

    coverage = database.get_coverage_stats()
    sources = database.get_all_sources()

    source_details = []
    for s in sources:
        needs = database.needs_update(s["source_name"])
        source_details.append({
            "name": s["source_name"],
            "publisher": s["publisher"],
            "dataset": s["dataset_name"],
            "last_updated": s.get("last_updated", "never"),
            "frequency": s["update_frequency"],
            "needs_update": needs,
            "indicators": json.loads(s.get("indicators", "{}") or "{}"),
        })

    return {
        "coverage": coverage,
        "sources": source_details,
        "generated_at": datetime.utcnow().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Update data sources")
    parser.add_argument("--force", action="store_true", help="Update all sources regardless of schedule")
    parser.add_argument("--source", type=str, default="", help="Update a specific source only")
    parser.add_argument("--summary", action="store_true", help="Show update status summary")
    parser.add_argument("--db", type=str, default="data/global_debt.db", help="Database path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.summary:
        summary = get_update_summary(args.db)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    if args.source:
        db = get_database(args.db)
        sources_cfg = db.get_all_sources()
        target = None
        for s in sources_cfg:
            if s["source_name"] == args.source:
                target = s
                break
        if not target:
            print(f"Unknown source: {args.source}")
            print(f"Available: {[s['source_name'] for s in sources_cfg]}")
            sys.exit(1)

        year_end = datetime.utcnow().year
        country_data = _fetch_for_source(args.source)

        db.register_source(target)
        log_id = db.log_update(target["source_name"], target["publisher"])
        total = 0
        for cc, cd in country_data.items():
            for yr, inds in cd.items():
                prov = SourceProvenance(
                    source_name=target["source_name"],
                    source_type="api",
                    publisher=target["publisher"],
                    dataset_name=target["dataset_name"],
                    fetched_at=datetime.utcnow().isoformat(),
                    update_frequency=target["update_frequency"],
                    url=target["url"],
                )
                total += db.write_records(cc, yr, inds, prov)
        db.complete_update(log_id, total, "success", "", target["source_name"])
        print(f"Updated {args.source}: {total} records written")
    else:
        results = update_all_sources(args.db, force=args.force)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
