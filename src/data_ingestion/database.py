"""
Data Database
=============
SQLite-backed local database for all fetched macro-debt data.
Tracks source provenance for every record — API endpoint, indicator,
fetch timestamp, and source name. Enables offline operation and
automated periodic updates.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("debt_framework")

_DB_LOCK = threading.Lock()


@dataclass
class SourceProvenance:
    """Provenance metadata for a single data record."""
    source_name: str
    source_type: str          # "api", "api_derived", "calculated"
    api_endpoint: str = ""
    indicator_code: str = ""
    publisher: str = ""       # e.g. "World Bank", "IMF", "BIS"
    dataset_name: str = ""    # e.g. "WDI", "WEO", "IDS"
    fetched_at: str = ""
    update_frequency: str = "" # "daily", "quarterly", "annual"
    url: str = ""


@dataclass
class UpdateSchedule:
    """Schedule configuration for a data source."""
    source_name: str
    frequency_hours: int
    last_updated: str = ""
    last_record_count: int = 0
    last_status: str = "never_run"


class DataDatabase:
    """SQLite database for macro-debt data with full source provenance.

    Tables:
        raw_records: One row per (country, year, indicator) with source tracking
        update_log: History of data refresh runs
        source_config: Registered data sources and their schedules
    """

    def __init__(self, db_path: str = "data/global_debt.db"):
        self.db_path = db_path
        self._ensure_dir()
        self._init_schema()
        logger.info(f"Database initialized at {db_path}")

    def _ensure_dir(self):
        import os
        d = os.path.dirname(self.db_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with _DB_LOCK:
            conn = self._conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS raw_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        country TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        indicator TEXT NOT NULL,
                        value REAL,
                        source_name TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        publisher TEXT DEFAULT '',
                        dataset_name TEXT DEFAULT '',
                        indicator_code TEXT DEFAULT '',
                        api_endpoint TEXT DEFAULT '',
                        fetched_at TEXT NOT NULL,
                        update_frequency TEXT DEFAULT 'annual',
                        url TEXT DEFAULT '',
                        UNIQUE(country, year, indicator, source_name)
                    );

                    CREATE TABLE IF NOT EXISTS update_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        records_written INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'running',
                        error_message TEXT DEFAULT '',
                        publisher TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS source_config (
                        source_name TEXT PRIMARY KEY,
                        publisher TEXT NOT NULL,
                        dataset_name TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        api_endpoint TEXT DEFAULT '',
                        url TEXT DEFAULT '',
                        update_frequency TEXT DEFAULT 'annual',
                        frequency_hours INTEGER DEFAULT 8760,
                        last_updated TEXT DEFAULT '',
                        is_active INTEGER DEFAULT 1,
                        indicators TEXT DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_records_country_year
                        ON raw_records(country, year);
                    CREATE INDEX IF NOT EXISTS idx_records_indicator
                        ON raw_records(indicator);
                    CREATE INDEX IF NOT EXISTS idx_records_source
                        ON raw_records(source_name);
                    CREATE INDEX IF NOT EXISTS idx_update_log_source
                        ON update_log(source_name, started_at);
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    #  Writing                                                             #
    # ------------------------------------------------------------------ #

    def write_records(
        self,
        country: str,
        year: int,
        records: Dict[str, Any],
        provenance: SourceProvenance,
    ) -> int:
        """Write indicator records for a single (country, year) with provenance.

        Args:
            country: ISO 3-letter country code
            year: Calendar year
            records: Dict of {indicator_name: value}
            provenance: Source metadata

        Returns:
            Number of records written
        """
        fetched_at = provenance.fetched_at or datetime.utcnow().isoformat()
        rows = []
        for indicator, value in records.items():
            if indicator in ("country", "year", "source"):
                continue
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            rows.append((
                country, year, indicator, float(value),
                provenance.source_name, provenance.source_type,
                provenance.publisher, provenance.dataset_name,
                provenance.indicator_code, provenance.api_endpoint,
                fetched_at, provenance.update_frequency, provenance.url,
            ))

        if not rows:
            return 0

        with _DB_LOCK:
            conn = self._conn()
            try:
                conn.executemany(
                    """INSERT OR REPLACE INTO raw_records
                    (country, year, indicator, value, source_name, source_type,
                     publisher, dataset_name, indicator_code, api_endpoint,
                     fetched_at, update_frequency, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug(f"DB: wrote {len(rows)} records for {country} {year} ({provenance.source_name})")
        return len(rows)

    def write_records_batch(
        self,
        db_rows: list,
    ) -> int:
        """Batch write: list of (country, year, value, indicator, provenance).

        Groups by (country, year, source_name) for efficient executemany.
        Each item in db_rows is a tuple:
            (country, year, value, indicator, SourceProvenance)
        """
        from collections import defaultdict
        groups: Dict[tuple, list] = defaultdict(list)

        for country, year, value, indicator, prov in db_rows:
            key = (country, year, prov.source_name)
            groups[key].append((
                country, year, indicator, value,
                prov.source_name, prov.source_type,
                prov.publisher, prov.dataset_name,
                prov.indicator_code, prov.api_endpoint,
                prov.fetched_at, prov.update_frequency, prov.url,
            ))

        total = 0
        with _DB_LOCK:
            conn = self._conn()
            try:
                for rows in groups.values():
                    conn.executemany(
                        """INSERT OR REPLACE INTO raw_records
                        (country, year, indicator, value, source_name, source_type,
                         publisher, dataset_name, indicator_code, api_endpoint,
                         fetched_at, update_frequency, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    total += len(rows)
                conn.commit()
            finally:
                conn.close()
        return total

    def log_update(self, source_name: str, publisher: str = "") -> int:
        """Start an update log entry. Returns the log entry ID."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                cur = conn.execute(
                    "INSERT INTO update_log (source_name, publisher, started_at) VALUES (?, ?, ?)",
                    (source_name, publisher, datetime.utcnow().isoformat()),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def complete_update(self, log_id: int, records: int, status: str = "success", error: str = "", source_name: str = ""):
        """Mark an update log entry as complete."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE update_log SET finished_at=?, records_written=?, status=?, error_message=? WHERE id=?",
                    (datetime.utcnow().isoformat(), records, status, error, log_id),
                )
                if status == "success" and source_name:
                    conn.execute(
                        "UPDATE source_config SET last_updated=? WHERE source_name=?",
                        (datetime.utcnow().isoformat(), source_name),
                    )
                conn.commit()
            finally:
                conn.close()

    def register_source(self, schedule: "SourceSchedule"):
        """Register or update a data source configuration."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO source_config
                    (source_name, publisher, dataset_name, description,
                     api_endpoint, url, update_frequency, frequency_hours,
                     indicators, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (
                        schedule.source_name,
                        schedule.publisher,
                        schedule.dataset_name,
                        schedule.description,
                        schedule.api_endpoint,
                        schedule.url,
                        schedule.update_frequency,
                        schedule.frequency_hours,
                        json.dumps(schedule.indicators),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    #  Reading                                                             #
    # ------------------------------------------------------------------ #

    def get_indicator(
        self,
        country: str,
        year: int,
        indicator: str,
    ) -> Optional[Dict]:
        """Get the most recent record for an indicator with full provenance."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                row = conn.execute(
                    """SELECT value, source_name, source_type, publisher, dataset_name,
                              indicator_code, api_endpoint, fetched_at, update_frequency, url
                       FROM raw_records
                       WHERE country=? AND year=? AND indicator=?
                       ORDER BY fetched_at DESC LIMIT 1""",
                    (country, year, indicator),
                ).fetchone()
                if not row:
                    return None
                return dict(row)
            finally:
                conn.close()

    def get_country_panel(self, country: str, start_year: int, end_year: int) -> List[Dict]:
        """Get all data for a country across a year range, pivoted to row-per-year."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                rows = conn.execute(
                    """SELECT year, indicator, value, source_name, publisher,
                              dataset_name, fetched_at
                       FROM raw_records
                       WHERE country=? AND year BETWEEN ? AND ?
                       ORDER BY year, indicator""",
                    (country, start_year, end_year),
                ).fetchall()
            finally:
                conn.close()

        panel = {}
        for row in rows:
            year = row["year"]
            if year not in panel:
                panel[year] = {"country": country, "year": year}
            panel[year][row["indicator"]] = row["value"]
            panel[year][f"{row['indicator']}__source"] = row["source_name"]
            panel[year][f"{row['indicator']}__publisher"] = row["publisher"]
            panel[year][f"{row['indicator']}__dataset"] = row["dataset_name"]
            panel[year][f"{row['indicator']}__fetched_at"] = row["fetched_at"]

        return [panel[y] for y in sorted(panel)]

    def get_all_sources(self) -> List[Dict]:
        """Get all registered data sources."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM source_config WHERE is_active=1 ORDER BY source_name"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_source_indicators(self, source_name: str) -> List[str]:
        """Get indicator list for a data source."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT indicators FROM source_config WHERE source_name=?",
                    (source_name,),
                ).fetchone()
                if row:
                    return json.loads(row["indicators"])
                return []
            finally:
                conn.close()

    def get_update_history(self, source_name: str = "", limit: int = 20) -> List[Dict]:
        """Get recent update history."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                if source_name:
                    rows = conn.execute(
                        "SELECT * FROM update_log WHERE source_name=? ORDER BY started_at DESC LIMIT ?",
                        (source_name, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM update_log ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_coverage_stats(self) -> Dict:
        """Get coverage statistics: countries, years, records per source."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                country_count = conn.execute(
                    "SELECT COUNT(DISTINCT country) FROM raw_records"
                ).fetchone()[0]
                total_records = conn.execute(
                    "SELECT COUNT(*) FROM raw_records"
                ).fetchone()[0]
                sources = conn.execute(
                    """SELECT source_name, publisher, COUNT(*) as cnt,
                              COUNT(DISTINCT country) as countries,
                              MAX(fetched_at) as latest_fetch
                       FROM raw_records
                       GROUP BY source_name, publisher"""
                ).fetchall()
                indicators = conn.execute(
                    """SELECT indicator, COUNT(DISTINCT country) as countries,
                              COUNT(*) as records,
                              GROUP_CONCAT(DISTINCT source_name) as sources
                       FROM raw_records
                       GROUP BY indicator ORDER BY indicator"""
                ).fetchall()
            finally:
                conn.close()

        return {
            "total_countries": country_count,
            "total_records": total_records,
            "sources": [dict(s) for s in sources],
            "indicators": [dict(i) for i in indicators],
        }

    def needs_update(self, source_name: str) -> bool:
        """Check if a source needs updating based on its schedule."""
        with _DB_LOCK:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT last_updated, frequency_hours FROM source_config WHERE source_name=?",
                    (source_name,),
                ).fetchone()
                if not row or not row["last_updated"]:
                    return True
                last = datetime.fromisoformat(row["last_updated"])
                freq_h = row["frequency_hours"]
                return datetime.utcnow() - last > timedelta(hours=freq_h)
            finally:
                conn.close()

    def close(self):
        pass


@dataclass
class SourceSchedule:
    """Configuration for a registered data source."""
    source_name: str
    publisher: str
    dataset_name: str
    description: str = ""
    api_endpoint: str = ""
    url: str = ""
    update_frequency: str = "annual"
    frequency_hours: int = 8760
    indicators: List[str] = field(default_factory=list)


# Global database instance
db: Optional[DataDatabase] = None


def get_database(db_path: str = "data/global_debt.db") -> DataDatabase:
    """Get or create the global database instance."""
    global db
    if db is None:
        db = DataDatabase(db_path)
    return db
