"""
Data Source Registry
=====================
Abstraction layer over all external data sources.
All sources are real public APIs — no synthetic or mock data.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.config.settings import ADVANCED_ECONOMIES, EMERGING_MARKETS

logger = logging.getLogger("debt_framework")


class DataSource(ABC):
    """Abstract base class for all data sources."""

    source_name: str = "base"
    update_frequency: str = "unknown"

    @abstractmethod
    def fetch(self, country: str, start_year: int, end_year: int) -> List[Dict]:
        ...

    @abstractmethod
    def get_metadata(self) -> Dict:
        ...


class DataSourceRegistry:
    """Registry managing all real-data source connections."""

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self._register_default_sources()

    def _register_default_sources(self):
        """Register all data sources from real public APIs and curated local data."""
        from src.data_ingestion.real_sources import (
            WorldBankDataSource,
            IMFWEOSource,
        )
        from src.data_ingestion.local_data import LocalDataSource

        sources = {
            "macro_wb": WorldBankDataSource(),
            "fiscal_imf": IMFWEOSource(),
            "curated": LocalDataSource(),
        }
        for name, source in sources.items():
            self._sources[name] = source

    def register(self, name: str, source: DataSource):
        """Register a custom data source."""
        self._sources[name] = source

    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        return self._sources.get(name)

    def get_source_names(self) -> List[str]:
        """Return names of all registered sources."""
        return list(self._sources.keys())

    def fetch_all(self, country: str, start_year: int, end_year: int) -> Dict[str, List[Dict]]:
        """Fetch from all registered sources and merge results."""
        results = {}
        for name, source in self._sources.items():
            try:
                data = source.fetch(country, start_year, end_year)
                results[name] = data
                logger.info(f"Fetched {name} for {country}: {len(data)} records")
            except Exception as e:
                logger.warning(f"Failed to fetch {name} for {country}: {e}")
                results[name] = []
        return results

    def get_available_countries(self) -> List[str]:
        """Return list of countries available across all sources."""
        return list(set(
            list(ADVANCED_ECONOMIES) + list(EMERGING_MARKETS)
        ))


# Global registry instance
registry = DataSourceRegistry()
