"""
Global Debt Cycle Assessment Framework
========================================
Configuration: Source URLs, thresholds, scoring weights, pipeline parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

# ---------------------------------------------------------------------------
# Country groups
# ---------------------------------------------------------------------------
ADVANCED_ECONOMIES = [
    "USA", "JPN", "DEU", "GBR", "FRA", "ITA", "CAN", "AUS", "KOR", "ESP",
    "NLD", "CHE", "SWE", "NOR", "AUT", "BEL", "DNK", "FIN", "PRT", "IRL",
    "GRC", "SGP", "HKG", "NZL",
]

EMERGING_MARKETS = [
    "CHN", "IND", "BRA", "RUS", "MEX", "ZAF", "TUR", "IDN", "THA", "MYS",
    "POL", "COL", "ARG", "CHL", "PER", "EGY", "NGA", "VNM", "PAK", "PHL",
    "HUN", "CZE", "ROU", "KAZ", "UKR", "MAR", "SAU", "ARE", "QAT", "ISR",
    "DOM", "GTM", "ECU", "BOL", "PRY", "URY", "TZA", "KEN", "ETH", "GHA",
]

COMMODITY_EXPORTERS = [
    "SAU", "ARE", "QAT", "KWT", "IRN", "RUS", "BRA", "CHL", "PER", "COL",
    "AUS", "CAN", "NOR", "NGA", "ZAF", "MEX", "IDN", "MYS",
]

EUROZONE = [
    "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "GRC", "PRT",
    "FIN", "LUX", "SVK", "SVN", "EST", "LVA", "LTU", "CYP", "MLT",
]

SMALL_OPEN_ECONOMIES = [
    "SGP", "HKG", "CH", "LUX", "ISL", "FIN", "NOR", "NZL",
]

# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------
@dataclass
class ScoringWeights:
    """Weights for composite score dimensions."""
    fiscal: float = 0.25
    private_sector: float = 0.20
    external: float = 0.20
    market_pricing: float = 0.15
    macro_context: float = 0.10
    structural_resilience: float = 0.10

SCORING_WEIGHTS = ScoringWeights()


@dataclass
class DebtCyclePhaseConfig:
    """Phase classification thresholds (composite score 0-100)."""
    green_stable_low_debt: Tuple[float, float] = (70.0, 100.0)
    blue_expansion: Tuple[float, float] = (55.0, 70.0)
    yellow_overextension_peak: Tuple[float, float] = (40.0, 55.0)
    orange_stress_early_deleveraging: Tuple[float, float] = (25.0, 40.0)
    red_crisis_deep_deleveraging: Tuple[float, float] = (0.0, 25.0)

PHASE_CONFIG = DebtCyclePhaseConfig()

# ---------------------------------------------------------------------------
# Early warning thresholds
# ---------------------------------------------------------------------------
@dataclass
class EarlyWarningThresholds:
    """Thresholds for early warning signal engine."""
    bis_credit_gap_em: float = 10.0        # pp above trend for EMs
    bis_credit_gap_dm: float = 8.0         # pp above trend for DMs
    current_account_deficit_dm: float = 4.0  # % of GDP sustained 2+ years
    current_account_deficit_em: float = 5.0
    sovereign_cds_em: float = 300.0        # bps
    sovereign_cds_dm: float = 100.0        # bps
    gov_debt_gdp_threshold: float = 90.0   # % of GDP (Reinhart-Rogoff)
    st_debt_reserves_ratio: float = 100.0  # % short-term ext debt / reserves
    reer_appreciation_3y: float = 15.0     # % over 3 years
    fiscal_deficit_dm: float = 5.0         # % of GDP
    fiscal_deficit_em: float = 3.0         # % of GDP

EW_THRESHOLDS = EarlyWarningThresholds()

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
@dataclass
class AlertThresholds:
    """Thresholds for alert engine."""
    score_drop_quarterly: float = 10.0    # points drop in 1 quarter
    cds_widening_quarterly: float = 100.0  # bps widening in 1 quarter
    bis_gap_upper: float = 10.0           # pp above trend
    bis_gap_lower: float = -5.0           # pp below trend
    fx_reserves_months: float = 3.0       # months of imports
    gfn_reserves_ratio: float = 80.0      # % GFN / reserves
    crisis_score_em: float = 500.0        # bps CDS for EM
    crisis_score_dm: float = 150.0        # bps CDS for DM
    divergence_threshold: float = 30.0    # points between sub-scores

ALERT_THRESHOLDS = AlertThresholds()

# ---------------------------------------------------------------------------
# Data source configuration
# ---------------------------------------------------------------------------
@dataclass
class DataSourceConfig:
    """Configuration for external data sources."""
    imf_weo_api: str = "https://dataservices.imf.org/fruapim?indicator_id="
    bis_credit_gap_url: str = "https://www.bis.org/statistics/credit_gaps.htm"
    world_bank_api: str = "https://api.worldbank.org/v2/country/"
    imf_qeds_url: str = "https://data.imf.org/regular.aspx?key=61545843"
    bis_ids_url: str = "https://stats.bis.org/statx/srs/"
    update_frequency_market: str = "daily"
    update_frequency_macro: str = "quarterly"
    update_frequency_structural: str = "annual"

DATA_SOURCE_CONFIG = DataSourceConfig()

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass
class ModelParameters:
    """Parameters for analytical models."""
    # HP filter lambda (BIS convention)
    hp_lambda_annual: int = 400_000
    hp_lambda_quarterly: int = 1_600_000
    # Rolling windows
    debt_trajectory_window: int = 10  # years
    rolling_macro_window: int = 5    # years
    # Snowball model
    dsm_projection_years: int = 5
    dsm_simulation_runs: int = 10_000
    # Cycle analysis
    typical_ae_cycle_length: int = 18  # years
    typical_em_cycle_length: int = 10  # years
    # Backtesting
    backtest_start_year: int = 1995
    crisis_horizon_years: int = 3

MODEL_PARAMS = ModelParameters()

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Overall pipeline execution configuration."""
    countries_tier1_count: int = 60
    countries_tier2_count: int = 80
    min_data_completeness: float = 0.70
    min_consecutive_years: int = 5
    outlier_threshold_sigma: float = 3.0
    winsorize_bounds: Tuple[float, float] = (-3.0, 3.0)

PIPELINE_CONFIG = PipelineConfig()
