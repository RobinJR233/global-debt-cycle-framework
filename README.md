# Global Debt Cycle Assessment Framework

An end-to-end macro-financial analysis tool that assesses sovereign debt sustainability and systemic risk across 100+ countries. It pulls real-time data from IMF, World Bank, BIS, and FRED, runs three analytical engines (Debt Sustainability Model, Early Warning Signals, External Vulnerability), and produces a composite risk score mapped to a five-phase debt cycle framework.

## Theoretical Framework

Based on the sovereign debt cycle model developed by Reinhart & Rogoff and later refined by the IMF's Debt Sustainability Framework. The core insight: sovereign debt typically moves through five distinct phases driven by the interaction of interest rates (r), economic growth (g), primary balances, and external financing conditions.

### The g - r Channel

The spread between the nominal interest rate (r) and nominal GDP growth (g) is the primary driver of debt dynamics:

- **g > r**: Debt naturally stabilizes or declines — the " virtuous cycle"
- **g < r**: Debt compounds upward — the "debt spiral" requiring primary surpluses to offset

### Five-Phase Debt Cycle

| Score | Phase | Color | Economic Characteristics |
|------:|-------|:-----:|--------------------------|
| 70–100 | Stable, Low Debt | Green | Low leverage, fiscal surpluses, strong external position |
| 55–70  | Expansion | Blue | Accelerating credit, widening deficits, rising optimism |
| 40–55  | Overextension / Peak | Yellow | Credit gap above trend, external vulnerabilities emerging |
| 25–40  | Stress, Early Deleveraging | Orange | Market repricing, CDS widening, reserves under pressure |
| 0–25   | Crisis, Deep Deleveraging | Red | Capital flight, IMF program likely, forced adjustment |

## Analytical Engines

### 1. Debt Sustainability Model (DSM)

Projects debt/GDP forward using a stochastic snowball model with three scenarios:

```
Δd_t = (r_t - g_t) * d_{t-1} + pb_t
```

- **Baseline**: Current trend continuation
- **Adverse**: 2pp GDP growth shock, 1.5pp rate shock, 2pp primary balance deterioration, FX depreciation
- **Severely Adverse**: 4pp GDP shock, 3pp rate shock, 4pp primary balance shock, 30% currency depreciation

Each scenario runs 10,000 Monte Carlo simulations to generate fan charts and probability distributions.

### 2. Early Warning Signals (EWS)

Implements the Kaminsky-Lizondo-Reinhart signal approach with 12 monitored indicators:

| Indicator | EM Threshold | DM Threshold |
|-----------|-------------|-------------|
| BIS credit-to-GDP gap | +10pp | +8pp |
| Current account deficit | -5% GDP | -4% GDP |
| Sovereign CDS | 300 bps | 100 bps |
| Gov debt / GDP | 90% | 90% |
| ST debt / reserves | 100% | 100% |
| REER appreciation (3y) | 15% | 15% |
| Fiscal deficit | -3% GDP | -5% GDP |

### 3. External Vulnerability Model

Assesses currency mismatch and external financing risk:

- Gross Financing Need (GFN) / FX reserves ratio
- Currency mismatch index (foreign currency debt exposure)
- Short-term external debt / reserves coverage
- 30% FX depreciation stress test on external debt burden

### 4. Composite Scoring

Six weighted dimensions produce a 0–100 score:

| Dimension | Weight | Rationale |
|-----------|------:|-----------|
| Fiscal Health | 25% | Primary balance trajectory, debt/GDP level |
| Private Sector | 20% | Credit growth, household/corporate leverage |
| External Position | 20% | Current account, reserves, external debt |
| Market Pricing | 15% | CDS spreads, rating migration, sovereign spreads |
| Macro Context | 10% | g - r, growth outlook, inflation |
| Structural Resilience | 10% | Institutional quality, governance, diversification |

## Installation

```bash
git clone https://github.com/RobinJR233/global-debt-cycle-framework.git
cd global-debt-cycle-framework
pip install -r requirements.txt
```

### Dependencies

**Required:**
- `numpy >= 1.24`
- `scipy >= 1.10`
- `pandas >= 2.0`

**Optional (visualization):**
- `matplotlib >= 3.7`
- `plotly >= 5.15`
- `folium >= 0.14`

**Optional (utilities):**
- `requests >= 2.31` (for live API data)
- `openpyxl >= 3.1` (for Excel I/O)
- `rich >= 13.0` (for CLI output)
- `tqdm >= 4.65` (for progress bars)

No API keys required — all data sources are public.

## Usage

### CLI

```bash
# Single country analysis
python main.py country USA --start 2000 --end 2026

# Table output (default)
python main.py country USA

# JSON output for programmatic use
python main.py country CHN --format json --start 2010 --end 2026

# Batch analysis
python main.py batch "USA,JPN,DEU,GBR,CHN,IND,BRA"

# Show data source status
python main.py sources

# Verbose debug logging
python main.py country USA -v
```

### Python API

```python
from src import (
    run_pipeline_for_country,
    run_pipeline_batch,
    get_global_summary,
    registry,
)

# Single country
result = run_pipeline_for_country("USA", start_year=2000, end_year=2026)

# Access results
print(result["latest_year"]["composite_score"])   # 0-100
print(result["latest_year"]["phase"])              # green/blue/yellow/orange/red
print(result["latest_year"]["g_minus_r"])          # core debt dynamics indicator
print(result["dsm"]["sustainability_verdict"])     # sustainable / at_risk / unsustainable
print(result["external_vulnerability"]["overall_risk"])  # low / moderate / high / severe

# Batch analysis
countries = ["USA", "JPN", "DEU", "GBR", "CHN", "IND", "BRA", "RUS", "MEX", "ZAF"]
results = run_pipeline_batch(countries, start_year=2000, end_year=2026)
summary = get_global_summary(results)

print(summary["top_5_resilient"])       # [(code, score, phase), ...]
print(summary["bottom_5_vulnerable"])   # [(code, score, phase), ...]
print(summary["phase_distribution"])    # {phase: count}
```

### Individual Engines

```python
from src.data_ingestion.pipeline import fetch_country_data
from src.feature_engineering.engineer import engineer_features
from src.modeling.early_warning import run_early_warning_signals
from src.modeling.debt_sustainability import run_debt_sustainability_analysis
from src.modeling.external_vulnerability import assess_external_vulnerability
from src.scoring.composite_scorer import compute_composite_score

# Stage 1: Data
raw = fetch_country_data("USA", 2000, 2026)

# Stage 2: Features
featured = engineer_features(raw["data"])

# Stage 3: EWS
is_em = raw["group"] == "EM"
for row in featured:
    ews = run_early_warning_signals(row, is_em=is_em)

# Stage 4: DSM
dsm = run_debt_sustainability_analysis("USA", featured[-1], featured[:-1])

# Stage 5: External Vulnerability
ext = assess_external_vulnerability(featured[-1])

# Stage 6: Composite Score
scored = compute_composite_score(featured, ews_results, dsm, ext)
```

## Data Sources

All data is sourced live from public APIs. No API keys required.

| Source | Indicators | Frequency |
|--------|-----------|-----------|
| IMF WEO | Government debt, fiscal balance, g - r | Annual |
| World Bank | GDP, current account, external debt | Annual |
| BIS | Credit-to-GDP gaps | Annual |
| FRED | US Treasury yields, VIX (market context) | Daily → Annual |

## Project Structure

```
global-debt-cycle-framework/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/
│   ├── __init__.py                  # Package entry, public API
│   │
│   ├── config/
│   │   └── settings.py              # All tunable parameters
│   │                               # - Scoring weights
│   │                               # - Phase thresholds
│   │                               # - Early warning thresholds
│   │                               # - Alert thresholds
│   │                               # - Data source URLs
│   │                               # - Model parameters (HP filter, windows)
│   │                               # - Pipeline configuration
│   │
│   ├── data_ingestion/
│   │   ├── real_sources.py          # IMF, World Bank, BIS API clients
│   │   ├── database.py              # SQLite storage layer
│   │   ├── local_data.py            # Embedded fallback data
│   │   ├── registry.py              # Country classification (AE/EM/commodity)
│   │   ├── updater.py               # Incremental data refresh
│   │   ├── pipeline.py              # Data fetching orchestration
│   │   └── __init__.py
│   │
│   ├── feature_engineering/
│   │   ├── engineer.py              # HP filter, credit gaps, derived indicators
│   │   └── __init__.py
│   │
│   ├── modeling/
│   │   ├── debt_sustainability.py   # Snowball model, Monte Carlo, scenarios
│   │   ├── early_warning.py         # Kaminsky-Lizondo-Reinhart signals
│   │   ├── external_vulnerability.py # GFN/reserves, currency mismatch
│   │   └── __init__.py
│   │
│   ├── scoring/
│   │   ├── composite_scorer.py      # Weighted aggregation, phase classification
│   │   └── __init__.py
│   │
│   └── orchestration/
│       ├── pipeline.py              # End-to-end pipeline wiring
│       └── __init__.py
│
├── data/
│   ├── global_debt.db               # SQLite database (auto-created)
│   ├── raw/                         # Raw API responses
│   ├── curated/                     # Cleaned datasets
│   ├── features/                    # Engineered features
│   └── output/                      # Reports and exports
│
├── tests/
│   └── test_framework.py            # Unit tests (HP filter, scoring, etc.)
│
└── notebooks/                       # Analysis notebooks (empty — add your own)
```

## Key Design Decisions

### HP Filter for Trend Decomposition

The Hodrick-Prescott filter separates trend from cyclical components in credit-to-GDP ratios. Lambda values follow BIS convention:
- Annual data: λ = 400,000
- Quarterly data: λ = 1,600,000

### Monte Carlo Simulation

The DSM projects 10,000 paths per scenario using random shocks to growth, rates, and primary balances. This produces empirically grounded fan charts rather than single-point forecasts.

### No Hardcoded Data

All macro indicators come from public APIs. A local fallback module (`local_data.py`) provides embedded data for offline use or when APIs are unavailable.

## Testing

```bash
python -m pytest tests/
```

Covers: HP filter correctness, one-sided HP filter, credit gap computation, scoring engine, and phase classification.

## Extending the Framework

### Adding a new country group

Edit `src/config/settings.py` — all country lists are defined there (Advanced Economies, Emerging Markets, Commodity Exporters, Eurozone, Small Open Economies).

### Adjusting scoring weights

Modify the `SCORING_WEIGHTS` dataclass in `src/config/settings.py`. Weights must sum to 1.0.

### Adding a data source

1. Add the API client in `src/data_ingestion/real_sources.py`
2. Register the source in `src/data_ingestion/registry.py`
3. Add indicator extraction logic in `src/data_ingestion/pipeline.py`

### Adding a scoring dimension

1. Define the dimension and weight in `src/config/settings.py`
2. Implement the sub-score function in `src/scoring/composite_scorer.py`
3. Wire it into `compute_composite_score()`

## Limitations

- Annual frequency only (quarterly data available but not yet integrated into scoring)
- Sovereign debt focus — does not cover corporate or household debt cycles directly
- Data availability varies by country; some indicators may be missing for smaller economies
- External debt data from World Bank has a 2-3 year lag
- The framework does not generate trading signals or investment recommendations

## License

MIT
