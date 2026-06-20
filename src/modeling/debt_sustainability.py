"""
Debt Sustainability Model (DSM)
================================
Implements the IMF-style debt sustainability analysis:
  - Snowball projection (5-year horizon)
  - Baseline, adverse, and severely adverse scenarios
  - Monte Carlo simulation for fan charts
  - Convergence test
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("debt_framework")


@dataclass
class ScenarioParams:
    """Parameters for a stress scenario."""
    name: str
    gdp_growth_delta: float        # deviation from baseline (pp)
    interest_rate_delta: float     # deviation from baseline (pp)
    primary_balance_delta: float   # deviation from baseline (pp of GDP)
    fx_depreciation: float = 0.0   # currency depreciation (pp)
    external_shock: float = 0.0    # external sector shock (pp of GDP)


BASELINE = ScenarioParams("Baseline", 0, 0, 0)
ADVERSE = ScenarioParams("Adverse", -2.0, 1.5, -2.0, 0.0, -2.0)
SEVERELY_ADVERSE = ScenarioParams("Severely Adverse", -4.0, 3.0, -4.0, -30.0, -4.0)


@dataclass
class DSMResult:
    """Result of a debt sustainability analysis for one country."""
    country: str
    baseline_path: List[float] = field(default_factory=list)
    adverse_path: List[float] = field(default_factory=list)
    severe_path: List[float] = field(default_factory=list)
    fan_chart_low: List[float] = field(default_factory=list)
    fan_chart_high: List[float] = field(default_factory=list)
    required_pb_adjustment: float = 0.0
    sustainability_verdict: str = "unknown"
    baseline_final_debt_gdp: float = 0.0
    adverse_final_debt_gdp: float = 0.0
    severe_final_debt_gdp: float = 0.0


def project_debt_snowball(
    initial_debt_gdp: float,
    baseline_params: Dict,
    scenario: ScenarioParams,
    years: int = 5,
) -> List[float]:
    """Project debt/GDP forward using the snowball formula.

    Δd = (r - g) * d + pb

    Args:
        initial_debt_gdp: Starting debt/GDP ratio
        baseline_params: Dict with baseline projections
            - gdp_growth: list of projected nominal GDP growth rates
            - interest_rate: list of projected effective interest rates
            - primary_balance: list of projected primary balances
        scenario: ScenarioParams with deltas from baseline
        years: Number of years to project

    Returns:
        List of projected debt/GDP ratios
    """
    path = [initial_debt_gdp]

    for t in range(1, years + 1):
        d_prev = path[t - 1]

        g = baseline_params.get("gdp_growth", [3.0] * years)[min(t - 1, len(baseline_params.get("gdp_growth", [])) - 1)]
        r = baseline_params.get("interest_rate", [4.0] * years)[min(t - 1, len(baseline_params.get("interest_rate", [])) - 1)]
        pb = baseline_params.get("primary_balance", [-3.0] * years)[min(t - 1, len(baseline_params.get("primary_balance", [])) - 1)]

        # Apply scenario deltas
        g += scenario.gdp_growth_delta
        r += scenario.interest_rate_delta
        pb += scenario.primary_balance_delta

        # Snowball formula: rates are in percentage points, divide by 100
        delta_d = (r - g) / 100.0 * d_prev + pb

        # FX shock impact on foreign-currency debt (simplified)
        if scenario.fx_depreciation != 0:
            # Assume 40% of debt is foreign currency denominated
            fc_share = 0.40
            delta_d += fc_share * d_prev * scenario.fx_depreciation / 100.0

        d_new = d_prev + delta_d
        path.append(round(max(0, d_new), 1))

    return path


def monte_carlo_fan_chart(
    initial_debt_gdp: float,
    baseline_params: Dict,
    years: int = 5,
    n_sims: int = 10000,
) -> Dict:
    """Generate fan chart via Monte Carlo simulation.

    Assumes normally distributed shocks to growth, rates, and primary balance.
    """
    paths = []

    # Shock standard deviations
    g_std = 1.0
    r_std = 0.5
    pb_std = 1.5

    for _ in range(n_sims):
        path = [initial_debt_gdp]
        for t in range(1, years + 1):
            d_prev = path[t - 1]

            g = baseline_params.get("gdp_growth", [3.0] * years)[min(t - 1, len(baseline_params.get("gdp_growth", [])) - 1)]
            r = baseline_params.get("interest_rate", [4.0] * years)[min(t - 1, len(baseline_params.get("interest_rate", [])) - 1)]
            pb = baseline_params.get("primary_balance", [-3.0] * years)[min(t - 1, len(baseline_params.get("primary_balance", [])) - 1)]

            # Draw shocks (rates are in pct points, divide by 100)
            g_shock = np.random.normal(0, g_std)
            r_shock = np.random.normal(0, r_std)
            pb_shock = np.random.normal(0, pb_std)

            g += g_shock
            r += r_shock
            pb += pb_shock

            delta_d = (r - g) / 100.0 * d_prev + pb
            d_new = d_prev + delta_d
            path.append(max(0, d_new))
        paths.append(path)

    paths = np.array(paths)

    result = {
        "median": [round(float(np.median(paths[:, t])), 1) for t in range(years + 1)],
        "p10": [round(float(np.percentile(paths[:, t], 10)), 1) for t in range(years + 1)],
        "p25": [round(float(np.percentile(paths[:, t], 25)), 1) for t in range(years + 1)],
        "p75": [round(float(np.percentile(paths[:, t], 75)), 1) for t in range(years + 1)],
        "p90": [round(float(np.percentile(paths[:, t], 90)), 1) for t in range(years + 1)],
    }

    return result


def compute_required_pb_adjustment(
    initial_debt_gdp: float,
    baseline_params: Dict,
    target_debt_gdp: float = 60.0,
    years: int = 5,
) -> float:
    """Compute the required primary balance adjustment to reach a target debt level.

    This answers: "What primary surplus is needed to bring debt/GDP to target?"
    """
    # Simplified: use the formula Δd = pb → pb = Δd / years
    delta_d_needed = target_debt_gdp - initial_debt_gdp
    pb_needed = delta_d_needed / years

    # But also need to account for the snowball effect
    # Approximate: avg (r-g) over the horizon
    avg_r_g = 0.5  # baseline assumption
    total_snowball = avg_r_g * initial_debt_gdp * years / 2  # Approximate triangular area

    pb_adjustment = (delta_d_needed + total_snowball) / years

    return round(pb_adjustment, 1)


def assess_sustainability(
    baseline_path: List[float],
    adverse_path: List[float],
    severe_path: List[float],
    fan_chart_p90: List[float],
    initial_debt_gdp: float,
) -> str:
    """Assess debt sustainability based on projection outcomes."""
    final_baseline = baseline_path[-1] if baseline_path else initial_debt_gdp
    final_adverse = adverse_path[-1] if adverse_path else initial_debt_gdp
    final_p90 = fan_chart_p90[-1] if fan_chart_p90 else initial_debt_gdp

    if final_baseline < initial_debt_gdp * 0.9 and final_adverse < initial_debt_gdp:
        return "sustainable"
    elif final_baseline > initial_debt_gdp and final_adverse > initial_debt_gdp * 1.5:
        return "unsustainable"
    elif final_baseline > initial_debt_gdp or final_adverse > initial_debt_gdp * 1.2:
        return "at_risk"
    elif final_p90 > initial_debt_gdp * 1.5:
        return "at_risk"
    else:
        return "sustainable"


def run_debt_sustainability_analysis(
    country: str,
    current_row: Dict,
    historical_data: List[Dict],
) -> DSMResult:
    """Run full DSM for a country given current data and historical context."""
    initial_debt_gdp = current_row.get("gov_gross_debt_pct_gdp", 70.0)
    is_em = current_row.get("group", "EM") == "EM"

    # Build baseline parameters from current state (deterministic projections)
    g_growth = current_row.get("real_gdp_growth_pct", 3.0)
    inflation = current_row.get("inflation_pct", 2.0)
    nominal_g = g_growth + inflation
    r_eff = current_row.get("policy_rate_pct", 4.0)
    pb = current_row.get("primary_balance_pct_gdp", -3.0)

    baseline_params = {
        "gdp_growth": [nominal_g for _ in range(5)],
        "interest_rate": [r_eff for _ in range(5)],
        "primary_balance": [pb for _ in range(5)],
    }

    # Run projections for each scenario
    baseline_path = project_debt_snowball(initial_debt_gdp, baseline_params, BASELINE)
    adverse_path = project_debt_snowball(initial_debt_gdp, baseline_params, ADVERSE)
    severe_path = project_debt_snowball(initial_debt_gdp, baseline_params, SEVERELY_ADVERSE)

    # Fan chart
    fan_chart = monte_carlo_fan_chart(initial_debt_gdp, baseline_params)

    # Required adjustment
    required_pb = compute_required_pb_adjustment(initial_debt_gdp, baseline_params)

    # Sustainability verdict
    verdict = assess_sustainability(baseline_path, adverse_path, severe_path, fan_chart["p90"], initial_debt_gdp)

    return DSMResult(
        country=country,
        baseline_path=baseline_path,
        adverse_path=adverse_path,
        severe_path=severe_path,
        fan_chart_low=fan_chart["p10"],
        fan_chart_high=fan_chart["p90"],
        required_pb_adjustment=required_pb,
        sustainability_verdict=verdict,
        baseline_final_debt_gdp=baseline_path[-1],
        adverse_final_debt_gdp=adverse_path[-1],
        severe_final_debt_gdp=severe_path[-1],
    )
