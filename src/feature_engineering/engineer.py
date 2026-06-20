"""
Feature Engineering
===================
Derives debt cycle indicators from raw data:
  - BIS Credit-to-GDP Gap
  - Debt trajectory classification
  - Snowball indicator (r-g)
  - Fiscal space indicator
  - Cycle phase classification
  - Z-score normalization
"""

import logging
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("debt_framework")


# ---------------------------------------------------------------------------
# HP Filter (one-sided, for real-time estimation)
# ---------------------------------------------------------------------------

def _hp_filter(series: np.ndarray, lam: float) -> np.ndarray:
    """Apply Hodrick-Prescott filter to extract trend component.

    Uses the standard symmetric HP filter for post-sample data.
    For real-time estimation (most recent points), use _one_sided_hp_filter.
    """
    n = len(series)
    if n < 4:
        return series.copy()

    # Build the HP filter matrix: (I + lambda * D'D) * trend = data
    I = np.eye(n)
    D = np.diff(I, n=2, axis=0)
    DTD = D.T @ D

    try:
        trend = np.linalg.solve(I + lam * DTD, series)
        return trend
    except np.linalg.LinAlgError:
        logger.warning("HP filter failed, returning raw series")
        return series.copy()


def _one_sided_hp_filter(series: np.ndarray, lam: float) -> np.ndarray:
    """One-sided HP filter using Kalman filter approach for real-time estimation."""
    from scipy import linalg

    n = len(series)
    if n < 5:
        return series.copy()

    # State space representation
    # y_t = mu_t + eps_t,  eps_t ~ N(0, sigma_eps^2)
    # mu_{t+1} = mu_t + beta_t
    # beta_{t+1} = beta_t + eta_t,  eta_t ~ N(0, sigma_eta^2)
    # sigma_eta^2 / sigma_eps^2 = 1 / lambda

    sig_ratio = 1.0 / lam

    # State vector: [level, slope]
    F = np.array([[1, 1], [0, 1]])          # Transition matrix
    H = np.array([[1, 0]])                   # Observation matrix
    Q = sig_ratio * np.eye(2)                # State noise
    R = np.array([[1.0]])                    # Observation noise

    # Initial conditions
    x = np.array([series[0], 0])
    P = 1e6 * np.eye(2)

    trend = np.zeros(n)
    for t in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        y = series[t] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T / S
        x = x + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P

        trend[t] = (H @ x).item()

    return trend


def compute_bis_credit_gap(
    credit_to_gdp: np.ndarray,
    lam_annual: float = 400_000,
) -> np.ndarray:
    """Compute the BIS Credit-to-GDP Gap.

    gap(t) = credit_to_gdp(t) - HP_trend(t)
    Uses one-sided HP filter for the most recent observations.
    """
    n = len(credit_to_gdp)
    gap = np.full(n, np.nan)

    # Need enough observations for the filter
    min_obs = 11
    if n < min_obs:
        return gap

    # Use symmetric HP filter for full sample, then replace last 3 with one-sided
    trend = _hp_filter(credit_to_gdp, lam_annual)
    gap = credit_to_gdp - trend

    # Replace last 3 points with one-sided estimate for better real-time accuracy
    if n >= min_obs:
        one_sided = _one_sided_hp_filter(credit_to_gdp, lam_annual)
        gap[-3:] = credit_to_gdp[-3:] - one_sided[-3:]

    return gap


def compute_debt_snowball(
    debt_to_gdp: np.ndarray,
    nominal_gdp_growth: np.ndarray,
    interest_rate: np.ndarray,
    primary_balance_pct_gdp: np.ndarray,
) -> np.ndarray:
    """Compute the debt snowball effect.

    The change in debt/GDP = (r - g) * d + pb
    where r = interest rate, g = GDP growth, d = debt/GDP, pb = primary balance
    """
    n = len(debt_to_gdp)
    snowball = np.full(n, np.nan)

    for t in range(1, n):
        if not any(np.isnan([debt_to_gdp[t], nominal_gdp_growth[t],
                             interest_rate[t], primary_balance_pct_gdp[t]])):
            r_g_spread = (interest_rate[t] - nominal_gdp_growth[t]) / 100.0
            mechanical = r_g_spread * debt_to_gdp[t - 1]
            snowball[t] = mechanical + primary_balance_pct_gdp[t]

    return snowball


def compute_fiscal_space_indicator(
    debt_to_gdp: np.ndarray,
    primary_balance_pct_gdp: np.ndarray,
    nominal_gdp_growth: np.ndarray,
    interest_rate: np.ndarray,
    debt_limit: float = 90.0,
) -> np.ndarray:
    """Compute fiscal space: distance from assumed fiscal limit.

    Positive = space remaining; negative = exceeded limit
    Adjusts for growth-debt differential and primary balance trajectory.
    """
    n = len(debt_to_gdp)
    fiscal_space = np.full(n, np.nan)

    for t in range(n):
        if not any(np.isnan([debt_to_gdp[t], primary_balance_pct_gdp[t],
                             nominal_gdp_growth[t], interest_rate[t]])):
            space = debt_limit - debt_to_gdp[t]
            # Adjust for structural position (rates in pct points)
            r_g = (interest_rate[t] - nominal_gdp_growth[t]) / 100.0
            pb_adj = primary_balance_pct_gdp[t] - r_g * debt_to_gdp[t]
            fiscal_space[t] = space + pb_adj * 3  # 3-year horizon adjustment

    return fiscal_space


def compute_growth_debt_differential(
    nominal_gdp_growth: np.ndarray,
    effective_interest_rate: np.ndarray,
) -> np.ndarray:
    """Compute g - r: the fundamental debt rollover condition.

    Positive = debt naturally stabilizes (growth exceeds cost of debt)
    Negative = snowball effect worsens debt dynamics
    """
    return nominal_gdp_growth - effective_interest_rate


def classify_debt_trajectory(
    debt_to_gdp: np.ndarray,
    window: int = 10,
) -> List[str]:
    """Classify debt trajectory as rising, peak, declining, or plateau."""
    n = len(debt_to_gdp)
    classifications = ["insufficient_data"] * n

    for t in range(window, n):
        window_data = debt_to_gdp[t - window:t + 1]

        # Check for NaN
        if np.any(np.isnan(window_data)):
            continue

        # Linear trend over window
        x = np.arange(len(window_data))
        slope = np.polyfit(x, window_data, 1)[0]

        recent_avg = np.mean(window_data[-3:])
        earlier_avg = np.mean(window_data[:3])
        max_val = np.max(window_data)

        # Peak: was rising but now slowing/flat
        if abs(slope) < 0.5 and recent_avg > debt_to_gdp[t - window]:
            classifications[t] = "plateau"
        # Peak: near historical max and slope turning negative
        elif abs(slope) < 0.3 and recent_avg > max_val * 0.95:
            classifications[t] = "peak"
        elif slope > 1.0:
            classifications[t] = "rising"
        elif slope < -1.0:
            classifications[t] = "declining"
        else:
            classifications[t] = "plateau"

    return classifications


def classify_cycle_phase(
    bis_gap: float,
    debt_to_gdp_change_5y: float,
    cds_spread: float,
    cds_threshold_em: float = 300.0,
    cds_threshold_dm: float = 100.0,
    is_em: bool = True,
) -> str:
    """Classify the current phase of the debt cycle.

    Phases:
    - expansion: credit gap rising, debt accumulating
    - peak: gap above threshold, market stress emerging
    - contraction: gap falling, deleveraging underway
    - trough: gap deeply negative, stabilizing
    """
    cds_thresh = cds_threshold_em if is_em else cds_threshold_dm

    if bis_gap > 10:
        if cds_spread > cds_thresh:
            return "peak_high_stress"
        return "peak"
    elif 5 < bis_gap <= 10:
        return "peak"
    elif -5 <= bis_gap <= 5:
        if debt_to_gdp_change_5y > 5:
            return "expansion"
        elif debt_to_gdp_change_5y < -5:
            return "contraction"
        return "plateau"
    elif -10 <= bis_gap < -5:
        return "contraction"
    else:
        return "trough"


def compute_zscore_cross_section(
    value: float,
    cross_section_values: np.ndarray,
    winsorize_bounds: Tuple[float, float] = (-3.0, 3.0),
) -> float:
    """Compute cross-sectional z-score: standardize against peer group at a point in time."""
    clean = cross_section_values[~np.isnan(cross_section_values)]
    if len(clean) < 3:
        return 0.0

    mean = np.mean(clean)
    std = np.std(clean)
    if std < 1e-9:
        return 0.0

    z = (value - mean) / std
    z = max(winsorize_bounds[0], min(winsorize_bounds[1], z))
    return float(z)


def compute_zscore_historical(
    value: float,
    historical_values: np.ndarray,
    winsorize_bounds: Tuple[float, float] = (-3.0, 3.0),
) -> float:
    """Compute country-specific historical z-score."""
    clean = historical_values[~np.isnan(historical_values)]
    if len(clean) < 3:
        return 0.0

    mean = np.mean(clean)
    std = np.std(clean)
    if std < 1e-9:
        return 0.0

    z = (value - mean) / std
    z = max(winsorize_bounds[0], min(winsorize_bounds[1], z))
    return float(z)


# ---------------------------------------------------------------------------
# Main Feature Engineering Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeatureEngineeringConfig:
    hp_lambda_annual: float = 400_000.0
    hp_lambda_quarterly: float = 1_600_000.0
    trajectory_window: int = 10
    winsorize_bounds: Tuple[float, float] = (-3.0, 3.0)


def engineer_features(panel_data: List[Dict], config: FeatureEngineeringConfig = None) -> List[Dict]:
    """Run full feature engineering pipeline on a country's panel data.

    Args:
        panel_data: List of dicts, each representing one (country, year) row
        config: Feature engineering parameters

    Returns:
        Panel with additional engineered features
    """
    if config is None:
        config = FeatureEngineeringConfig()

    if not panel_data:
        return []

    country = panel_data[0]["country"]
    n = len(panel_data)

    # Extract arrays
    def _get_array(key: str) -> np.ndarray:
        return np.array([row.get(key, np.nan) for row in panel_data])

    debt_gdp = _get_array("gov_gross_debt_pct_gdp")
    hh_debt = _get_array("hh_debt_pct_gdp")
    corp_debt = _get_array("corp_debt_pct_gdp")
    credit_gdp = _get_array("total_credit_pct_gdp")
    ca = _get_array("current_account_pct_gdp")
    cds = _get_array("cds_5y_bps")
    gdp_growth = _get_array("real_gdp_growth_pct")
    inflation = _get_array("inflation_pct")
    policy_rate = _get_array("policy_rate_pct")
    pb = _get_array("primary_balance_pct_gdp")
    npl = _get_array("corporate_npl_pct")
    ext_debt = _get_array("external_debt_pct_gdp")
    reserves = _get_array("fx_reserves_months_imports")
    inst_quality = _get_array("institutional_quality")
    rating = _get_array("credit_rating")

    # Derived arrays
    nominal_growth = gdp_growth + inflation
    effective_rate = policy_rate  # Approximation
    net_debt = debt_gdp * np.array([row.get("net_ratio", 0.7) for row in panel_data])

    # --- Feature 1: BIS Credit-to-GDP Gap ---
    bis_gap = compute_bis_credit_gap(credit_gdp, config.hp_lambda_annual)

    # --- Feature 2: Debt trajectory ---
    trajectory = classify_debt_trajectory(debt_gdp, config.trajectory_window)

    # --- Feature 3: Debt acceleration (5-year change) ---
    debt_accel_5y = np.full(n, np.nan)
    for t in range(5, n):
        if not (np.isnan(debt_gdp[t]) or np.isnan(debt_gdp[t - 5])):
            debt_accel_5y[t] = debt_gdp[t] - debt_gdp[t - 5]

    # --- Feature 4: Snowball indicator ---
    snowball = compute_debt_snowball(debt_gdp, nominal_growth, effective_rate, pb)

    # --- Feature 5: Growth-debt differential ---
    g_minus_r = compute_growth_debt_differential(nominal_growth, effective_rate)

    # --- Feature 6: Fiscal space ---
    fiscal_space = compute_fiscal_space_indicator(debt_gdp, pb, nominal_growth, effective_rate)

    # --- Feature 7: Cycle phase classification ---
    is_em = country in {"CHN", "IND", "BRA", "RUS", "MEX", "ZAF", "TUR", "IDN", "THA",
                         "POL", "COL", "ARG", "CHL", "PER", "EGY", "NGA", "VNM", "PAK",
                         "PHL", "HUN", "CZE", "ROU", "KAZ", "UKR", "MAR", "SAU", "ARE",
                         "QAT", "ISR", "DOM", "GTM", "ECU", "BOL", "PRY", "URY", "TZA",
                         "KEN", "ETH", "GHA"}
    phases = []
    for t in range(n):
        gap_val = bis_gap[t] if not np.isnan(bis_gap[t]) else 0
        d5y = debt_accel_5y[t] if not np.isnan(debt_accel_5y[t]) else 0
        cds_val = cds[t] if not np.isnan(cds[t]) else 0
        phase = classify_cycle_phase(gap_val, d5y, cds_val, is_em=is_em)
        phases.append(phase)

    # --- Feature 8: Total credit / GDP ---
    total_credit = hh_debt + corp_debt + debt_gdp

    # --- Feature 9: GFN proxy (Current account + short-term debt / reserves) ---
    st_debt_res = _get_array("st_ext_debt_pct_reserves")
    gfn_proxy = np.abs(ca) + st_debt_res

    # --- Feature 10: Currency mismatch index ---
    ext_debt_stock = _get_array("external_debt_pct_gdp")
    nii = _get_array("nii_pct_gdp")
    currency_mismatch = np.abs(ext_debt_stock - nii) if not np.all(np.isnan(ext_debt_stock - nii)) else np.full(n, np.nan)

    # --- Feature 11: Interest rate burden ---
    ir_burden = np.full(n, np.nan)
    for t in range(n):
        if not (np.isnan(debt_gdp[t]) or np.isnan(effective_rate[t])):
            ir_burden[t] = effective_rate[t] * debt_gdp[t] / 100.0

    # --- Feature 12: Credit concentration ratio ---
    hh_share = np.where(total_credit > 0.1, hh_debt / total_credit, np.nan)

    # --- Merge features back into panel ---
    feature_keys = [
        "bis_credit_gap", "debt_trajectory", "debt_acceleration_5y",
        "snowball_effect", "g_minus_r", "fiscal_space",
        "cycle_phase", "total_credit_pct_gdp", "gfn_proxy",
        "currency_mismatch_index", "interest_rate_burden_pct_gdp",
        "hh_debt_share_total_credit",
    ]

    for i, row in enumerate(panel_data):
        row["bis_credit_gap"] = round(float(bis_gap[i]), 2) if not np.isnan(bis_gap[i]) else None
        row["debt_trajectory"] = trajectory[i]
        row["debt_acceleration_5y"] = round(float(debt_accel_5y[i]), 2) if not np.isnan(debt_accel_5y[i]) else None
        row["snowball_effect"] = round(float(snowball[i]), 2) if not np.isnan(snowball[i]) else None
        row["g_minus_r"] = round(float(g_minus_r[i]), 2) if not np.isnan(g_minus_r[i]) else None
        row["fiscal_space"] = round(float(fiscal_space[i]), 2) if not np.isnan(fiscal_space[i]) else None
        row["cycle_phase"] = phases[i]
        row["total_credit_pct_gdp"] = round(float(total_credit[i]), 2) if not np.isnan(total_credit[i]) else None
        row["gfn_proxy"] = round(float(gfn_proxy[i]), 2) if not np.isnan(gfn_proxy[i]) else None
        row["currency_mismatch_index"] = round(float(currency_mismatch[i]), 2) if not np.isnan(currency_mismatch[i]) else None
        row["interest_rate_burden_pct_gdp"] = round(float(ir_burden[i]), 2) if not np.isnan(ir_burden[i]) else None
        row["hh_debt_share_total_credit"] = round(float(hh_share[i]), 3) if not np.isnan(hh_share[i]) else None
        row["net_debt_pct_gdp"] = round(float(net_debt[i]), 2) if not np.isnan(net_debt[i]) else None

    logger.info(f"Engineered {len(feature_keys)} features for {country}")
    return panel_data
