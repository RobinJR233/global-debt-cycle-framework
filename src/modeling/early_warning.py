"""
Early Warning Signal Engine
============================
Implements the Kaminsky-Lizondo-Reinhart signal approach for
early detection of debt crisis conditions.
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("debt_framework")


@dataclass
class SignalConfig:
    """Configuration for signal thresholds."""
    bis_credit_gap_em: float = 10.0
    bis_credit_gap_dm: float = 8.0
    current_account_deficit_dm: float = 4.0
    current_account_deficit_em: float = 5.0
    sovereign_cds_em: float = 300.0
    sovereign_cds_dm: float = 100.0
    gov_debt_gdp: float = 90.0
    st_debt_reserves_ratio: float = 100.0
    reer_appreciation_3y: float = 15.0
    fiscal_deficit_dm: float = 5.0
    fiscal_deficit_em: float = 3.0
    npl_threshold: float = 5.0
    reserves_months_min: float = 3.0


SIGNAL_CONFIG = SignalConfig()


@dataclass
class SignalResult:
    """Result from signal evaluation for a single indicator."""
    indicator: str
    triggered: bool
    current_value: float
    threshold: float
    direction: str  # "above" or "below"
    severity: float  # 0-1, how far beyond threshold


def evaluate_signal(
    current_value: float,
    threshold: float,
    direction: str = "above",
) -> SignalResult:
    """Evaluate a single signal against its threshold."""
    triggered = False
    severity = 0.0

    if direction == "above" and current_value is not None:
        if current_value > threshold:
            triggered = True
            denom = max(abs(threshold * 0.5), 0.1)
            severity = min(1.0, (current_value - threshold) / denom)
    elif direction == "below" and current_value is not None:
        if current_value < threshold:
            triggered = True
            denom = max(abs(threshold * 0.5), 0.1)
            severity = min(1.0, (threshold - current_value) / denom)

    return SignalResult(
        indicator="",  # Set by caller
        triggered=triggered,
        current_value=current_value if current_value is not None else 0.0,
        threshold=threshold,
        direction=direction,
        severity=round(severity, 3),
    )


def run_early_warning_signals(
    row: Dict,
    is_em: bool = True,
    config: SignalConfig = None,
) -> Dict:
    """Run all early warning signals for a single country-year observation.

    Returns:
        Dict with signal count, triggered signals, and composite EWS score.
    """
    if config is None:
        config = SignalConfig()

    signals = []

    # Signal 1: BIS Credit-to-GDP Gap
    bis_gap = row.get("bis_credit_gap")
    if bis_gap is not None:
        thresh = config.bis_credit_gap_em if is_em else config.bis_credit_gap_dm
        sig = evaluate_signal(bis_gap, thresh, "above")
        sig.indicator = "bis_credit_gap"
        sig.threshold = thresh
        signals.append(sig)

    # Signal 2: Current Account Deficit (sustained)
    ca = row.get("current_account_pct_gdp")
    if ca is not None:
        thresh = config.current_account_deficit_em if is_em else config.current_account_deficit_dm
        sig = evaluate_signal(abs(ca) if ca < 0 else 0, thresh, "above")
        sig.indicator = "current_account_deficit"
        sig.threshold = thresh
        sig.current_value = abs(ca) if ca < 0 else 0
        signals.append(sig)

    # Signal 3: Sovereign CDS
    cds = row.get("cds_5y_bps")
    if cds is not None:
        thresh = config.sovereign_cds_em if is_em else config.sovereign_cds_dm
        sig = evaluate_signal(cds, thresh, "above")
        sig.indicator = "sovereign_cds"
        sig.threshold = thresh
        signals.append(sig)

    # Signal 4: Government Debt / GDP
    debt_gdp = row.get("gov_gross_debt_pct_gdp")
    if debt_gdp is not None:
        sig = evaluate_signal(debt_gdp, config.gov_debt_gdp, "above")
        sig.indicator = "gov_debt_to_gdp"
        sig.threshold = config.gov_debt_gdp
        signals.append(sig)

    # Signal 5: Short-term External Debt / Reserves
    st_res = row.get("st_ext_debt_pct_reserves")
    if st_res is not None:
        sig = evaluate_signal(st_res, config.st_debt_reserves_ratio, "above")
        sig.indicator = "st_ext_debt_reserves_ratio"
        sig.threshold = config.st_debt_reserves_ratio
        signals.append(sig)

    # Signal 6: FX Reserves (inverse signal — low reserves is bad)
    reserves = row.get("fx_reserves_months_imports")
    if reserves is not None:
        sig = evaluate_signal(reserves, config.reserves_months_min, "below")
        sig.indicator = "fx_reserves_coverage"
        sig.threshold = config.reserves_months_min
        signals.append(sig)

    # Signal 7: Corporate NPLs
    npl = row.get("corporate_npl_pct")
    if npl is not None:
        sig = evaluate_signal(npl, config.npl_threshold, "above")
        sig.indicator = "corporate_npls"
        sig.threshold = config.npl_threshold
        signals.append(sig)

    # Signal 8: Fiscal Deficit
    fiscal_bal = row.get("overall_fiscal_balance_pct_gdp")
    if fiscal_bal is not None:
        deficit = abs(fiscal_bal) if fiscal_bal < 0 else 0
        thresh = config.fiscal_deficit_em if is_em else config.fiscal_deficit_dm
        sig = evaluate_signal(deficit, thresh, "above")
        sig.indicator = "fiscal_deficit"
        sig.threshold = thresh
        sig.current_value = deficit
        signals.append(sig)

    # Signal 9: g - r differential (inverse — negative is bad)
    g_minus_r = row.get("g_minus_r")
    if g_minus_r is not None:
        sig = evaluate_signal(g_minus_r, 0.0, "below")
        sig.indicator = "g_minus_r_differential"
        sig.threshold = 0.0
        signals.append(sig)

    # Signal 10: Interest rate burden
    ir_burden = row.get("interest_rate_burden_pct_gdp")
    if ir_burden is not None:
        sig = evaluate_signal(ir_burden, 5.0, "above")
        sig.indicator = "interest_rate_burden"
        sig.threshold = 5.0
        signals.append(sig)

    # Composite
    triggered_count = sum(1 for s in signals if s.triggered)
    total_signals = len(signals)
    avg_severity = np.mean([s.severity for s in signals if s.triggered]) if triggered_count > 0 else 0.0

    # EWS score: 0-100 (100 = all signals triggered at max severity)
    ews_score = round((triggered_count / total_signals) * 60 + avg_severity * 40, 1) if total_signals > 0 else 0.0

    return {
        "ews_signal_count": triggered_count,
        "ews_total_signals": total_signals,
        "ews_score": ews_score,
        "ews_severity_avg": round(float(avg_severity), 3),
        "ews_triggered_signals": [s.indicator for s in signals if s.triggered],
        "ews_signal_details": [
            {
                "indicator": s.indicator,
                "triggered": s.triggered,
                "current_value": s.current_value,
                "threshold": s.threshold,
                "direction": s.direction,
                "severity": s.severity,
            }
            for s in signals
        ],
    }
