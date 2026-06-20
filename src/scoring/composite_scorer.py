"""
Composite Scoring Engine
========================
Aggregates all indicators into a unified 0-100 debt cycle score
with five-color phase classification.
"""

import logging
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from src.config.settings import SCORING_WEIGHTS, PHASE_CONFIG

logger = logging.getLogger("debt_framework")


# Pre-built rating map (created once at import time)
_RATING_SCORES: Dict[str, int] = {
    "AAA": 95, "AA+": 90, "AA": 85, "AA-": 80,
    "A+": 75, "A": 70, "A-": 65,
    "BBB+": 60, "BBB": 55, "BBB-": 50,
    "BB+": 40, "BB": 35, "BB-": 30,
    "B+": 20, "B": 15, "B-": 10,
    "CCC+": 5, "CCC": 3, "CCC-": 2, "CC": 1, "C": 0, "D": 0,
}


@dataclass
class SubScoreResult:
    """Result for a single scoring dimension."""
    dimension: str
    score: float           # 0-100 (100 = most resilient)
    raw_indicators: Dict
    weight: float
    weighted_score: float


@dataclass
class CompositeScoreResult:
    """Full composite scoring result."""
    country: str
    year: int
    composite_score: float      # 0-100
    phase: str                  # green/blue/yellow/orange/red
    sub_scores: List[SubScoreResult]
    velocity: float             # Change from previous year
    peer_percentile: float      # Within peer group
    ews_score: float


# ---------------------------------------------------------------------------
# Dimension scoring functions
# ---------------------------------------------------------------------------

def _score_fiscal(row: Dict) -> SubScoreResult:
    """Score the fiscal dimension (weight 25%)."""
    indicators = {}

    # Debt/GDP (lower = better, inverted scoring)
    debt_gdp = row.get("gov_gross_debt_pct_gdp", 70)
    if debt_gdp is not None:
        # 0-100 scale: 0% debt = 100 pts, 300% debt = 0 pts
        debt_score = max(0, min(100, 100 - (debt_gdp / 300) * 100))
        indicators["debt_to_gdp"] = {"value": debt_gdp, "score": round(debt_score, 1)}
    else:
        debt_score = 50
        indicators["debt_to_gdp"] = {"value": None, "score": 50}

    # Primary balance (higher surplus = better)
    pb = row.get("primary_balance_pct_gdp", -3)
    if pb is not None:
        # -10% = 0 pts, +5% = 100 pts
        pb_score = max(0, min(100, (pb + 10) / 15 * 100))
        indicators["primary_balance"] = {"value": pb, "score": round(pb_score, 1)}
    else:
        pb_score = 50
        indicators["primary_balance"] = {"value": None, "score": 50}

    # Debt service ratio
    dsr = row.get("debt_service_ratio_pct_revenue", 15)
    if dsr is not None:
        # 5% = 100 pts, 30% = 0 pts
        dsr_score = max(0, min(100, 100 - (dsr - 5) / 25 * 100))
        indicators["debt_service_ratio"] = {"value": dsr, "score": round(dsr_score, 1)}
    else:
        dsr_score = 50
        indicators["debt_service_ratio"] = {"value": None, "score": 50}

    # Fiscal space
    fs = row.get("fiscal_space", 20)
    if fs is not None:
        # Map fiscal space to score
        fs_score = max(0, min(100, 50 + fs * 2))
        indicators["fiscal_space"] = {"value": fs, "score": round(fs_score, 1)}
    else:
        fs_score = 50
        indicators["fiscal_space"] = {"value": None, "score": 50}

    # Composite fiscal score (weighted average)
    fiscal_score = np.mean([debt_score, pb_score, dsr_score, fs_score])

    return SubScoreResult(
        dimension="fiscal",
        score=round(fiscal_score, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.fiscal,
        weighted_score=round(fiscal_score * SCORING_WEIGHTS.fiscal, 1),
    )


def _score_private_sector(row: Dict) -> SubScoreResult:
    """Score the private sector dimension (weight 20%)."""
    indicators = {}

    # BIS Credit-to-GDP Gap (lower gap = better)
    gap = row.get("bis_credit_gap")
    if gap is not None:
        # -20 = 100 pts, +20 = 0 pts
        gap_score = max(0, min(100, 100 - (gap + 20) / 40 * 100))
        indicators["bis_credit_gap"] = {"value": gap, "score": round(gap_score, 1)}
    else:
        gap_score = 50
        indicators["bis_credit_gap"] = {"value": None, "score": 50}

    # Household DSR
    hh_dsr = row.get("hh_dsr_pct_income", 12)
    if hh_dsr is not None:
        # 5% = 100 pts, 25% = 0 pts
        dsr_score = max(0, min(100, 100 - (hh_dsr - 5) / 20 * 100))
        indicators["hh_debt_service_ratio"] = {"value": hh_dsr, "score": round(dsr_score, 1)}
    else:
        dsr_score = 50
        indicators["hh_debt_service_ratio"] = {"value": None, "score": 50}

    # Corporate NPLs (lower = better)
    npl = row.get("corporate_npl_pct", 2)
    if npl is not None:
        npl_score = max(0, min(100, 100 - npl / 10 * 100))
        indicators["corporate_npls"] = {"value": npl, "score": round(npl_score, 1)}
    else:
        npl_score = 50
        indicators["corporate_npls"] = {"value": None, "score": 50}

    # Household debt / GDP
    hh_debt = row.get("hh_debt_pct_gdp", 40)
    if hh_debt is not None:
        hh_score = max(0, min(100, 100 - hh_debt / 100 * 100))
        indicators["hh_debt_to_gdp"] = {"value": hh_debt, "score": round(hh_score, 1)}
    else:
        hh_score = 50
        indicators["hh_debt_to_gdp"] = {"value": None, "score": 50}

    private_score = np.mean([gap_score, dsr_score, npl_score, hh_score])

    return SubScoreResult(
        dimension="private_sector",
        score=round(private_score, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.private_sector,
        weighted_score=round(private_score * SCORING_WEIGHTS.private_sector, 1),
    )


def _score_external(row: Dict) -> SubScoreResult:
    """Score the external sector dimension (weight 20%)."""
    indicators = {}

    # External debt / GDP
    ext = row.get("external_debt_pct_gdp", 40)
    if ext is not None:
        ext_score = max(0, min(100, 100 - ext / 100 * 100))
        indicators["external_debt_to_gdp"] = {"value": ext, "score": round(ext_score, 1)}
    else:
        ext_score = 50
        indicators["external_debt_to_gdp"] = {"value": None, "score": 50}

    # Reserves coverage (higher = better)
    reserves = row.get("fx_reserves_months_imports", 6)
    if reserves is not None:
        # 2 months = 0 pts, 12 months = 100 pts
        res_score = max(0, min(100, (reserves - 2) / 10 * 100))
        indicators["fx_reserves_coverage"] = {"value": reserves, "score": round(res_score, 1)}
    else:
        res_score = 50
        indicators["fx_reserves_coverage"] = {"value": None, "score": 50}

    # Current account (surplus = better)
    ca = row.get("current_account_pct_gdp", 0)
    if ca is not None:
        # -10% CA = 0 pts, +10% CA = 100 pts
        ca_score = max(0, min(100, (ca + 10) / 20 * 100))
        indicators["current_account"] = {"value": ca, "score": round(ca_score, 1)}
    else:
        ca_score = 50
        indicators["current_account"] = {"value": None, "score": 50}

    # GFN / Reserves (lower = better)
    gfn = row.get("gfn_proxy", 50)
    if gfn is not None:
        gfn_score = max(0, min(100, 100 - gfn / 200 * 100))
        indicators["gfn_proxy"] = {"value": gfn, "score": round(gfn_score, 1)}
    else:
        gfn_score = 50
        indicators["gfn_proxy"] = {"value": None, "score": 50}

    ext_score_combined = np.mean([ext_score, res_score, ca_score, gfn_score])

    return SubScoreResult(
        dimension="external",
        score=round(ext_score_combined, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.external,
        weighted_score=round(ext_score_combined * SCORING_WEIGHTS.external, 1),
    )


def _score_market_pricing(row: Dict) -> SubScoreResult:
    """Score the market pricing dimension (weight 15%)."""
    indicators = {}

    # CDS spread (lower = better)
    cds = row.get("cds_5y_bps", 100)
    if cds is not None:
        # 0 bps = 100 pts, 1000 bps = 0 pts
        cds_score = max(0, min(100, 100 - cds / 1000 * 100))
        indicators["cds_5y"] = {"value": cds, "score": round(cds_score, 1)}
    else:
        cds_score = 50
        indicators["cds_5y"] = {"value": None, "score": 50}

    # Yield spread (lower = better)
    spread = row.get("yield_spread_vs_ust_bps", 100)
    if spread is not None:
        spread_score = max(0, min(100, 100 - abs(spread) / 500 * 100))
        indicators["yield_spread"] = {"value": spread, "score": round(spread_score, 1)}
    else:
        spread_score = 50
        indicators["yield_spread"] = {"value": None, "score": 50}

    # Credit rating
    rating = row.get("credit_rating", "BBB")
    rating_score = _RATING_SCORES.get(rating, 50)
    indicators["credit_rating"] = {"value": rating, "score": rating_score}

    # EWS score (already computed)
    ews = row.get("ews_score", 50)
    ews_score = max(0, min(100, 100 - ews))  # Invert: high EWS = low market health
    indicators["early_warning_signals"] = {"value": ews, "score": round(ews_score, 1)}

    market_score = np.mean([cds_score, spread_score, rating_score, ews_score])

    return SubScoreResult(
        dimension="market_pricing",
        score=round(market_score, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.market_pricing,
        weighted_score=round(market_score * SCORING_WEIGHTS.market_pricing, 1),
    )


def _score_macro_context(row: Dict) -> SubScoreResult:
    """Score the macro context dimension (weight 10%)."""
    indicators = {}

    # g - r differential
    g_minus_r = row.get("g_minus_r", 0)
    if g_minus_r is not None:
        # -5 = 0 pts, +5 = 100 pts
        gmr_score = max(0, min(100, (g_minus_r + 5) / 10 * 100))
        indicators["g_minus_r"] = {"value": g_minus_r, "score": round(gmr_score, 1)}
    else:
        gmr_score = 50
        indicators["g_minus_r"] = {"value": None, "score": 50}

    # Real interest rate burden
    ir_burden = row.get("interest_rate_burden_pct_gdp", 3)
    if ir_burden is not None:
        burden_score = max(0, min(100, 100 - ir_burden / 10 * 100))
        indicators["interest_rate_burden"] = {"value": ir_burden, "score": round(burden_score, 1)}
    else:
        burden_score = 50
        indicators["interest_rate_burden"] = {"value": None, "score": 50}

    # Output gap
    output_gap = row.get("output_gap_pct", 0)
    if output_gap is not None:
        gap_score = max(0, min(100, 50 + output_gap * 5))  # slight positive = good
        indicators["output_gap"] = {"value": output_gap, "score": round(gap_score, 1)}
    else:
        gap_score = 50
        indicators["output_gap"] = {"value": None, "score": 50}

    macro_score = np.mean([gmr_score, burden_score, gap_score])

    return SubScoreResult(
        dimension="macro_context",
        score=round(macro_score, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.macro_context,
        weighted_score=round(macro_score * SCORING_WEIGHTS.macro_context, 1),
    )


def _score_structural(row: Dict) -> SubScoreResult:
    """Score the structural resilience dimension (weight 10%)."""
    indicators = {}

    # Institutional quality (average of WGI sub-indices)
    inst = row.get("institutional_quality", 50)
    gov_eff = row.get("gov_effectiveness", inst)
    reg_qual = row.get("regulatory_quality", inst)
    rol = row.get("rule_of_law", inst)
    cc = row.get("corruption_control", inst)
    pol_stab = row.get("political_stability", inst)

    inst_score = max(0, min(100, inst))
    indicators["institutional_quality"] = {"value": round(inst, 1), "score": round(inst_score, 1)}

    # Financial development
    fin_dev = row.get("financial_development_index", 60)
    fin_score = max(0, min(100, fin_dev))
    indicators["financial_development"] = {"value": fin_dev, "score": round(fin_score, 1)}

    # Debt trajectory
    trajectory = row.get("debt_trajectory", "plateau")
    traj_scores = {"rising": 30, "peak": 25, "plateau": 50, "declining": 65}
    traj_score = traj_scores.get(trajectory, 50)
    indicators["debt_trajectory"] = {"value": trajectory, "score": traj_score}

    # Commodity dependence (lower = better for resilience)
    comm = row.get("commodity_dependence_pct_exports", 20)
    comm_score = max(0, min(100, 100 - comm / 100 * 80))
    indicators["commodity_dependence"] = {"value": comm, "score": round(comm_score, 1)}

    struct_score = np.mean([inst_score, fin_score, traj_score, comm_score])

    return SubScoreResult(
        dimension="structural_resilience",
        score=round(struct_score, 1),
        raw_indicators=indicators,
        weight=SCORING_WEIGHTS.structural_resilience,
        weighted_score=round(struct_score * SCORING_WEIGHTS.structural_resilience, 1),
    )


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

PHASE_MAP = {
    "sustainable": "green",
    "stable_low_debt": "green",
    "expansion": "blue",
    "peak": "yellow",
    "plateau": "yellow",
    "overextension": "yellow",
    "stress": "orange",
    "early_deleveraging": "orange",
    "crisis": "red",
    "deep_deleveraging": "red",
    "trough": "green",  # Bottoming out is healthy
    "contraction": "orange",  # Deleveraging can be painful
}

PHASE_COLORS = {
    "green": ("🟢", "Stable / Low Debt"),
    "blue": ("🔵", "Expansion Phase"),
    "yellow": ("🟡", "Overextension / Peak"),
    "orange": ("🟠", "Stress / Deleveraging"),
    "red": ("🔴", "Crisis / Deep Deleveraging"),
}


def classify_phase(composite_score: float) -> str:
    """Classify phase based on composite score."""
    if composite_score >= PHASE_CONFIG.green_stable_low_debt[0]:
        return "green"
    elif composite_score >= PHASE_CONFIG.blue_expansion[0]:
        return "blue"
    elif composite_score >= PHASE_CONFIG.yellow_overextension_peak[0]:
        return "yellow"
    elif composite_score >= PHASE_CONFIG.orange_stress_early_deleveraging[0]:
        return "orange"
    else:
        return "red"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

SCORING_FUNCTIONS = [
    _score_fiscal,
    _score_private_sector,
    _score_external,
    _score_market_pricing,
    _score_macro_context,
    _score_structural,
]


def compute_composite_score(
    row: Dict,
    previous_row: Dict = None,
) -> CompositeScoreResult:
    """Compute the full composite score for a country-year.

    Args:
        row: Current year's feature-engineered data
        previous_row: Previous year's data (for velocity calculation)

    Returns:
        CompositeScoreResult with all scoring details
    """
    country = row.get("country", "Unknown")
    year = row.get("year", 2024)

    # Run all dimension scorers
    sub_scores = [fn(row) for fn in SCORING_FUNCTIONS]

    # Composite score (weighted sum, capped at 0-100)
    composite = sum(ss.weighted_score for ss in sub_scores)
    composite = max(0, min(100, round(composite, 1)))

    # Phase classification
    phase = classify_phase(composite)

    # Velocity (change from previous year)
    velocity = 0.0
    if previous_row:
        prev_score = previous_row.get("composite_score")
        if prev_score is not None:
            velocity = round(composite - prev_score, 1)

    # EWS score
    ews = row.get("ews_score", 50)

    # Peer percentile (simplified — would use actual peer group in production)
    peer_pct = round(composite, 1)

    return CompositeScoreResult(
        country=country,
        year=year,
        composite_score=composite,
        phase=phase,
        sub_scores=sub_scores,
        velocity=velocity,
        peer_percentile=peer_pct,
        ews_score=ews,
    )
