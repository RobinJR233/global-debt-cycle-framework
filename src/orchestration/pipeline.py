"""
Pipeline Orchestrator
=====================
Wires all pipeline stages together for end-to-end execution.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime

from src.scoring.composite_scorer import compute_composite_score
from src.feature_engineering.engineer import engineer_features, FeatureEngineeringConfig
from src.modeling.early_warning import run_early_warning_signals
from src.modeling.debt_sustainability import run_debt_sustainability_analysis
from src.modeling.external_vulnerability import assess_external_vulnerability
from src.data_ingestion.pipeline import fetch_country_data, get_country_group

logger = logging.getLogger("debt_framework")


def run_pipeline_for_country(
    country: str,
    start_year: int = 2000,
    end_year: int = 2026,
) -> Dict:
    """Run the complete pipeline for a single country.

    Stages:
    1. Data Ingestion → raw panel
    2. Feature Engineering → panel with derived features
    3. Early Warning Signals → EWS scores
    4. Debt Sustainability Analysis → projections
    5. External Vulnerability → external risk
    6. Composite Scoring → final scores + classification
    """
    logger.info("=" * 60)
    logger.info("Starting pipeline for %s", country)
    logger.info("=" * 60)

    # Stage 1: Data Ingestion
    logger.info("[Stage 1] Data Ingestion")
    raw_data = fetch_country_data(country, start_year, end_year)

    # Stage 2: Feature Engineering
    logger.info("[Stage 2] Feature Engineering")
    fe_config = FeatureEngineeringConfig()
    featured_data = engineer_features(raw_data["data"], fe_config)

    if not featured_data:
        return {
            "country": country,
            "group": raw_data["group"],
            "years": [],
            "latest_year": None,
            "dsm": {"sustainability_verdict": "unknown", "baseline_path": [], "severe_path": []},
            "external_vulnerability": {"overall_risk": "unknown"},
            "sub_scores": [],
            "composite_score": None,
            "phase": "unknown",
            "velocity": 0.0,
            "ews_score": 0.0,
        }

    # Stage 3: Early Warning Signals
    logger.info("[Stage 3] Early Warning Signals")
    is_em = raw_data["group"] == "EM"
    for row in featured_data:
        ews_result = run_early_warning_signals(row, is_em=is_em)
        row.update(ews_result)

    # Stage 4: Debt Sustainability Analysis (only for latest year)
    logger.info("[Stage 4] Debt Sustainability Analysis")
    latest_row = featured_data[-1]
    historical = featured_data[:-1] if len(featured_data) > 1 else featured_data
    dsm_result = run_debt_sustainability_analysis(country, latest_row, historical)

    # Stage 5: External Vulnerability
    logger.info("[Stage 5] External Vulnerability Assessment")
    ext_vuln = assess_external_vulnerability(latest_row)

    # Stage 6: Composite Scoring (year by year)
    logger.info("[Stage 6] Composite Scoring")
    scored_data = []
    for i, row in enumerate(featured_data):
        prev = featured_data[i - 1] if i > 0 else None
        result = compute_composite_score(row, prev)
        scored_data.append({
            "year": row["year"],
            "country": country,
            "composite_score": result.composite_score,
            "phase": result.phase,
            "sub_scores": [
                {
                    "dimension": ss.dimension,
                    "score": ss.score,
                    "weight": ss.weight,
                    "weighted_score": ss.weighted_score,
                    "indicators": ss.raw_indicators,
                }
                for ss in result.sub_scores
            ],
            "velocity": result.velocity,
            "ews_score": result.ews_score,
            "cycle_phase": row.get("cycle_phase", "unknown"),
            "bis_credit_gap": row.get("bis_credit_gap"),
            "debt_trajectory": row.get("debt_trajectory"),
            "snowball_effect": row.get("snowball_effect"),
            "g_minus_r": row.get("g_minus_r"),
            "fiscal_space": row.get("fiscal_space"),
            "gov_gross_debt_pct_gdp": row.get("gov_gross_debt_pct_gdp"),
            "hh_debt_pct_gdp": row.get("hh_debt_pct_gdp"),
            "corp_debt_pct_gdp": row.get("corp_debt_pct_gdp"),
            "external_debt_pct_gdp": row.get("external_debt_pct_gdp"),
            "fx_reserves_months_imports": row.get("fx_reserves_months_imports"),
            "current_account_pct_gdp": row.get("current_account_pct_gdp"),
            "cds_5y_bps": row.get("cds_5y_bps"),
            "credit_rating": row.get("credit_rating"),
            "real_gdp_growth_pct": row.get("real_gdp_growth_pct"),
            "inflation_pct": row.get("inflation_pct"),
            "policy_rate_pct": row.get("policy_rate_pct"),
            "primary_balance_pct_gdp": row.get("primary_balance_pct_gdp"),
            "institutional_quality": row.get("institutional_quality"),
            "commodity_dependence_pct_exports": row.get("commodity_dependence_pct_exports"),
        })

    logger.info(f"Pipeline complete for {country}")

    return {
        "country": country,
        "group": raw_data["group"],
        "years": scored_data,
        "latest_year": scored_data[-1] if scored_data else None,
        "dsm": {
            "baseline_path": dsm_result.baseline_path,
            "adverse_path": dsm_result.adverse_path,
            "severe_path": dsm_result.severe_path,
            "fan_chart_low": dsm_result.fan_chart_low,
            "fan_chart_high": dsm_result.fan_chart_high,
            "required_pb_adjustment": dsm_result.required_pb_adjustment,
            "sustainability_verdict": dsm_result.sustainability_verdict,
            "baseline_final_debt_gdp": dsm_result.baseline_final_debt_gdp,
            "adverse_final_debt_gdp": dsm_result.adverse_final_debt_gdp,
            "severe_final_debt_gdp": dsm_result.severe_final_debt_gdp,
        },
        "external_vulnerability": {
            "external_debt_pct_gdp": ext_vuln.external_debt_pct_gdp,
            "sustainability": ext_vuln.ext_debt_sustainability,
            "gfn_reserves_ratio": ext_vuln.gfn_reserves_ratio,
            "currency_mismatch_index": ext_vuln.currency_mismatch_index,
            "reserves_coverage_months": ext_vuln.reserves_coverage_months,
            "nii_pct_gdp": ext_vuln.nii_pct_gdp,
            "nii_vs_ext_debt_gap": ext_vuln.nii_vs_ext_debt_gap,
            "stress_30pct_depreciation_impact": ext_vuln.stress_30pct_depreciation_impact,
            "overall_risk": ext_vuln.overall_risk,
        },
        "completed_at": datetime.utcnow().isoformat(),
    }


def run_pipeline_batch(
    countries: List[str],
    start_year: int = 2000,
    end_year: int = 2026,
) -> List[Dict]:
    """Run pipeline for multiple countries."""
    results = []
    for country in countries:
        try:
            result = run_pipeline_for_country(country, start_year, end_year)
            results.append(result)
        except Exception as e:
            logger.error(f"Pipeline failed for {country}: {e}")
    return results


def get_global_summary(results: List[Dict]) -> Dict:
    """Generate a summary from batch pipeline results."""
    if not results:
        return {}

    latest_scores = [(r["country"], r["latest_year"]["composite_score"],
                      r["latest_year"]["phase"]) for r in results if r.get("latest_year")]

    latest_scores.sort(key=lambda x: x[1], reverse=True)

    # Bottom 5 = lowest scores (most vulnerable), worst-first order.
    # When total <= 10 there will be overlap with top 5 — unavoidable with few countries.
    bottom_5 = list(reversed(latest_scores[-5:]))

    return {
        "total_countries": len(results),
        "top_5_resilient": latest_scores[:5],
        "bottom_5_vulnerable": bottom_5,
        "phase_distribution": _count_phases(latest_scores),
        "avg_composite_score": round(
            sum(s[1] for s in latest_scores) / len(latest_scores), 1
        ),
    }


def _count_phases(scores: List[Tuple]) -> Dict[str, int]:
    counts = {"green": 0, "blue": 0, "yellow": 0, "orange": 0, "red": 0}
    for _, _, phase in scores:
        counts[phase] = counts.get(phase, 0) + 1
    return counts
