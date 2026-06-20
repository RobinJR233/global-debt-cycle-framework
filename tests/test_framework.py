"""
Unit tests for the Global Debt Cycle Assessment Framework.
"""

import math
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestHPFilter(unittest.TestCase):
    def test_hp_filter_returns_same_length(self):
        from src.feature_engineering.engineer import _hp_filter
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        trend = _hp_filter(series, 400_000)
        self.assertEqual(len(trend), len(series))

    def test_hp_filter_short_series(self):
        from src.feature_engineering.engineer import _hp_filter
        series = np.array([1.0, 2.0, 3.0])
        trend = _hp_filter(series, 400_000)
        np.testing.assert_array_equal(trend, series)

    def test_hp_filter_trend_smoother_than_data(self):
        from src.feature_engineering.engineer import _hp_filter
        np.random.seed(42)
        series = np.cumsum(np.random.randn(30)) + 50
        trend = _hp_filter(series, 400_000)
        # Trend should have lower variance than original
        self.assertLess(np.var(trend), np.var(series))


class TestOneSidedHPFilter(unittest.TestCase):
    def test_one_sided_hp_filter_length(self):
        from src.feature_engineering.engineer import _one_sided_hp_filter
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trend = _one_sided_hp_filter(series, 400_000)
        self.assertEqual(len(trend), len(series))

    def test_one_sided_hp_filter_short_series(self):
        from src.feature_engineering.engineer import _one_sided_hp_filter
        series = np.array([1.0, 2.0, 3.0])
        trend = _one_sided_hp_filter(series, 400_000)
        np.testing.assert_array_equal(trend, series)


class TestBISCreditGap(unittest.TestCase):
    def test_credit_gap_output_length(self):
        from src.feature_engineering.engineer import compute_bis_credit_gap
        credit_to_gdp = np.array([100.0] * 20)
        gap = compute_bis_credit_gap(credit_to_gdp)
        self.assertEqual(len(gap), 20)

    def test_credit_gap_flat_series_zeros(self):
        from src.feature_engineering.engineer import compute_bis_credit_gap
        credit_to_gdp = np.full(20, 100.0)
        gap = compute_bis_credit_gap(credit_to_gdp)
        # Flat series should have gap near zero
        non_nan = gap[~np.isnan(gap)]
        self.assertTrue(len(non_nan) > 0)
        for val in non_nan:
            self.assertAlmostEqual(val, 0.0, delta=1.0)

    def test_credit_gap_rising_series_positive(self):
        from src.feature_engineering.engineer import compute_bis_credit_gap
        # Steadily rising credit should yield positive gap
        credit_to_gdp = np.linspace(80, 140, 20)
        gap = compute_bis_credit_gap(credit_to_gdp)
        non_nan = gap[~np.isnan(gap)]
        if len(non_nan) > 0:
            self.assertTrue(np.mean(non_nan) > 0)

    def test_credit_gap_short_series_all_nan(self):
        from src.feature_engineering.engineer import compute_bis_credit_gap
        credit_to_gdp = np.array([100.0, 101.0, 102.0])
        gap = compute_bis_credit_gap(credit_to_gdp)
        self.assertTrue(all(np.isnan(gap)))


class TestDebtSnowball(unittest.TestCase):
    def test_snowball_basic(self):
        from src.feature_engineering.engineer import compute_debt_snowball
        debt = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        growth = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        rate = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
        pb = np.array([-2.0, -2.0, -2.0, -2.0, -2.0])
        snowball = compute_debt_snowball(debt, growth, rate, pb)
        self.assertEqual(len(snowball), 5)
        # First element should be NaN (needs previous period)
        self.assertTrue(np.isnan(snowball[0]))

    def test_snowball_formula(self):
        from src.feature_engineering.engineer import compute_debt_snowball
        debt = np.array([100.0, 100.0, 100.0])
        growth = np.array([3.0, 3.0, 3.0])
        rate = np.array([4.0, 4.0, 4.0])
        pb = np.array([-2.0, -2.0, -2.0])
        snowball = compute_debt_snowball(debt, growth, rate, pb)
        # delta_d = (4-3)/100*100 + (-2) = 1-2 = -1
        self.assertAlmostEqual(snowball[1], -1.0, delta=0.1)
        self.assertTrue(np.isnan(snowball[0]))


class TestFiscalSpace(unittest.TestCase):
    def test_fiscal_space_basic(self):
        from src.feature_engineering.engineer import compute_fiscal_space_indicator
        debt = np.array([60.0, 70.0, 80.0])
        pb = np.array([-2.0, -3.0, -4.0])
        growth = np.array([3.0, 3.0, 3.0])
        rate = np.array([4.0, 4.0, 4.0])
        fs = compute_fiscal_space_indicator(debt, pb, growth, rate)
        self.assertEqual(len(fs), 3)
        # Higher debt = lower fiscal space
        self.assertLess(fs[2], fs[0])


class TestDebtTrajectory(unittest.TestCase):
    def test_trajectory_rising(self):
        from src.feature_engineering.engineer import classify_debt_trajectory
        debt = np.array([40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0])
        traj = classify_debt_trajectory(debt, window=5)
        self.assertIn(traj[-1], ["rising", "plateau"])

    def test_trajectory_insufficient_data(self):
        from src.feature_engineering.engineer import classify_debt_trajectory
        debt = np.array([50.0, 55.0, 60.0])
        traj = classify_debt_trajectory(debt, window=5)
        self.assertTrue(all(t == "insufficient_data" for t in traj))


class TestCyclePhase(unittest.TestCase):
    def test_peak_phase(self):
        from src.feature_engineering.engineer import classify_cycle_phase
        phase = classify_cycle_phase(bis_gap=15.0, debt_to_gdp_change_5y=10.0,
                                     cds_spread=350.0, is_em=True)
        self.assertEqual(phase, "peak_high_stress")

    def test_expansion_phase(self):
        from src.feature_engineering.engineer import classify_cycle_phase
        phase = classify_cycle_phase(bis_gap=7.0, debt_to_gdp_change_5y=8.0,
                                     cds_spread=50.0, is_em=True)
        self.assertEqual(phase, "peak")

    def test_plateau_phase(self):
        from src.feature_engineering.engineer import classify_cycle_phase
        phase = classify_cycle_phase(bis_gap=0.0, debt_to_gdp_change_5y=0.0,
                                     cds_spread=50.0, is_em=True)
        self.assertEqual(phase, "plateau")

    def test_trough_phase(self):
        from src.feature_engineering.engineer import classify_cycle_phase
        phase = classify_cycle_phase(bis_gap=-15.0, debt_to_gdp_change_5y=-5.0,
                                     cds_spread=50.0, is_em=True)
        self.assertEqual(phase, "trough")


class TestZScore(unittest.TestCase):
    def test_zscore_cross_section(self):
        from src.feature_engineering.engineer import compute_zscore_cross_section
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        z = compute_zscore_cross_section(30.0, values)
        # Mean=30, std≈14.14, z=(30-30)/14.14=0
        self.assertAlmostEqual(z, 0.0, delta=0.1)

    def test_zscore_insufficient_data(self):
        from src.feature_engineering.engineer import compute_zscore_cross_section
        values = np.array([10.0, 20.0])
        z = compute_zscore_cross_section(15.0, values)
        self.assertEqual(z, 0.0)


class TestEngineerFeatures(unittest.TestCase):
    def _make_panel(self, n_years=15):
        """Create a minimal panel for testing."""
        panel = []
        for i in range(n_years):
            panel.append({
                "country": "TEST",
                "year": 2000 + i,
                "gov_gross_debt_pct_gdp": 60.0 + i * 2,
                "hh_debt_pct_gdp": 40.0 + i,
                "corp_debt_pct_gdp": 50.0 + i * 0.5,
                "total_credit_pct_gdp": 150.0 + i * 3.5,
                "current_account_pct_gdp": -2.0,
                "external_debt_pct_gdp": 40.0,
                "ext_debt_service_pct_exports": 15.0,
                "st_ext_debt_pct_reserves": 40.0,
                "fx_reserves_months_imports": 5.0,
                "nii_pct_gdp": -5.0,
                "real_gdp_growth_pct": 2.5,
                "inflation_pct": 2.0,
                "policy_rate_pct": 3.5,
                "primary_balance_pct_gdp": -3.0,
                "corporate_npl_pct": 2.0,
                "institutional_quality": 60.0,
                "credit_rating": "BBB",
                "gov_effectiveness": 60.0,
                "regulatory_quality": 60.0,
                "rule_of_law": 60.0,
                "corruption_control": 60.0,
                "political_stability": 60.0,
                "voice_accountability": 60.0,
                "commodity_dependence_pct_exports": 20.0,
                "financial_development_index": 60.0,
                "old_age_dependency_pct": 20.0,
                "output_gap_pct": 0.0,
            })
        return panel

    def test_engineer_features_runs(self):
        from src.feature_engineering.engineer import engineer_features
        panel = self._make_panel()
        result = engineer_features(panel)
        self.assertEqual(len(result), 15)
        # Check that features were added
        self.assertIn("bis_credit_gap", result[0])
        self.assertIn("snowball_effect", result[0])
        self.assertIn("fiscal_space", result[0])
        self.assertIn("g_minus_r", result[0])

    def test_engineer_features_expected_keys(self):
        from src.feature_engineering.engineer import engineer_features
        panel = self._make_panel()
        result = engineer_features(panel)
        expected_keys = [
            "bis_credit_gap", "debt_trajectory", "debt_acceleration_5y",
            "snowball_effect", "g_minus_r", "fiscal_space",
            "cycle_phase", "total_credit_pct_gdp", "gfn_proxy",
            "currency_mismatch_index", "interest_rate_burden_pct_gdp",
            "hh_debt_share_total_credit", "net_debt_pct_gdp",
        ]
        for key in expected_keys:
            self.assertIn(key, result[-1], f"Missing feature: {key}")

    def test_engineer_features_empty_panel(self):
        from src.feature_engineering.engineer import engineer_features
        result = engineer_features([])
        self.assertEqual(result, [])

    def test_g_minus_r_computed_correctly(self):
        from src.feature_engineering.engineer import engineer_features
        panel = self._make_panel(n_years=1)
        # nominal growth = real + inflation = 2.5 + 2.0 = 4.5
        # effective rate = policy_rate = 3.5
        # g - r = 4.5 - 3.5 = 1.0
        result = engineer_features(panel)
        self.assertAlmostEqual(result[0]["g_minus_r"], 1.0, delta=0.5)


# ---------------------------------------------------------------------------
# Composite Scoring Tests
# ---------------------------------------------------------------------------

class TestCompositeScorer(unittest.TestCase):
    def _make_row(self, overrides=None):
        """Create a minimal row dict for scoring."""
        row = {
            "gov_gross_debt_pct_gdp": 80.0,
            "primary_balance_pct_gdp": -3.0,
            "debt_service_ratio_pct_revenue": 12.0,
            "fiscal_space": 10.0,
            "bis_credit_gap": 2.0,
            "hh_dsr_pct_income": 15.0,
            "corporate_npl_pct": 2.0,
            "hh_debt_pct_gdp": 50.0,
            "external_debt_pct_gdp": 40.0,
            "fx_reserves_months_imports": 6.0,
            "current_account_pct_gdp": -2.0,
            "gfn_proxy": 30.0,
            "cds_5y_bps": 50.0,
            "yield_spread_vs_ust_bps": 50.0,
            "credit_rating": "A+",
            "ews_score": 30.0,
            "g_minus_r": 1.0,
            "interest_rate_burden_pct_gdp": 3.0,
            "output_gap_pct": 1.0,
            "institutional_quality": 70.0,
            "financial_development_index": 70.0,
            "debt_trajectory": "plateau",
            "commodity_dependence_pct_exports": 15.0,
        }
        if overrides:
            row.update(overrides)
        return row

    def test_composite_score_range(self):
        from src.scoring.composite_scorer import compute_composite_score
        row = self._make_row()
        result = compute_composite_score(row)
        self.assertGreaterEqual(result.composite_score, 0.0)
        self.assertLessEqual(result.composite_score, 100.0)

    def test_composite_score_returns_all_fields(self):
        from src.scoring.composite_scorer import compute_composite_score
        row = self._make_row()
        result = compute_composite_score(row)
        self.assertEqual(result.country, "Unknown")
        self.assertEqual(result.year, 2024)
        self.assertIsInstance(result.sub_scores, list)
        self.assertEqual(len(result.sub_scores), 6)
        self.assertIsInstance(result.phase, str)
        self.assertIn(result.phase, ["green", "blue", "yellow", "orange", "red"])

    def test_phase_green_high_score(self):
        from src.scoring.composite_scorer import compute_composite_score, classify_phase
        self.assertEqual(classify_phase(85.0), "green")
        self.assertEqual(classify_phase(70.0), "green")

    def test_phase_blue(self):
        from src.scoring.composite_scorer import classify_phase
        self.assertEqual(classify_phase(65.0), "blue")

    def test_phase_yellow(self):
        from src.scoring.composite_scorer import classify_phase
        self.assertEqual(classify_phase(50.0), "yellow")

    def test_phase_orange(self):
        from src.scoring.composite_scorer import classify_phase
        self.assertEqual(classify_phase(30.0), "orange")

    def test_phase_red(self):
        from src.scoring.composite_scorer import classify_phase
        self.assertEqual(classify_phase(15.0), "red")

    def test_velocity_calculation(self):
        from src.scoring.composite_scorer import compute_composite_score
        prev = self._make_row()
        prev["composite_score"] = 50.0
        curr = self._make_row()
        result = compute_composite_score(curr, prev)
        self.assertIsInstance(result.velocity, float)

    def test_sub_scores_weights_sum_to_one(self):
        from src.scoring.composite_scorer import compute_composite_score
        row = self._make_row()
        result = compute_composite_score(row)
        total_weight = sum(ss.weight for ss in result.sub_scores)
        self.assertAlmostEqual(total_weight, 1.0, delta=0.01)

    def test_low_resilience_row(self):
        from src.scoring.composite_scorer import compute_composite_score
        row = self._make_row({
            "gov_gross_debt_pct_gdp": 200.0,
            "primary_balance_pct_gdp": -15.0,
            "debt_service_ratio_pct_revenue": 30.0,
            "fiscal_space": -40.0,
            "bis_credit_gap": 20.0,
            "hh_dsr_pct_income": 25.0,
            "corporate_npl_pct": 10.0,
            "hh_debt_pct_gdp": 100.0,
            "external_debt_pct_gdp": 100.0,
            "fx_reserves_months_imports": 1.0,
            "current_account_pct_gdp": -10.0,
            "gfn_proxy": 100.0,
            "cds_5y_bps": 1000.0,
            "yield_spread_vs_ust_bps": 300.0,
            "credit_rating": "C",
            "ews_score": 80.0,
            "g_minus_r": -8.0,
            "interest_rate_burden_pct_gdp": 10.0,
            "output_gap_pct": -5.0,
            "institutional_quality": 10.0,
            "financial_development_index": 10.0,
            "debt_trajectory": "rising",
            "commodity_dependence_pct_exports": 80.0,
        })
        result = compute_composite_score(row)
        self.assertLess(result.composite_score, 20.0)

    def test_high_resilience_row(self):
        from src.scoring.composite_scorer import compute_composite_score
        row = self._make_row({
            "gov_gross_debt_pct_gdp": 30.0,
            "primary_balance_pct_gdp": 3.0,
            "corporate_npl_pct": 0.5,
            "external_debt_pct_gdp": 15.0,
            "cds_5y_bps": 10.0,
            "g_minus_r": 4.0,
            "institutional_quality": 85.0,
        })
        result = compute_composite_score(row)
        self.assertGreater(result.composite_score, 60.0)


# ---------------------------------------------------------------------------
# Early Warning Signal Tests
# ---------------------------------------------------------------------------

class TestEarlyWarningSignals(unittest.TestCase):
    def _make_row(self, overrides=None):
        row = {
            "bis_credit_gap": 2.0,
            "current_account_pct_gdp": -2.0,
            "cds_5y_bps": 50.0,
            "gov_gross_debt_pct_gdp": 70.0,
            "st_ext_debt_pct_reserves": 40.0,
            "fx_reserves_months_imports": 6.0,
            "corporate_npl_pct": 2.0,
            "overall_fiscal_balance_pct_gdp": -3.0,
            "g_minus_r": 1.0,
            "interest_rate_burden_pct_gdp": 3.0,
        }
        if overrides:
            row.update(overrides)
        return row

    def test_ews_returns_all_fields(self):
        from src.modeling.early_warning import run_early_warning_signals
        row = self._make_row()
        result = run_early_warning_signals(row, is_em=True)
        self.assertIn("ews_score", result)
        self.assertIn("ews_signal_count", result)
        self.assertIn("ews_total_signals", result)
        self.assertIn("ews_triggered_signals", result)
        self.assertIn("ews_signal_details", result)

    def test_ews_score_range(self):
        from src.modeling.early_warning import run_early_warning_signals
        row = self._make_row()
        result = run_early_warning_signals(row, is_em=True)
        self.assertGreaterEqual(result["ews_score"], 0.0)
        self.assertLessEqual(result["ews_score"], 100.0)

    def test_ews_no_signals_triggered(self):
        from src.modeling.early_warning import run_early_warning_signals
        row = self._make_row({
            "bis_credit_gap": -5.0,
            "cds_5y_bps": 10.0,
            "gov_gross_debt_pct_gdp": 30.0,
            "st_ext_debt_pct_reserves": 20.0,
            "fx_reserves_months_imports": 12.0,
            "corporate_npl_pct": 0.5,
            "overall_fiscal_balance_pct_gdp": 2.0,
            "g_minus_r": 5.0,
            "interest_rate_burden_pct_gdp": 1.0,
        })
        result = run_early_warning_signals(row, is_em=True)
        self.assertEqual(result["ews_signal_count"], 0)
        self.assertEqual(result["ews_score"], 0.0)

    def test_ews_many_signals_triggered(self):
        from src.modeling.early_warning import run_early_warning_signals
        row = self._make_row({
            "bis_credit_gap": 15.0,
            "current_account_pct_gdp": -10.0,
            "cds_5y_bps": 500.0,
            "gov_gross_debt_pct_gdp": 120.0,
            "st_ext_debt_pct_reserves": 150.0,
            "fx_reserves_months_imports": 1.0,
            "corporate_npl_pct": 8.0,
            "overall_fiscal_balance_pct_gdp": -10.0,
            "g_minus_r": -4.0,
            "interest_rate_burden_pct_gdp": 8.0,
        })
        result = run_early_warning_signals(row, is_em=True)
        self.assertGreater(result["ews_signal_count"], 5)
        self.assertGreater(result["ews_score"], 30.0)

    def test_ews_dm_thresholds(self):
        from src.modeling.early_warning import run_early_warning_signals
        row = self._make_row()
        # Same values, DM thresholds should trigger fewer signals
        result_em = run_early_warning_signals(row, is_em=True)
        result_dm = run_early_warning_signals(row, is_em=False)
        # EMs have lower thresholds, so same values trigger more signals for EM
        self.assertGreaterEqual(result_em["ews_score"], result_dm["ews_score"])


# ---------------------------------------------------------------------------
# Debt Sustainability Model Tests
# ---------------------------------------------------------------------------

class TestDebtSustainability(unittest.TestCase):
    def _make_current_row(self):
        return {
            "country": "TEST",
            "group": "EM",
            "gov_gross_debt_pct_gdp": 70.0,
            "real_gdp_growth_pct": 3.0,
            "inflation_pct": 4.0,
            "policy_rate_pct": 6.0,
            "primary_balance_pct_gdp": -2.0,
        }

    def test_dsm_returns_all_fields(self):
        from src.modeling.debt_sustainability import run_debt_sustainability_analysis
        row = self._make_current_row()
        result = run_debt_sustainability_analysis("TEST", row, [])
        self.assertEqual(result.country, "TEST")
        self.assertEqual(len(result.baseline_path), 6)  # 5 years + initial
        self.assertEqual(len(result.adverse_path), 6)
        self.assertEqual(len(result.severe_path), 6)
        self.assertEqual(len(result.fan_chart_low), 6)
        self.assertEqual(len(result.fan_chart_high), 6)
        self.assertIsInstance(result.sustainability_verdict, str)

    def test_dsm_baseline_path_starts_at_initial(self):
        from src.modeling.debt_sustainability import run_debt_sustainability_analysis
        row = self._make_current_row()
        result = run_debt_sustainability_analysis("TEST", row, [])
        self.assertAlmostEqual(result.baseline_path[0], 70.0, delta=0.1)

    def test_dsm_severe_different_from_baseline(self):
        np.random.seed(42)
        from src.modeling.debt_sustainability import run_debt_sustainability_analysis
        row = self._make_current_row()
        result = run_debt_sustainability_analysis("TEST", row, [])
        # Severe adverse scenario should produce a different final debt than baseline
        self.assertNotAlmostEqual(result.severe_final_debt_gdp, result.baseline_final_debt_gdp, delta=1.0)

    def test_dsm_verdict_sustainable(self):
        from src.modeling.debt_sustainability import run_debt_sustainability_analysis
        row = {
            "country": "GOOD",
            "group": "DM",
            "gov_gross_debt_pct_gdp": 40.0,
            "real_gdp_growth_pct": 3.0,
            "inflation_pct": 2.0,
            "policy_rate_pct": 2.0,
            "primary_balance_pct_gdp": 1.0,
        }
        result = run_debt_sustainability_analysis("GOOD", row, [])
        self.assertIn(result.sustainability_verdict, ["sustainable", "at_risk"])

    def test_dsm_project_debt_snowball(self):
        from src.modeling.debt_sustainability import project_debt_snowball, ScenarioParams
        params = {
            "gdp_growth": [3.0, 3.0, 3.0],
            "interest_rate": [4.0, 4.0, 4.0],
            "primary_balance": [-2.0, -2.0, -2.0],
        }
        scenario = ScenarioParams(name="Test", gdp_growth_delta=0, interest_rate_delta=0,
                                  primary_balance_delta=0)
        path = project_debt_snowball(100.0, params, scenario, years=3)
        self.assertEqual(len(path), 4)  # initial + 3 years


# ---------------------------------------------------------------------------
# External Vulnerability Tests
# ---------------------------------------------------------------------------

class TestExternalVulnerability(unittest.TestCase):
    def _make_row(self, overrides=None):
        row = {
            "country": "TEST",
            "group": "EM",
            "external_debt_pct_gdp": 40.0,
            "fx_reserves_months_imports": 6.0,
            "current_account_pct_gdp": -2.0,
            "st_ext_debt_pct_reserves": 40.0,
            "nii_pct_gdp": -5.0,
            "gfn_proxy": 30.0,
        }
        if overrides:
            row.update(overrides)
        return row

    def test_external_vulnerability_returns_all_fields(self):
        from src.modeling.external_vulnerability import assess_external_vulnerability
        row = self._make_row()
        result = assess_external_vulnerability(row)
        self.assertEqual(result.country, "TEST")
        self.assertIsInstance(result.overall_risk, str)
        self.assertIn(result.overall_risk, ["low", "moderate", "high", "severe"])

    def test_high_external_debt_risky(self):
        from src.modeling.external_vulnerability import assess_external_vulnerability
        row = self._make_row({
            "external_debt_pct_gdp": 80.0,
            "st_ext_debt_pct_reserves": 150.0,
            "fx_reserves_months_imports": 1.0,
        })
        result = assess_external_vulnerability(row)
        self.assertIn(result.overall_risk, ["high", "severe"])

    def test_low_risk_scenario(self):
        from src.modeling.external_vulnerability import assess_external_vulnerability
        row = self._make_row({
            "external_debt_pct_gdp": 15.0,
            "st_ext_debt_pct_reserves": 20.0,
            "fx_reserves_months_imports": 12.0,
            "current_account_pct_gdp": 5.0,
        })
        result = assess_external_vulnerability(row)
        self.assertIn(result.overall_risk, ["low", "moderate"])

    def test_dm_threshold_higher(self):
        from src.modeling.external_vulnerability import assess_external_vulnerability
        row_em = self._make_row({"external_debt_pct_gdp": 50.0, "group": "EM"})
        row_dm = self._make_row({"external_debt_pct_gdp": 50.0, "group": "DM"})
        result_em = assess_external_vulnerability(row_em)
        result_dm = assess_external_vulnerability(row_dm)
        # 50% ext debt is at_risk for EM but only at_risk for DM too (threshold 60)
        self.assertEqual(result_em.ext_debt_sustainability, "at_risk")
        self.assertEqual(result_dm.ext_debt_sustainability, "sustainable")


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestDataIngestion(unittest.TestCase):
    def test_fetch_country_data(self):
        from src.data_ingestion.pipeline import fetch_country_data
        result = fetch_country_data("USA", 2000, 2024)
        self.assertEqual(result["country"], "USA")
        self.assertEqual(result["group"], "AE")
        self.assertGreater(len(result["data"]), 0)
        self.assertIn("sources", result)

    def test_fetch_em_country(self):
        from src.data_ingestion.pipeline import fetch_country_data
        result = fetch_country_data("CHN", 2000, 2024)
        self.assertEqual(result["group"], "EM")

    def test_fetch_unknown_country(self):
        from src.data_ingestion.pipeline import fetch_country_data
        result = fetch_country_data("XXX", 2000, 2024)
        self.assertEqual(result["country"], "XXX")
        self.assertEqual(result["group"], "other")

    def test_get_country_group(self):
        from src.data_ingestion.pipeline import get_country_group
        self.assertEqual(get_country_group("USA"), "AE")
        self.assertEqual(get_country_group("JPN"), "AE")
        self.assertEqual(get_country_group("CHN"), "EM")
        self.assertEqual(get_country_group("IND"), "EM")
        self.assertEqual(get_country_group("XXX"), "other")

    def test_registry_available_countries(self):
        from src.data_ingestion.registry import registry
        countries = registry.get_available_countries()
        self.assertIn("USA", countries)
        self.assertIn("CHN", countries)
        self.assertGreater(len(countries), 50)


class TestPipelineOrchestration(unittest.TestCase):
    def test_run_pipeline_country(self):
        from src.orchestration.pipeline import run_pipeline_for_country
        result = run_pipeline_for_country("USA", 2010, 2020)
        self.assertEqual(result["country"], "USA")
        self.assertIn("years", result)
        self.assertIn("latest_year", result)
        self.assertIn("dsm", result)
        self.assertIn("external_vulnerability", result)
        self.assertIsNotNone(result["latest_year"])

    def test_run_pipeline_em_country(self):
        from src.orchestration.pipeline import run_pipeline_for_country
        result = run_pipeline_for_country("CHN", 2010, 2020)
        self.assertEqual(result["country"], "CHN")
        self.assertIsNotNone(result["latest_year"])

    def test_run_pipeline_batch(self):
        from src.orchestration.pipeline import run_pipeline_batch
        results = run_pipeline_batch(["USA", "JPN", "DEU"], 2010, 2020)
        self.assertEqual(len(results), 3)
        countries = [r["country"] for r in results]
        self.assertIn("USA", countries)
        self.assertIn("JPN", countries)
        self.assertIn("DEU", countries)

    def test_get_global_summary(self):
        from src.orchestration.pipeline import run_pipeline_batch, get_global_summary
        results = run_pipeline_batch(["USA", "JPN", "DEU"], 2010, 2020)
        summary = get_global_summary(results)
        self.assertEqual(summary["total_countries"], 3)
        self.assertIn("avg_composite_score", summary)
        self.assertIn("top_5_resilient", summary)
        self.assertIn("bottom_5_vulnerable", summary)
        self.assertIn("phase_distribution", summary)

    def test_batch_with_many_countries(self):
        from src.orchestration.pipeline import run_pipeline_batch, get_global_summary
        countries = ["USA", "JPN", "DEU", "GBR", "CHN", "IND", "BRA", "FRA", "ITA", "CAN"]
        results = run_pipeline_batch(countries, 2010, 2020)
        self.assertEqual(len(results), 10)
        summary = get_global_summary(results)
        self.assertEqual(summary["total_countries"], 10)
        # With 10 countries, top 5 and bottom 5 should be different
        top_countries = {c for c, _, _ in summary["top_5_resilient"]}
        bottom_countries = {c for c, _, _ in summary["bottom_5_vulnerable"]}
        # They might overlap at rank 5/6 boundary, but shouldn't be identical
        self.assertFalse(top_countries == bottom_countries)

    def test_pipeline_latest_year_fields(self):
        from src.orchestration.pipeline import run_pipeline_for_country
        result = run_pipeline_for_country("USA", 2000, 2024)
        latest = result["latest_year"]
        expected_fields = [
            "composite_score", "phase", "sub_scores", "velocity",
            "ews_score", "gov_gross_debt_pct_gdp", "g_minus_r",
            "fiscal_space", "real_gdp_growth_pct", "inflation_pct",
        ]
        for field in expected_fields:
            self.assertIn(field, latest, f"Missing field: {field}")

    def test_dsm_fields(self):
        from src.orchestration.pipeline import run_pipeline_for_country
        result = run_pipeline_for_country("USA", 2000, 2024)
        dsm = result["dsm"]
        expected_fields = [
            "baseline_path", "adverse_path", "severe_path",
            "fan_chart_low", "fan_chart_high",
            "required_pb_adjustment", "sustainability_verdict",
            "baseline_final_debt_gdp", "adverse_final_debt_gdp",
            "severe_final_debt_gdp",
        ]
        for field in expected_fields:
            self.assertIn(field, dsm, f"Missing DSM field: {field}")

    def test_external_vulnerability_fields(self):
        from src.orchestration.pipeline import run_pipeline_for_country
        result = run_pipeline_for_country("USA", 2000, 2024)
        ext = result["external_vulnerability"]
        expected_fields = [
            "external_debt_pct_gdp", "sustainability",
            "gfn_reserves_ratio", "currency_mismatch_index",
            "reserves_coverage_months", "nii_pct_gdp",
            "nii_vs_ext_debt_gap", "stress_30pct_depreciation_impact",
            "overall_risk",
        ]
        for field in expected_fields:
            self.assertIn(field, ext, f"Missing external vulnerability field: {field}")


# ---------------------------------------------------------------------------
# Edge Cases & Regression Tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_all_nan_credit_gap(self):
        from src.feature_engineering.engineer import compute_bis_credit_gap
        series = np.full(20, np.nan)
        gap = compute_bis_credit_gap(series, 400_000)
        self.assertTrue(all(np.isnan(gap)))

    def test_snowball_with_nans(self):
        from src.feature_engineering.engineer import compute_debt_snowball
        debt = np.array([100.0, np.nan, 104.0])
        growth = np.array([3.0, 3.0, 3.0])
        rate = np.array([4.0, 4.0, 4.0])
        pb = np.array([-2.0, -2.0, -2.0])
        snowball = compute_debt_snowball(debt, growth, rate, pb)
        self.assertTrue(np.isnan(snowball[1]))

    def test_scoring_with_none_values(self):
        from src.scoring.composite_scorer import _score_fiscal
        row = {"gov_gross_debt_pct_gdp": None, "primary_balance_pct_gdp": None,
               "debt_service_ratio_pct_revenue": None, "fiscal_space": None}
        result = _score_fiscal(row)
        self.assertIsNotNone(result)
        self.assertEqual(result.score, 50.0)  # Default score for None

    def test_phase_colors_exist(self):
        from src.scoring.composite_scorer import PHASE_COLORS
        for phase in ["green", "blue", "yellow", "orange", "red"]:
            self.assertIn(phase, PHASE_COLORS)
            emoji, label = PHASE_COLORS[phase]
            self.assertIsInstance(emoji, str)
            self.assertIsInstance(label, str)


if __name__ == "__main__":
    unittest.main()
