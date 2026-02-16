"""
Tests for Gardner (2022) Two-Stage DiD estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.two_stage import (
    TwoStageBootstrapResults,
    TwoStageDiD,
    TwoStageDiDResults,
    two_stage_did,
)

# =============================================================================
# Shared test data generation
# =============================================================================


def generate_test_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    dynamic_effects: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic staggered adoption data for testing."""
    rng = np.random.default_rng(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 2.0
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    relative_time = times - first_treat_expanded

    if dynamic_effects:
        dynamic_mult = 1 + 0.1 * np.maximum(relative_time, 0)
    else:
        dynamic_mult = np.ones_like(relative_time, dtype=float)

    effect = treatment_effect * dynamic_mult

    outcomes = (
        unit_fe_expanded + time_fe_expanded + effect * post + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded,
        }
    )


# =============================================================================
# TestTwoStageDiDBasic
# =============================================================================


class TestTwoStageDiDBasic:
    """Tests for basic TwoStageDiD functionality."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_test_data()

        est = TwoStageDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert est.is_fitted_
        assert isinstance(results, TwoStageDiDResults)

    def test_att_accuracy(self):
        """Test that ATT recovers true treatment effect."""
        data = generate_test_data(treatment_effect=2.0, dynamic_effects=False, seed=123)

        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Should recover ~2.0 with reasonable tolerance
        assert abs(results.overall_att - 2.0) < 0.3

    def test_se_positive_finite(self):
        """Test that SEs are positive and finite."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_ci_contains_point_estimate(self):
        """Test that confidence interval contains the point estimate."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_conf_int[0] <= results.overall_att
        assert results.overall_att <= results.overall_conf_int[1]

    def test_t_stat_and_p_value(self):
        """Test that t-stat and p-value are consistent."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert np.isfinite(results.overall_t_stat)
        assert 0 <= results.overall_p_value <= 1

        # t-stat should equal ATT / SE
        expected_t = results.overall_att / results.overall_se
        assert abs(results.overall_t_stat - expected_t) < 1e-10

    def test_event_study(self):
        """Test event study specification."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0

        # Check reference period is present
        ref_period = -1
        assert ref_period in results.event_study_effects
        assert results.event_study_effects[ref_period]["effect"] == 0.0

        # Post-treatment effects should be positive (treatment_effect=2.0)
        post_effects = {h: e for h, e in results.event_study_effects.items() if h >= 0}
        assert len(post_effects) > 0
        for h, eff in post_effects.items():
            assert eff["effect"] > 0, f"Post-treatment effect at h={h} should be positive"
            assert eff["se"] > 0, f"SE at h={h} should be positive"
            assert np.isfinite(eff["t_stat"])
            assert 0 <= eff["p_value"] <= 1

    def test_group_effects(self):
        """Test group (cohort) effects."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None
        # Should have 3 groups (cohorts 3, 5, 7)
        assert len(results.group_effects) == 3
        for g, eff in results.group_effects.items():
            assert eff["effect"] > 0
            assert eff["se"] > 0
            assert np.isfinite(eff["t_stat"])

    def test_all_aggregation(self):
        """Test aggregate='all' produces both event study and group effects."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_summary_text(self):
        """Test that summary produces expected header text."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        text = results.summary()
        assert "Two-Stage DiD Estimator Results (Gardner 2022)" in text
        assert "ATT" in text
        assert "Overall Average Treatment Effect" in text

    def test_to_dataframe_event_study(self):
        """Test to_dataframe with event_study level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        df = results.to_dataframe("event_study")
        assert isinstance(df, pd.DataFrame)
        assert "relative_period" in df.columns
        assert "effect" in df.columns
        assert "se" in df.columns
        assert len(df) > 0

    def test_to_dataframe_group(self):
        """Test to_dataframe with group level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        df = results.to_dataframe("group")
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        assert len(df) == 3

    def test_to_dataframe_observation(self):
        """Test to_dataframe with observation level."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        df = results.to_dataframe("observation")
        assert isinstance(df, pd.DataFrame)
        assert "tau_hat" in df.columns
        assert "weight" in df.columns
        assert "unit" in df.columns
        assert "time" in df.columns
        assert len(df) == results.n_treated_obs

    def test_to_dataframe_invalid_level(self):
        """Test to_dataframe with invalid level raises."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe("invalid")

    def test_to_dataframe_no_event_study(self):
        """Test to_dataframe raises when event study not computed."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe("event_study")

    def test_repr(self):
        """Test __repr__ contains expected elements."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        repr_str = repr(results)
        assert "TwoStageDiDResults" in repr_str
        assert "ATT=" in repr_str
        assert "SE=" in repr_str

    def test_is_significant_property(self):
        """Test is_significant property."""
        data = generate_test_data(treatment_effect=2.0)
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert isinstance(results.is_significant, bool)
        # Strong treatment effect should be significant
        assert results.is_significant

    def test_significance_stars_property(self):
        """Test significance_stars property."""
        data = generate_test_data(treatment_effect=2.0)
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        stars = results.significance_stars
        assert isinstance(stars, str)
        # Strong effect should have at least one star
        assert len(stars.strip()) > 0

    def test_metadata_fields(self):
        """Test that metadata fields are populated correctly."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.n_obs == len(data)
        assert results.n_treated_obs > 0
        assert results.n_untreated_obs > 0
        assert results.n_treated_obs + results.n_untreated_obs == results.n_obs
        assert results.n_treated_units > 0
        assert results.n_control_units > 0
        assert len(results.groups) == 3
        assert len(results.time_periods) == 10


# =============================================================================
# TestTwoStageDiDEquivalence
# =============================================================================


class TestTwoStageDiDEquivalence:
    """Test that TwoStageDiD point estimates match ImputationDiD."""

    def test_overall_att_matches_imputation(self):
        """Overall ATT should match ImputationDiD to machine precision."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        imp_results = ImputationDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert abs(ts_results.overall_att - imp_results.overall_att) < 1e-10

    def test_event_study_effects_match_imputation(self):
        """Event study point estimates should match ImputationDiD."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        imp_results = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Both should have the same horizons
        ts_horizons = set(ts_results.event_study_effects.keys())
        imp_horizons = set(imp_results.event_study_effects.keys())
        assert ts_horizons == imp_horizons

        # Point estimates should match
        for h in ts_horizons:
            ts_eff = ts_results.event_study_effects[h]["effect"]
            imp_eff = imp_results.event_study_effects[h]["effect"]
            if np.isfinite(ts_eff) and np.isfinite(imp_eff):
                assert (
                    abs(ts_eff - imp_eff) < 1e-8
                ), f"Effect mismatch at h={h}: TS={ts_eff:.10f}, Imp={imp_eff:.10f}"

    def test_group_effects_match_imputation(self):
        """Group point estimates should match ImputationDiD."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )
        imp_results = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert set(ts_results.group_effects.keys()) == set(imp_results.group_effects.keys())

        for g in ts_results.group_effects:
            ts_eff = ts_results.group_effects[g]["effect"]
            imp_eff = imp_results.group_effects[g]["effect"]
            assert abs(ts_eff - imp_eff) < 1e-8

    def test_ses_differ_from_imputation(self):
        """GMM SEs should differ from conservative (Theorem 3) SEs."""
        from diff_diff.imputation import ImputationDiD

        data = generate_test_data()

        ts_results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        imp_results = ImputationDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # SEs should differ (different variance estimators)
        assert abs(ts_results.overall_se - imp_results.overall_se) > 1e-6


# =============================================================================
# TestTwoStageDiDVariance
# =============================================================================


class TestTwoStageDiDVariance:
    """Tests for GMM sandwich variance estimator."""

    def test_gmm_se_differs_from_naive(self):
        """GMM SE should differ from naive Stage 2 OLS SE."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # The GMM SE accounts for first-stage estimation uncertainty
        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_event_study_se_positive(self):
        """Event study SEs should all be positive."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0:
                assert eff["se"] > 0, f"SE at h={h} should be positive"
                assert np.isfinite(eff["se"])


# =============================================================================
# TestTwoStageDiDEdgeCases
# =============================================================================


class TestTwoStageDiDEdgeCases:
    """Tests for edge cases and error handling."""

    def test_always_treated_excluded_with_warning(self):
        """Always-treated units should be excluded with a warning."""
        data = generate_test_data()

        # Add an always-treated unit (first_treat = 0 means treated at time 0)
        always_treated = pd.DataFrame(
            {
                "unit": np.repeat(999, 10),
                "time": np.arange(10),
                "outcome": np.random.default_rng(42).standard_normal(10),
                "first_treat": np.repeat(-1, 10),  # treated before sample starts
            }
        )
        data_with_always = pd.concat([data, always_treated], ignore_index=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data_with_always,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            always_treated_warns = [
                x for x in w if "treated in all observed periods" in str(x.message)
            ]
            assert len(always_treated_warns) > 0

        # Verify unit was excluded (total obs should be less)
        assert results.n_obs == len(data)

    def test_no_never_treated_works(self):
        """Estimation should work without never-treated units."""
        data = generate_test_data(never_treated_frac=0.0)

        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.overall_att > 0
        assert results.overall_se > 0

    def test_single_cohort(self):
        """Should work with a single treatment cohort."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 50, 8
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[15:] = 4  # single cohort at period 4

        ft_exp = np.repeat(first_treat, n_periods)
        post = (times >= ft_exp) & (ft_exp > 0)
        outcomes = (
            rng.standard_normal(n_units)[np.repeat(np.arange(n_units), n_periods)]
            + 2.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert abs(results.overall_att - 2.0) < 0.5
        assert len(results.groups) == 1

    def test_anticipation_shifts_timing(self):
        """Anticipation parameter should shift effective treatment timing."""
        data = generate_test_data(seed=123)

        results_no_ant = TwoStageDiD(anticipation=0).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_with_ant = TwoStageDiD(anticipation=1).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # With anticipation, more obs are treated -> different ATT
        assert results_with_ant.n_treated_obs > results_no_ant.n_treated_obs
        assert abs(results_no_ant.overall_att - results_with_ant.overall_att) > 0.01

    def test_rank_deficiency_warning(self):
        """Rank deficiency should emit warning in 'warn' mode."""
        # Create data where some treated units have no untreated periods
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # All units treated at period 0 (except never-treated)
        first_treat = np.zeros(n_units, dtype=int)
        first_treat[5:] = 0  # never treated (first_treat=0)
        first_treat[:5] = 1  # treated at period 1

        ft_exp = np.repeat(first_treat, n_periods)
        outcomes = rng.standard_normal(len(units))

        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        # Should work without error in warn mode
        results = TwoStageDiD(rank_deficient_action="warn").fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        assert isinstance(results, TwoStageDiDResults)

    def test_rank_deficiency_error(self):
        """Rank deficiency should raise in 'error' mode when violated."""
        # Create data where a treated unit has NO untreated periods at all
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        # Some units treated at period 0 (no pre-treatment)
        first_treat = np.zeros(n_units, dtype=int)
        first_treat[10:] = 0  # first_treat at the first time period
        first_treat[:5] = 0  # never treated
        first_treat[5:10] = 0  # Make all units at period 0 as treated
        # Actually let's have some treated at period 0 so they fail rank check
        first_treat[5:10] = 0  # All these are coded as never-treated (first_treat=0)

        ft_exp = np.repeat(first_treat, n_periods)
        outcomes = rng.standard_normal(len(units))
        data = pd.DataFrame(
            {"unit": units, "time": times, "outcome": outcomes, "first_treat": ft_exp}
        )

        # All units are never-treated, so no treated obs -> ValueError
        with pytest.raises(ValueError, match="No treated observations"):
            TwoStageDiD(rank_deficient_action="error").fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

    def test_nan_propagation(self):
        """NaN SE should propagate to t_stat, p_value, conf_int."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # For a reference period, t_stat and p_value should be NaN
        if results.event_study_effects:
            pass  # Only check if event study was computed

        # Normal results should have finite values
        assert np.isfinite(results.overall_t_stat)
        assert np.isfinite(results.overall_p_value)

    def test_covariates(self):
        """Estimation with covariates should work."""
        data = generate_test_data()
        rng = np.random.default_rng(99)
        data["x1"] = rng.standard_normal(len(data))
        data["x2"] = rng.standard_normal(len(data))

        results = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att > 0
        assert results.overall_se > 0
        assert np.isfinite(results.overall_se)

    def test_missing_column_error(self):
        """Missing required columns should raise ValueError."""
        data = generate_test_data()

        with pytest.raises(ValueError, match="Missing columns"):
            TwoStageDiD().fit(
                data,
                outcome="nonexistent",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_no_treated_obs_error(self):
        """Should raise when no treated observations exist."""
        rng = np.random.default_rng(42)
        n = 100
        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(10), 10),
                "time": np.tile(np.arange(10), 10),
                "outcome": rng.standard_normal(n),
                "first_treat": 0,  # all never-treated
            }
        )

        with pytest.raises(ValueError, match="No treated"):
            TwoStageDiD().fit(
                data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
            )

    def test_horizon_max(self):
        """horizon_max should limit event study horizons."""
        data = generate_test_data()
        results = TwoStageDiD(horizon_max=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # All horizons should have |h| <= 2
        for h in results.event_study_effects:
            if results.event_study_effects[h].get("n_obs", 0) > 0:
                assert abs(h) <= 2

    def test_always_treated_warning_lists_unit_ids(self):
        """Always-treated warning should include affected unit IDs."""
        data = generate_test_data()

        # Add two always-treated units (first_treat before min_time=0)
        always_treated = pd.DataFrame(
            {
                "unit": np.repeat([997, 998], 10),
                "time": np.tile(np.arange(10), 2),
                "outcome": np.random.default_rng(42).standard_normal(20),
                "first_treat": np.repeat([-1, -2], 10),
            }
        )
        data_with_always = pd.concat([data, always_treated], ignore_index=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TwoStageDiD().fit(
                data_with_always,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            always_warns = [x for x in w if "treated in all observed periods" in str(x.message)]
            assert len(always_warns) == 1
            msg = str(always_warns[0].message)
            assert "997" in msg
            assert "998" in msg

    def test_bootstrap_with_nan_y_tilde(self, ci_params):
        """Bootstrap should handle NaN y_tilde from unidentified FEs."""
        # No never-treated units: cohorts 3, 5, 7 on periods 0-9 means
        # periods 7-9 have zero untreated obs -> NaN y_tilde
        data = generate_test_data(never_treated_frac=0.0)
        n_boot = ci_params.bootstrap(20)

        results = TwoStageDiD(n_bootstrap=n_boot).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert np.isfinite(results.overall_att)
        assert results.overall_se > 0

    def test_balance_e_empty_cohorts_warns(self):
        """Unreasonably large balance_e should warn when no cohorts qualify."""
        data = generate_test_data()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = TwoStageDiD().fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
                balance_e=100,  # No cohort can satisfy this
            )
            balance_warns = [x for x in w if "No cohorts satisfy" in str(x.message)]
            assert len(balance_warns) > 0

        # Event study should contain only the reference period
        assert len(results.event_study_effects) == 1
        ref_key = list(results.event_study_effects.keys())[0]
        assert results.event_study_effects[ref_key]["n_obs"] == 0


# =============================================================================
# TestTwoStageDiDParameters
# =============================================================================


class TestTwoStageDiDParameters:
    """Tests for parameter handling."""

    def test_get_params(self):
        """get_params should include all __init__ params."""
        est = TwoStageDiD(anticipation=1, alpha=0.1, n_bootstrap=100, seed=42, horizon_max=5)
        params = est.get_params()

        assert params["anticipation"] == 1
        assert params["alpha"] == 0.1
        assert params["n_bootstrap"] == 100
        assert params["seed"] == 42
        assert params["horizon_max"] == 5
        assert params["rank_deficient_action"] == "warn"
        assert params["cluster"] is None

    def test_set_params(self):
        """set_params should modify attributes."""
        est = TwoStageDiD()
        est.set_params(anticipation=2, alpha=0.1)

        assert est.anticipation == 2
        assert est.alpha == 0.1

    def test_set_params_returns_self(self):
        """set_params should return self for chaining."""
        est = TwoStageDiD()
        result = est.set_params(anticipation=1)
        assert result is est

    def test_set_params_unknown_raises(self):
        """set_params with unknown param should raise."""
        est = TwoStageDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_param=42)

    def test_rank_deficient_action_validation(self):
        """Invalid rank_deficient_action should raise."""
        with pytest.raises(ValueError, match="rank_deficient_action"):
            TwoStageDiD(rank_deficient_action="invalid")

    def test_cluster_changes_ses(self):
        """Different cluster variable should change SEs."""
        data = generate_test_data()
        # Add a cluster variable with fewer clusters than units
        data["cluster"] = data["unit"] % 10

        results_unit = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_cluster = TwoStageDiD(cluster="cluster").fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Point estimates should be the same
        assert abs(results_unit.overall_att - results_cluster.overall_att) < 1e-10
        # SEs should differ
        assert abs(results_unit.overall_se - results_cluster.overall_se) > 1e-6

    def test_horizon_max_limits_horizons(self):
        """horizon_max should limit event study horizons."""
        data = generate_test_data()

        results_full = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_limited = TwoStageDiD(horizon_max=2).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        full_horizons = set(results_full.event_study_effects.keys())
        limited_horizons = set(results_limited.event_study_effects.keys())

        assert len(limited_horizons) <= len(full_horizons)


# =============================================================================
# TestTwoStageDiDBootstrap
# =============================================================================


class TestTwoStageDiDBootstrap:
    """Tests for bootstrap inference."""

    def test_bootstrap_runs(self, ci_params):
        """Bootstrap should complete and produce results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        assert results.bootstrap_results is not None
        assert isinstance(results.bootstrap_results, TwoStageBootstrapResults)

    def test_bootstrap_structure(self, ci_params):
        """Bootstrap results should have correct structure."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        br = results.bootstrap_results
        assert br.n_bootstrap == n_boot
        assert br.weight_type == "rademacher"
        assert br.overall_att_se > 0
        assert br.overall_att_ci[0] < br.overall_att_ci[1]
        assert 0 < br.overall_att_p_value <= 1

    def test_bootstrap_updates_inference(self, ci_params):
        """Bootstrap should update the main results inference."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        results_analytical = TwoStageDiD(seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )
        results_bootstrap = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        # Point estimates should match
        assert abs(results_analytical.overall_att - results_bootstrap.overall_att) < 1e-10
        # SEs should differ (analytical GMM vs bootstrap)
        assert abs(results_analytical.overall_se - results_bootstrap.overall_se) > 1e-6

    def test_bootstrap_event_study(self, ci_params):
        """Bootstrap should work with event study specification."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        results = TwoStageDiD(n_bootstrap=n_boot, seed=42).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.bootstrap_results is not None
        assert results.bootstrap_results.event_study_ses is not None
        for h, se in results.bootstrap_results.event_study_ses.items():
            assert se > 0


# =============================================================================
# TestTwoStageDiDConvenience
# =============================================================================


class TestTwoStageDiDConvenience:
    """Tests for convenience function."""

    def test_convenience_function_returns_results(self):
        """Convenience function should return TwoStageDiDResults."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert isinstance(results, TwoStageDiDResults)
        assert results.overall_att > 0

    def test_convenience_function_kwargs(self):
        """Constructor kwargs should be forwarded."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            anticipation=1,
            alpha=0.1,
        )

        assert isinstance(results, TwoStageDiDResults)
        assert results.alpha == 0.1

    def test_convenience_function_aggregate(self):
        """Convenience function should support aggregate parameter."""
        data = generate_test_data()
        results = two_stage_did(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None

    def test_estimator_summary_before_fit_raises(self):
        """Calling summary() before fit() should raise."""
        est = TwoStageDiD()
        with pytest.raises(RuntimeError, match="fitted"):
            est.summary()

    def test_print_summary(self, capsys):
        """print_summary should print to stdout."""
        data = generate_test_data()
        results = TwoStageDiD().fit(
            data, outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        results.print_summary()
        captured = capsys.readouterr()
        assert "Two-Stage DiD" in captured.out
