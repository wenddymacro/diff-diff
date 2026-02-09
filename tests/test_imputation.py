"""
Tests for Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.imputation import (
    ImputationBootstrapResults,
    ImputationDiD,
    ImputationDiDResults,
    imputation_did,
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
# TestImputationDiD
# =============================================================================


class TestImputationDiD:
    """Tests for ImputationDiD estimator."""

    def test_basic_fit(self):
        """Test basic model fitting."""
        data = generate_test_data()

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert est.is_fitted_
        assert isinstance(results, ImputationDiDResults)
        assert results.overall_att is not None
        assert results.overall_se > 0
        assert results.n_treated_obs > 0
        assert results.n_untreated_obs > 0
        assert results.n_treated_units > 0
        assert results.n_control_units > 0
        assert len(results.groups) == 3
        assert len(results.time_periods) == 10

    def test_positive_treatment_effect(self):
        """Test recovery of positive treatment effect."""
        data = generate_test_data(treatment_effect=3.0, seed=123)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att > 0
        # Effect should be close to 3.0 (dynamic effects add some)
        assert abs(results.overall_att - 3.0) < 2 * results.overall_se + 1.5

    def test_zero_treatment_effect(self):
        """Test with no treatment effect."""
        data = generate_test_data(treatment_effect=0.0, seed=456)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert abs(results.overall_att) < 3 * results.overall_se + 0.5

    def test_aggregate_simple(self):
        """Test that default aggregate computes overall ATT."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert results.overall_se > 0
        assert results.event_study_effects is None
        assert results.group_effects is None

    def test_aggregate_event_study(self):
        """Test event study aggregation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        assert len(results.event_study_effects) > 0
        assert results.group_effects is None

        for h, eff in results.event_study_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert "t_stat" in eff
            assert "p_value" in eff
            assert "conf_int" in eff
            assert "n_obs" in eff

    def test_aggregate_group(self):
        """Test group aggregation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        assert results.group_effects is not None
        assert len(results.group_effects) == 3  # 3 cohorts
        assert results.event_study_effects is None

        for g, eff in results.group_effects.items():
            assert "effect" in eff
            assert "se" in eff
            assert eff["se"] > 0

    def test_aggregate_all(self):
        """Test 'all' aggregation computes both event study and group."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        assert results.event_study_effects is not None
        assert results.group_effects is not None

    def test_covariates(self):
        """Test estimation with covariates."""
        data = generate_test_data()
        rng = np.random.default_rng(99)
        data["x1"] = rng.standard_normal(len(data))
        data["x2"] = rng.standard_normal(len(data))

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        )

        assert results.overall_att is not None
        assert results.overall_se > 0

    def test_anticipation(self):
        """Test anticipation parameter."""
        data = generate_test_data()

        est0 = ImputationDiD(anticipation=0)
        results0 = est0.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        est1 = ImputationDiD(anticipation=1)
        results1 = est1.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # With anticipation=1, more obs are treated, fewer untreated
        assert results1.n_treated_obs > results0.n_treated_obs

        # Reference period changes
        ref0 = [h for h, e in results0.event_study_effects.items() if e.get("n_obs", 1) == 0]
        ref1 = [h for h, e in results1.event_study_effects.items() if e.get("n_obs", 1) == 0]
        assert -1 in ref0
        assert -2 in ref1

    def test_balance_e(self):
        """Test balance_e restricts event study to balanced cohorts."""
        data = generate_test_data()

        est = ImputationDiD()
        results_unbal = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_bal = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=2,
        )

        # Balanced should have same or fewer horizons
        assert len(results_bal.event_study_effects) <= len(results_unbal.event_study_effects) + 5

    def test_horizon_max(self):
        """Test horizon_max caps event study horizons."""
        data = generate_test_data()

        est = ImputationDiD(horizon_max=3)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            if results.event_study_effects[h].get("n_obs", 0) > 0:
                assert abs(h) <= 3

    def test_summary(self):
        """Test summary output."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )

        summary = results.summary()
        assert "Imputation DiD" in summary
        assert "ATT" in summary
        assert "Event Study" in summary
        assert "Group" in summary

    def test_to_dataframe_observation(self):
        """Test to_dataframe at observation level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        df = results.to_dataframe("observation")
        assert "tau_hat" in df.columns
        assert "weight" in df.columns
        assert len(df) == results.n_treated_obs

    def test_to_dataframe_event_study(self):
        """Test to_dataframe at event study level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        df = results.to_dataframe("event_study")
        assert "relative_period" in df.columns
        assert "effect" in df.columns
        assert "se" in df.columns

    def test_to_dataframe_group(self):
        """Test to_dataframe at group level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        df = results.to_dataframe("group")
        assert "group" in df.columns
        assert len(df) == 3

    def test_to_dataframe_errors(self):
        """Test to_dataframe raises on invalid level."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe("invalid")

        with pytest.raises(ValueError, match="Event study effects not computed"):
            results.to_dataframe("event_study")

    def test_get_params(self):
        """Test get_params returns all constructor parameters."""
        est = ImputationDiD(
            anticipation=1,
            alpha=0.10,
            n_bootstrap=100,
            seed=42,
            horizon_max=5,
            aux_partition="cohort",
        )
        params = est.get_params()

        assert params["anticipation"] == 1
        assert params["alpha"] == 0.10
        assert params["n_bootstrap"] == 100
        assert params["seed"] == 42
        assert params["horizon_max"] == 5
        assert params["aux_partition"] == "cohort"
        assert params["cluster"] is None
        assert params["rank_deficient_action"] == "warn"

    def test_set_params(self):
        """Test set_params modifies attributes."""
        est = ImputationDiD()
        est.set_params(alpha=0.10, anticipation=2)

        assert est.alpha == 0.10
        assert est.anticipation == 2

    def test_set_params_unknown(self):
        """Test set_params raises on unknown parameter."""
        est = ImputationDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent=True)

    def test_missing_columns(self):
        """Test error on missing columns."""
        data = generate_test_data()

        est = ImputationDiD()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="nonexistent",
            )

    def test_significance_properties(self):
        """Test is_significant and significance_stars properties."""
        data = generate_test_data(treatment_effect=5.0)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.is_significant
        assert results.significance_stars in ("***", "**", "*", ".")

    def test_repr(self):
        """Test string representation."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        r = repr(results)
        assert "ImputationDiDResults" in r
        assert "ATT=" in r

    def test_convenience_function(self):
        """Test imputation_did convenience function."""
        data = generate_test_data()
        results = imputation_did(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
        )

        assert isinstance(results, ImputationDiDResults)
        assert results.event_study_effects is not None

    def test_convenience_function_kwargs(self):
        """Test imputation_did passes kwargs to constructor."""
        data = generate_test_data()
        results = imputation_did(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            alpha=0.10,
        )

        assert results.alpha == 0.10

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (some units missing periods)."""
        data = generate_test_data(seed=99)
        rng = np.random.default_rng(99)

        # Drop some observations randomly
        keep = rng.random(len(data)) > 0.1
        data_unbal = data[keep].reset_index(drop=True)

        est = ImputationDiD()
        results = est.fit(
            data_unbal,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_att is not None
        assert results.overall_se > 0


# =============================================================================
# TestImputationDiDResults
# =============================================================================


class TestImputationDiDResults:
    """Tests for ImputationDiDResults."""

    def test_pretrend_test(self):
        """Test pre-trend test on data with parallel trends."""
        data = generate_test_data(dynamic_effects=False, seed=77, n_units=200)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test()
        assert "f_stat" in pt
        assert "p_value" in pt
        assert "n_leads" in pt
        assert pt["n_leads"] > 0

        # Under parallel trends, should not reject
        assert pt["p_value"] > 0.01

    def test_pretrend_with_violation(self):
        """Test pre-trend test detects trend violation."""
        data = generate_test_data(seed=88, n_units=200)

        # Add a pre-treatment trend for treated units
        rng = np.random.default_rng(88)
        for idx in data.index:
            if data.loc[idx, "first_treat"] > 0:
                t = data.loc[idx, "time"]
                ft = data.loc[idx, "first_treat"]
                if t < ft:
                    data.loc[idx, "outcome"] += 0.5 * (t - ft)

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test()
        # With pre-trend violation, should reject (low p-value)
        assert pt["p_value"] < 0.10

    def test_pretrend_unbalanced_panel(self):
        """Test pretrend_test uses iterative demeaning for unbalanced panels."""
        data = generate_test_data(dynamic_effects=False, seed=77, n_units=200)
        # Make unbalanced by dropping ~15% of observations
        rng = np.random.default_rng(77)
        keep = rng.random(len(data)) > 0.15
        data_unbal = data[keep].reset_index(drop=True)

        est = ImputationDiD()
        results = est.fit(
            data_unbal,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        pt = results.pretrend_test()
        assert pt["n_leads"] > 0
        # Under parallel trends, should not reject
        assert pt["p_value"] > 0.01

    def test_pretrend_n_leads(self):
        """Test pre-trend test with specified number of leads."""
        data = generate_test_data(n_units=200, seed=55)
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        pt = results.pretrend_test(n_leads=2)
        assert pt["n_leads"] == 2


# =============================================================================
# TestImputationVariance
# =============================================================================


class TestImputationVariance:
    """Tests for conservative variance estimation (Theorem 3)."""

    def test_se_positive(self):
        """Test that SE is positive."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.overall_se > 0

    def test_se_positive_event_study(self):
        """Test that event study SEs are positive."""
        data = generate_test_data()
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0 and np.isfinite(eff["se"]):
                assert eff["se"] > 0

    def test_aux_partition_cohort_horizon(self):
        """Test cohort_horizon partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="cohort_horizon")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_aux_partition_cohort(self):
        """Test cohort partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="cohort")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_aux_partition_horizon(self):
        """Test horizon partition produces valid SEs."""
        data = generate_test_data()
        est = ImputationDiD(aux_partition="horizon")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_coarser_partition_more_conservative(self):
        """Test that coarser partition gives more conservative (larger) SEs."""
        data = generate_test_data(n_units=200, seed=42)

        est_fine = ImputationDiD(aux_partition="cohort_horizon")
        results_fine = est_fine.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        est_coarse = ImputationDiD(aux_partition="cohort")
        results_coarse = est_coarse.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Coarser partition should give >= SE (approximately)
        # Allow small tolerance for numerical issues
        assert results_coarse.overall_se >= results_fine.overall_se * 0.95

    def test_invalid_aux_partition(self):
        """Test that invalid aux_partition raises ValueError."""
        with pytest.raises(ValueError, match="aux_partition"):
            ImputationDiD(aux_partition="invalid")


# =============================================================================
# TestImputationBootstrap
# =============================================================================


class TestImputationBootstrap:
    """Tests for bootstrap inference."""

    def test_basic_bootstrap(self, ci_params):
        """Test basic bootstrap inference."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert results.bootstrap_results is not None
        assert isinstance(results.bootstrap_results, ImputationBootstrapResults)
        assert results.bootstrap_results.n_bootstrap == n_boot
        assert results.bootstrap_results.overall_att_se > 0

    def test_bootstrap_reproducibility(self, ci_params):
        """Test that same seed gives same results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        est1 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r1 = est1.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        est2 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r2 = est2.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        assert r1.overall_se == r2.overall_se

    def test_bootstrap_different_seeds(self, ci_params):
        """Test that different seeds give different results."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)

        est1 = ImputationDiD(n_bootstrap=n_boot, seed=42)
        r1 = est1.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        est2 = ImputationDiD(n_bootstrap=n_boot, seed=99)
        r2 = est2.fit(data, outcome="outcome", unit="unit", time="time", first_treat="first_treat")

        # Results should differ (at least slightly)
        assert r1.overall_se != r2.overall_se

    def test_bootstrap_event_study(self, ci_params):
        """Test bootstrap with event study aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        br = results.bootstrap_results
        assert br.event_study_ses is not None
        assert len(br.event_study_ses) > 0

    def test_bootstrap_group(self, ci_params):
        """Test bootstrap with group aggregation."""
        data = generate_test_data()
        n_boot = ci_params.bootstrap(50)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="group",
        )

        br = results.bootstrap_results
        assert br.group_ses is not None
        assert len(br.group_ses) == 3

    def test_bootstrap_balance_e_consistency(self, ci_params):
        """Test bootstrap event study respects balance_e filtering."""
        data = generate_test_data(n_units=150, seed=42)
        n_boot = ci_params.bootstrap(50)

        # Run WITH balance_e
        est_bal = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_bal = est_bal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=2,
        )

        # Run WITHOUT balance_e
        est_nobal = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_nobal = est_nobal.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results_bal.bootstrap_results is not None
        assert results_bal.bootstrap_results.event_study_ses is not None

        # Verify SEs are finite
        for h in results_bal.event_study_effects:
            eff = results_bal.event_study_effects[h]
            if eff.get("n_obs", 0) > 0 and np.isfinite(eff["effect"]):
                if h in results_bal.bootstrap_results.event_study_ses:
                    assert np.isfinite(results_bal.bootstrap_results.event_study_ses[h])

        # Verify balance_e changed bootstrap SEs at some horizon
        if results_nobal.bootstrap_results is not None:
            bal_ses = results_bal.bootstrap_results.event_study_ses
            nobal_ses = results_nobal.bootstrap_results.event_study_ses
            shared_h = set(bal_ses.keys()) & set(nobal_ses.keys())
            any_different = any(
                not np.isclose(bal_ses[h], nobal_ses[h], rtol=0.05)
                for h in shared_h
                if np.isfinite(bal_ses[h]) and np.isfinite(nobal_ses[h])
            )
            assert any_different, "balance_e should change bootstrap SEs for at least one horizon"

    def test_bootstrap_p_value_significance(self, ci_params):
        """Test bootstrap p-value for significant effect."""
        data = generate_test_data(treatment_effect=5.0, n_units=200)
        n_boot = ci_params.bootstrap(199, min_n=99)
        est = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Strong effect should be significant
        assert results.overall_p_value < 0.05


# =============================================================================
# TestImputationVsOtherEstimators
# =============================================================================


class TestImputationVsOtherEstimators:
    """Cross-validation with CallawaySantAnna and SunAbraham."""

    def test_similar_point_estimates_vs_cs(self):
        """Test that point estimates are similar to CallawaySantAnna."""
        from diff_diff import CallawaySantAnna

        data = generate_test_data(n_units=200, treatment_effect=2.0, seed=42, dynamic_effects=False)

        imp_est = ImputationDiD()
        imp_results = imp_est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        cs = CallawaySantAnna()
        cs_results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Point estimates should be reasonably close
        cs_att = cs_results.overall_att
        imp_att = imp_results.overall_att
        assert abs(imp_att - cs_att) < 1.0

    def test_similar_point_estimates_vs_sa(self):
        """Test that point estimates are similar to SunAbraham."""
        from diff_diff import SunAbraham

        data = generate_test_data(n_units=200, treatment_effect=2.0, seed=42, dynamic_effects=False)

        imp_est = ImputationDiD()
        imp_results = imp_est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        sa = SunAbraham()
        sa_results = sa.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        # Point estimates should be reasonably close
        assert abs(imp_results.overall_att - sa_results.overall_att) < 1.0

    def test_shorter_cis_under_homogeneous_effects(self):
        """Under homogeneous effects, imputation CIs should be shorter."""
        data = generate_test_data(
            n_units=300,
            treatment_effect=2.0,
            seed=42,
            dynamic_effects=False,
        )

        imp_est = ImputationDiD()
        imp_results = imp_est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        from diff_diff import CallawaySantAnna

        cs = CallawaySantAnna()
        cs_results = cs.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        imp_ci_width = imp_results.overall_conf_int[1] - imp_results.overall_conf_int[0]
        cs_ci_width = cs_results.overall_conf_int[1] - cs_results.overall_conf_int[0]

        # Imputation CIs should be shorter (or at least not much longer)
        assert imp_ci_width < cs_ci_width * 1.5


# =============================================================================
# TestImputationEdgeCases
# =============================================================================


class TestImputationEdgeCases:
    """Tests for edge cases."""

    def test_single_cohort(self):
        """Test with a single treatment cohort."""
        rng = np.random.default_rng(42)
        n_units = 50
        n_periods = 8

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[25:] = 4  # Single cohort at period 4

        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)

        outcomes = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + np.tile(np.linspace(0, 1, n_periods), n_units)
            + 2.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_exp,
            }
        )

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert len(results.groups) == 1
        assert results.overall_se > 0
        assert abs(results.overall_att - 2.0) < 1.0

    def test_no_never_treated(self):
        """Test with no never-treated units (Proposition 5)."""
        data = generate_test_data(never_treated_frac=0.0, seed=42)

        est = ImputationDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Should still estimate
        assert results.overall_att is not None
        assert results.overall_se > 0

        # Proposition 5: long-run horizons should be NaN
        prop5_nans = [
            h
            for h, eff in results.event_study_effects.items()
            if np.isnan(eff["effect"]) and eff.get("n_obs", 0) > 0
        ]
        assert len(prop5_nans) > 0, "Should have Prop 5 NaN horizons"

        # Check all inference fields are NaN for Prop 5 horizons
        for h in prop5_nans:
            eff = results.event_study_effects[h]
            assert np.isnan(eff["se"])
            assert np.isnan(eff["t_stat"])
            assert np.isnan(eff["p_value"])
            assert np.isnan(eff["conf_int"][0])
            assert np.isnan(eff["conf_int"][1])

    def test_two_periods(self):
        """Test with just two periods (basic 2x2 DiD)."""
        rng = np.random.default_rng(42)
        n_units = 60
        n_periods = 2

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[30:] = 1  # Treated in period 1

        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0)

        outcomes = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + 3.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "outcome": outcomes,
                "first_treat": first_treat_exp,
            }
        )

        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        assert abs(results.overall_att - 3.0) < 1.0

    def test_rank_deficiency_warn(self):
        """Test rank_deficient_action='warn' doesn't error."""
        data = generate_test_data()
        est = ImputationDiD(rank_deficient_action="warn")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_rank_deficiency_error(self):
        """Test rank_deficient_action='error' works."""
        est = ImputationDiD(rank_deficient_action="error")
        # Should work fine on good data
        data = generate_test_data()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.overall_se > 0

    def test_invalid_rank_deficient_action(self):
        """Test invalid rank_deficient_action raises ValueError."""
        with pytest.raises(ValueError, match="rank_deficient_action"):
            ImputationDiD(rank_deficient_action="ignore")

    def test_always_treated_warning(self):
        """Test warning for units treated in all periods."""
        rng = np.random.default_rng(42)
        n_units = 40
        n_periods = 6

        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)

        first_treat = np.zeros(n_units, dtype=int)
        first_treat[10:20] = 0  # period 0 = always treated
        first_treat[20:] = 3  # treated at period 3

        # Make some units treated in all periods
        first_treat[10:20] = 0  # Never treated (actually)
        # To make always-treated: first_treat <= min_time (0)
        first_treat[0:5] = 0  # These are never-treated
        first_treat[5:10] = -1  # Treated before panel starts!

        first_treat_exp = np.repeat(first_treat, n_periods)
        post = (times >= first_treat_exp) & (first_treat_exp > 0) & (first_treat_exp != np.inf)

        outcomes = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + 2.0 * post
            + rng.standard_normal(len(units)) * 0.5
        )

        # Fix: first_treat with -1 won't trigger the never_treated check properly
        # Let's use first_treat = 0 for some units to trigger always-treated
        first_treat_2 = np.zeros(n_units, dtype=int)
        first_treat_2[:10] = 0  # never treated
        first_treat_2[10:15] = 0  # also never treated (we need >= 1 always-treated)
        first_treat_2[15:] = 3
        # Actually, to trigger always-treated, we need first_treat <= min(time) = 0
        # But first_treat == 0 means never-treated in the code
        # We need first_treat > 0 but <= min(time)
        # min(time) = 0, so first_treat must be <= 0 and > 0, impossible
        # Let's start times at 1
        times_shifted = np.tile(np.arange(1, n_periods + 1), n_units)

        first_treat_3 = np.zeros(n_units, dtype=int)
        first_treat_3[:10] = 0  # never treated
        first_treat_3[10:15] = 1  # treated from the very beginning (always treated)
        first_treat_3[15:] = 4

        first_treat_exp_3 = np.repeat(first_treat_3, n_periods)
        post_3 = (times_shifted >= first_treat_exp_3) & (first_treat_exp_3 > 0)

        outcomes_3 = (
            np.repeat(rng.standard_normal(n_units) * 2, n_periods)
            + 2.0 * post_3
            + rng.standard_normal(len(units)) * 0.5
        )

        data = pd.DataFrame(
            {
                "unit": units,
                "time": times_shifted,
                "outcome": outcomes_3,
                "first_treat": first_treat_exp_3,
            }
        )

        est = ImputationDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

        # Should have issued a warning about always-treated
        always_treated_warnings = [
            x for x in w if "treated in all observed periods" in str(x.message)
        ]
        assert len(always_treated_warnings) > 0

    def test_no_treated_units(self):
        """Test error when no treated units."""
        data = generate_test_data()
        data["first_treat"] = 0  # All never-treated

        est = ImputationDiD()
        with pytest.raises(ValueError, match="No treated"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_nan_propagation_all_nan_horizon(self):
        """Test NaN propagation when all tau_hat at a horizon are NaN."""
        data = generate_test_data(never_treated_frac=0.0, seed=42)

        est = ImputationDiD()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Check that NaN horizons have all-NaN inference
        for h, eff in results.event_study_effects.items():
            if eff.get("n_obs", 0) > 0 and np.isnan(eff["effect"]):
                assert np.isnan(eff["se"])
                assert np.isnan(eff["t_stat"])
                assert np.isnan(eff["p_value"])
                assert np.isnan(eff["conf_int"][0])
                assert np.isnan(eff["conf_int"][1])

    def test_summary_not_fitted(self):
        """Test error when calling summary before fit."""
        est = ImputationDiD()
        with pytest.raises(RuntimeError, match="must be fitted"):
            est.summary()

    def test_rank_condition_missing_untreated_period(self):
        """Test warning when a post-treatment period has no untreated units."""
        # Construct data where ALL units are treated from period 2 onward,
        # so periods 2+ have no untreated observations
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        rows = []
        for i in range(n_units):
            ft = 2  # all units treated at period 2
            for t in range(n_periods):
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0  # treatment effect
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="event_study",
            )
            rank_warnings = [x for x in w if "Rank condition" in str(x.message)]
            assert len(rank_warnings) > 0, "Should warn about rank condition violation"

        # Affected horizons should have NaN effects (periods with no untreated units)
        if results.event_study_effects:
            nan_effects = [
                h
                for h, d in results.event_study_effects.items()
                if np.isnan(d["effect"]) and d.get("n_obs", 1) > 0
            ]
            assert len(nan_effects) > 0, "Some horizons should have NaN effects"

    def test_rank_condition_error_mode(self):
        """Test error raised when rank condition fails with action='error'."""
        # Same setup as test_rank_condition_missing_untreated_period
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        rows = []
        for i in range(n_units):
            ft = 2
            for t in range(n_periods):
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="error")
        with pytest.raises(ValueError, match="Rank condition"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_bootstrap_cluster_not_unit(self, ci_params):
        """Test bootstrap uses cluster column when cluster != unit."""
        data = generate_test_data(n_units=100, n_periods=8, seed=42)
        # Create cluster column grouping every 5 units
        unit_to_cluster = {u: u // 5 for u in data["unit"].unique()}
        data["cluster_id"] = data["unit"].map(unit_to_cluster)

        n_boot = ci_params.bootstrap(99, min_n=49)
        est = ImputationDiD(cluster="cluster_id", n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert results.bootstrap_results is not None
        assert results.bootstrap_results.overall_att_se > 0

        # Bootstrap SE with cluster should differ from unit-level bootstrap
        est_unit = ImputationDiD(n_bootstrap=n_boot, seed=42)
        results_unit = est_unit.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        assert (
            results.bootstrap_results.overall_att_se
            != results_unit.bootstrap_results.overall_att_se
        )

    def test_bootstrap_invalid_cluster_column(self):
        """Test error when cluster column doesn't exist."""
        data = generate_test_data(n_units=50, seed=42)
        est = ImputationDiD(cluster="nonexistent_col")
        with pytest.raises(ValueError, match="not found"):
            est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )

    def test_plot_reference_with_anticipation(self):
        """Test event study plot detects reference period with anticipation."""
        data = generate_test_data(n_units=100, n_periods=10, seed=42)
        est = ImputationDiD(anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        # Reference should be at -2 (= -1 - anticipation)
        assert -2 in results.event_study_effects
        assert results.event_study_effects[-2]["n_obs"] == 0  # reference marker

        # Test that plot_event_study auto-detects it
        pytest.importorskip("matplotlib")
        from diff_diff import plot_event_study

        fig = plot_event_study(results)
        assert fig is not None

    def test_overall_se_with_partial_nan_tau_hat(self):
        """Test overall SE uses finite-only weights when some tau_hat are NaN."""
        # Create staggered data: cohort A treated at t=2, cohort B never-treated
        # but drop all never-treated obs at t=5, so t=5 time FE is unidentified
        # -> tau_hat for (cohort A, t=5) will be NaN
        rng = np.random.default_rng(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            if i < 20:
                ft = 2  # early-treated
            else:
                ft = 99  # never-treated
            for t in range(n_periods):
                # Drop never-treated at t=5 to create unidentified time FE
                if ft == 99 and t == 5:
                    continue
                y = rng.standard_normal() + i * 0.1 + t * 0.05
                if t >= ft:
                    y += 1.0
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": ft,
                    }
                )
        data = pd.DataFrame(rows)

        est = ImputationDiD(rank_deficient_action="silent")
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )

        tau_hat = results.treatment_effects["tau_hat"]
        n_nan = tau_hat.isna().sum()
        n_finite = tau_hat.notna().sum()

        # Verify the scenario actually produces partial NaN
        assert n_nan > 0, "Expected some NaN tau_hat (missing time FE at t=5)"
        assert n_finite > 0, "Expected some finite tau_hat"

        # Partial NaN case: SE should be finite (computed from finite-only weights)
        assert np.isfinite(
            results.overall_se
        ), f"overall_se should be finite with {n_finite} finite and {n_nan} NaN tau_hat"
        assert np.isfinite(results.overall_att)

    def test_iterative_demean_balanced_matches_one_pass(self):
        """Test _iterative_demean matches one-pass for balanced panels."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 20, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        vals = rng.standard_normal(n_units * n_periods)
        idx = pd.RangeIndex(len(vals))

        result_iter = ImputationDiD._iterative_demean(vals, units, times, idx)

        # One-pass for balanced panel
        s = pd.DataFrame({"val": vals, "unit": units, "time": times})
        gm = s["val"].mean()
        um = s.groupby("unit")["val"].transform("mean").values
        tm = s.groupby("time")["val"].transform("mean").values
        result_onepass = vals - um - tm + gm

        np.testing.assert_allclose(result_iter, result_onepass, atol=1e-8)

    def test_unbalanced_panel_fe_correctness(self):
        """Test FE estimates match OLS for unbalanced panel."""
        # Create small unbalanced panel with known FE structure
        rng = np.random.default_rng(42)
        n_units, n_periods = 8, 5
        unit_fe_true = rng.standard_normal(n_units) * 2.0
        time_fe_true = np.linspace(0, 1, n_periods)

        rows = []
        for i in range(n_units):
            for t in range(n_periods):
                # Drop ~20% of obs to make unbalanced
                if rng.random() < 0.2:
                    continue
                y = unit_fe_true[i] + time_fe_true[t] + rng.standard_normal() * 0.01
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "outcome": y,
                        "first_treat": n_periods,  # all never-treated -> Omega_0
                    }
                )

        df_0 = pd.DataFrame(rows)

        # Compute FE via iterative method (what we're testing)
        est = ImputationDiD()
        unit_fe_iter, time_fe_iter = est._iterative_fe(
            df_0["outcome"].values,
            df_0["unit"].values,
            df_0["time"].values,
            df_0.index,
        )

        # Compute exact OLS FE via lstsq with dummy variables
        unique_units = sorted(df_0["unit"].unique())
        unique_times = sorted(df_0["time"].unique())
        n = len(df_0)
        n_u = len(unique_units)
        n_t = len(unique_times)
        u_map = {u: i for i, u in enumerate(unique_units)}
        t_map = {t: i for i, t in enumerate(unique_times)}

        X = np.zeros((n, 1 + (n_u - 1) + (n_t - 1)))
        X[:, 0] = 1.0  # intercept
        for j in range(n):
            uid = u_map[df_0["unit"].iloc[j]]
            tid = t_map[df_0["time"].iloc[j]]
            if uid > 0:
                X[j, uid] = 1.0
            if tid > 0:
                X[j, n_u + tid - 1] = 1.0

        beta_ols = np.linalg.lstsq(X, df_0["outcome"].values, rcond=None)[0]

        # Reconstruct OLS fitted values
        intercept = beta_ols[0]
        unit_fe_ols = {unique_units[0]: intercept}
        for i in range(1, n_u):
            unit_fe_ols[unique_units[i]] = intercept + beta_ols[i]
        time_fe_ols = {unique_times[0]: 0.0}
        for i in range(1, n_t):
            time_fe_ols[unique_times[i]] = beta_ols[n_u + i - 1]

        # Compare fitted values (parameterization-invariant check)
        for j in range(n):
            u = df_0["unit"].iloc[j]
            t = df_0["time"].iloc[j]
            y_hat_iter = unit_fe_iter[u] + time_fe_iter[t]
            y_hat_ols = unit_fe_ols[u] + time_fe_ols[t]
            assert abs(y_hat_iter - y_hat_ols) < 1e-6, (
                f"Fitted values differ at unit={u}, time={t}: "
                f"iterative={y_hat_iter:.8f} vs OLS={y_hat_ols:.8f}"
            )
