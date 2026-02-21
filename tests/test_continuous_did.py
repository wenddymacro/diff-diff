"""
Unit and integration tests for ContinuousDiD estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.continuous_did import ContinuousDiD
from diff_diff.continuous_did_bspline import (
    bspline_derivative_design_matrix,
    bspline_design_matrix,
    build_bspline_basis,
    default_dose_grid,
)
from diff_diff.continuous_did_results import ContinuousDiDResults
from diff_diff.prep_dgp import generate_continuous_did_data

# =============================================================================
# B-Spline Basis Tests
# =============================================================================


class TestBSplineBasis:
    """Test B-spline utility functions."""

    def test_knot_construction_no_interior(self):
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        assert deg == 3
        # Boundary knots repeated degree+1 times
        assert knots[0] == 1.0
        assert knots[-1] == 5.0
        assert len(knots) == 3 + 1 + 3 + 1  # (degree+1)*2

    def test_knot_construction_with_interior(self):
        dose = np.linspace(1, 10, 100)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=2)
        # Interior knots at 1/3 and 2/3 quantiles
        n_expected = 2 * (3 + 1) + 2  # boundary + interior
        assert len(knots) == n_expected

    def test_design_matrix_shape(self):
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        n_basis = len(knots) - deg - 1  # Total basis functions
        assert B.shape == (5, n_basis)  # Same columns (intercept replaces first)

    def test_design_matrix_intercept_column(self):
        dose = np.linspace(1, 5, 20)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        # First column should be all ones
        np.testing.assert_array_equal(B[:, 0], np.ones(20))

    def test_design_matrix_no_intercept(self):
        dose = np.linspace(1, 5, 20)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=0)
        B_no = bspline_design_matrix(dose, knots, deg, include_intercept=False)
        n_basis = len(knots) - deg - 1
        assert B_no.shape == (20, n_basis)
        # First column should NOT be all ones
        assert not np.allclose(B_no[:, 0], 1.0)

    def test_derivative_numerical_check(self):
        """Verify B-spline derivatives match finite differences."""
        dose = np.linspace(1, 5, 50)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=1)

        # Evaluate at interior points (avoid boundaries)
        x = np.linspace(1.5, 4.5, 30)
        dB = bspline_derivative_design_matrix(x, knots, deg, include_intercept=True)

        # Finite difference check
        h = 1e-6
        x_plus = x + h
        x_minus = x - h
        B_plus = bspline_design_matrix(x_plus, knots, deg, include_intercept=True)
        B_minus = bspline_design_matrix(x_minus, knots, deg, include_intercept=True)
        fd = (B_plus - B_minus) / (2 * h)

        # Intercept derivative should be 0
        np.testing.assert_allclose(dB[:, 0], 0.0, atol=1e-10)
        # Other columns should match finite differences
        np.testing.assert_allclose(dB[:, 1:], fd[:, 1:], atol=1e-4)

    def test_partition_of_unity(self):
        """B-spline basis without intercept should sum to ~1 at interior points."""
        dose = np.linspace(1, 5, 50)
        knots, deg = build_bspline_basis(dose, degree=3, num_knots=2)
        x = np.linspace(1.1, 4.9, 30)
        B = bspline_design_matrix(x, knots, deg, include_intercept=False)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_linear_basis(self):
        """Degree 1 with 0 knots: 2 basis functions (intercept + linear)."""
        dose = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knots, deg = build_bspline_basis(dose, degree=1, num_knots=0)
        B = bspline_design_matrix(dose, knots, deg, include_intercept=True)
        assert B.shape[1] == 2  # intercept + 1 basis fn


class TestDoseGrid:
    """Test dose grid computation."""

    def test_default_grid_size(self):
        dose = np.random.default_rng(42).lognormal(0.5, 0.5, size=100)
        grid = default_dose_grid(dose)
        assert len(grid) == 90  # quantiles 0.10 to 0.99

    def test_default_grid_sorted(self):
        dose = np.random.default_rng(42).lognormal(0.5, 0.5, size=100)
        grid = default_dose_grid(dose)
        assert np.all(np.diff(grid) >= 0)

    def test_custom_grid_passthrough(self):
        custom = np.array([1.0, 2.0, 3.0])
        est = ContinuousDiD(dvals=custom)
        np.testing.assert_array_equal(est.dvals, custom)

    def test_empty_dose(self):
        grid = default_dose_grid(np.array([0.0, 0.0]))
        assert len(grid) == 0


# =============================================================================
# ContinuousDiD Estimator Tests
# =============================================================================


class TestContinuousDiDInit:
    """Test constructor, get_params, set_params."""

    def test_default_params(self):
        est = ContinuousDiD()
        params = est.get_params()
        assert params["degree"] == 3
        assert params["num_knots"] == 0
        assert params["control_group"] == "never_treated"
        assert params["alpha"] == 0.05
        assert params["n_bootstrap"] == 0

    def test_set_params(self):
        est = ContinuousDiD()
        est.set_params(degree=1, num_knots=2)
        assert est.degree == 1
        assert est.num_knots == 2

    def test_set_invalid_param(self):
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(nonexistent_param=42)


class TestContinuousDiDDataValidation:
    """Test data validation in fit()."""

    def test_missing_column(self):
        data = pd.DataFrame({"unit": [1], "period": [1], "outcome": [1.0]})
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="Column.*not found"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_non_time_invariant_dose(self):
        data = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "period": [1, 2, 1, 2],
            "outcome": [1.0, 2.0, 1.0, 2.0],
            "first_treat": [2, 2, 0, 0],
            "dose": [1.0, 2.0, 0.0, 0.0],  # Dose changes over time!
        })
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="time-invariant"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_drop_zero_dose_treated(self):
        """Units with positive first_treat but zero dose should be dropped."""
        # Need enough treated units for OLS: degree=1 → 2 basis fns → need >2 treated
        rows = []
        uid = 0
        # 1 treated unit with zero dose (should be dropped)
        rows += [{"unit": uid, "period": 1, "outcome": 1.0, "first_treat": 2, "dose": 0.0},
                 {"unit": uid, "period": 2, "outcome": 3.0, "first_treat": 2, "dose": 0.0}]
        uid += 1
        # 4 treated units with positive dose (should remain)
        for d in [1.0, 2.0, 3.0, 4.0]:
            rows += [{"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 2, "dose": d},
                     {"unit": uid, "period": 2, "outcome": 2 * d, "first_treat": 2, "dose": d}]
            uid += 1
        # 3 control units
        for _ in range(3):
            rows += [{"unit": uid, "period": 1, "outcome": 0.0, "first_treat": 0, "dose": 0.0},
                     {"unit": uid, "period": 2, "outcome": 0.0, "first_treat": 0, "dose": 0.0}]
            uid += 1

        data = pd.DataFrame(rows)
        est = ContinuousDiD(degree=1, num_knots=0)
        with pytest.warns(UserWarning, match="Dropping.*units"):
            results = est.fit(data, "outcome", "unit", "period", "first_treat", "dose")
        # Unit 0 dropped (zero dose but treated), 4 treated remain
        assert results.n_treated_units == 4

    def test_unbalanced_panel_error(self):
        data = pd.DataFrame({
            "unit": [1, 1, 2],
            "period": [1, 2, 1],
            "outcome": [1.0, 2.0, 1.0],
            "first_treat": [2, 2, 0],
            "dose": [1.0, 1.0, 0.0],
        })
        est = ContinuousDiD()
        with pytest.raises(ValueError, match="[Uu]nbalanced"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")

    def test_no_never_treated_error(self):
        data = pd.DataFrame({
            "unit": [1, 1, 2, 2],
            "period": [1, 2, 1, 2],
            "outcome": [1.0, 3.0, 1.0, 4.0],
            "first_treat": [2, 2, 2, 2],
            "dose": [1.0, 1.0, 2.0, 2.0],
        })
        est = ContinuousDiD(control_group="never_treated")
        with pytest.raises(ValueError, match="[Nn]ever-treated"):
            est.fit(data, "outcome", "unit", "period", "first_treat", "dose")


class TestContinuousDiDFit:
    """Test basic fit returns correct types and shapes."""

    @pytest.fixture
    def basic_data(self):
        return generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5
        )

    def test_fit_returns_results(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(
            basic_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert isinstance(results, ContinuousDiDResults)

    def test_dose_response_shapes(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(
            basic_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        n_grid = len(results.dose_grid)
        assert results.dose_response_att.effects.shape == (n_grid,)
        assert results.dose_response_acrt.effects.shape == (n_grid,)
        assert results.dose_response_att.se.shape == (n_grid,)
        assert results.dose_response_acrt.se.shape == (n_grid,)

    def test_overall_parameters(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(
            basic_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_acrt)

    def test_group_time_effects_populated(self, basic_data):
        est = ContinuousDiD()
        results = est.fit(
            basic_data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert len(results.group_time_effects) > 0

    def test_not_yet_treated_control(self):
        data = generate_continuous_did_data(
            n_units=100, n_periods=4, cohort_periods=[2, 3], seed=42,
        )
        est = ContinuousDiD(control_group="not_yet_treated")
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert isinstance(results, ContinuousDiDResults)


class TestContinuousDiDResults:
    """Test results object methods."""

    @pytest.fixture
    def results(self):
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.1
        )
        est = ContinuousDiD(n_bootstrap=49, seed=42)
        return est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )

    def test_summary(self, results):
        s = results.summary()
        assert "ATT_glob" in s
        assert "ACRT_glob" in s
        assert "Continuous" in s

    def test_print_summary(self, results, capsys):
        results.print_summary()
        captured = capsys.readouterr()
        assert "ATT_glob" in captured.out

    def test_to_dataframe_dose_response(self, results):
        df = results.to_dataframe(level="dose_response")
        assert "dose" in df.columns
        assert "att" in df.columns
        assert "acrt" in df.columns
        assert len(df) == len(results.dose_grid)

    def test_to_dataframe_group_time(self, results):
        df = results.to_dataframe(level="group_time")
        assert "group" in df.columns
        assert "time" in df.columns
        assert "att_glob" in df.columns

    def test_to_dataframe_event_study_error(self, results):
        """Should error if event study not computed."""
        with pytest.raises(ValueError, match="[Ee]vent study"):
            results.to_dataframe(level="event_study")

    def test_to_dataframe_invalid_level(self, results):
        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe(level="invalid")

    def test_is_significant(self, results):
        assert isinstance(results.is_significant, bool)

    def test_significance_stars(self, results):
        stars = results.significance_stars
        assert stars in ("", ".", "*", "**", "***")

    def test_repr(self, results):
        r = repr(results)
        assert "ContinuousDiDResults" in r


class TestDoseAggregation:
    """Test dose-response aggregation across (g,t) cells."""

    def test_multi_period_aggregation(self):
        data = generate_continuous_did_data(
            n_units=200, n_periods=5, cohort_periods=[2, 4],
            seed=42, noise_sd=0.1,
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose",
            aggregate="dose",
        )
        # With linear DGP (ATT(d) = 1 + 2d) and degree=1, should recover well
        # ACRT should be close to 2.0
        assert abs(results.overall_acrt - 2.0) < 0.3

    def test_single_cohort_aggregation(self):
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.1,
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose",
            aggregate="dose",
        )
        assert len(results.groups) == 1
        assert np.isfinite(results.overall_att)


class TestEventStudyAggregation:
    """Test event-study aggregation path."""

    def test_event_study_computed(self):
        data = generate_continuous_did_data(
            n_units=200, n_periods=5, cohort_periods=[2, 4],
            seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=49, seed=42)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose",
            aggregate="eventstudy",
        )
        assert results.event_study_effects is not None
        # Should have pre and post relative periods
        rel_periods = sorted(results.event_study_effects.keys())
        assert min(rel_periods) < 0  # Pre-treatment
        assert max(rel_periods) >= 0  # Post-treatment

    def test_event_study_to_dataframe(self):
        data = generate_continuous_did_data(
            n_units=200, n_periods=4, cohort_periods=[2, 3],
            seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD()
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose",
            aggregate="eventstudy",
        )
        df = results.to_dataframe(level="event_study")
        assert "relative_period" in df.columns
        assert "att_glob" in df.columns


class TestBootstrap:
    """Test bootstrap inference."""

    def test_bootstrap_ses_positive(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert results.overall_att_se > 0
        assert results.overall_acrt_se > 0
        # Dose-response SEs should be positive
        assert np.all(results.dose_response_att.se > 0)

    def test_bootstrap_ci_contains_estimate(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        lo, hi = results.overall_att_conf_int
        assert lo <= results.overall_att <= hi

    def test_bootstrap_p_values_valid(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=n_boot, seed=42)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert 0 <= results.overall_att_p_value <= 1
        assert 0 <= results.overall_acrt_p_value <= 1


class TestAnalyticalSE:
    """Test analytical standard errors (n_bootstrap=0)."""

    def test_analytical_se_positive(self):
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=0)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert results.overall_att_se > 0
        assert results.overall_acrt_se > 0

    def test_analytical_ci(self):
        data = generate_continuous_did_data(
            n_units=100, n_periods=3, seed=42, noise_sd=0.5,
        )
        est = ContinuousDiD(n_bootstrap=0)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        lo, hi = results.overall_att_conf_int
        assert lo < results.overall_att < hi


class TestEdgeCases:
    """Test edge cases."""

    def test_few_treated_units(self):
        """Estimator should handle very few treated units."""
        data = generate_continuous_did_data(
            n_units=30, n_periods=3, seed=42,
            never_treated_frac=0.8,  # Only ~6 treated
        )
        est = ContinuousDiD(degree=1, num_knots=0)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert isinstance(results, ContinuousDiDResults)

    def test_inf_first_treat_normalization(self):
        """first_treat=inf should be treated as never-treated."""
        data = generate_continuous_did_data(n_units=50, n_periods=3, seed=42)
        data.loc[data["first_treat"] == 0, "first_treat"] = np.inf
        est = ContinuousDiD()
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        assert results.n_control_units > 0

    def test_custom_dvals(self):
        data = generate_continuous_did_data(n_units=100, n_periods=3, seed=42)
        custom_grid = np.array([1.0, 2.0, 3.0])
        est = ContinuousDiD(dvals=custom_grid)
        results = est.fit(
            data, "outcome", "unit", "period", "first_treat", "dose"
        )
        np.testing.assert_array_equal(results.dose_grid, custom_grid)
        assert len(results.dose_response_att.effects) == 3
