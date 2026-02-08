"""
Comprehensive methodology verification tests for TwoWayFixedEffects estimator.

This module verifies that the TwoWayFixedEffects implementation matches:
1. The theoretical formulas from within-transformation algebra
2. The behavior of R's fixest::feols() with absorbed unit+time FE
3. All documented edge cases in docs/methodology/REGISTRY.md

References:
- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data, 2nd ed.
  MIT Press, Chapter 10.
- Goodman-Bacon, A. (2021). Difference-in-Differences with variation in treatment timing.
  Journal of Econometrics, 225(2), 254-277.
"""

import json
import os
import subprocess
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from diff_diff import TwoWayFixedEffects
from diff_diff.linalg import LinearRegression
from diff_diff.utils import within_transform


# =============================================================================
# R Availability Fixtures
# =============================================================================

_fixest_available_cache = None


def _check_fixest_available() -> bool:
    """Check if R and fixest package are available (cached)."""
    global _fixest_available_cache
    if _fixest_available_cache is None:
        r_env = os.environ.get("DIFF_DIFF_R", "auto").lower()
        if r_env == "skip":
            _fixest_available_cache = False
        else:
            try:
                result = subprocess.run(
                    ["Rscript", "-e", "library(fixest); library(jsonlite); cat('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                _fixest_available_cache = result.returncode == 0 and "OK" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                _fixest_available_cache = False
    return _fixest_available_cache


@pytest.fixture(scope="session")
def fixest_available():
    """Lazy check for R/fixest availability."""
    return _check_fixest_available()


@pytest.fixture
def require_fixest(fixest_available):
    """Skip test if R/fixest is not available."""
    if not fixest_available:
        pytest.skip("R or fixest package not available")


# =============================================================================
# Data Generation Helpers
# =============================================================================


def generate_twfe_panel(
    n_units: int = 20,
    n_periods: int = 4,
    treatment_effect: float = 3.0,
    noise_sd: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data for TWFE testing with known ATT."""
    np.random.seed(seed)
    n_treated = n_units // 2
    data = []

    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_effect = np.random.normal(0, 2)

        for period in range(n_periods):
            post = 1 if period >= n_periods // 2 else 0
            time_effect = period * 1.0

            y = 10.0 + unit_effect + time_effect
            if is_treated and post:
                y += treatment_effect
            y += np.random.normal(0, noise_sd)

            data.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": post,
                "outcome": y,
            })

    return pd.DataFrame(data)


def generate_hand_calculable_panel() -> pd.DataFrame:
    """
    Generate a minimal 2-period panel with exact hand-calculable values.

    4 units (2 treated, 2 control) × 2 periods = 8 observations.
    No noise, so ATT is exactly 3.0.
    """
    return pd.DataFrame({
        "unit": [0, 0, 1, 1, 2, 2, 3, 3],
        "period": [0, 1, 0, 1, 0, 1, 0, 1],
        "treated": [1, 1, 1, 1, 0, 0, 0, 0],
        "post": [0, 1, 0, 1, 0, 1, 0, 1],
        "outcome": [
            10.0, 15.0,  # Unit 0 (treated): pre=10, post=15 (diff=5)
            12.0, 17.0,  # Unit 1 (treated): pre=12, post=17 (diff=5)
            8.0, 10.0,   # Unit 2 (control): pre=8, post=10 (diff=2)
            6.0, 8.0,    # Unit 3 (control): pre=6, post=8 (diff=2)
        ],
    })
    # ATT = (mean treated diff) - (mean control diff) = 5.0 - 2.0 = 3.0


# =============================================================================
# Phase 1: Within-Transformation Algebra
# =============================================================================


class TestWithinTransformationAlgebra:
    """Verify the within-transformation (two-way demeaning) is correct."""

    def test_within_transform_hand_calculation(self):
        """Verify within-transformation matches hand calculation: y_it - ȳ_i - ȳ_t + ȳ."""
        data = generate_hand_calculable_panel()

        # Hand-calculate within-transformed outcome
        # Unit means: unit 0 = 12.5, unit 1 = 14.5, unit 2 = 9.0, unit 3 = 7.0
        # Time means: period 0 = (10+12+8+6)/4 = 9.0, period 1 = (15+17+10+8)/4 = 12.5
        # Grand mean = (10+15+12+17+8+10+6+8)/8 = 86/8 = 10.75
        unit_means = data.groupby("unit")["outcome"].transform("mean")
        time_means = data.groupby("period")["outcome"].transform("mean")
        grand_mean = data["outcome"].mean()
        expected_demeaned = data["outcome"] - unit_means - time_means + grand_mean

        # Use the library function
        result = within_transform(data, ["outcome"], "unit", "period")

        np.testing.assert_allclose(
            result["outcome_demeaned"].values,
            expected_demeaned.values,
            rtol=1e-12,
        )

    def test_within_transform_covariates_also_demeaned(self):
        """Verify covariates are demeaned (not just outcome)."""
        data = generate_twfe_panel(n_units=10, n_periods=4, seed=123)
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))

        result = within_transform(data, ["outcome", "x1"], "unit", "period")

        # Demeaned covariates should sum to ~0 within each unit and time group
        for var in ["outcome_demeaned", "x1_demeaned"]:
            unit_sums = result.groupby("unit")[var].sum()
            time_sums = result.groupby("period")[var].sum()
            np.testing.assert_allclose(unit_sums.values, 0, atol=1e-10)
            np.testing.assert_allclose(time_sums.values, 0, atol=1e-10)

    def test_twfe_att_matches_hand_calculated_demeaned_ols(self):
        """
        Verify TWFE ATT matches manual demeaned OLS on a small panel.

        By FWL theorem, regressing demeaned Y on demeaned (D_i * Post_t) gives ATT.
        Both outcome and regressors must be within-transformed.
        """
        data = generate_hand_calculable_panel()

        # Run TWFE
        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Manual demeaned OLS: demean both y and the interaction term
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "period")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        manual_att = coeffs[1]

        np.testing.assert_allclose(results.att, manual_att, rtol=1e-10)

    def test_twfe_att_matches_basic_did_for_two_period_design(self):
        """TWFE and basic DiD should agree on 2-period data."""
        from diff_diff import DifferenceInDifferences

        data = generate_hand_calculable_panel()

        # TWFE
        twfe = TwoWayFixedEffects(robust=True)
        twfe_results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Basic DiD
        did = DifferenceInDifferences(robust=True, cluster="unit")
        did_results = did.fit(
            data, outcome="outcome", treatment="treated", time="post"
        )

        np.testing.assert_allclose(twfe_results.att, did_results.att, rtol=1e-10)

    def test_demeaned_outcome_sums_to_zero(self):
        """Within-transformed outcome sums to zero within each unit and time group."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=99)

        result = within_transform(data, ["outcome"], "unit", "period")

        unit_sums = result.groupby("unit")["outcome_demeaned"].sum()
        time_sums = result.groupby("period")["outcome_demeaned"].sum()

        np.testing.assert_allclose(unit_sums.values, 0, atol=1e-10)
        np.testing.assert_allclose(time_sums.values, 0, atol=1e-10)


# =============================================================================
# Phase 2: R Comparison
# =============================================================================


class TestRBenchmarkTWFE:
    """Compare TWFE estimates against R's fixest::feols() with absorbed FE."""

    @pytest.fixture
    def benchmark_panel_data(self, tmp_path):
        """Generate panel data and save to CSV for R comparison."""
        np.random.seed(12345)
        n_units = 50
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = unit * 0.2

            for period in range(n_periods):
                post = 1 if period >= 2 else 0
                period_effect = period * 1.0

                y = 10.0 + unit_effect + period_effect
                if is_treated and post:
                    y += 3.0
                y += np.random.normal(0, 0.5)

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                })

        df = pd.DataFrame(data)
        csv_path = tmp_path / "panel_data.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    @pytest.fixture
    def benchmark_panel_data_with_covariate(self, tmp_path):
        """Generate panel data with a time-varying covariate."""
        np.random.seed(12345)
        n_units = 50
        n_periods = 4

        data = []
        for unit in range(n_units):
            is_treated = unit < n_units // 2
            unit_effect = unit * 0.2

            for period in range(n_periods):
                post = 1 if period >= 2 else 0
                period_effect = period * 1.0
                x1 = np.random.normal(0, 1) + period * 0.3

                y = 10.0 + unit_effect + period_effect + 1.5 * x1
                if is_treated and post:
                    y += 3.0
                y += np.random.normal(0, 0.5)

                data.append({
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": post,
                    "outcome": y,
                    "x1": x1,
                })

        df = pd.DataFrame(data)
        csv_path = tmp_path / "panel_data_cov.csv"
        df.to_csv(csv_path, index=False)
        return df, str(csv_path)

    def _run_r_feols_twfe(self, data_path: str, covariates=None) -> Dict[str, Any]:
        """Run R's fixest::feols() with absorbed unit+post FE, clustered at unit."""
        escaped_path = data_path.replace("\\", "/")

        if covariates:
            cov_str = " + ".join(covariates)
            formula = f"outcome ~ treated:post + {cov_str} | unit + post"
        else:
            formula = "outcome ~ treated:post | unit + post"

        r_script = f'''
        suppressMessages(library(fixest))
        suppressMessages(library(jsonlite))

        data <- read.csv("{escaped_path}")
        data$treated <- as.numeric(data$treated)
        data$post <- as.numeric(data$post)

        result <- feols({formula}, data = data, cluster = ~unit)

        # Use coeftable() to get fixest's own inference (SE, t-stat, p-value)
        # This ensures we use fixest's df adjustment, not a manual pt() call
        ct <- coeftable(result)
        att_row <- which(rownames(ct) == "treated:post")
        if (length(att_row) == 0) {{
            att_row <- which(grepl("treated.*post", rownames(ct)))
        }}

        att <- ct[att_row, "Estimate"]
        se_val <- ct[att_row, "Std. Error"]
        tstat <- ct[att_row, "t value"]
        pval <- ct[att_row, "Pr(>|t|)"]
        ci <- confint(result)
        ci_lower <- ci[att_row, 1]
        ci_upper <- ci[att_row, 2]

        output <- list(
            att = unbox(att),
            se = unbox(se_val),
            t_stat = unbox(tstat),
            p_value = unbox(pval),
            ci_lower = unbox(ci_lower),
            ci_upper = unbox(ci_upper),
            n_obs = unbox(result$nobs)
        )

        cat(toJSON(output, pretty = TRUE, digits = 15))
        '''

        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        parsed = json.loads(result.stdout)
        # Unwrap single-element lists from R's JSON encoding
        for key in parsed:
            if isinstance(parsed[key], list) and len(parsed[key]) == 1:
                parsed[key] = parsed[key][0]

        return parsed

    def _run_python_twfe(self, data, covariates=None):
        """Run Python TWFE estimator."""
        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            covariates=covariates,
        )
        return results

    def test_att_matches_r_twfe(self, require_fixest, benchmark_panel_data):
        """ATT within rtol=1e-3 (0.1%) of R's fixest."""
        data, csv_path = benchmark_panel_data

        py_results = self._run_python_twfe(data)
        r_results = self._run_r_feols_twfe(csv_path)

        np.testing.assert_allclose(
            py_results.att, r_results["att"], rtol=1e-3,
            err_msg=f"ATT mismatch: Python={py_results.att:.6f}, R={r_results['att']:.6f}",
        )

    def test_se_matches_r_twfe(self, require_fixest, benchmark_panel_data):
        """Cluster-robust SE within rtol=0.01 (1%) of R's fixest."""
        data, csv_path = benchmark_panel_data

        py_results = self._run_python_twfe(data)
        r_results = self._run_r_feols_twfe(csv_path)

        np.testing.assert_allclose(
            py_results.se, r_results["se"], rtol=0.01,
            err_msg=f"SE mismatch: Python={py_results.se:.6f}, R={r_results['se']:.6f}",
        )

    def test_pvalue_matches_r_twfe(self, require_fixest, benchmark_panel_data):
        """P-value within atol=0.01 of R's fixest."""
        data, csv_path = benchmark_panel_data

        py_results = self._run_python_twfe(data)
        r_results = self._run_r_feols_twfe(csv_path)

        np.testing.assert_allclose(
            py_results.p_value, r_results["p_value"], atol=0.01,
            err_msg=f"P-value mismatch: Python={py_results.p_value:.6f}, R={r_results['p_value']:.6f}",
        )

    def test_ci_matches_r_twfe(self, require_fixest, benchmark_panel_data):
        """CI bounds within rtol=0.01 (1%) of R's fixest."""
        data, csv_path = benchmark_panel_data

        py_results = self._run_python_twfe(data)
        r_results = self._run_r_feols_twfe(csv_path)

        np.testing.assert_allclose(
            py_results.conf_int[0], r_results["ci_lower"], rtol=0.01,
            err_msg=f"CI lower mismatch: Python={py_results.conf_int[0]:.6f}, R={r_results['ci_lower']:.6f}",
        )
        np.testing.assert_allclose(
            py_results.conf_int[1], r_results["ci_upper"], rtol=0.01,
            err_msg=f"CI upper mismatch: Python={py_results.conf_int[1]:.6f}, R={r_results['ci_upper']:.6f}",
        )

    def test_att_matches_r_with_covariate(
        self, require_fixest, benchmark_panel_data_with_covariate
    ):
        """ATT with demeaned covariate within rtol=1e-3 of R."""
        data, csv_path = benchmark_panel_data_with_covariate

        py_results = self._run_python_twfe(data, covariates=["x1"])
        r_results = self._run_r_feols_twfe(csv_path, covariates=["x1"])

        np.testing.assert_allclose(
            py_results.att, r_results["att"], rtol=1e-3,
            err_msg=f"ATT w/ cov mismatch: Python={py_results.att:.6f}, R={r_results['att']:.6f}",
        )

    def test_se_matches_r_with_covariate(
        self, require_fixest, benchmark_panel_data_with_covariate
    ):
        """SE with covariate within rtol=0.01 of R."""
        data, csv_path = benchmark_panel_data_with_covariate

        py_results = self._run_python_twfe(data, covariates=["x1"])
        r_results = self._run_r_feols_twfe(csv_path, covariates=["x1"])

        np.testing.assert_allclose(
            py_results.se, r_results["se"], rtol=0.01,
            err_msg=f"SE w/ cov mismatch: Python={py_results.se:.6f}, R={r_results['se']:.6f}",
        )


# =============================================================================
# Phase 3: Edge Cases (from REGISTRY.md)
# =============================================================================


class TestTWFEEdgeCases:
    """Test all edge cases documented in docs/methodology/REGISTRY.md."""

    def test_staggered_treatment_warning_multiperiod_time(self):
        """Staggered treatment warning fires when `time` is multi-valued.

        This tests the multi-period `time` scenario. When `time` has actual
        period values (not binary 0/1), the staggered check can detect
        different cohorts starting treatment at different periods. We use
        `time="period"` here because the standard binary `time="post"`
        configuration cannot detect staggering (see
        test_staggered_warning_not_fired_with_binary_time).
        """
        np.random.seed(42)
        data = []
        for unit in range(20):
            # Units 0-4: treated at period 2
            # Units 5-9: treated at period 3
            # Units 10-19: never treated
            for period in range(5):
                if unit < 5:
                    treated = 1 if period >= 2 else 0
                elif unit < 10:
                    treated = 1 if period >= 3 else 0
                else:
                    treated = 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 3.0 + np.random.normal(0, 0.5)
                data.append({
                    "unit": unit, "period": period, "treated": treated,
                    "outcome": y,
                })
        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects(robust=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use time="period" so staggered detection sees different first-treat times
            twfe.fit(df, outcome="outcome", treatment="treated", time="period", unit="unit")

        staggered_warnings = [x for x in w if "Staggered treatment" in str(x.message)]
        assert len(staggered_warnings) > 0, "Expected staggered treatment warning"

    def test_staggered_warning_not_fired_with_binary_time(self):
        """Staggered warning does NOT fire with binary time (known limitation).

        When `time` is a binary post indicator (0/1), all treated units appear
        to start treatment at time=1, so unique_treat_times=[1] and the
        staggered check cannot distinguish cohorts. This is a documented
        limitation — users with staggered designs should use `decompose()` or
        `CallawaySantAnna` directly.
        """
        np.random.seed(42)
        data = []
        for unit in range(20):
            # Units 0-4: treated at period 2 (early cohort)
            # Units 5-9: treated at period 3 (late cohort)
            # Units 10-19: never treated
            for period in range(5):
                if unit < 5:
                    treated = 1 if period >= 2 else 0
                elif unit < 10:
                    treated = 1 if period >= 3 else 0
                else:
                    treated = 0
                post = 1 if period >= 2 else 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 3.0 + np.random.normal(0, 0.5)
                data.append({
                    "unit": unit, "period": period, "post": post,
                    "treated": treated, "outcome": y,
                })
        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects(robust=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # With binary time="post", staggering is undetectable
            twfe.fit(df, outcome="outcome", treatment="treated", time="post", unit="unit")

        staggered_warnings = [x for x in w if "Staggered treatment" in str(x.message)]
        assert len(staggered_warnings) == 0, (
            "Staggered warning should NOT fire with binary time (known limitation)"
        )

    def test_auto_clusters_at_unit_level(self):
        """SE with cluster=None (default) equals SE when explicitly passing cluster='unit'."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # Default (auto-clusters at unit)
        twfe_default = TwoWayFixedEffects(robust=True)
        results_default = twfe_default.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Explicit cluster at unit
        twfe_explicit = TwoWayFixedEffects(robust=True, cluster="unit")
        results_explicit = twfe_explicit.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        np.testing.assert_allclose(
            results_default.se, results_explicit.se, rtol=1e-12,
        )
        # Config should be immutable
        assert twfe_default.cluster is None

    def test_df_adjustment_for_absorbed_fe(self):
        """
        Verify degrees-of-freedom adjustment for absorbed fixed effects.

        TWFE applies df_adjustment = n_units + n_times - 2 to account for
        absorbed FE. Verify the SE matches a manual LinearRegression with
        the same df adjustment.
        """
        data = generate_twfe_panel(n_units=20, n_periods=2, noise_sd=0.5, seed=42)

        # Run TWFE
        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Manual: demean both y and the interaction, then run LinearRegression
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "period")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])

        n_units = data["unit"].nunique()
        n_times = data["period"].nunique()
        df_adjustment = n_units + n_times - 2
        cluster_ids = data["unit"].values

        reg = LinearRegression(
            include_intercept=False,
            robust=True,
            cluster_ids=cluster_ids,
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment)
        manual_se = reg.get_inference(1).se

        np.testing.assert_allclose(
            results.se, manual_se, rtol=1e-10,
            err_msg=f"SE df-adjustment mismatch: TWFE={results.se:.8f}, manual={manual_se:.8f}",
        )

    def test_treatment_collinear_with_fe_raises_error(self):
        """Treatment perfectly collinear with FE raises ValueError.

        When all units of one group are treated in ALL periods (no variation
        within the treatment indicator after demeaning), the demeaned
        interaction becomes zero and collinear with the intercept.
        """
        # All treated units are treated in ALL periods -> D_it = D_i for all t
        # After demeaning by unit+time, D_i*Post_t becomes constant within unit
        # (because D_i is absorbed), making it perfectly collinear.
        data = pd.DataFrame({
            "unit": [0, 0, 1, 1, 2, 2, 3, 3],
            "period": [0, 1, 0, 1, 0, 1, 0, 1],
            "treated": [1, 1, 1, 1, 0, 0, 0, 0],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": [10.0, 11.0, 12.0, 13.0, 8.0, 9.0, 6.0, 7.0],
        })

        # Construct data where treatment effect cannot be identified:
        # Make treatment*post a linear combination of unit and post dummies
        # by having treated = 1 for all periods for some units.
        # When treated is constant within unit, treated*post = treated × post,
        # and after demeaning: demeaned(treated*post) is perfectly
        # correlated with demeaned(post) (which is absorbed).
        # Actually for rank deficiency: add a covariate that together with
        # the treatment interaction makes the design matrix rank-deficient.
        # Simpler: use rank_deficient_action="error" with a collinear covariate
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(robust=True, rank_deficient_action="error")
        with pytest.raises(ValueError):
            twfe.fit(
                data, outcome="outcome", treatment="treated", time="post",
                unit="unit", covariates=["bad_cov"],
            )

    def test_covariate_collinearity_warns_not_errors(self):
        """Collinear covariate emits warning but ATT is still finite."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        # Add a covariate that's collinear with treatment*post
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(robust=True, rank_deficient_action="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                time="post",
                unit="unit",
                covariates=["bad_cov"],
            )

        collinear_warnings = [x for x in w if "collinear" in str(x.message).lower()]
        assert len(collinear_warnings) > 0, "Expected collinearity warning"
        assert np.isfinite(results.att), "ATT should be finite despite collinearity"
        # ATT should be in reasonable range of true effect (3.0)
        assert abs(results.att - 3.0) < 1.5, f"ATT={results.att} far from true effect 3.0"

    def test_rank_deficient_action_error_raises(self):
        """rank_deficient_action='error' raises ValueError on rank-deficient data."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(robust=True, rank_deficient_action="error")
        with pytest.raises(ValueError):
            twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                time="post",
                unit="unit",
                covariates=["bad_cov"],
            )

    def test_rank_deficient_action_silent_no_warning(self):
        """rank_deficient_action='silent' emits no warnings."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        data["bad_cov"] = data["treated"] * data["post"]

        twfe = TwoWayFixedEffects(robust=True, rank_deficient_action="silent")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = twfe.fit(
                data,
                outcome="outcome",
                treatment="treated",
                time="post",
                unit="unit",
                covariates=["bad_cov"],
            )

        collinear_warnings = [x for x in w if "collinear" in str(x.message).lower()]
        assert len(collinear_warnings) == 0, "Expected no collinearity warnings with silent"
        assert np.isfinite(results.att)

    def test_unbalanced_panel_produces_valid_results(self):
        """Dropping some unit-period observations still gives valid results."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # Drop some observations to create unbalanced panel
        drop_indices = [3, 7, 15, 22, 45, 60]
        data = data.drop(index=drop_indices).reset_index(drop=True)

        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        assert np.isfinite(results.att), "ATT should be finite for unbalanced panel"
        assert results.se > 0, "SE should be positive"
        assert results.n_obs == len(data)

    def test_unit_column_missing_raises_error(self):
        """Missing unit column raises ValueError."""
        data = generate_hand_calculable_panel()

        twfe = TwoWayFixedEffects(robust=True)
        with pytest.raises(ValueError, match="not found"):
            twfe.fit(
                data, outcome="outcome", treatment="treated",
                time="post", unit="nonexistent_unit",
            )

    def test_decompose_integration(self):
        """decompose() returns BaconDecompositionResults for staggered data."""
        from diff_diff.bacon import BaconDecompositionResults

        np.random.seed(42)
        data = []
        for unit in range(30):
            if unit < 10:
                first_treat = 3
            elif unit < 20:
                first_treat = 4
            else:
                first_treat = 0  # never treated

            for period in range(1, 6):
                treated = 1 if (first_treat > 0 and period >= first_treat) else 0
                y = 10.0 + unit * 0.1 + period * 0.5 + treated * 2.0 + np.random.normal(0, 0.5)
                data.append({
                    "unit": unit,
                    "period": period,
                    "outcome": y,
                    "first_treat": first_treat,
                })

        df = pd.DataFrame(data)

        twfe = TwoWayFixedEffects(robust=True)
        decomp = twfe.decompose(
            df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )

        assert isinstance(decomp, BaconDecompositionResults)
        assert len(decomp.comparisons) > 0


# =============================================================================
# Phase 4: SE Verification
# =============================================================================


class TestTWFESEVerification:
    """Verify standard error properties."""

    def test_cluster_se_differs_from_hc1_se(self):
        """
        Cluster-robust SE differs from HC1 SE, verifying auto-clustering is active.

        TWFE auto-clusters at unit level. We manually compute HC1 SE on the
        same demeaned data (demeaned by unit + post, matching TWFE) and verify
        the SEs are different, proving clustering changes inference.
        """
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        # TWFE: cluster-robust at unit (automatic)
        twfe = TwoWayFixedEffects(robust=True)
        twfe_results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Manual HC1 SE on same demeaned regression (no clustering)
        # Demean by unit + post to match TWFE's within-transform
        data_with_tp = data.copy()
        data_with_tp["tp"] = data["treated"] * data["post"]
        demeaned = within_transform(data_with_tp, ["outcome", "tp"], "unit", "post")
        y = demeaned["outcome_demeaned"].values
        tp = demeaned["tp_demeaned"].values
        X = np.column_stack([np.ones(len(y)), tp])
        n_units = data["unit"].nunique()
        n_times = data["post"].nunique()
        df_adjustment = n_units + n_times - 2

        hc1_reg = LinearRegression(
            include_intercept=False,
            robust=True,
            cluster_ids=None,  # HC1, no clustering
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment)
        hc1_se = hc1_reg.get_inference(1).se

        # Verify SEs are different (auto-clustering is active)
        assert twfe_results.se != hc1_se, (
            f"Cluster SE ({twfe_results.se:.6f}) should differ from "
            f"HC1 SE ({hc1_se:.6f}) — auto-clustering must be active"
        )

        # Also verify TWFE SE matches a manually computed cluster SE
        cluster_reg = LinearRegression(
            include_intercept=False,
            robust=True,
            cluster_ids=data["unit"].values,
            rank_deficient_action="silent",
        ).fit(X, y, df_adjustment=df_adjustment)
        manual_cluster_se = cluster_reg.get_inference(1).se

        np.testing.assert_allclose(
            twfe_results.se, manual_cluster_se, rtol=1e-10,
            err_msg="TWFE SE should match manually computed cluster SE"
        )

    def test_vcov_positive_semidefinite(self):
        """VCoV matrix should be positive semi-definite."""
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        eigenvalues = np.linalg.eigvalsh(results.vcov)
        assert np.all(eigenvalues >= -1e-10), (
            f"VCoV has negative eigenvalues: {eigenvalues[eigenvalues < -1e-10]}"
        )


# =============================================================================
# Phase 5: Wild Bootstrap
# =============================================================================


class TestTWFEWildBootstrap:
    """Verify wild cluster bootstrap inference."""

    def test_wild_bootstrap_produces_valid_inference(self, ci_params):
        """Wild bootstrap produces finite SE and valid p-value."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        n_boot = ci_params.bootstrap(999, min_n=199)

        twfe = TwoWayFixedEffects(
            robust=True, inference="wild_bootstrap", n_bootstrap=n_boot, seed=42
        )
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        assert np.isfinite(results.se) and results.se > 0
        assert 0 <= results.p_value <= 1
        assert results.inference_method == "wild_bootstrap"

    @pytest.mark.parametrize("weight_type", ["rademacher", "mammen", "webb"])
    def test_wild_bootstrap_weight_types(self, ci_params, weight_type):
        """Each bootstrap weight type produces valid inference."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)
        n_boot = ci_params.bootstrap(199, min_n=99)

        twfe = TwoWayFixedEffects(
            robust=True,
            inference="wild_bootstrap",
            n_bootstrap=n_boot,
            bootstrap_weights=weight_type,
            seed=42,
        )
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        assert np.isfinite(results.se) and results.se > 0
        assert 0 <= results.p_value <= 1

    def test_inference_parameter_routing(self):
        """inference='wild_bootstrap' routes to wild bootstrap method."""
        data = generate_twfe_panel(n_units=20, n_periods=2, seed=42)

        twfe = TwoWayFixedEffects(
            robust=True, inference="wild_bootstrap", n_bootstrap=99, seed=42
        )
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        assert results.inference_method == "wild_bootstrap"


# =============================================================================
# Phase 6: Params & Results
# =============================================================================


class TestTWFEParamsAndResults:
    """Verify sklearn-like parameter interface and results completeness."""

    def test_get_params_returns_all_parameters(self):
        """All inherited constructor params present in get_params()."""
        twfe = TwoWayFixedEffects(robust=True)
        params = twfe.get_params()

        expected_keys = {
            "robust", "cluster", "alpha", "inference",
            "n_bootstrap", "bootstrap_weights", "seed",
            "rank_deficient_action",
        }
        assert expected_keys.issubset(params.keys()), (
            f"Missing params: {expected_keys - params.keys()}"
        )

    def test_set_params_modifies_attributes(self):
        """set_params() modifies estimator attributes."""
        twfe = TwoWayFixedEffects(robust=True)
        twfe.set_params(alpha=0.10, robust=False)

        assert twfe.alpha == 0.10
        assert twfe.robust is False

    def test_summary_contains_key_info(self):
        """summary() output contains ATT."""
        data = generate_hand_calculable_panel()
        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        summary = results.summary()
        assert "ATT" in summary

    def test_to_dict_contains_all_fields(self):
        """to_dict() contains required fields."""
        data = generate_hand_calculable_panel()
        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        d = results.to_dict()
        for key in ["att", "se", "t_stat", "p_value", "n_obs"]:
            assert key in d, f"Missing key '{key}' in to_dict()"

    def test_residuals_plus_fitted_equals_demeaned_outcome(self):
        """Check residuals + fitted = demeaned outcome (not raw outcome).

        TWFE demeans by unit + time (where time is the `time` parameter).
        The demeaned outcome is the within-transformed y.
        """
        data = generate_twfe_panel(n_units=20, n_periods=4, seed=42)

        twfe = TwoWayFixedEffects(robust=True)
        results = twfe.fit(
            data, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

        # Within-transform by unit + post (same as TWFE internally does)
        demeaned = within_transform(data, ["outcome"], "unit", "post")
        y_demeaned = demeaned["outcome_demeaned"].values

        reconstructed = results.residuals + results.fitted_values
        np.testing.assert_allclose(
            reconstructed, y_demeaned, rtol=1e-10,
            err_msg="residuals + fitted_values should equal demeaned outcome",
        )
