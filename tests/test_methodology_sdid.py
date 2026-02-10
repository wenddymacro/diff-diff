"""
Methodology tests for Synthetic Difference-in-Differences (SDID).

Tests verify the implementation matches Arkhangelsky et al. (2021) and
R's synthdid package behavior: Frank-Wolfe solver, collapsed form,
auto-regularization, sparsification, and correct bootstrap/placebo SE.
"""

import warnings

import numpy as np
import pytest

from diff_diff.synthetic_did import SyntheticDiD
from diff_diff.utils import (
    _compute_noise_level,
    _compute_noise_level_numpy,
    _compute_regularization,
    _fw_step,
    _sc_weight_fw,
    _sparsify,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_time_weights,
)


# =============================================================================
# Test Helpers
# =============================================================================


def _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3,
                att=5.0, seed=42):
    """Create a simple panel dataset for testing."""
    rng = np.random.default_rng(seed)
    data = []
    for unit in range(n_control + n_treated):
        is_treated = unit >= n_control
        unit_fe = rng.normal(0, 2)
        for t in range(n_pre + n_post):
            y = 10.0 + unit_fe + t * 0.3 + rng.normal(0, 0.5)
            if is_treated and t >= n_pre:
                y += att
            data.append({
                "unit": unit,
                "period": t,
                "treated": int(is_treated),
                "outcome": y,
            })
    import pandas as pd
    return pd.DataFrame(data)


# =============================================================================
# Phase A: Noise Level and Regularization
# =============================================================================


class TestNoiseLevel:
    """Verify _compute_noise_level matches hand-computed first-diff sd."""

    def test_known_values(self):
        """Test with a simple matrix where first-diffs are known."""
        # 3 time periods, 2 control units
        # Unit 0: [1, 3, 6] -> diffs: [2, 3]
        # Unit 1: [2, 2, 5] -> diffs: [0, 3]
        # All diffs: [2, 3, 0, 3], sd(ddof=1) = std([2,3,0,3], ddof=1)
        Y = np.array([[1.0, 2.0],
                       [3.0, 2.0],
                       [6.0, 5.0]])
        expected = np.std([2.0, 3.0, 0.0, 3.0], ddof=1)
        result = _compute_noise_level(Y)
        assert abs(result - expected) < 1e-10

    def test_single_period(self):
        """Single period -> no diffs possible -> noise level = 0."""
        Y = np.array([[1.0, 2.0, 3.0]])
        assert _compute_noise_level(Y) == 0.0

    def test_two_periods(self):
        """Two periods -> one diff per unit."""
        Y = np.array([[1.0, 4.0],
                       [3.0, 7.0]])
        # Diffs: [2.0, 3.0], sd(ddof=1)
        expected = np.std([2.0, 3.0], ddof=1)
        assert abs(_compute_noise_level(Y) - expected) < 1e-10


class TestRegularization:
    """Verify _compute_regularization formula with known inputs."""

    def test_formula(self):
        """Check zeta_omega = (N1*T1)^0.25 * sigma, zeta_lambda = 1e-6 * sigma."""
        # Use a simple Y_pre_control where sigma is easy to compute
        Y = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [6.0, 7.0]])
        sigma = _compute_noise_level(Y)
        n_treated, n_post = 2, 3

        zeta_omega, zeta_lambda = _compute_regularization(Y, n_treated, n_post)

        expected_omega = (n_treated * n_post) ** 0.25 * sigma
        expected_lambda = 1e-6 * sigma

        assert abs(zeta_omega - expected_omega) < 1e-10
        assert abs(zeta_lambda - expected_lambda) < 1e-15

    def test_zero_noise(self):
        """Constant outcomes -> zero noise -> zero regularization."""
        Y = np.array([[5.0, 5.0],
                       [5.0, 5.0],
                       [5.0, 5.0]])
        zo, zl = _compute_regularization(Y, 2, 3)
        assert zo == 0.0
        assert zl == 0.0


# =============================================================================
# Phase B: Frank-Wolfe Solver
# =============================================================================


class TestFrankWolfe:
    """Verify Frank-Wolfe step and solver behavior."""

    def test_fw_step_descent(self):
        """A single FW step should not increase the half-gradient objective.

        The FW step minimizes the linearized objective at the current point.
        After the step with exact line search, the true objective should
        not increase when measured correctly.
        """
        rng = np.random.default_rng(42)
        N, T0 = 10, 5
        A = rng.standard_normal((N, T0))
        b = rng.standard_normal(N)
        eta = N * 0.1 ** 2  # eta = N * zeta^2 matching R's formulation

        x = np.ones(T0) / T0

        # Run a few steps to get away from the initial uniform point
        # (first step from uniform can have numerical issues)
        for _ in range(5):
            x = _fw_step(A, x, b, eta)

        def objective(lam):
            err = A @ lam - b
            return (eta / N) * np.sum(lam**2) + np.sum(err**2) / N

        obj_before = objective(x)
        x_new = _fw_step(A, x, b, eta)
        obj_after = objective(x_new)

        assert obj_after <= obj_before + 1e-8

    def test_fw_step_on_simplex(self):
        """FW step should return a vector on the simplex."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 4))
        b = rng.standard_normal(8)
        x = np.array([0.25, 0.25, 0.25, 0.25])

        x_new = _fw_step(A, x, b, eta=1.0)
        assert np.all(x_new >= -1e-10)
        assert abs(np.sum(x_new) - 1.0) < 1e-10

    def test_sc_weight_fw_converges(self):
        """Full FW solver should converge on a known QP."""
        rng = np.random.default_rng(42)
        N, T0 = 15, 6
        Y = rng.standard_normal((N, T0 + 1))  # last col is target

        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1000)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_sc_weight_fw_max_iter_zero(self):
        """max_iter=0 should return initial uniform weights."""
        Y = np.random.randn(5, 4)  # (N, T0+1) with T0=3
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=0)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(lam, expected, atol=1e-10)

    def test_sc_weight_fw_max_iter_one(self):
        """max_iter=1 should return weights after one step (still on simplex)."""
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((8, 5))
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-10

    def test_intercept_centering(self):
        """With intercept=True, column-centering should occur."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((10, 5)) + 100  # large offset
        lam_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=True)
        lam_no_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=False)
        # Both should be on simplex but may differ
        assert abs(np.sum(lam_intercept) - 1.0) < 1e-6
        assert abs(np.sum(lam_no_intercept) - 1.0) < 1e-6
        # They should be different because centering matters
        assert not np.allclose(lam_intercept, lam_no_intercept, atol=1e-3)


class TestSparsify:
    """Verify sparsification behavior."""

    def test_basic(self):
        """Weights below max/4 should be zeroed."""
        v = np.array([0.8, 0.1, 0.05, 0.05])
        result = _sparsify(v)
        assert result[0] > 0
        # 0.1 < 0.8/4 = 0.2, so should be zeroed
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_all_zero(self):
        """All-zero input should return uniform weights."""
        v = np.zeros(5)
        result = _sparsify(v)
        np.testing.assert_allclose(result, np.ones(5) / 5)

    def test_single_nonzero(self):
        """Single nonzero element -> that element becomes 1.0."""
        v = np.array([0.0, 0.0, 0.5, 0.0])
        result = _sparsify(v)
        assert result[2] == 1.0
        assert np.sum(result) == 1.0

    def test_equal_weights(self):
        """Equal weights: all equal max, so max/4 threshold keeps them all."""
        v = np.array([0.25, 0.25, 0.25, 0.25])
        result = _sparsify(v)
        # 0.25 > 0.25/4 = 0.0625, so all kept
        np.testing.assert_allclose(result, v, atol=1e-10)


class TestSumNormalize:
    """Verify _sum_normalize helper."""

    def test_basic(self):
        v = np.array([2.0, 3.0, 5.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5])

    def test_zero_sum(self):
        """Zero-sum vector -> uniform weights."""
        v = np.array([0.0, 0.0, 0.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [1.0/3, 1.0/3, 1.0/3])


# =============================================================================
# Phase C/D: Unit and Time Weights
# =============================================================================


class TestUnitWeights:
    """Verify compute_sdid_unit_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex (sum to 1, non-negative)."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_pre_treated_mean = rng.standard_normal(8)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert np.all(omega >= -1e-10)
        assert abs(np.sum(omega) - 1.0) < 1e-6

    def test_single_control(self):
        """Single control unit -> weight = [1.0]."""
        Y_pre = np.array([[1.0], [2.0], [3.0]])
        Y_pre_treated_mean = np.array([1.5, 2.5, 3.5])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 1
        assert abs(omega[0] - 1.0) < 1e-10

    def test_empty_control(self):
        """No control units -> empty array."""
        Y_pre = np.zeros((5, 0))
        Y_pre_treated_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 0

    def test_sparsification_occurs(self):
        """With enough controls, some weights should be zeroed by sparsification."""
        rng = np.random.default_rng(42)
        # Many controls with varied patterns — expect some to get zeroed
        Y_pre = rng.standard_normal((10, 50))
        Y_pre_treated_mean = rng.standard_normal(10)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=0.5)

        # At least some weights should be exactly zero after sparsification
        assert np.sum(omega == 0) > 0


class TestTimeWeights:
    """Verify compute_time_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_post = rng.standard_normal((3, 15))

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_single_pre_period(self):
        """Single pre-period -> weight = [1.0]."""
        Y_pre = np.array([[1.0, 2.0, 3.0]])
        Y_post = np.array([[4.0, 5.0, 6.0]])

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert len(lam) == 1
        assert abs(lam[0] - 1.0) < 1e-10

    def test_collapsed_form_correctness(self):
        """Verify the collapsed form matrix is built correctly.

        The time weight optimization solves on (N_co, T_pre+1) where the
        last column is the per-control post-period mean.
        """
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((4, 3))  # 4 pre-periods, 3 controls
        Y_post = rng.standard_normal((2, 3))  # 2 post-periods, 3 controls

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        # Should have 4 weights (one per pre-period)
        assert len(lam) == 4
        assert abs(np.sum(lam) - 1.0) < 1e-6


# =============================================================================
# Full Pipeline
# =============================================================================


class TestATTFullPipeline:
    """Test full SDID estimation pipeline."""

    def test_estimation_produces_reasonable_att(self, ci_params):
        """Full estimation on canonical data should produce reasonable ATT."""
        df = _make_panel(n_control=20, n_treated=3, n_pre=6, n_post=3,
                         att=5.0, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42, variance_method="placebo")
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(6, 9)),
        )

        # ATT should be positive and reasonably close to true value
        assert results.att > 0
        assert abs(results.att - 5.0) < 3.0

    def test_results_have_regularization_info(self):
        """Results should include noise_level, zeta_omega, zeta_lambda."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.noise_level is not None
        assert results.noise_level >= 0
        assert results.zeta_omega is not None
        assert results.zeta_omega >= 0
        assert results.zeta_lambda is not None
        assert results.zeta_lambda >= 0

    def test_user_override_regularization(self):
        """User-specified zeta_omega/zeta_lambda should be used instead of auto."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(
            zeta_omega=99.0, zeta_lambda=0.5,
            variance_method="placebo", seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.zeta_omega == 99.0
        assert results.zeta_lambda == 0.5


# =============================================================================
# Placebo SE
# =============================================================================


class TestPlaceboSE:
    """Verify placebo variance formula."""

    def test_placebo_se_formula(self):
        """SE should be sqrt((r-1)/r) * sd(estimates, ddof=1)."""
        df = _make_panel(n_control=15, n_treated=2, seed=42)
        sdid = SyntheticDiD(
            variance_method="placebo", n_bootstrap=100, seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "placebo"

        # Verify the formula: se = sqrt((r-1)/r) * sd(placebo_estimates)
        if results.placebo_effects is not None:
            r = len(results.placebo_effects)
            expected_se = np.sqrt((r - 1) / r) * np.std(results.placebo_effects, ddof=1)
            assert abs(results.se - expected_se) < 1e-10


# =============================================================================
# Bootstrap SE
# =============================================================================


class TestBootstrapSE:
    """Verify bootstrap SE with fixed weights."""

    def test_bootstrap_se_positive(self, ci_params):
        """Bootstrap SE should be positive."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=n_boot, seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "bootstrap"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_single_treated_unit(self, ci_params):
        """Estimation should work with a single treated unit."""
        df = _make_panel(n_control=10, n_treated=1, n_pre=5, n_post=2,
                         att=3.0, seed=42)
        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=[5, 6],
        )
        assert np.isfinite(results.att)

    def test_insufficient_controls_for_placebo(self):
        """Placebo with n_control <= n_treated should warn and return SE=0."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=[5, 6],
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Should warn about insufficient controls for placebo
            assert len(user_warnings) > 0

        assert results.se == 0.0

    def test_se_zero_propagation(self):
        """When SE=0, t_stat and p_value should be NaN, CI should be NaN."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=[5, 6],
            )

        if results.se == 0.0:
            assert np.isnan(results.t_stat)
            assert np.isnan(results.p_value)
            assert np.isnan(results.conf_int[0])
            assert np.isnan(results.conf_int[1])


# =============================================================================
# get_params / set_params
# =============================================================================


class TestGetSetParams:
    """Verify parameter accessors."""

    def test_get_params_includes_new_names(self):
        """get_params should include zeta_omega/zeta_lambda."""
        sdid = SyntheticDiD(zeta_omega=1.0, zeta_lambda=0.5)
        params = sdid.get_params()
        assert "zeta_omega" in params
        assert "zeta_lambda" in params
        assert params["zeta_omega"] == 1.0
        assert params["zeta_lambda"] == 0.5

    def test_get_params_excludes_old_names(self):
        """get_params should NOT include lambda_reg or zeta."""
        sdid = SyntheticDiD()
        params = sdid.get_params()
        assert "lambda_reg" not in params
        assert "zeta" not in params

    def test_set_params_new_names(self):
        """set_params with new names should work."""
        sdid = SyntheticDiD()
        sdid.set_params(zeta_omega=2.0, zeta_lambda=0.1)
        assert sdid.zeta_omega == 2.0
        assert sdid.zeta_lambda == 0.1

    def test_set_params_deprecated_names_warn(self):
        """set_params with old names should emit DeprecationWarning."""
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.set_params(lambda_reg=1.0)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_set_params_unknown_raises(self):
        """set_params with unknown name should raise ValueError."""
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            sdid.set_params(nonexistent_param=1.0)


class TestDeprecatedParams:
    """Test deprecated parameter handling in __init__."""

    def test_lambda_reg_warns(self):
        """SyntheticDiD(lambda_reg=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(lambda_reg=0.1)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "lambda_reg" in str(dep[0].message)

        # Deprecated param is ignored — auto-computed used
        assert sdid.zeta_omega is None

    def test_zeta_warns(self):
        """SyntheticDiD(zeta=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(zeta=2.0)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "zeta" in str(dep[0].message)

        assert sdid.zeta_lambda is None

    def test_both_deprecated_params(self):
        """Both deprecated params at once should emit two warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SyntheticDiD(lambda_reg=0.5, zeta=1.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 2

    def test_default_variance_method_is_placebo(self):
        """Default variance_method should be 'placebo' (matching R)."""
        sdid = SyntheticDiD()
        assert sdid.variance_method == "placebo"


class TestNoiseLevelEdgeCases:
    """Edge case tests for _compute_noise_level_numpy."""

    def test_noise_level_single_control_two_periods(self):
        """noise_level returns 0.0 (not NaN) for 1 control, 2 pre-periods.

        With shape (2, 1), first_diffs has size=1, and np.std([x], ddof=1)
        would divide by zero → NaN. Guard ensures 0.0 is returned instead,
        matching the Rust backend behavior.
        """
        Y = np.array([[1.0], [2.0]])  # (2, 1)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0
        assert not np.isnan(result)

    def test_noise_level_single_element_returns_zero(self):
        """noise_level returns 0.0 when first_diffs has exactly 1 element."""
        # (2, 1) → diff → (1, 1) → size=1 → return 0.0
        Y = np.array([[5.0], [10.0]])
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0

    def test_noise_level_empty_returns_zero(self):
        """noise_level returns 0.0 for single time period (no diffs possible)."""
        Y = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0


class TestPlaceboReestimation:
    """Tests verifying placebo variance re-estimates weights (not fixed)."""

    def test_placebo_reestimates_weights_not_fixed(self):
        """Placebo variance re-estimates omega/lambda per replication (matching R).

        Verifies the methodology choice: R's vcov(method='placebo') passes
        update.omega=TRUE, update.lambda=TRUE, so weights are re-estimated
        via Frank-Wolfe on each permutation — NOT renormalized from originals.

        We verify this by comparing the actual placebo SE against a manual
        fixed-weight computation; if they differ, re-estimation is happening.
        """
        # Need enough controls for placebo to work (n_control > n_treated)
        # and enough variation for weights to differ between re-estimation
        # and renormalization.
        df = _make_panel(n_control=15, n_treated=2, n_pre=6, n_post=3,
                         att=5.0, seed=123)
        post_periods = list(range(6, 9))

        # Fit SDID to get original weights and matrices
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )
        actual_se = results.se

        # Now compute a "fixed weight" placebo manually:
        # permute controls, renormalize original omega (no Frank-Wolfe),
        # keep original lambda unchanged.
        rng = np.random.default_rng(42)

        # Build the outcome matrix (T, N) as the estimator does
        pivot = df.pivot(index="period", columns="unit", values="outcome")
        control_units = sorted(results.unit_weights.keys())
        treated_mask = df.groupby("unit")["treated"].max().values.astype(bool)
        control_idx = np.where(~treated_mask)[0]
        treated_idx = np.where(treated_mask)[0]
        Y = pivot.values  # (T, N)
        pre_periods_arr = np.array(post_periods)
        pre_mask = ~np.isin(pivot.index.values, pre_periods_arr)

        Y_pre_control = Y[np.ix_(pre_mask, control_idx)]
        Y_post_control = Y[np.ix_(~pre_mask, control_idx)]

        # Extract numpy arrays from result dicts (ordered by control unit)
        unit_weights_arr = np.array([results.unit_weights[u] for u in control_units])
        time_weights_arr = np.array([results.time_weights[t]
                                     for t in sorted(results.time_weights.keys())])

        n_control = len(control_idx)
        n_treated_count = len(treated_idx)
        n_pseudo_control = n_control - n_treated_count

        fixed_estimates = []
        for _ in range(50):
            perm = rng.permutation(n_control)
            pc_idx = perm[:n_pseudo_control]
            pt_idx = perm[n_pseudo_control:]

            # Fixed weights: renormalize original omega for pseudo-controls
            fixed_omega = _sum_normalize(unit_weights_arr[pc_idx])
            fixed_lambda = time_weights_arr  # unchanged

            Y_pre_pc = Y_pre_control[:, pc_idx]
            Y_post_pc = Y_post_control[:, pc_idx]
            Y_pre_pt_mean = np.mean(Y_pre_control[:, pt_idx], axis=1)
            Y_post_pt_mean = np.mean(Y_post_control[:, pt_idx], axis=1)

            try:
                tau = compute_sdid_estimator(
                    Y_pre_pc, Y_post_pc,
                    Y_pre_pt_mean, Y_post_pt_mean,
                    fixed_omega, fixed_lambda,
                )
                fixed_estimates.append(tau)
            except (ValueError, np.linalg.LinAlgError):
                continue

        if len(fixed_estimates) >= 2:
            n_s = len(fixed_estimates)
            fixed_se = (np.sqrt((n_s - 1) / n_s)
                        * np.std(fixed_estimates, ddof=1))
            # The two SEs should differ because re-estimation produces
            # different weights than renormalization
            assert actual_se != pytest.approx(fixed_se, rel=0.01), (
                f"Placebo SE ({actual_se:.6f}) matches fixed-weight SE "
                f"({fixed_se:.6f}), suggesting weights are NOT being "
                f"re-estimated as R's synthdid does."
            )
