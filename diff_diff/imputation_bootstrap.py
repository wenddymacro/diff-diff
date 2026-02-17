"""
Bootstrap inference methods for the Imputation DiD estimator.

This module contains ImputationDiDBootstrapMixin, which provides multiplier
bootstrap inference. Extracted from imputation.py for module size management.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.imputation_results import ImputationBootstrapResults
from diff_diff.staggered_bootstrap import _generate_bootstrap_weights_batch

__all__ = [
    "ImputationDiDBootstrapMixin",
]


def _compute_target_weights(
    tau_hat: np.ndarray,
    target_mask: np.ndarray,
) -> "tuple[np.ndarray, int]":
    """
    Equal weights for finite tau_hat observations within target_mask.

    Used by both aggregation and bootstrap paths to avoid weight logic
    duplication.

    Parameters
    ----------
    tau_hat : np.ndarray
        Per-observation treatment effects (may contain NaN).
    target_mask : np.ndarray
        Boolean mask selecting the target subset within tau_hat.

    Returns
    -------
    weights : np.ndarray
        Weight array (same length as tau_hat). 1/n_valid for finite
        observations in target_mask, 0 elsewhere.
    n_valid : int
        Number of finite observations in the target subset.
    """
    finite_target = np.isfinite(tau_hat) & target_mask
    n_valid = int(finite_target.sum())
    weights = np.zeros(len(tau_hat))
    if n_valid > 0:
        weights[np.where(finite_target)[0]] = 1.0 / n_valid
    return weights, n_valid


class ImputationDiDBootstrapMixin:
    """Mixin providing bootstrap inference methods for ImputationDiD."""

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        lower = float(np.percentile(boot_dist, alpha / 2 * 100))
        upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return (lower, upper)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        n_valid: Optional[int] = None,
    ) -> float:
        """
        Compute two-sided bootstrap p-value.

        Uses the percentile method: p-value is the proportion of bootstrap
        estimates on the opposite side of zero from the original estimate,
        doubled for two-sided test.

        Parameters
        ----------
        original_effect : float
            Original point estimate.
        boot_dist : np.ndarray
            Bootstrap distribution of the effect.
        n_valid : int, optional
            Number of valid bootstrap samples. If None, uses self.n_bootstrap.
        """
        if original_effect >= 0:
            p_one_sided = float(np.mean(boot_dist <= 0))
        else:
            p_one_sided = float(np.mean(boot_dist >= 0))
        p_value = min(2 * p_one_sided, 1.0)
        n_for_floor = n_valid if n_valid is not None else self.n_bootstrap
        p_value = max(p_value, 1 / (n_for_floor + 1))
        return p_value

    def _precompute_bootstrap_psi(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        cluster_var: str,
        kept_cov_mask: Optional[np.ndarray],
        overall_weights: np.ndarray,
        event_study_effects: Optional[Dict[int, Dict[str, Any]]],
        group_effects: Optional[Dict[Any, Dict[str, Any]]],
        treatment_groups: List[Any],
        tau_hat: np.ndarray,
        balance_e: Optional[int],
    ) -> Dict[str, Any]:
        """
        Pre-compute cluster-level influence function sums for each bootstrap target.

        For each aggregation target (overall, per-horizon, per-group), computes
        psi_i = sum_t v_it * epsilon_tilde_it for each cluster. The multiplier
        bootstrap then perturbs these psi sums with Rademacher weights.

        Computational cost scales with the number of aggregation targets, since
        each target requires its own v_untreated computation (weight-dependent).
        """
        result: Dict[str, Any] = {}

        common = dict(
            df=df,
            outcome=outcome,
            unit=unit,
            time=time,
            first_treat=first_treat,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            omega_1_mask=omega_1_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            grand_mean=grand_mean,
            delta_hat=delta_hat,
            cluster_var=cluster_var,
            kept_cov_mask=kept_cov_mask,
        )

        # Overall ATT
        overall_psi, cluster_ids = self._compute_cluster_psi_sums(**common, weights=overall_weights)
        result["overall"] = (overall_psi, cluster_ids)

        # Event study: per-horizon weights
        if event_study_effects:
            result["event_study"] = {}
            df_1 = df.loc[omega_1_mask]
            rel_times = df_1["_rel_time"].values

            # Balanced cohort mask (same logic as _aggregate_event_study)
            balanced_mask = None
            if balance_e is not None:
                all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))
                if self.horizon_max is not None:
                    all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]
                cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
                balanced_mask = self._compute_balanced_cohort_mask(
                    df_1, first_treat, all_horizons, balance_e, cohort_rel_times
                )

            ref_period = -1 - self.anticipation
            for h in event_study_effects:
                if event_study_effects[h].get("n_obs", 0) == 0:
                    continue
                if h == ref_period:
                    continue
                if not np.isfinite(event_study_effects[h].get("effect", np.nan)):
                    continue
                h_mask = rel_times == h
                if balanced_mask is not None:
                    h_mask = h_mask & balanced_mask
                weights_h, n_valid_h = _compute_target_weights(tau_hat, h_mask)
                if n_valid_h == 0:
                    continue

                psi_h, _ = self._compute_cluster_psi_sums(**common, weights=weights_h)
                result["event_study"][h] = psi_h

        # Group effects: per-group weights
        if group_effects:
            result["group"] = {}
            df_1 = df.loc[omega_1_mask]
            cohorts = df_1[first_treat].values

            for g in group_effects:
                if group_effects[g].get("n_obs", 0) == 0:
                    continue
                if not np.isfinite(group_effects[g].get("effect", np.nan)):
                    continue
                g_mask = cohorts == g
                weights_g, n_valid_g = _compute_target_weights(tau_hat, g_mask)
                if n_valid_g == 0:
                    continue

                psi_g, _ = self._compute_cluster_psi_sums(**common, weights=weights_g)
                result["group"][g] = psi_g

        return result

    def _run_bootstrap(
        self,
        original_att: float,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
        psi_data: Dict[str, Any],
    ) -> ImputationBootstrapResults:
        """
        Run multiplier bootstrap on pre-computed influence function sums.

        Uses T_b = sum_i w_b_i * psi_i where w_b_i are Rademacher weights
        and psi_i are cluster-level influence function sums from Theorem 3.
        SE = std(T_b, ddof=1).
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        overall_psi, cluster_ids = psi_data["overall"]
        n_clusters = len(cluster_ids)

        # Generate ALL weights upfront: shape (n_bootstrap, n_clusters)
        all_weights = _generate_bootstrap_weights_batch(
            self.n_bootstrap, n_clusters, self.bootstrap_weights, rng
        )

        # Overall ATT bootstrap draws
        boot_overall = np.dot(all_weights, overall_psi)  # (n_bootstrap,)

        # Event study: loop over horizons
        boot_event_study: Optional[Dict[int, np.ndarray]] = None
        if original_event_study and "event_study" in psi_data:
            boot_event_study = {}
            for h, psi_h in psi_data["event_study"].items():
                boot_event_study[h] = np.dot(all_weights, psi_h)

        # Group effects: loop over groups
        boot_group: Optional[Dict[Any, np.ndarray]] = None
        if original_group and "group" in psi_data:
            boot_group = {}
            for g, psi_g in psi_data["group"].items():
                boot_group[g] = np.dot(all_weights, psi_g)

        # --- Inference (percentile bootstrap, matching CS/SA convention) ---
        # Shift perturbation-centered draws to effect-centered draws.
        # The multiplier bootstrap produces T_b = sum w_b_i * psi_i centered at 0.
        # CS adds the original effect back (L411 of staggered_bootstrap.py).
        # We do the same here so percentile CIs and empirical p-values work correctly.
        boot_overall_shifted = boot_overall + original_att

        overall_se = float(np.std(boot_overall, ddof=1))
        overall_ci = (
            self._compute_percentile_ci(boot_overall_shifted, self.alpha)
            if overall_se > 0
            else (np.nan, np.nan)
        )
        overall_p = (
            self._compute_bootstrap_pvalue(original_att, boot_overall_shifted)
            if overall_se > 0
            else np.nan
        )

        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None
        if boot_event_study and original_event_study:
            event_study_ses = {}
            event_study_cis = {}
            event_study_p_values = {}
            for h in boot_event_study:
                se_h = float(np.std(boot_event_study[h], ddof=1))
                event_study_ses[h] = se_h
                orig_eff = original_event_study[h]["effect"]
                if se_h > 0 and np.isfinite(orig_eff):
                    shifted_h = boot_event_study[h] + orig_eff
                    event_study_p_values[h] = self._compute_bootstrap_pvalue(orig_eff, shifted_h)
                    event_study_cis[h] = self._compute_percentile_ci(shifted_h, self.alpha)
                else:
                    event_study_p_values[h] = np.nan
                    event_study_cis[h] = (np.nan, np.nan)

        group_ses = None
        group_cis = None
        group_p_values = None
        if boot_group and original_group:
            group_ses = {}
            group_cis = {}
            group_p_values = {}
            for g in boot_group:
                se_g = float(np.std(boot_group[g], ddof=1))
                group_ses[g] = se_g
                orig_eff = original_group[g]["effect"]
                if se_g > 0 and np.isfinite(orig_eff):
                    shifted_g = boot_group[g] + orig_eff
                    group_p_values[g] = self._compute_bootstrap_pvalue(orig_eff, shifted_g)
                    group_cis[g] = self._compute_percentile_ci(shifted_g, self.alpha)
                else:
                    group_p_values[g] = np.nan
                    group_cis[g] = (np.nan, np.nan)

        return ImputationBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_ses=group_ses,
            group_cis=group_cis,
            group_p_values=group_p_values,
            bootstrap_distribution=boot_overall_shifted,
        )
