"""
Gardner (2022) Two-Stage Difference-in-Differences Estimator.

Implements the two-stage DiD estimator from Gardner (2022), "Two-stage
differences in differences". The method:
1. Estimates unit + time fixed effects on untreated observations only
2. Residualizes ALL outcomes using estimated FEs
3. Regresses residualized outcomes on treatment indicators (Stage 2)

Inference uses the GMM sandwich variance estimator from Butts & Gardner
(2022) that correctly accounts for first-stage estimation uncertainty.

Point estimates are identical to ImputationDiD (Borusyak et al. 2024);
the key difference is the variance estimator (GMM sandwich vs. conservative).

References
----------
Gardner, J. (2022). Two-stage differences in differences.
    arXiv:2207.05943.
Butts, K. & Gardner, J. (2022). did2s: Two-Stage
    Difference-in-Differences. R Journal, 14(1), 162-173.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff.linalg import solve_ols
from diff_diff.results import _get_significance_stars
from diff_diff.utils import compute_confidence_interval, compute_p_value

# =============================================================================
# Results Dataclasses
# =============================================================================


@dataclass
class TwoStageBootstrapResults:
    """
    Results from TwoStageDiD bootstrap inference.

    Bootstrap uses multiplier bootstrap on the GMM influence function,
    consistent with other library estimators. The R `did2s` package uses
    block bootstrap by default; multiplier bootstrap is asymptotically
    equivalent.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights (currently "rademacher" only).
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : tuple
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    event_study_ses : dict, optional
        Bootstrap SEs for event study effects.
    event_study_cis : dict, optional
        Bootstrap CIs for event study effects.
    event_study_p_values : dict, optional
        Bootstrap p-values for event study effects.
    group_ses : dict, optional
        Bootstrap SEs for group effects.
    group_cis : dict, optional
        Bootstrap CIs for group effects.
    group_p_values : dict, optional
        Bootstrap p-values for group effects.
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of overall ATT.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_ses: Optional[Dict[Any, float]] = None
    group_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TwoStageDiDResults:
    """
    Results from Gardner (2022) two-stage DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Per-observation treatment effects with columns: unit, time,
        tau_hat, weight. tau_hat is the residualized outcome y_tilde
        for treated observations; weight is 1/n_treated.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT (GMM sandwich).
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    event_study_effects : dict, optional
        Dictionary mapping relative time h to effect dict with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_obs'.
    group_effects : dict, optional
        Dictionary mapping cohort g to effect dict.
    groups : list
        List of treatment cohorts.
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_obs : int
        Number of treated observations.
    n_untreated_obs : int
        Number of untreated observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to untreated observations.
    alpha : float
        Significance level used.
    bootstrap_results : TwoStageBootstrapResults, optional
        Bootstrap inference results.
    """

    treatment_effects: pd.DataFrame
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    event_study_effects: Optional[Dict[int, Dict[str, Any]]]
    group_effects: Optional[Dict[Any, Dict[str, Any]]]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_obs: int
    n_untreated_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    bootstrap_results: Optional[TwoStageBootstrapResults] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"TwoStageDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_treated_obs={self.n_treated_obs})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to alpha used in estimation.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Two-Stage DiD Estimator Results (Gardner 2022)".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated observations:':<30} {self.n_treated_obs:>10}",
            f"{'Untreated observations:':<30} {self.n_untreated_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            "",
        ]

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ]
        )

        t_str = (
            f"{self.overall_t_stat:>10.3f}" if np.isfinite(self.overall_t_stat) else f"{'NaN':>10}"
        )
        p_str = (
            f"{self.overall_p_value:>10.4f}"
            if np.isfinite(self.overall_p_value)
            else f"{'NaN':>10}"
        )
        sig = _get_significance_stars(self.overall_p_value)

        lines.extend(
            [
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{t_str} {p_str} {sig:>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
                "",
            ]
        )

        # Event study effects
        if self.event_study_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Event Study (Dynamic) Effects".center(85),
                    "-" * 85,
                    f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for h in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[h]
                if eff.get("n_obs", 1) == 0:
                    # Reference period marker
                    lines.append(
                        f"[ref: {h}]" f"{'0.0000':>17} {'---':>12} {'---':>10} {'---':>10} {'':>6}"
                    )
                elif np.isnan(eff["effect"]):
                    lines.append(f"{h:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    e_sig = _get_significance_stars(eff["p_value"])
                    e_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    e_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{h:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{e_t} {e_p} {e_sig:>6}"
                    )

            lines.extend(["-" * 85, ""])

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Group (Cohort) Effects".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for g in sorted(self.group_effects.keys()):
                eff = self.group_effects[g]
                if np.isnan(eff["effect"]):
                    lines.append(f"{g:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    g_sig = _get_significance_stars(eff["p_value"])
                    g_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    g_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{g:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{g_t} {g_p} {g_sig:>6}"
                    )

            lines.extend(["-" * 85, ""])

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 85,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="event_study"
            Level of aggregation:
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects
            - "observation": Per-observation treatment effects

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "observation":
            return self.treatment_effects.copy()

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. "
                    "Use aggregate='event_study' or aggregate='all'."
                )
            rows = []
            for h, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": h,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "n_obs": data.get("n_obs", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed. " "Use aggregate='group' or aggregate='all'."
                )
            rows = []
            for g, data in sorted(self.group_effects.items()):
                rows.append(
                    {
                        "group": g,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "n_obs": data.get("n_obs", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        else:
            raise ValueError(
                f"Unknown level: {level}. Use 'event_study', 'group', or 'observation'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)


# =============================================================================
# Main Estimator
# =============================================================================


class TwoStageDiD:
    """
    Gardner (2022) two-stage Difference-in-Differences estimator.

    This estimator addresses TWFE bias under heterogeneous treatment
    effects by:
    1. Estimating unit + time FEs on untreated observations only
    2. Residualizing ALL outcomes using estimated FEs
    3. Regressing residualized outcomes on treatment indicators

    Point estimates are identical to ImputationDiD (Borusyak et al. 2024).
    The key difference is the variance estimator: TwoStageDiD uses a GMM
    sandwich variance that accounts for first-stage estimation uncertainty,
    while ImputationDiD uses the conservative variance from Theorem 3.

    Parameters
    ----------
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default.
    n_bootstrap : int, default=0
        Number of bootstrap iterations. If 0, uses analytical GMM
        sandwich inference.
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently
    horizon_max : int, optional
        Maximum event-study horizon. If set, event study effects are only
        computed for |h| <= horizon_max.

    Attributes
    ----------
    results_ : TwoStageDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import TwoStageDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> results.print_summary()

    With event study:

    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat',
    ...                   aggregate='event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The two-stage estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model.

    References
    ----------
    Gardner, J. (2022). Two-stage differences in differences.
        arXiv:2207.05943.
    Butts, K. & Gardner, J. (2022). did2s: Two-Stage
        Difference-in-Differences. R Journal, 14(1), 162-173.
    """

    def __init__(
        self,
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        horizon_max: Optional[int] = None,
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max

        self.is_fitted_ = False
        self.results_: Optional[TwoStageDiDResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
    ) -> TwoStageDiDResults:
        """
        Fit the two-stage DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list of str, optional
            List of covariate column names.
        aggregate : str, optional
            Aggregation mode: None/"simple" (overall ATT only),
            "event_study", "group", or "all".
        balance_e : int, optional
            When computing event study, restrict to cohorts observed at all
            relative times in [-balance_e, max_h].

        Returns
        -------
        TwoStageDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # ---- Data validation ----
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = data.copy()
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Validate absorbing treatment
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        non_constant = ft_nunique[ft_nunique > 1]
        if len(non_constant) > 0:
            example_unit = non_constant.index[0]
            example_vals = sorted(df.loc[df[unit] == example_unit, first_treat].unique())
            warnings.warn(
                f"{len(non_constant)} unit(s) have non-constant '{first_treat}' "
                f"values (e.g., unit '{example_unit}' has values {example_vals}). "
                f"TwoStageDiD assumes treatment is an absorbing state "
                f"(once treated, always treated) with a single treatment onset "
                f"time per unit. Non-constant first_treat violates this assumption "
                f"and may produce unreliable estimates.",
                UserWarning,
                stacklevel=2,
            )
            df[first_treat] = df.groupby(unit)[first_treat].transform("first")

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        always_treated_units = df.loc[always_treated_mask, unit].unique()
        n_always_treated = len(always_treated_units)
        if n_always_treated > 0:
            unit_list = ", ".join(str(u) for u in always_treated_units[:10])
            suffix = f" (and {n_always_treated - 10} more)" if n_always_treated > 10 else ""
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}): [{unit_list}{suffix}]. "
                "These units have no untreated observations and cannot contribute "
                "to the counterfactual model. Excluding from estimation.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[unit].isin(always_treated_units)].copy()

        # Treatment indicator with anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Partition into Omega_0 (untreated) and Omega_1 (treated)
        omega_0_mask = ~df["_treated"]
        omega_1_mask = df["_treated"]

        n_omega_0 = int(omega_0_mask.sum())
        n_omega_1 = int(omega_1_mask.sum())

        if n_omega_0 == 0:
            raise ValueError(
                "No untreated observations found. Cannot estimate counterfactual model."
            )
        if n_omega_1 == 0:
            raise ValueError("No treated observations found. Nothing to estimate.")

        # Groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
        units_in_omega_0 = df.loc[omega_0_mask, unit].unique()
        n_control_units = len(units_in_omega_0)

        # Cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit
        if self.cluster is not None and self.cluster not in df.columns:
            raise ValueError(
                f"Cluster column '{self.cluster}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Stage 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask
        )

        # ---- Rank condition checks ----
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

        post_period_ids = df.loc[omega_1_mask, time].unique()
        periods_with_fe = set(time_fe.keys())
        periods_missing_fe = set(post_period_ids) - periods_with_fe

        if units_missing_fe or periods_missing_fe:
            parts = []
            if units_missing_fe:
                sorted_missing = sorted(units_missing_fe)
                parts.append(
                    f"{len(units_missing_fe)} treated unit(s) have no untreated "
                    f"periods (units: {sorted_missing[:5]}"
                    f"{'...' if len(units_missing_fe) > 5 else ''})"
                )
            if periods_missing_fe:
                sorted_missing = sorted(periods_missing_fe)
                parts.append(
                    f"{len(periods_missing_fe)} post-treatment period(s) have no "
                    f"untreated units (periods: {sorted_missing[:5]}"
                    f"{'...' if len(periods_missing_fe) > 5 else ''})"
                )
            msg = (
                "Rank condition violated: "
                + "; ".join(parts)
                + ". Affected treatment effects will be NaN."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(msg)
            elif self.rank_deficient_action == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

        # ---- Residualize ALL observations ----
        y_tilde = self._residualize(
            df, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )
        df["_y_tilde"] = y_tilde

        # ---- Stage 2: OLS of y_tilde on treatment indicators ----
        # Build design matrices and compute effects + GMM variance
        ref_period = -1 - self.anticipation

        # Always compute overall ATT (static specification)
        overall_att, overall_se = self._stage2_static(
            df=df,
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

        overall_t = (
            overall_att / overall_se if np.isfinite(overall_se) and overall_se > 0 else np.nan
        )
        overall_p = compute_p_value(overall_t)
        overall_ci = (
            compute_confidence_interval(overall_att, overall_se, self.alpha)
            if np.isfinite(overall_se) and overall_se > 0
            else (np.nan, np.nan)
        )

        # Event study and group aggregation
        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._stage2_event_study(
                df=df,
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
                treatment_groups=treatment_groups,
                ref_period=ref_period,
                balance_e=balance_e,
                kept_cov_mask=kept_cov_mask,
            )

        if aggregate in ("group", "all"):
            group_effects = self._stage2_group(
                df=df,
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
                treatment_groups=treatment_groups,
                kept_cov_mask=kept_cov_mask,
            )

        # Build treatment effects DataFrame
        treated_df = df.loc[omega_1_mask, [unit, time, "_y_tilde", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_y_tilde": "tau_hat", "_rel_time": "rel_time"})
        tau_finite = treated_df["tau_hat"].notna() & np.isfinite(treated_df["tau_hat"].values)
        n_valid_te = int(tau_finite.sum())
        if n_valid_te > 0:
            treated_df["weight"] = np.where(tau_finite, 1.0 / n_valid_te, 0.0)
        else:
            treated_df["weight"] = 0.0

        # ---- Bootstrap ----
        bootstrap_results = None
        if self.n_bootstrap > 0:
            try:
                bootstrap_results = self._run_bootstrap(
                    df=df,
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
                    treatment_groups=treatment_groups,
                    ref_period=ref_period,
                    balance_e=balance_e,
                    original_att=overall_att,
                    original_event_study=event_study_effects,
                    original_group=group_effects,
                    aggregate=aggregate,
                )
            except Exception as e:
                warnings.warn(
                    f"Bootstrap failed: {e}. Skipping bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )

            if bootstrap_results is not None:
                # Update inference with bootstrap results
                overall_se = bootstrap_results.overall_att_se
                overall_t = (
                    overall_att / overall_se
                    if np.isfinite(overall_se) and overall_se > 0
                    else np.nan
                )
                overall_p = bootstrap_results.overall_att_p_value
                overall_ci = bootstrap_results.overall_att_ci

                # Update event study
                if event_study_effects and bootstrap_results.event_study_ses:
                    for h in event_study_effects:
                        if (
                            h in bootstrap_results.event_study_ses
                            and event_study_effects[h].get("n_obs", 1) > 0
                        ):
                            event_study_effects[h]["se"] = bootstrap_results.event_study_ses[h]
                            event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[
                                h
                            ]
                            event_study_effects[h]["p_value"] = (
                                bootstrap_results.event_study_p_values[h]
                            )
                            eff_val = event_study_effects[h]["effect"]
                            se_val = event_study_effects[h]["se"]
                            event_study_effects[h]["t_stat"] = (
                                eff_val / se_val if np.isfinite(se_val) and se_val > 0 else np.nan
                            )

                # Update group effects
                if group_effects and bootstrap_results.group_ses:
                    for g in group_effects:
                        if g in bootstrap_results.group_ses:
                            group_effects[g]["se"] = bootstrap_results.group_ses[g]
                            group_effects[g]["conf_int"] = bootstrap_results.group_cis[g]
                            group_effects[g]["p_value"] = bootstrap_results.group_p_values[g]
                            eff_val = group_effects[g]["effect"]
                            se_val = group_effects[g]["se"]
                            group_effects[g]["t_stat"] = (
                                eff_val / se_val if np.isfinite(se_val) and se_val > 0 else np.nan
                            )

        # Construct results
        self.results_ = TwoStageDiDResults(
            treatment_effects=treated_df,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_obs=n_omega_1,
            n_untreated_obs=n_omega_0,
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            bootstrap_results=bootstrap_results,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Stage 1: OLS on untreated observations
    # =========================================================================

    def _iterative_fe(
        self,
        y: np.ndarray,
        unit_vals: np.ndarray,
        time_vals: np.ndarray,
        idx: pd.Index,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Estimate unit and time FE via iterative alternating projection.

        Returns
        -------
        unit_fe : dict
            Mapping from unit -> unit fixed effect.
        time_fe : dict
            Mapping from time -> time fixed effect.
        """
        n = len(y)
        alpha = np.zeros(n)
        beta = np.zeros(n)

        with np.errstate(invalid="ignore", divide="ignore"):
            for iteration in range(max_iter):
                resid_after_alpha = y - alpha
                beta_new = (
                    pd.Series(resid_after_alpha, index=idx)
                    .groupby(time_vals)
                    .transform("mean")
                    .values
                )

                resid_after_beta = y - beta_new
                alpha_new = (
                    pd.Series(resid_after_beta, index=idx)
                    .groupby(unit_vals)
                    .transform("mean")
                    .values
                )

                max_change = max(
                    np.max(np.abs(alpha_new - alpha)),
                    np.max(np.abs(beta_new - beta)),
                )
                alpha = alpha_new
                beta = beta_new
                if max_change < tol:
                    break

        unit_fe = pd.Series(alpha, index=idx).groupby(unit_vals).first().to_dict()
        time_fe = pd.Series(beta, index=idx).groupby(time_vals).first().to_dict()
        return unit_fe, time_fe

    @staticmethod
    def _iterative_demean(
        vals: np.ndarray,
        unit_vals: np.ndarray,
        time_vals: np.ndarray,
        idx: pd.Index,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Demean a vector by iterative alternating projection."""
        result = vals.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(max_iter):
                time_means = (
                    pd.Series(result, index=idx).groupby(time_vals).transform("mean").values
                )
                result_after_time = result - time_means
                unit_means = (
                    pd.Series(result_after_time, index=idx)
                    .groupby(unit_vals)
                    .transform("mean")
                    .values
                )
                result_new = result_after_time - unit_means
                if np.max(np.abs(result_new - result)) < tol:
                    result = result_new
                    break
                result = result_new
        return result

    def _fit_untreated_model(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
    ) -> Tuple[
        Dict[Any, float], Dict[Any, float], float, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Stage 1: Estimate unit + time FE on untreated observations.

        Returns
        -------
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask
        """
        df_0 = df.loc[omega_0_mask]

        if covariates is None or len(covariates) == 0:
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index
            )
            return unit_fe, time_fe, 0.0, None, None

        else:
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values
            n_cov = len(covariates)

            y_dm = self._iterative_demean(y, units, times, df_0.index)
            X_dm = np.column_stack(
                [
                    self._iterative_demean(X_raw[:, j], units, times, df_0.index)
                    for j in range(n_cov)
                ]
            )

            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
            )
            delta_hat = result[0]
            kept_cov_mask = np.isfinite(delta_hat)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            y_adj = y - X_raw @ delta_hat_clean
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index)

            return unit_fe, time_fe, 0.0, delta_hat_clean, kept_cov_mask

    # =========================================================================
    # Residualization
    # =========================================================================

    def _residualize(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute residualized outcome y_tilde for ALL observations.

        y_tilde_i = y_i - mu_hat_i - eta_hat_t [- X_i @ delta_hat]
        """
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values

        # Handle missing FE (NaN for units/periods not in untreated sample)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + df[covariates].values @ delta_hat

        y_tilde = df[outcome].values - y_hat
        return y_tilde

    # =========================================================================
    # Stage 2 specifications
    # =========================================================================

    def _stage2_static(
        self,
        df: pd.DataFrame,
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
    ) -> Tuple[float, float]:
        """
        Static (simple ATT) Stage 2: OLS of y_tilde on D_it.

        Returns (att, se).
        """
        y_tilde = df["_y_tilde"].values.copy()

        # Handle NaN y_tilde (from unidentified FEs — e.g., rank condition violations)
        # Set to 0 so solve_ols doesn't reject; these obs have X_2=0 (untreated)
        # or contribute NaN treatment effects (excluded from point estimate).
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0

        D = omega_1_mask.values.astype(float)
        # Zero out treatment indicator for NaN y_tilde obs (don't count in ATT)
        D[nan_mask] = 0.0

        # X_2: treatment indicator (no intercept)
        X_2 = D.reshape(-1, 1)

        # Avoid degenerate case where all treated obs have NaN y_tilde
        if D.sum() == 0:
            return np.nan, np.nan

        # Stage 2 OLS for point estimate (discard naive SE)
        coef, residuals, _ = solve_ols(X_2, y_tilde, return_vcov=False)
        att = float(coef[0])

        # GMM sandwich variance
        eps_2 = y_tilde - X_2 @ coef  # Stage 2 residuals

        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
        )

        se = float(np.sqrt(max(V[0, 0], 0.0)))
        return att, se

    def _stage2_event_study(
        self,
        df: pd.DataFrame,
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
        treatment_groups: List[Any],
        ref_period: int,
        balance_e: Optional[int],
        kept_cov_mask: Optional[np.ndarray],
    ) -> Dict[int, Dict[str, Any]]:
        """Event study Stage 2: OLS of y_tilde on relative-time dummies."""
        y_tilde = df["_y_tilde"].values.copy()
        # Handle NaN y_tilde (unidentified FEs)
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0
        rel_times = df["_rel_time"].values
        n = len(df)

        # Get all horizons from treated observations
        treated_rel = rel_times[omega_1_mask.values]
        all_horizons = sorted(set(int(h) for h in treated_rel if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
        if balance_e is not None:
            cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
            balanced_cohorts = set()
            if all_horizons:
                max_h = max(all_horizons)
                required_range = set(range(-balance_e, max_h + 1))
                for g, horizons in cohort_rel_times.items():
                    if required_range.issubset(horizons):
                        balanced_cohorts.add(g)
            if not balanced_cohorts:
                warnings.warn(
                    f"No cohorts satisfy balance_e={balance_e} requirement. "
                    "Event study results will contain only the reference period. "
                    "Consider reducing balance_e.",
                    UserWarning,
                    stacklevel=2,
                )
                return {
                    ref_period: {
                        "effect": 0.0,
                        "se": 0.0,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "conf_int": (0.0, 0.0),
                        "n_obs": 0,
                    }
                }
            balance_mask = df[first_treat].isin(balanced_cohorts).values
        else:
            balance_mask = np.ones(n, dtype=bool)

        # Remove reference period from estimation horizons
        est_horizons = [h for h in all_horizons if h != ref_period]

        if len(est_horizons) == 0:
            # No horizons to estimate — return just reference period
            return {
                ref_period: {
                    "effect": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (0.0, 0.0),
                    "n_obs": 0,
                }
            }

        # Build Stage 2 design: one column per horizon (no intercept)
        # Never-treated obs get all-zero rows (undefined relative time -> NaN)
        # With no intercept, they contribute zero to X'_2 X_2 and X'_2 y_tilde
        horizon_to_col = {h: j for j, h in enumerate(est_horizons)}
        k = len(est_horizons)
        X_2 = np.zeros((n, k))

        for i in range(n):
            if not balance_mask[i]:
                continue
            if nan_mask[i]:
                continue  # NaN y_tilde -> don't include in event study
            h = rel_times[i]
            if np.isfinite(h):
                h_int = int(h)
                if h_int in horizon_to_col:
                    X_2[i, horizon_to_col[h_int]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(X_2, y_tilde, return_vcov=False)
        eps_2 = y_tilde - X_2 @ coef

        # GMM variance for full coefficient vector
        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
        )

        # Build results dict
        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        for h in est_horizons:
            j = horizon_to_col[h]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat = effect / se if np.isfinite(se) and se > 0 else np.nan
            p_val = compute_p_value(t_stat)
            ci = (
                compute_confidence_interval(effect, se, self.alpha)
                if np.isfinite(se) and se > 0
                else (np.nan, np.nan)
            )

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        return event_study_effects

    def _stage2_group(
        self,
        df: pd.DataFrame,
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
        treatment_groups: List[Any],
        kept_cov_mask: Optional[np.ndarray],
    ) -> Dict[Any, Dict[str, Any]]:
        """Group (cohort) Stage 2: OLS of y_tilde on cohort dummies."""
        y_tilde = df["_y_tilde"].values.copy()
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0
        n = len(df)

        # Build Stage 2 design: one column per cohort (no intercept)
        group_to_col = {g: j for j, g in enumerate(treatment_groups)}
        k = len(treatment_groups)
        X_2 = np.zeros((n, k))

        ft_vals = df[first_treat].values
        treated_mask = omega_1_mask.values
        for i in range(n):
            if treated_mask[i] and not nan_mask[i]:
                g = ft_vals[i]
                if g in group_to_col:
                    X_2[i, group_to_col[g]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(X_2, y_tilde, return_vcov=False)
        eps_2 = y_tilde - X_2 @ coef

        # GMM variance
        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
        )

        group_effects: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            j = group_to_col[g]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat = effect / se if np.isfinite(se) and se > 0 else np.nan
            p_val = compute_p_value(t_stat)
            ci = (
                compute_confidence_interval(effect, se, self.alpha)
                if np.isfinite(se) and se > 0
                else (np.nan, np.nan)
            )

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        return group_effects

    # =========================================================================
    # GMM Sandwich Variance (Butts & Gardner 2022)
    # =========================================================================

    def _compute_gmm_variance(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        delta_hat: Optional[np.ndarray],
        kept_cov_mask: Optional[np.ndarray],
        X_2: np.ndarray,
        eps_2: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Compute GMM sandwich variance (Butts & Gardner 2022).

        Matches the R `did2s` source code implementation: uses the GLOBAL
        Hessian inverse (not per-cluster) and NO finite-sample adjustments.

        The per-observation influence function is:
            IF_i = (X'_2 X_2)^{-1} [gamma_hat' x_{10i} eps_{10i} - x_{2i} eps_{2i}]

        where gamma_hat = (X'_{10} X_{10})^{-1} (X'_1 X_2) uses the GLOBAL
        cross-moment.

        The cluster-robust variance is:
            V = (X'_2 X_2)^{-1} (sum_g S_g S'_g) (X'_2 X_2)^{-1}
            S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
            c_g = X'_{10g} eps_{10g}

        Parameters
        ----------
        X_2 : np.ndarray, shape (n, k)
            Stage 2 design matrix (treatment indicators).
        eps_2 : np.ndarray, shape (n,)
            Stage 2 residuals.
        cluster_ids : np.ndarray, shape (n,)
            Cluster identifiers.

        Returns
        -------
        np.ndarray, shape (k, k)
            Variance-covariance matrix.
        """
        n = len(df)
        k = X_2.shape[1]

        # Exclude rank-deficient covariates
        cov_list = covariates
        if covariates and kept_cov_mask is not None and not np.all(kept_cov_mask):
            cov_list = [c for c, k_ in zip(covariates, kept_cov_mask) if k_]

        # Build sparse FE design matrices X_1 (all obs) and X_10 (untreated only)
        X_1_sparse, X_10_sparse, unit_to_idx, time_to_idx = self._build_fe_design(
            df, unit, time, cov_list, omega_0_mask
        )

        p = X_1_sparse.shape[1]

        # eps_10 = Y - X_10 @ gamma_hat
        # Untreated: stage 1 residual (Y - fitted). Treated: Y (X_10 rows = 0).
        # Reconstruct Y from y_tilde: Y = y_tilde + fitted_stage1
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values
        alpha_i = np.where(pd.isna(alpha_i), 0.0, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), 0.0, beta_t).astype(float)
        fitted_1 = alpha_i + beta_t
        if delta_hat is not None and cov_list:
            if kept_cov_mask is not None and not np.all(kept_cov_mask):
                fitted_1 = fitted_1 + df[cov_list].values @ delta_hat[kept_cov_mask]
            else:
                fitted_1 = fitted_1 + df[cov_list].values @ delta_hat

        y_tilde = df["_y_tilde"].values
        y_vals = y_tilde + fitted_1  # reconstruct Y

        # eps_10: for untreated, stage 1 residual; for treated, Y_i (since X_10 rows = 0)
        eps_10 = np.empty(n)
        omega_0 = omega_0_mask.values
        eps_10[omega_0] = y_vals[omega_0] - fitted_1[omega_0]  # Stage 1 residual
        eps_10[~omega_0] = y_vals[~omega_0]  # x_{10i} = 0, so eps_10 = Y

        # 1. gamma_hat = (X'_{10} X_{10})^{-1} (X'_1 X_2)  [p x k]
        XtX_10 = X_10_sparse.T @ X_10_sparse  # (p x p) sparse
        Xt1_X2 = X_1_sparse.T @ X_2  # (p x k) dense

        try:
            solve_XtX = sparse_factorized(XtX_10.tocsc())
            if Xt1_X2.ndim == 1:
                gamma_hat = solve_XtX(Xt1_X2).reshape(-1, 1)
            else:
                gamma_hat = np.column_stack(
                    [solve_XtX(Xt1_X2[:, j]) for j in range(Xt1_X2.shape[1])]
                )
        except RuntimeError:
            # Singular matrix — fall back to dense least-squares
            gamma_hat = np.linalg.lstsq(XtX_10.toarray(), Xt1_X2, rcond=None)[0]
            if gamma_hat.ndim == 1:
                gamma_hat = gamma_hat.reshape(-1, 1)

        # 2. Per-cluster Stage 1 scores: c_g = X'_{10g} eps_{10g}
        # Only untreated obs have non-zero X_10 rows
        weighted_X10 = X_10_sparse.multiply(eps_10[:, None])  # sparse element-wise

        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        G = len(unique_clusters)

        # Aggregate sparse rows by cluster using column-wise np.add.at
        weighted_X10_csc = weighted_X10.tocsc()
        c_by_cluster = np.zeros((G, p))
        for j_col in range(p):
            col_data = weighted_X10_csc.getcol(j_col).toarray().ravel()
            np.add.at(c_by_cluster[:, j_col], cluster_indices, col_data)

        # 3. Per-cluster Stage 2 scores: X'_{2g} eps_{2g}
        weighted_X2 = X_2 * eps_2[:, None]  # (n x k) dense
        s2_by_cluster = np.zeros((G, k))
        for j_col in range(k):
            np.add.at(s2_by_cluster[:, j_col], cluster_indices, weighted_X2[:, j_col])

        # 4. S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            correction = c_by_cluster @ gamma_hat  # (G x p) @ (p x k) = (G x k)
        # Replace NaN/inf from overflow (rank-deficient FE) with 0
        np.nan_to_num(correction, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        S = correction - s2_by_cluster  # (G x k)

        # 5. Meat: sum_g S_g S'_g = S' S
        with np.errstate(invalid="ignore", over="ignore"):
            meat = S.T @ S  # (k x k)

        # 6. Bread: (X'_2 X_2)^{-1}
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            XtX_2 = X_2.T @ X_2
        try:
            bread = np.linalg.solve(XtX_2, np.eye(k))
        except np.linalg.LinAlgError:
            bread = np.linalg.lstsq(XtX_2, np.eye(k), rcond=None)[0]

        # 7. V = bread @ meat @ bread
        V = bread @ meat @ bread
        return V

    def _build_fe_design(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[Any, int], Dict[Any, int]]:
        """
        Build sparse FE design matrices X_1 (all obs) and X_10 (untreated rows only).

        Column layout: [unit_0, ..., unit_{U-2}, time_0, ..., time_{T-2}, cov_1, ..., cov_C]
        (Drop first unit and first time for identification.)

        X_10 is identical to X_1 except that rows for treated observations are zeroed out.

        Returns
        -------
        X_1_sparse : sparse.csr_matrix, shape (n, p)
        X_10_sparse : sparse.csr_matrix, shape (n, p)
        unit_to_idx : dict
        time_to_idx : dict
        """
        n = len(df)
        unit_vals = df[unit].values
        time_vals = df[time].values
        omega_0 = omega_0_mask.values

        all_units = np.unique(unit_vals)
        all_times = np.unique(time_vals)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates) if covariates else 0
        n_fe_cols = (n_units - 1) + (n_times - 1)

        def _build_rows(mask=None):
            """Build sparse matrix for given observation mask."""
            # Unit dummies (drop first)
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_mask = u_indices > 0
            if mask is not None:
                u_mask = u_mask & mask

            u_rows = np.arange(n)[u_mask]
            u_cols = u_indices[u_mask] - 1

            # Time dummies (drop first)
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            if mask is not None:
                t_mask = t_mask & mask

            t_rows = np.arange(n)[t_mask]
            t_cols = (n_units - 1) + t_indices[t_mask] - 1

            rows = np.concatenate([u_rows, t_rows])
            cols = np.concatenate([u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            if n_cov > 0:
                cov_data = df[covariates].values.copy()
                if mask is not None:
                    cov_data[~mask] = 0.0
                A_cov = sparse.csr_matrix(cov_data)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        X_1 = _build_rows(mask=None)
        X_10 = _build_rows(mask=omega_0)

        return X_1, X_10, unit_to_idx, time_to_idx

    # =========================================================================
    # Bootstrap
    # =========================================================================

    def _compute_cluster_S_scores(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        delta_hat: Optional[np.ndarray],
        kept_cov_mask: Optional[np.ndarray],
        X_2: np.ndarray,
        eps_2: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-cluster S_g scores for bootstrap.

        Returns
        -------
        S : np.ndarray, shape (G, k)
            Per-cluster influence scores.
        bread : np.ndarray, shape (k, k)
            (X'_2 X_2)^{-1}.
        unique_clusters : np.ndarray
            Unique cluster identifiers.
        """
        n = len(df)
        k = X_2.shape[1]

        cov_list = covariates
        if covariates and kept_cov_mask is not None and not np.all(kept_cov_mask):
            cov_list = [c for c, k_ in zip(covariates, kept_cov_mask) if k_]

        X_1_sparse, X_10_sparse, _, _ = self._build_fe_design(
            df, unit, time, cov_list, omega_0_mask
        )
        p = X_1_sparse.shape[1]

        # Reconstruct Y and compute eps_10
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values
        alpha_i = np.where(pd.isna(alpha_i), 0.0, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), 0.0, beta_t).astype(float)
        fitted_1 = alpha_i + beta_t
        if delta_hat is not None and cov_list:
            if kept_cov_mask is not None and not np.all(kept_cov_mask):
                fitted_1 = fitted_1 + df[cov_list].values @ delta_hat[kept_cov_mask]
            else:
                fitted_1 = fitted_1 + df[cov_list].values @ delta_hat

        y_tilde = df["_y_tilde"].values
        y_vals = y_tilde + fitted_1

        eps_10 = np.empty(n)
        omega_0 = omega_0_mask.values
        eps_10[omega_0] = y_vals[omega_0] - fitted_1[omega_0]
        eps_10[~omega_0] = y_vals[~omega_0]

        # gamma_hat
        XtX_10 = X_10_sparse.T @ X_10_sparse
        Xt1_X2 = X_1_sparse.T @ X_2

        try:
            solve_XtX = sparse_factorized(XtX_10.tocsc())
            if Xt1_X2.ndim == 1:
                gamma_hat = solve_XtX(Xt1_X2).reshape(-1, 1)
            else:
                gamma_hat = np.column_stack(
                    [solve_XtX(Xt1_X2[:, j]) for j in range(Xt1_X2.shape[1])]
                )
        except RuntimeError:
            gamma_hat = np.linalg.lstsq(XtX_10.toarray(), Xt1_X2, rcond=None)[0]
            if gamma_hat.ndim == 1:
                gamma_hat = gamma_hat.reshape(-1, 1)

        # Per-cluster aggregation
        weighted_X10 = X_10_sparse.multiply(eps_10[:, None])
        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        G = len(unique_clusters)

        weighted_X10_csc = weighted_X10.tocsc()
        c_by_cluster = np.zeros((G, p))
        for j_col in range(p):
            col_data = weighted_X10_csc.getcol(j_col).toarray().ravel()
            np.add.at(c_by_cluster[:, j_col], cluster_indices, col_data)

        weighted_X2 = X_2 * eps_2[:, None]
        s2_by_cluster = np.zeros((G, k))
        for j_col in range(k):
            np.add.at(s2_by_cluster[:, j_col], cluster_indices, weighted_X2[:, j_col])

        correction = c_by_cluster @ gamma_hat
        S = correction - s2_by_cluster

        # Bread
        XtX_2 = X_2.T @ X_2
        try:
            bread = np.linalg.solve(XtX_2, np.eye(k))
        except np.linalg.LinAlgError:
            bread = np.linalg.lstsq(XtX_2, np.eye(k), rcond=None)[0]

        return S, bread, unique_clusters

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
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
        treatment_groups: List[Any],
        ref_period: int,
        balance_e: Optional[int],
        original_att: float,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
        aggregate: Optional[str],
    ) -> Optional[TwoStageBootstrapResults]:
        """Run multiplier bootstrap on GMM influence function."""
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        from diff_diff.staggered_bootstrap import _generate_bootstrap_weights_batch

        y_tilde = df["_y_tilde"].values.copy()  # .copy() to avoid mutating df column
        n = len(df)
        cluster_ids = df[cluster_var].values

        # Handle NaN y_tilde (from unidentified FEs) — matches _stage2_static logic
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0

        # --- Static specification bootstrap ---
        D = omega_1_mask.values.astype(float)  # .astype() already creates a copy
        D[nan_mask] = 0.0  # Exclude NaN y_tilde obs from bootstrap estimation

        # Degenerate case: all treated obs have NaN y_tilde
        if D.sum() == 0:
            return None

        X_2_static = D.reshape(-1, 1)
        coef_static = solve_ols(X_2_static, y_tilde, return_vcov=False)[0]
        eps_2_static = y_tilde - X_2_static @ coef_static

        S_static, bread_static, unique_clusters = self._compute_cluster_S_scores(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2_static,
            eps_2=eps_2_static,
            cluster_ids=cluster_ids,
        )

        n_clusters = len(unique_clusters)
        all_weights = _generate_bootstrap_weights_batch(
            self.n_bootstrap, n_clusters, "rademacher", rng
        )

        # T_b = bread @ (sum_g w_bg * S_g) = bread @ (W @ S)'  per boot
        # IF_b = bread @ S_g for each cluster, then perturb
        # boot_coef = all_weights @ S_static @ bread_static.T  → (B, k)
        # For static (k=1): boot_att = all_weights @ S_static @ bread_static.T
        boot_att_vec = all_weights @ S_static  # (B, 1)
        boot_att_vec = boot_att_vec @ bread_static.T  # (B, 1)
        boot_overall = boot_att_vec[:, 0]

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

        # --- Event study bootstrap ---
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if original_event_study and aggregate in ("event_study", "all"):
            # Recompute S scores for event study specification
            rel_times = df["_rel_time"].values
            treated_rel = rel_times[omega_1_mask.values]
            all_horizons = sorted(set(int(h) for h in treated_rel if np.isfinite(h)))
            if self.horizon_max is not None:
                all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

            if balance_e is not None:
                cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
                balanced_cohorts = set()
                if all_horizons:
                    max_h = max(all_horizons)
                    required_range = set(range(-balance_e, max_h + 1))
                    for g, horizons in cohort_rel_times.items():
                        if required_range.issubset(horizons):
                            balanced_cohorts.add(g)
                if not balanced_cohorts:
                    all_horizons = []  # No qualifying cohorts -> skip event study bootstrap
                else:
                    balance_mask = df[first_treat].isin(balanced_cohorts).values
            else:
                balance_mask = np.ones(n, dtype=bool)

            est_horizons = [h for h in all_horizons if h != ref_period]
            if est_horizons:
                horizon_to_col = {h: j for j, h in enumerate(est_horizons)}
                k_es = len(est_horizons)
                X_2_es = np.zeros((n, k_es))
                for i in range(n):
                    if not balance_mask[i]:
                        continue
                    if nan_mask[i]:
                        continue  # NaN y_tilde -> exclude from bootstrap event study
                    h = rel_times[i]
                    if np.isfinite(h):
                        h_int = int(h)
                        if h_int in horizon_to_col:
                            X_2_es[i, horizon_to_col[h_int]] = 1.0

                coef_es = solve_ols(X_2_es, y_tilde, return_vcov=False)[0]
                eps_2_es = y_tilde - X_2_es @ coef_es

                S_es, bread_es, _ = self._compute_cluster_S_scores(
                    df=df,
                    unit=unit,
                    time=time,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    delta_hat=delta_hat,
                    kept_cov_mask=kept_cov_mask,
                    X_2=X_2_es,
                    eps_2=eps_2_es,
                    cluster_ids=cluster_ids,
                )

                # boot_coef_es: (B, k_es)
                boot_coef_es = (all_weights @ S_es) @ bread_es.T

                event_study_ses = {}
                event_study_cis = {}
                event_study_p_values = {}
                for h in original_event_study:
                    if original_event_study[h].get("n_obs", 0) == 0:
                        continue
                    if h not in horizon_to_col:
                        continue
                    j = horizon_to_col[h]
                    orig_eff = original_event_study[h]["effect"]
                    boot_h = boot_coef_es[:, j]
                    se_h = float(np.std(boot_h, ddof=1))
                    event_study_ses[h] = se_h
                    if se_h > 0 and np.isfinite(orig_eff):
                        shifted_h = boot_h + orig_eff
                        event_study_p_values[h] = self._compute_bootstrap_pvalue(
                            orig_eff, shifted_h
                        )
                        event_study_cis[h] = self._compute_percentile_ci(shifted_h, self.alpha)
                    else:
                        event_study_p_values[h] = np.nan
                        event_study_cis[h] = (np.nan, np.nan)

        # --- Group bootstrap ---
        group_ses = None
        group_cis = None
        group_p_values = None

        if original_group and aggregate in ("group", "all"):
            group_to_col = {g: j for j, g in enumerate(treatment_groups)}
            k_grp = len(treatment_groups)
            X_2_grp = np.zeros((n, k_grp))
            ft_vals = df[first_treat].values
            treated_mask = omega_1_mask.values
            for i in range(n):
                if treated_mask[i]:
                    if nan_mask[i]:
                        continue  # NaN y_tilde -> exclude from group bootstrap
                    g = ft_vals[i]
                    if g in group_to_col:
                        X_2_grp[i, group_to_col[g]] = 1.0

            coef_grp = solve_ols(X_2_grp, y_tilde, return_vcov=False)[0]
            eps_2_grp = y_tilde - X_2_grp @ coef_grp

            S_grp, bread_grp, _ = self._compute_cluster_S_scores(
                df=df,
                unit=unit,
                time=time,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                delta_hat=delta_hat,
                kept_cov_mask=kept_cov_mask,
                X_2=X_2_grp,
                eps_2=eps_2_grp,
                cluster_ids=cluster_ids,
            )

            boot_coef_grp = (all_weights @ S_grp) @ bread_grp.T

            group_ses = {}
            group_cis = {}
            group_p_values = {}
            for g in original_group:
                if g not in group_to_col:
                    continue
                j = group_to_col[g]
                orig_eff = original_group[g]["effect"]
                boot_g = boot_coef_grp[:, j]
                se_g = float(np.std(boot_g, ddof=1))
                group_ses[g] = se_g
                if se_g > 0 and np.isfinite(orig_eff):
                    shifted_g = boot_g + orig_eff
                    group_p_values[g] = self._compute_bootstrap_pvalue(orig_eff, shifted_g)
                    group_cis[g] = self._compute_percentile_ci(shifted_g, self.alpha)
                else:
                    group_p_values[g] = np.nan
                    group_cis[g] = (np.nan, np.nan)

        return TwoStageBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type="rademacher",
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

    # =========================================================================
    # Bootstrap helpers
    # =========================================================================

    @staticmethod
    def _compute_percentile_ci(
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        lower = float(np.percentile(boot_dist, alpha / 2 * 100))
        upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return (lower, upper)

    @staticmethod
    def _compute_bootstrap_pvalue(
        original_effect: float,
        boot_dist: np.ndarray,
    ) -> float:
        """Compute two-sided bootstrap p-value."""
        if original_effect >= 0:
            p_one_sided = float(np.mean(boot_dist <= 0))
        else:
            p_one_sided = float(np.mean(boot_dist >= 0))
        p_value = min(2 * p_one_sided, 1.0)
        p_value = max(p_value, 1 / (len(boot_dist) + 1))
        return p_value

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _build_cohort_rel_times(
        df: pd.DataFrame,
        first_treat: str,
    ) -> Dict[Any, Set[int]]:
        """Build mapping of cohort -> set of observed relative times."""
        treated_mask = ~df["_never_treated"]
        treated_df = df.loc[treated_mask]
        result: Dict[Any, Set[int]] = {}
        ft_vals = treated_df[first_treat].values
        rt_vals = treated_df["_rel_time"].values
        for i in range(len(treated_df)):
            h = rt_vals[i]
            if np.isfinite(h):
                result.setdefault(ft_vals[i], set()).add(int(h))
        return result

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "horizon_max": self.horizon_max,
        }

    def set_params(self, **params) -> "TwoStageDiD":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


# =============================================================================
# Convenience function
# =============================================================================


def two_stage_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]] = None,
    aggregate: Optional[str] = None,
    balance_e: Optional[int] = None,
    **kwargs,
) -> TwoStageDiDResults:
    """
    Convenience function for two-stage DiD estimation.

    This is a shortcut for creating a TwoStageDiD estimator and calling fit().

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    first_treat : str
        Column indicating first treatment period (0 for never-treated).
    covariates : list of str, optional
        Covariate column names.
    aggregate : str, optional
        Aggregation mode: None, "simple", "event_study", "group", "all".
    balance_e : int, optional
        Balance event study to cohorts observed at all relative times.
    **kwargs
        Additional keyword arguments passed to TwoStageDiD constructor.

    Returns
    -------
    TwoStageDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import two_stage_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = two_stage_did(data, 'outcome', 'unit', 'period',
    ...                         'first_treat', aggregate='event_study')
    >>> results.print_summary()
    """
    est = TwoStageDiD(**kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        covariates=covariates,
        aggregate=aggregate,
        balance_e=balance_e,
    )
