"""
Borusyak-Jaravel-Spiess (2024) Imputation DiD Estimator.

Implements the efficient imputation estimator for staggered
Difference-in-Differences from Borusyak, Jaravel & Spiess (2024),
"Revisiting Event-Study Designs: Robust and Efficient Estimation",
Review of Economic Studies.

The estimator:
1. Runs OLS on untreated observations to estimate unit + time fixed effects
2. Imputes counterfactual Y(0) for treated observations
3. Aggregates imputed treatment effects with researcher-chosen weights

Inference uses the conservative clustered variance estimator (Theorem 3).
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from diff_diff.linalg import solve_ols
from diff_diff.results import _get_significance_stars
from diff_diff.utils import compute_confidence_interval, compute_p_value


# =============================================================================
# Results Dataclasses
# =============================================================================


@dataclass
class ImputationBootstrapResults:
    """
    Results from ImputationDiD bootstrap inference.

    Bootstrap is a library extension beyond Borusyak et al. (2024), which
    proposes only analytical inference via the conservative variance estimator.
    Provided for consistency with CallawaySantAnna and SunAbraham.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights ("rademacher", "mammen", "webb").
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
class ImputationDiDResults:
    """
    Results from Borusyak-Jaravel-Spiess (2024) imputation DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Unit-level treatment effects with columns: unit, time, tau_hat, weight.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT.
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
        Number of treated observations (|Omega_1|).
    n_untreated_obs : int
        Number of untreated observations (|Omega_0|).
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to Omega_0.
    alpha : float
        Significance level used.
    pretrend_results : dict, optional
        Populated by pretrend_test().
    bootstrap_results : ImputationBootstrapResults, optional
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
    pretrend_results: Optional[Dict[str, Any]] = field(default=None, repr=False)
    bootstrap_results: Optional[ImputationBootstrapResults] = field(default=None, repr=False)
    # Internal: stores data needed for pretrend_test()
    _estimator_ref: Optional[Any] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"ImputationDiDResults(ATT={self.overall_att:.4f}{sig}, "
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
            "Imputation DiD Estimator Results (Borusyak et al. 2024)".center(85),
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

        # Pre-trend test
        if self.pretrend_results is not None:
            pt = self.pretrend_results
            lines.extend(
                [
                    "-" * 85,
                    "Pre-Trend Test (Equation 9)".center(85),
                    "-" * 85,
                    f"{'F-statistic:':<30} {pt['f_stat']:>10.3f}",
                    f"{'P-value:':<30} {pt['p_value']:>10.4f}",
                    f"{'Degrees of freedom:':<30} {pt['df']:>10}",
                    f"{'Number of leads:':<30} {pt['n_leads']:>10}",
                    "-" * 85,
                    "",
                ]
            )

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

    def to_dataframe(self, level: str = "observation") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="observation"
            Level of aggregation:
            - "observation": Unit-level treatment effects
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects

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
                f"Unknown level: {level}. Use 'observation', 'event_study', or 'group'."
            )

    def pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a pre-trend test (Equation 9 of Borusyak et al. 2024).

        Adds pre-treatment lead indicators to the Step 1 OLS and tests
        their joint significance via a cluster-robust Wald F-test.

        Parameters
        ----------
        n_leads : int, optional
            Number of pre-treatment leads to include. If None, uses all
            available pre-treatment periods minus one (for the reference period).

        Returns
        -------
        dict
            Dictionary with keys: 'f_stat', 'p_value', 'df', 'n_leads',
            'lead_coefficients'.
        """
        if self._estimator_ref is None:
            raise RuntimeError(
                "Pre-trend test requires internal estimator reference. "
                "Re-fit the model to use this method."
            )
        result = self._estimator_ref._pretrend_test(n_leads=n_leads)
        self.pretrend_results = result
        return result

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


class ImputationDiD:
    """
    Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.

    This is the efficient estimator for staggered Difference-in-Differences
    under parallel trends. It produces shorter confidence intervals than
    Callaway-Sant'Anna (~50% shorter) and Sun-Abraham (2-3.5x shorter)
    under homogeneous treatment effects.

    The estimation procedure:
    1. Run OLS on untreated observations to estimate unit + time fixed effects
    2. Impute counterfactual Y(0) for treated observations
    3. Aggregate imputed treatment effects with researcher-chosen weights

    Inference uses the conservative clustered variance estimator from Theorem 3
    of the paper.

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
        Number of bootstrap iterations. If 0, uses analytical inference
        (conservative variance from Theorem 3).
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
    aux_partition : str, default="cohort_horizon"
        Controls the auxiliary model partition for Theorem 3 variance:
        - "cohort_horizon": Groups by cohort x relative time (tightest SEs)
        - "cohort": Groups by cohort only (more conservative)
        - "horizon": Groups by relative time only (more conservative)

    Attributes
    ----------
    results_ : ImputationDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import ImputationDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat')
    >>> results.print_summary()

    With event study:

    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat',
    ...                   aggregate='event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The imputation estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model. There is no ``control_group`` parameter because this
    is fundamental to the method's efficiency.

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study
    Designs: Robust and Efficient Estimation. Review of Economic Studies,
    91(6), 3253-3285.
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
        aux_partition: str = "cohort_horizon",
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if aux_partition not in ("cohort_horizon", "cohort", "horizon"):
            raise ValueError(
                f"aux_partition must be 'cohort_horizon', 'cohort', or 'horizon', "
                f"got '{aux_partition}'"
            )

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max
        self.aux_partition = aux_partition

        self.is_fitted_ = False
        self.results_: Optional[ImputationDiDResults] = None

        # Internal state preserved for pretrend_test()
        self._fit_data: Optional[Dict[str, Any]] = None

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
    ) -> ImputationDiDResults:
        """
        Fit the imputation DiD estimator.

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
        ImputationDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units (treated in all observed periods)
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        n_always_treated = df.loc[always_treated_mask, unit].nunique()
        if n_always_treated > 0:
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}). These units have no untreated "
                "observations and cannot contribute to the counterfactual model. "
                "Their treatment effects will be imputed but may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Create treatment indicator D_it
        # D_it = 1 if t >= first_treat and first_treat > 0
        # With anticipation: D_it = 1 if t >= first_treat - anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Identify Omega_0 (untreated) and Omega_1 (treated)
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

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
        # Control units = units with at least one untreated observation
        units_in_omega_0 = df.loc[omega_0_mask, unit].unique()
        n_control_units = len(units_in_omega_0)

        # Cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit
        if self.cluster is not None and self.cluster not in df.columns:
            raise ValueError(
                f"Cluster column '{self.cluster}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Compute relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Step 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask
        )

        # ---- Rank condition checks ----
        # Check: every treated unit should have >= 1 untreated period (for unit FE)
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

        # Check: every post-treatment period should have >= 1 untreated unit (for time FE)
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
            # "silent": continue without warning

        # ---- Step 2: Impute treatment effects ----
        tau_hat, y_hat_0 = self._impute_treatment_effects(
            df,
            outcome,
            unit,
            time,
            covariates,
            omega_1_mask,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
        )

        # Store tau_hat in dataframe
        df["_tau_hat"] = np.nan
        df.loc[omega_1_mask, "_tau_hat"] = tau_hat

        # ---- Step 3: Aggregate ----
        # Always compute overall ATT (simple aggregation)
        n_treated = int(omega_1_mask.sum())
        valid_tau = tau_hat[np.isfinite(tau_hat)]

        if len(valid_tau) == 0:
            overall_att = np.nan
        else:
            overall_att = float(np.mean(valid_tau))

        # ---- Conservative variance (Theorem 3) ----
        # Build weights matching the ATT: uniform over finite tau_hat, zero for NaN
        overall_weights = np.zeros(n_omega_1)
        finite_mask = np.isfinite(tau_hat)
        n_valid = int(finite_mask.sum())
        if n_valid > 0:
            overall_weights[finite_mask] = 1.0 / n_valid

        if n_valid == 0:
            overall_se = np.nan
        else:
            overall_se = self._compute_conservative_variance(
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
                weights=overall_weights,
                cluster_var=cluster_var,
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
            event_study_effects = self._aggregate_event_study(
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
                treatment_groups=treatment_groups,
                balance_e=balance_e,
            )

        if aggregate in ("group", "all"):
            group_effects = self._aggregate_group(
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
                treatment_groups=treatment_groups,
            )

        # Build treatment effects dataframe
        treated_df = df.loc[omega_1_mask, [unit, time, "_tau_hat", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_tau_hat": "tau_hat", "_rel_time": "rel_time"})
        treated_df["weight"] = 1.0 / n_treated

        # Store fit data for pretrend_test
        self._fit_data = {
            "df": df,
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "first_treat": first_treat,
            "covariates": covariates,
            "omega_0_mask": omega_0_mask,
            "omega_1_mask": omega_1_mask,
            "cluster_var": cluster_var,
            "unit_fe": unit_fe,
            "time_fe": time_fe,
            "grand_mean": grand_mean,
            "delta_hat": delta_hat,
        }

        # Bootstrap
        bootstrap_results = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_bootstrap(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                aggregate=aggregate,
                balance_e=balance_e,
                original_att=overall_att,
                original_event_study=event_study_effects,
                original_group=group_effects,
            )

            # Update inference with bootstrap results
            overall_se = bootstrap_results.overall_att_se
            overall_t = (
                overall_att / overall_se if np.isfinite(overall_se) and overall_se > 0 else np.nan
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
                        event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[h]
                        event_study_effects[h]["p_value"] = bootstrap_results.event_study_p_values[
                            h
                        ]
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
        self.results_ = ImputationDiDResults(
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
            _estimator_ref=self,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Step 1: OLS on untreated observations
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
        Estimate unit and time FE via iterative alternating projection (Gauss-Seidel).

        Converges to the exact OLS solution for both balanced and unbalanced panels.
        For balanced panels, converges in 1-2 iterations (identical to one-pass).
        For unbalanced panels, typically 5-20 iterations.

        Returns
        -------
        unit_fe : dict
            Mapping from unit -> unit fixed effect.
        time_fe : dict
            Mapping from time -> time fixed effect.
        """
        n = len(y)
        alpha = np.zeros(n)  # unit FE broadcast to obs level
        beta = np.zeros(n)  # time FE broadcast to obs level

        with np.errstate(invalid="ignore", divide="ignore"):
            for iteration in range(max_iter):
                # Update time FE: beta_t = mean_i(y_it - alpha_i)
                resid_after_alpha = y - alpha
                beta_new = (
                    pd.Series(resid_after_alpha, index=idx)
                    .groupby(time_vals)
                    .transform("mean")
                    .values
                )

                # Update unit FE: alpha_i = mean_t(y_it - beta_t)
                resid_after_beta = y - beta_new
                alpha_new = (
                    pd.Series(resid_after_beta, index=idx)
                    .groupby(unit_vals)
                    .transform("mean")
                    .values
                )

                # Check convergence on FE changes
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
        """Demean a vector by iterative alternating projection (unit + time FE removal).

        Converges to the exact within-transformation for both balanced and
        unbalanced panels. For balanced panels, converges in 1-2 iterations.
        """
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

    @staticmethod
    def _compute_balanced_cohort_mask(
        df_treated: pd.DataFrame,
        first_treat: str,
        rel_times: np.ndarray,
        all_horizons: List[int],
        balance_e: int,
    ) -> np.ndarray:
        """Compute boolean mask selecting treated obs from balanced cohorts.

        A cohort is 'balanced' if it has observations at every relative time
        in [-balance_e, max(all_horizons)].

        Returns an array of shape (len(df_treated),) with True for obs in
        balanced cohorts.
        """
        if not all_horizons:
            return np.ones(len(df_treated), dtype=bool)

        max_h = max(all_horizons)
        required_range = set(h for h in all_horizons if -balance_e <= h <= max_h)

        cohort_horizons: Dict[Any, set] = {}
        first_treat_vals = df_treated[first_treat].values
        for i in range(len(df_treated)):
            g = first_treat_vals[i]
            h = int(rel_times[i]) if np.isfinite(rel_times[i]) else None
            if h is not None:
                cohort_horizons.setdefault(g, set()).add(h)

        balanced_cohorts = set()
        for g, horizons in cohort_horizons.items():
            if required_range.issubset(horizons):
                balanced_cohorts.add(g)

        return df_treated[first_treat].isin(balanced_cohorts).values

    def _fit_untreated_model(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
    ) -> Tuple[Dict[Any, float], Dict[Any, float], float, Optional[np.ndarray]]:
        """
        Step 1: Estimate unit + time FE on untreated observations.

        Uses iterative alternating projection (Gauss-Seidel) to compute exact
        OLS fixed effects for both balanced and unbalanced panels. For balanced
        panels, converges in 1-2 iterations (identical to one-pass demeaning).

        Returns
        -------
        unit_fe : dict
            Unit fixed effects {unit_id: alpha_i}.
        time_fe : dict
            Time fixed effects {time_period: beta_t}.
        grand_mean : float
            Grand mean (0.0 — absorbed into iterative FE).
        delta_hat : np.ndarray or None
            Covariate coefficients (if covariates provided).
        """
        df_0 = df.loc[omega_0_mask]

        if covariates is None or len(covariates) == 0:
            # No covariates: estimate FE via iterative alternating projection
            # (exact OLS for both balanced and unbalanced panels)
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index
            )
            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, None

        else:
            # With covariates: iteratively demean Y and X, OLS for delta,
            # then recover FE from covariate-adjusted outcome
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values
            n_cov = len(covariates)

            # Step A: Iteratively demean Y and all X columns to remove unit+time FE
            y_dm = self._iterative_demean(y, units, times, df_0.index)
            X_dm = np.column_stack(
                [
                    self._iterative_demean(X_raw[:, j], units, times, df_0.index)
                    for j in range(n_cov)
                ]
            )

            # Step B: OLS for covariate coefficients on demeaned data
            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
            )
            delta_hat = result[0]

            # Replace NaN coefficients with 0 for adjustment
            # (rank-deficient covariates are dropped)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            # Step C: Recover FE from covariate-adjusted outcome using iterative FE
            y_adj = y - X_raw @ delta_hat_clean
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index)

            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, delta_hat_clean

    # =========================================================================
    # Step 2: Impute counterfactuals
    # =========================================================================

    def _impute_treatment_effects(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Impute Y(0) for treated observations and compute tau_hat.

        Returns
        -------
        tau_hat : np.ndarray
            Imputed treatment effects for each treated observation.
        y_hat_0 : np.ndarray
            Imputed counterfactual Y(0).
        """
        df_1 = df.loc[omega_1_mask]
        n_1 = len(df_1)

        # Look up unit and time FE
        alpha_i = df_1[unit].map(unit_fe).values
        beta_t = df_1[time].map(time_fe).values

        # Handle missing FE (set to NaN)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            X_1 = df_1[covariates].values
            y_hat_0 = y_hat_0 + X_1 @ delta_hat

        tau_hat = df_1[outcome].values - y_hat_0

        return tau_hat, y_hat_0

    # =========================================================================
    # Conservative Variance (Theorem 3)
    # =========================================================================

    def _compute_conservative_variance(
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
        weights: np.ndarray,
        cluster_var: str,
    ) -> float:
        """
        Compute conservative clustered variance (Theorem 3, Equation 7).

        Parameters
        ----------
        weights : np.ndarray
            Aggregation weights w_it for treated observations.
            Shape: (n_treated,), must sum to 1.

        Returns
        -------
        float
            Standard error.
        """
        df_0 = df.loc[omega_0_mask]
        df_1 = df.loc[omega_1_mask]
        n_0 = len(df_0)
        n_1 = len(df_1)

        # ---- Compute v_it for treated observations ----
        # v_it = w_it for treated obs
        v_treated = weights.copy()

        # ---- Compute v_it for untreated observations ----
        if covariates is None or len(covariates) == 0:
            # FE-only case: closed-form
            # v_it^0 = -(w_i. / n_{0,i} + w_.t / n_{0,t} - w.. / N_0)
            # where w_i. = sum of w over treated obs of unit i
            #       w_.t = sum of w over treated obs at time t
            #       w.. = sum of all w = 1

            # Compute w_i. and w_.t from treated observations
            treated_units = df_1[unit].values
            treated_times = df_1[time].values

            w_by_unit: Dict[Any, float] = {}
            for i_idx in range(n_1):
                u = treated_units[i_idx]
                w_by_unit[u] = w_by_unit.get(u, 0.0) + weights[i_idx]

            w_by_time: Dict[Any, float] = {}
            for i_idx in range(n_1):
                t = treated_times[i_idx]
                w_by_time[t] = w_by_time.get(t, 0.0) + weights[i_idx]

            w_total = float(np.sum(weights))  # Should be ~1

            # Count untreated obs per unit and per time
            n0_by_unit = df_0.groupby(unit).size().to_dict()
            n0_by_time = df_0.groupby(time).size().to_dict()

            # Compute v for each untreated observation
            untreated_units = df_0[unit].values
            untreated_times = df_0[time].values
            v_untreated = np.zeros(n_0)

            for j in range(n_0):
                u = untreated_units[j]
                t = untreated_times[j]
                w_i = w_by_unit.get(u, 0.0)
                w_t = w_by_time.get(t, 0.0)
                n0_i = n0_by_unit.get(u, 1)
                n0_t = n0_by_time.get(t, 1)
                v_untreated[j] = -(w_i / n0_i + w_t / n0_t - w_total / n_0)
        else:
            # With covariates: use projection matrix approach
            v_untreated = self._compute_v_untreated_with_covariates(
                df_0, df_1, unit, time, covariates, weights, delta_hat
            )

        # ---- Compute auxiliary model residuals (Equation 8) ----
        epsilon_treated = self._compute_auxiliary_residuals_treated(
            df_1,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
            v_treated,
        )
        epsilon_untreated = self._compute_residuals_untreated(
            df_0, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )

        # ---- Clustered variance: sigma^2 = sum_i (sum_t v_it * eps_it)^2 ----
        # Combine v and epsilon for all observations
        v_all = np.empty(len(df))
        v_all[omega_1_mask.values] = v_treated
        v_all[omega_0_mask.values] = v_untreated

        eps_all = np.empty(len(df))
        eps_all[omega_1_mask.values] = epsilon_treated
        eps_all[omega_0_mask.values] = epsilon_untreated

        ve_product = v_all * eps_all
        # NaN eps from missing FE (rank condition violation). Zero their variance
        # contribution — matches R's did_imputation which drops unimputable obs.
        np.nan_to_num(ve_product, copy=False, nan=0.0)

        # Sum within clusters
        cluster_ids = df[cluster_var].values
        ve_series = pd.Series(ve_product, index=df.index)
        cluster_sums = ve_series.groupby(cluster_ids).sum()

        sigma_sq = float((cluster_sums**2).sum())
        se = np.sqrt(max(sigma_sq, 0.0))

        return se

    def _compute_v_untreated_with_covariates(
        self,
        df_0: pd.DataFrame,
        df_1: pd.DataFrame,
        unit: str,
        time: str,
        covariates: List[str],
        weights: np.ndarray,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute v_it for untreated observations with covariates.

        Uses the projection: v_untreated = -A_0 (A_0'A_0)^{-1} A_1' w_treated

        Uses scipy.sparse for FE dummy columns to reduce memory from O(N*(U+T))
        to O(N) for the FE portion.
        """
        units_0 = df_0[unit].values
        times_0 = df_0[time].values
        units_1 = df_1[unit].values
        times_1 = df_1[time].values

        all_units = np.unique(np.concatenate([units_0, units_1]))
        all_times = np.unique(np.concatenate([times_0, times_1]))
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates)
        n_fe_cols = (n_units - 1) + (n_times - 1)

        def _build_A_sparse(df_sub, unit_vals, time_vals):
            n = len(df_sub)

            # Unit dummies (drop first) — vectorized
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_mask = u_indices > 0  # skip first unit (dropped)
            u_rows = np.arange(n)[u_mask]
            u_cols = u_indices[u_mask] - 1

            # Time dummies (drop first) — vectorized
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            t_rows = np.arange(n)[t_mask]
            t_cols = (n_units - 1) + t_indices[t_mask] - 1

            rows = np.concatenate([u_rows, t_rows])
            cols = np.concatenate([u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            # Covariates (dense, typically few columns)
            if n_cov > 0:
                A_cov = sparse.csr_matrix(df_sub[covariates].values)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        A_0 = _build_A_sparse(df_0, units_0, times_0)
        A_1 = _build_A_sparse(df_1, units_1, times_1)

        # Compute A_1' w (sparse.T @ dense -> dense)
        A1_w = A_1.T @ weights  # shape (p,)

        # Solve (A_0'A_0) z = A_1' w
        # A_0'A_0 is (p x p) where p = n_fe_cols + n_cov — small enough to be dense
        A0tA0 = (A_0.T @ A_0).toarray()
        try:
            z = np.linalg.solve(A0tA0, A1_w)
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            z, _, _, _ = np.linalg.lstsq(A0tA0, A1_w, rcond=None)

        # v_untreated = -A_0 z (sparse @ dense -> dense)
        v_untreated = -(A_0 @ z)
        return v_untreated

    def _compute_auxiliary_residuals_treated(
        self,
        df_1: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        v_treated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute v_it-weighted auxiliary residuals for treated obs (Equation 8).

        Computes v_it-weighted tau_tilde_g per Equation 8 of Borusyak et al. (2024):
        tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it) within group g.

        epsilon_tilde_it = Y_it - alpha_i - beta_t [- X'delta] - tau_tilde_g
        """
        n_1 = len(df_1)

        # Compute base residuals (Y - Y_hat(0) = tau_hat)
        # NaN for missing FE (consistent with _impute_treatment_effects)
        alpha_i = df_1[unit].map(unit_fe).values.astype(float)  # NaN for missing
        beta_t = df_1[time].map(time_fe).values.astype(float)  # NaN for missing
        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat_0 = y_hat_0 + df_1[covariates].values @ delta_hat

        tau_hat = df_1[outcome].values - y_hat_0

        # Partition Omega_1 and compute tau_tilde for each group
        if self.aux_partition == "cohort_horizon":
            group_keys = list(zip(df_1[first_treat].values, df_1["_rel_time"].values))
        elif self.aux_partition == "cohort":
            group_keys = list(df_1[first_treat].values)
        elif self.aux_partition == "horizon":
            group_keys = list(df_1["_rel_time"].values)
        else:
            group_keys = list(range(n_1))  # each obs is its own group

        # Compute v_it-weighted average tau within each partition group (Equation 8)
        # tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it) within group g
        group_series = pd.Series(group_keys, index=df_1.index)
        tau_series = pd.Series(tau_hat, index=df_1.index)
        v_series = pd.Series(v_treated, index=df_1.index)

        weighted_tau_sum = (v_series * tau_series).groupby(group_series).sum()
        weight_sum = v_series.groupby(group_series).sum()

        # Guard: zero-weight groups -> their tau_tilde doesn't affect variance
        # (v_it ~ 0 means these obs contribute nothing to the estimand)
        # Use simple mean as fallback. This is common for event-study SE computation
        # where weights target a specific horizon, making other partition groups zero.
        zero_weight_groups = weight_sum.abs() < 1e-15
        if zero_weight_groups.any():
            simple_means = tau_series.groupby(group_series).mean()
            tau_tilde_map = weighted_tau_sum / weight_sum
            tau_tilde_map = tau_tilde_map.where(~zero_weight_groups, simple_means)
        else:
            tau_tilde_map = weighted_tau_sum / weight_sum

        tau_tilde = group_series.map(tau_tilde_map).values

        # Auxiliary residuals
        epsilon_treated = tau_hat - tau_tilde

        return epsilon_treated

    def _compute_residuals_untreated(
        self,
        df_0: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute Step 1 residuals for untreated observations."""
        alpha_i = df_0[unit].map(unit_fe).fillna(0.0).values
        beta_t = df_0[time].map(time_fe).fillna(0.0).values
        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + df_0[covariates].values @ delta_hat

        return df_0[outcome].values - y_hat

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _aggregate_event_study(
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
        treatment_groups: List[Any],
        balance_e: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate treatment effects by event-study horizon."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        rel_times = df_1["_rel_time"].values

        # Get all horizons
        all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
        if balance_e is not None:
            balanced_mask = pd.Series(
                self._compute_balanced_cohort_mask(
                    df_1, first_treat, rel_times, all_horizons, balance_e
                ),
                index=df_1.index,
            )
        else:
            balanced_mask = pd.Series(True, index=df_1.index)

        # Check Proposition 5: no never-treated units
        has_never_treated = df["_never_treated"].any()
        h_bar = np.inf
        if not has_never_treated and len(treatment_groups) > 1:
            h_bar = max(treatment_groups) - min(treatment_groups)

        # Reference period
        ref_period = -1 - self.anticipation

        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Add reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        # Collect horizons with Proposition 5 violations
        prop5_horizons = []

        for h in all_horizons:
            if h == ref_period:
                continue

            # Select treated obs at this horizon from balanced cohorts
            h_mask = (rel_times == h) & balanced_mask.values
            n_h = int(h_mask.sum())

            if n_h == 0:
                continue

            # Proposition 5 check
            if not has_never_treated and h >= h_bar:
                prop5_horizons.append(h)
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            tau_h = tau_hat[h_mask]
            valid_tau = tau_h[np.isfinite(tau_h)]

            if len(valid_tau) == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            effect = float(np.mean(valid_tau))

            # Compute SE via conservative variance with horizon-specific weights
            weights_h = np.zeros(int(omega_1_mask.sum()))
            omega_1_indices = np.where(omega_1_mask.values)[0]
            # Map h_mask (relative to df_1) to weights array
            h_indices_in_omega1 = np.where(h_mask)[0]
            n_valid = len(valid_tau)
            # Only weight valid (finite) observations
            finite_mask = np.isfinite(tau_hat[h_mask])
            valid_h_indices = h_indices_in_omega1[finite_mask]
            for idx in valid_h_indices:
                weights_h[idx] = 1.0 / n_valid

            se = self._compute_conservative_variance(
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
                weights=weights_h,
                cluster_var=cluster_var,
            )

            t_stat = effect / se if np.isfinite(se) and se > 0 else np.nan
            p_value = compute_p_value(t_stat)
            conf_int = (
                compute_confidence_interval(effect, se, self.alpha)
                if np.isfinite(se) and se > 0
                else (np.nan, np.nan)
            )

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_h,
            }

        # Proposition 5 warning
        if prop5_horizons:
            warnings.warn(
                f"Horizons {prop5_horizons} are not identified without "
                f"never-treated units (Proposition 5). Set to NaN.",
                UserWarning,
                stacklevel=3,
            )

        return event_study_effects

    def _aggregate_group(
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
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate treatment effects by cohort."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        cohorts = df_1[first_treat].values

        group_effects: Dict[Any, Dict[str, Any]] = {}

        for g in treatment_groups:
            g_mask = cohorts == g
            n_g = int(g_mask.sum())

            if n_g == 0:
                continue

            tau_g = tau_hat[g_mask]
            valid_tau = tau_g[np.isfinite(tau_g)]

            if len(valid_tau) == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_g,
                }
                continue

            effect = float(np.mean(valid_tau))

            # Compute SE with group-specific weights
            weights_g = np.zeros(int(omega_1_mask.sum()))
            finite_mask = np.isfinite(tau_hat) & g_mask
            g_indices = np.where(finite_mask)[0]
            n_valid = len(valid_tau)
            for idx in g_indices:
                weights_g[idx] = 1.0 / n_valid

            se = self._compute_conservative_variance(
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
                weights=weights_g,
                cluster_var=cluster_var,
            )

            t_stat = effect / se if np.isfinite(se) and se > 0 else np.nan
            p_value = compute_p_value(t_stat)
            conf_int = (
                compute_confidence_interval(effect, se, self.alpha)
                if np.isfinite(se) and se > 0
                else (np.nan, np.nan)
            )

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_g,
            }

        return group_effects

    # =========================================================================
    # Pre-trend test (Equation 9)
    # =========================================================================

    def _pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run pre-trend test (Equation 9).

        Adds pre-treatment lead indicators to the Step 1 OLS on Omega_0
        and tests their joint significance via cluster-robust Wald F-test.
        """
        if self._fit_data is None:
            raise RuntimeError("Must call fit() before pretrend_test().")

        fd = self._fit_data
        df = fd["df"]
        outcome = fd["outcome"]
        unit = fd["unit"]
        time = fd["time"]
        first_treat = fd["first_treat"]
        covariates = fd["covariates"]
        omega_0_mask = fd["omega_0_mask"]
        cluster_var = fd["cluster_var"]

        df_0 = df.loc[omega_0_mask].copy()

        # Compute relative time for untreated obs
        # For not-yet-treated units in their pre-treatment periods
        rel_time_0 = np.where(
            ~df_0["_never_treated"],
            df_0[time] - df_0[first_treat],
            np.nan,
        )

        # Get available pre-treatment relative times (negative values)
        pre_rel_times = sorted(
            set(int(h) for h in rel_time_0 if np.isfinite(h) and h < -self.anticipation)
        )

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Exclude the reference period (last pre-treatment period)
        ref = -1 - self.anticipation
        pre_rel_times = [h for h in pre_rel_times if h != ref]

        if n_leads is not None:
            # Take the n_leads periods closest to treatment
            pre_rel_times = sorted(pre_rel_times, reverse=True)[:n_leads]
            pre_rel_times = sorted(pre_rel_times)

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Build lead indicators
        lead_cols = []
        for h in pre_rel_times:
            col_name = f"_lead_{h}"
            df_0[col_name] = ((rel_time_0 == h)).astype(float)
            lead_cols.append(col_name)

        # Within-transform via iterative demeaning (exact for unbalanced panels)
        y_dm = self._iterative_demean(
            df_0[outcome].values, df_0[unit].values, df_0[time].values, df_0.index
        )

        all_x_cols = lead_cols[:]
        if covariates:
            all_x_cols.extend(covariates)

        X_dm = np.column_stack(
            [
                self._iterative_demean(
                    df_0[col].values, df_0[unit].values, df_0[time].values, df_0.index
                )
                for col in all_x_cols
            ]
        )

        # OLS with cluster-robust SEs
        cluster_ids = df_0[cluster_var].values
        result = solve_ols(
            X_dm,
            y_dm,
            cluster_ids=cluster_ids,
            return_vcov=True,
            rank_deficient_action=self.rank_deficient_action,
            column_names=all_x_cols,
        )
        coefficients = result[0]
        vcov = result[2]

        # Extract lead coefficients and their sub-VCV
        n_leads_actual = len(lead_cols)
        gamma = coefficients[:n_leads_actual]
        V_gamma = vcov[:n_leads_actual, :n_leads_actual]

        # Wald F-test: F = (gamma' V^{-1} gamma) / n_leads
        try:
            V_inv_gamma = np.linalg.solve(V_gamma, gamma)
            wald_stat = float(gamma @ V_inv_gamma)
            f_stat = wald_stat / n_leads_actual
        except np.linalg.LinAlgError:
            f_stat = np.nan

        # P-value from F distribution
        if np.isfinite(f_stat) and f_stat >= 0:
            n_clusters = len(np.unique(cluster_ids))
            df_denom = max(n_clusters - 1, 1)
            p_value = float(stats.f.sf(f_stat, n_leads_actual, df_denom))
        else:
            p_value = np.nan

        # Store lead coefficients
        lead_coefficients = {}
        for j, h in enumerate(pre_rel_times):
            lead_coefficients[h] = float(gamma[j])

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "df": n_leads_actual,
            "n_leads": n_leads_actual,
            "lead_coefficients": lead_coefficients,
        }

    # =========================================================================
    # Bootstrap
    # =========================================================================

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        original_att: float,
        original_event_study: Optional[Dict[int, Dict[str, Any]]],
        original_group: Optional[Dict[Any, Dict[str, Any]]],
    ) -> ImputationBootstrapResults:
        """
        Run multiplier bootstrap for inference.

        Uses a unit-level wild bootstrap: perturbs unit-level scores with
        Rademacher weights and re-estimates the aggregated effects.
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        # Get unique clusters for multiplier bootstrap
        # Uses cluster_var (= self.cluster if set, else unit)
        cluster_var = self.cluster if self.cluster is not None else unit
        all_clusters = df[cluster_var].unique()
        n_clusters = len(all_clusters)

        # Pre-compute cluster masks for efficiency
        cluster_indices: Dict[Any, np.ndarray] = {}
        for c in all_clusters:
            cluster_indices[c] = np.where(df[cluster_var].values == c)[0]

        # Pre-compute tau_hat and observation weights for each target
        omega_1_mask = df["_treated"]
        omega_0_mask = ~omega_1_mask
        n_1 = int(omega_1_mask.sum())

        tau_hat_full = df["_tau_hat"].values  # NaN for untreated

        # Pre-compute balanced cohort mask for event study (if balance_e is set)
        es_balanced_mask_treated = None
        if balance_e is not None and original_event_study:
            df_1_pre = df.loc[omega_1_mask]
            rel_times_pre = df_1_pre["_rel_time"].values
            all_horizons_pre = sorted(set(int(h) for h in rel_times_pre if np.isfinite(h)))
            if self.horizon_max is not None:
                all_horizons_pre = [h for h in all_horizons_pre if abs(h) <= self.horizon_max]
            es_balanced_mask_treated = self._compute_balanced_cohort_mask(
                df_1_pre, first_treat, rel_times_pre, all_horizons_pre, balance_e
            )

        # Bootstrap distributions
        boot_overall = np.zeros(self.n_bootstrap)
        boot_event_study: Optional[Dict[int, np.ndarray]] = None
        boot_group: Optional[Dict[Any, np.ndarray]] = None

        if original_event_study:
            horizons = [
                h for h in original_event_study if original_event_study[h].get("n_obs", 0) > 0
            ]
            boot_event_study = {h: np.zeros(self.n_bootstrap) for h in horizons}

        if original_group:
            boot_group = {g: np.zeros(self.n_bootstrap) for g in original_group}

        for b in range(self.n_bootstrap):
            # Generate Rademacher weights per cluster
            weights_b = rng.choice([-1.0, 1.0], size=n_clusters)

            # Map to observation level
            obs_weights = np.ones(len(df))
            for i, c in enumerate(all_clusters):
                obs_weights[cluster_indices[c]] = weights_b[i]

            # Perturbed overall ATT
            treated_tau = tau_hat_full[omega_1_mask.values]
            treated_weights = obs_weights[omega_1_mask.values]
            valid = np.isfinite(treated_tau)
            if valid.sum() > 0:
                boot_overall[b] = float(np.mean(treated_tau[valid] * treated_weights[valid]))
            else:
                boot_overall[b] = original_att

            # Perturbed event study
            if boot_event_study is not None and original_event_study:
                df_1 = df.loc[omega_1_mask]
                rel_times = df_1["_rel_time"].values
                for h in boot_event_study:
                    h_mask = rel_times == h
                    if es_balanced_mask_treated is not None:
                        h_mask = h_mask & es_balanced_mask_treated
                    tau_h = treated_tau[h_mask]
                    w_h = treated_weights[h_mask]
                    valid_h = np.isfinite(tau_h)
                    if valid_h.sum() > 0:
                        boot_event_study[h][b] = float(np.mean(tau_h[valid_h] * w_h[valid_h]))
                    else:
                        boot_event_study[h][b] = original_event_study[h]["effect"]

            # Perturbed group effects
            if boot_group is not None and original_group:
                df_1 = df.loc[omega_1_mask]
                cohorts = df_1[first_treat].values
                for g in boot_group:
                    g_mask = cohorts == g
                    tau_g = treated_tau[g_mask]
                    w_g = treated_weights[g_mask]
                    valid_g = np.isfinite(tau_g)
                    if valid_g.sum() > 0:
                        boot_group[g][b] = float(np.mean(tau_g[valid_g] * w_g[valid_g]))
                    else:
                        boot_group[g][b] = original_group[g]["effect"]

        # Compute bootstrap statistics
        # For multiplier bootstrap, the distribution is centered around 0
        # (perturbation distribution). SE is the std, p-value and CI use the SE.
        overall_se = float(np.std(boot_overall, ddof=1))
        overall_t = original_att / overall_se if overall_se > 0 else np.nan
        overall_p = float(2 * stats.norm.sf(abs(overall_t))) if np.isfinite(overall_t) else np.nan
        overall_ci = (
            compute_confidence_interval(original_att, overall_se, self.alpha)
            if overall_se > 0
            else (np.nan, np.nan)
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
                    t_h = orig_eff / se_h
                    event_study_p_values[h] = float(2 * stats.norm.sf(abs(t_h)))
                    event_study_cis[h] = compute_confidence_interval(orig_eff, se_h, self.alpha)
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
                    t_g = orig_eff / se_g
                    group_p_values[g] = float(2 * stats.norm.sf(abs(t_g)))
                    group_cis[g] = compute_confidence_interval(orig_eff, se_g, self.alpha)
                else:
                    group_p_values[g] = np.nan
                    group_cis[g] = (np.nan, np.nan)

        return ImputationBootstrapResults(
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
            bootstrap_distribution=boot_overall,
        )

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
            "aux_partition": self.aux_partition,
        }

    def set_params(self, **params) -> "ImputationDiD":
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


def imputation_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]] = None,
    aggregate: Optional[str] = None,
    balance_e: Optional[int] = None,
    **kwargs,
) -> ImputationDiDResults:
    """
    Convenience function for imputation DiD estimation.

    This is a shortcut for creating an ImputationDiD estimator and calling fit().

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
        Additional keyword arguments passed to ImputationDiD constructor.

    Returns
    -------
    ImputationDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import imputation_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = imputation_did(data, 'outcome', 'unit', 'time', 'first_treat',
    ...                          aggregate='event_study')
    >>> results.print_summary()
    """
    est = ImputationDiD(**kwargs)
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
