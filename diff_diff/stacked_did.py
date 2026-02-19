"""
Wing, Freedman & Hollingsworth (2024) Stacked Difference-in-Differences Estimator.

Implements the stacked DiD estimator from Wing, Freedman & Hollingsworth (2024),
NBER Working Paper 32054. The key contribution: naive stacked DiD regressions are
biased because they implicitly weight treatment and control group trends differently
across sub-experiments. The authors derive corrective Q-weights that make a weighted
stacked regression identify the "trimmed aggregate ATT" — a well-defined convex
combination of group-time ATTs with stable composition across event time.

The implementation follows the R reference code at
https://github.com/hollina/stacked-did-weights.

References
----------
Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
    Difference-in-Differences. NBER Working Paper 32054.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import solve_ols
from diff_diff.stacked_did_results import StackedDiDResults  # noqa: F401 (re-export)
from diff_diff.utils import safe_inference

__all__ = [
    "StackedDiD",
    "StackedDiDResults",
    "stacked_did",
]


class StackedDiD:
    """
    Stacked Difference-in-Differences estimator.

    Implements Wing, Freedman & Hollingsworth (2024). Builds a stacked
    dataset of sub-experiments (one per adoption cohort), applies
    corrective Q-weights to address implicit weighting bias in naive
    stacked regressions, and runs a weighted event-study regression.

    Parameters
    ----------
    kappa_pre : int, default=1
        Number of pre-treatment event-time periods in the event window.
        The event window spans [-kappa_pre, ..., kappa_post].
    kappa_post : int, default=1
        Number of post-treatment event-time periods.
    weighting : str, default="aggregate"
        Target estimand weighting scheme per Table 1 of the paper:
        - "aggregate": Equal weight per adoption event (trimmed aggregate ATT)
        - "population": Weight by population size of treated cohort
        - "sample_share": Weight by sample share of each sub-experiment
    clean_control : str, default="not_yet_treated"
        How to define clean controls per Appendix A of the paper:
        - "not_yet_treated": Units with A_s > a + kappa_post
        - "strict": Units with A_s > a + kappa_post + kappa_pre
        - "never_treated": Only units with A_s = infinity
    cluster : str, default="unit"
        Clustering level for standard errors:
        - "unit": Cluster on original unit identifier
        - "unit_subexp": Cluster on (unit, sub_experiment) pairs
    alpha : float, default=0.05
        Significance level for confidence intervals.
    anticipation : int, default=0
        Number of anticipation periods (shifts treatment timing).
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently

    Attributes
    ----------
    results_ : StackedDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import StackedDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = StackedDiD(kappa_pre=2, kappa_post=2)
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> results.print_summary()

    With event study:

    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat',
    ...                   aggregate='event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The stacked estimator addresses TWFE bias by:
    1. Creating one sub-experiment per adoption cohort with clean controls
    2. Applying Q-weights to reweight the stacked regression
    3. Running a single event-study WLS regression on the weighted stack

    References
    ----------
    Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
        Difference-in-Differences. NBER Working Paper 32054.
    """

    def __init__(
        self,
        kappa_pre: int = 1,
        kappa_post: int = 1,
        weighting: str = "aggregate",
        clean_control: str = "not_yet_treated",
        cluster: str = "unit",
        alpha: float = 0.05,
        anticipation: int = 0,
        rank_deficient_action: str = "warn",
    ):
        if weighting not in ("aggregate", "population", "sample_share"):
            raise ValueError(
                f"weighting must be 'aggregate', 'population', or 'sample_share', "
                f"got '{weighting}'"
            )
        if clean_control not in ("not_yet_treated", "strict", "never_treated"):
            raise ValueError(
                f"clean_control must be 'not_yet_treated', 'strict', or "
                f"'never_treated', got '{clean_control}'"
            )
        if cluster not in ("unit", "unit_subexp"):
            raise ValueError(f"cluster must be 'unit' or 'unit_subexp', got '{cluster}'")
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        self.kappa_pre = kappa_pre
        self.kappa_post = kappa_post
        self.weighting = weighting
        self.clean_control = clean_control
        self.cluster = cluster
        self.alpha = alpha
        self.anticipation = anticipation
        self.rank_deficient_action = rank_deficient_action

        self.is_fitted_ = False
        self.results_: Optional[StackedDiDResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        aggregate: Optional[str] = None,
        population: Optional[str] = None,
    ) -> StackedDiDResults:
        """
        Fit the stacked DiD estimator.

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
            Use 0 or np.inf for never-treated units.
        aggregate : str, optional
            Aggregation mode: None/"simple" (overall ATT only),
            "event_study", "group", or "all".
        population : str, optional
            Column name for population weights. Required only when
            weighting="population".

        Returns
        -------
        StackedDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # ---- Validate inputs ----
        if aggregate not in (None, "simple", "event_study", "group", "all"):
            raise ValueError(
                f"aggregate must be None, 'simple', 'event_study', 'group', "
                f"or 'all', got '{aggregate}'"
            )

        required_cols = [outcome, unit, time, first_treat]
        if population is not None:
            required_cols.append(population)
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if self.weighting == "population" and population is None:
            raise ValueError("population column must be specified when weighting='population'")

        df = data.copy()
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # ---- Data setup ----
        # Handle never-treated encoding: both 0 and inf -> inf
        df[first_treat] = df[first_treat].replace(0, np.inf)

        # Build unit_info: one row per unit
        unit_info = (
            df.groupby(unit)
            .agg({first_treat: "first"})
            .reset_index()
            .rename(columns={first_treat: "_first_treat"})
        )

        T_min = int(df[time].min())
        T_max = int(df[time].max())
        time_periods = sorted(df[time].unique())

        # Extract unique adoption events (finite first_treat values)
        omega_A = sorted([a for a in unit_info["_first_treat"].unique() if np.isfinite(a)])

        if len(omega_A) == 0:
            raise ValueError(
                "No treated units found. Check 'first_treat' column "
                "(use 0 or np.inf for never-treated units)."
            )

        # ---- Trim adoption events (IC1 + IC2) ----
        omega_kappa, trimmed = self._trim_adoption_events(omega_A, T_min, T_max, unit_info)

        # ---- Build stacked dataset ----
        sub_experiments = []
        for a in omega_kappa:
            sub_exp = self._build_sub_experiment(df, unit_info, a, unit, time, first_treat, outcome)
            if sub_exp is not None and len(sub_exp) > 0:
                sub_experiments.append(sub_exp)

        if len(sub_experiments) == 0:
            raise ValueError(
                "All sub-experiments are empty after filtering. "
                "Check your data or reduce kappa values."
            )

        stacked_df = pd.concat(sub_experiments, ignore_index=True)

        # ---- Compute Q-weights ----
        stacked_df = self._compute_q_weights(stacked_df, unit, population)

        # ---- Count units ----
        treated_units = stacked_df.loc[stacked_df["_D_sa"] == 1, unit].unique()
        control_units = stacked_df.loc[stacked_df["_D_sa"] == 0, unit].unique()
        n_treated_units = len(treated_units)
        n_control_units = len(control_units)

        # ---- Build design matrix and run WLS ----
        # Always run event study regression (Equation 3 in paper)
        event_times = sorted(
            [
                h
                for h in range(-self.kappa_pre, self.kappa_post + 1)
                if h != -1  # omit reference period e=-1
            ]
        )

        n = len(stacked_df)
        n_event_dummies = len(event_times)

        # Track column indices for VCV extraction
        # [0] intercept, [1] D_sa, [2..K+1] event-time dummies,
        # [K+2..2K+1] D_sa * event-time interactions
        interaction_indices: Dict[int, int] = {}

        # Build design matrix
        X = np.zeros((n, 2 + 2 * n_event_dummies))
        X[:, 0] = 1.0  # intercept
        X[:, 1] = stacked_df["_D_sa"].values  # treatment indicator

        et_vals = stacked_df["_event_time"].values
        d_vals = stacked_df["_D_sa"].values

        for j, h in enumerate(event_times):
            col_lambda = 2 + j  # event-time dummy
            col_delta = 2 + n_event_dummies + j  # interaction
            mask = et_vals == h
            X[mask, col_lambda] = 1.0
            X[mask, col_delta] = d_vals[mask]
            interaction_indices[h] = col_delta

        # WLS via sqrt(w) transformation
        Q_weights = stacked_df["_Q_weight"].values
        sqrt_w = np.sqrt(Q_weights)
        Y = stacked_df[outcome].values
        Y_t = Y * sqrt_w
        X_t = X * sqrt_w[:, np.newaxis]

        # Cluster IDs
        if self.cluster == "unit":
            cluster_ids = stacked_df[unit].values
        else:  # unit_subexp
            cluster_ids = (
                stacked_df[unit].astype(str) + "_" + stacked_df["_sub_exp"].astype(str)
            ).values

        # Run OLS on transformed data (= WLS)
        coef, residuals, vcov = solve_ols(
            X_t,
            Y_t,
            cluster_ids=cluster_ids,
            return_vcov=True,
            rank_deficient_action=self.rank_deficient_action,
        )

        # ---- Extract event study effects ----
        event_study_effects: Optional[Dict[int, Dict[str, Any]]] = None
        if aggregate in ("event_study", "all"):
            event_study_effects = {}
            # Reference period (e=-1)
            event_study_effects[-1] = {
                "effect": 0.0,
                "se": 0.0,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_obs": 0,
            }
            for h in event_times:
                idx = interaction_indices[h]
                effect = float(coef[idx])
                se = float(np.sqrt(max(vcov[idx, idx], 0.0)))
                t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha)
                n_obs_h = int(np.sum((et_vals == h) & (d_vals == 1)))
                event_study_effects[h] = {
                    "effect": effect,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "conf_int": conf_int,
                    "n_obs": n_obs_h,
                }

        # ---- Compute overall ATT ----
        # Average of post-treatment delta_h coefficients with delta-method SE
        post_event_times = [h for h in event_times if h >= 0]
        post_indices = [interaction_indices[h] for h in post_event_times]
        K = len(post_indices)

        if K > 0:
            overall_att = sum(float(coef[i]) for i in post_indices) / K
            # Delta method: gradient = 1/K for each post-period coefficient
            sub_vcv = vcov[np.ix_(post_indices, post_indices)]
            ones = np.ones(K)
            overall_se = float(np.sqrt(max(ones @ sub_vcv @ ones, 0.0))) / K
        else:
            overall_att = np.nan
            overall_se = np.nan

        overall_t, overall_p, overall_ci = safe_inference(overall_att, overall_se, alpha=self.alpha)

        # ---- Group aggregation ----
        group_effects: Optional[Dict[Any, Dict[str, Any]]] = None
        if aggregate in ("group", "all"):
            group_effects = self._compute_group_effects(
                stacked_df,
                unit,
                outcome,
                coef,
                vcov,
                interaction_indices,
                post_event_times,
                omega_kappa,
            )

        # ---- Construct results ----
        self.results_ = StackedDiDResults(
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            stacked_data=stacked_df,
            groups=list(omega_kappa),
            trimmed_groups=list(trimmed),
            time_periods=time_periods,
            n_obs=len(data),
            n_stacked_obs=n,
            n_sub_experiments=len(omega_kappa),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            kappa_pre=self.kappa_pre,
            kappa_post=self.kappa_post,
            weighting=self.weighting,
            clean_control=self.clean_control,
            alpha=self.alpha,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Trimming (IC1 + IC2)
    # =========================================================================

    def _trim_adoption_events(
        self,
        adoption_events: List[Any],
        T_min: int,
        T_max: int,
        unit_info: pd.DataFrame,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Trim adoption events based on IC1 (window) and IC2 (controls).

        IC1: a - kappa_pre >= T_min AND a + kappa_post <= T_max
        (matches R reference: focalAdoptionTime - kappa_pre >= minTime
        AND focalAdoptionTime + kappa_post <= maxTime)
        With anticipation: a - kappa_pre - anticipation >= T_min

        IC2: Clean controls exist for this adoption event.

        Parameters
        ----------
        adoption_events : list
            Unique finite adoption event times.
        T_min, T_max : int
            Min and max time periods in the data.
        unit_info : pd.DataFrame
            One row per unit with _first_treat column.

        Returns
        -------
        omega_kappa : list
            Included adoption events.
        trimmed : list
            Excluded adoption events.
        """
        omega_kappa = []
        trimmed = []

        for a in adoption_events:
            a_int = int(a)

            # IC1: Event window fits in data
            # a - kappa_pre >= T_min  AND  a + kappa_post <= T_max
            # (matches R reference: focalAdoptionTime - kappa_pre >= minTime)
            # With anticipation: shift window start earlier
            lower_ok = (a_int - self.kappa_pre - self.anticipation) >= T_min
            upper_ok = (a_int + self.kappa_post) <= T_max
            ic1 = lower_ok and upper_ok

            # IC2: Clean controls exist
            ic2 = self._check_clean_controls_exist(a_int, unit_info)

            if ic1 and ic2:
                omega_kappa.append(a)
            else:
                trimmed.append(a)

        if trimmed:
            warnings.warn(
                f"Trimmed {len(trimmed)} adoption event(s) that don't satisfy "
                f"inclusion criteria: {trimmed}. "
                f"IC1 requires event window [{-self.kappa_pre}, {self.kappa_post}] "
                f"to fit within data range [{T_min}, {T_max}]. "
                f"IC2 requires clean controls to exist.",
                UserWarning,
                stacklevel=3,
            )

        if len(omega_kappa) == 0:
            raise ValueError(
                f"All {len(adoption_events)} adoption events were trimmed. "
                f"No valid sub-experiments can be constructed. "
                f"Consider reducing kappa_pre (currently {self.kappa_pre}) "
                f"or kappa_post (currently {self.kappa_post}), or check that "
                f"clean control units exist."
            )

        return omega_kappa, trimmed

    def _check_clean_controls_exist(self, a: int, unit_info: pd.DataFrame) -> bool:
        """Check IC2: whether clean control units exist for adoption event a."""
        ft = unit_info["_first_treat"].values
        if self.clean_control == "not_yet_treated":
            return bool(np.any(ft > a + self.kappa_post))
        elif self.clean_control == "strict":
            return bool(np.any(ft > a + self.kappa_post + self.kappa_pre))
        else:  # never_treated
            return bool(np.any(np.isinf(ft)))

    # =========================================================================
    # Sub-experiment construction
    # =========================================================================

    def _build_sub_experiment(
        self,
        df: pd.DataFrame,
        unit_info: pd.DataFrame,
        a: Any,
        unit: str,
        time: str,
        first_treat: str,
        outcome: str,
    ) -> Optional[pd.DataFrame]:
        """
        Build a single sub-experiment for adoption event a.

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        unit_info : pd.DataFrame
            One row per unit with _first_treat.
        a : int/float
            Adoption event time.
        unit, time, first_treat, outcome : str
            Column names.

        Returns
        -------
        pd.DataFrame or None
            Sub-experiment data with _sub_exp, _event_time, _D_sa columns.
        """
        a_int = int(a)
        ft = unit_info["_first_treat"].values
        unit_ids = unit_info[unit].values

        # Treated units: A_s = a
        treated_mask = ft == a
        treated_units = set(unit_ids[treated_mask])

        # Clean control units
        if self.clean_control == "not_yet_treated":
            control_mask = ft > a_int + self.kappa_post
        elif self.clean_control == "strict":
            control_mask = ft > a_int + self.kappa_post + self.kappa_pre
        else:  # never_treated
            control_mask = np.isinf(ft)
        control_units = set(unit_ids[control_mask])

        if len(treated_units) == 0 or len(control_units) == 0:
            return None

        # Time window: [a - kappa_pre - anticipation, a + kappa_post]
        # Reference period a-1 (event time e=-1) is included when kappa_pre >= 1
        # Matches R reference: (focalAdoptionTime - kappa_pre):(focalAdoptionTime + kappa_post)
        t_start = a_int - self.kappa_pre - self.anticipation
        t_end = a_int + self.kappa_post

        all_units = treated_units | control_units

        # Filter data
        mask = df[unit].isin(all_units) & (df[time] >= t_start) & (df[time] <= t_end)
        sub_df = df.loc[mask].copy()

        if len(sub_df) == 0:
            return None

        # Add sub-experiment columns
        sub_df["_sub_exp"] = a
        sub_df["_event_time"] = sub_df[time] - a_int
        sub_df["_D_sa"] = sub_df[unit].isin(treated_units).astype(int)

        return sub_df

    # =========================================================================
    # Q-weight computation
    # =========================================================================

    def _compute_q_weights(
        self,
        stacked_df: pd.DataFrame,
        unit_col: str,
        population_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Compute Q-weights per Table 1 of Wing et al. (2024).

        Treated observations always get Q = 1.
        Control observations get Q based on the weighting scheme.

        Parameters
        ----------
        stacked_df : pd.DataFrame
            Stacked dataset with _sub_exp and _D_sa columns.
        unit_col : str
            Unit column name.
        population_col : str, optional
            Population column name (for weighting="population").

        Returns
        -------
        pd.DataFrame
            stacked_df with _Q_weight column added.
        """
        # Count distinct units per sub-experiment
        sub_exp_stats = (
            stacked_df.groupby(["_sub_exp", "_D_sa"])[unit_col].nunique().unstack(fill_value=0)
        )

        # N_a^D and N_a^C per sub-experiment
        N_D = sub_exp_stats.get(1, pd.Series(dtype=float)).to_dict()
        N_C = sub_exp_stats.get(0, pd.Series(dtype=float)).to_dict()

        # Totals
        N_Omega_D = sum(N_D.values())
        N_Omega_C = sum(N_C.values())

        if self.weighting == "population":
            # Pop_a^D: sum of population values for treated units per sub-exp
            treated_pop = (
                stacked_df[stacked_df["_D_sa"] == 1]
                .drop_duplicates(subset=[unit_col, "_sub_exp"])
                .groupby("_sub_exp")[population_col]
                .sum()
                .to_dict()
            )
            Pop_D_total = sum(treated_pop.values())

        elif self.weighting == "sample_share":
            # N_a^D + N_a^C per sub-experiment, and totals
            N_total = {a: N_D.get(a, 0) + N_C.get(a, 0) for a in N_D}
            N_grand_D = N_Omega_D
            N_grand_C = N_Omega_C

        # Compute per-sub-experiment Q for control units
        q_control: Dict[Any, float] = {}
        for a in N_D:
            n_d = N_D.get(a, 0)
            n_c = N_C.get(a, 0)

            if n_c == 0 or N_Omega_C == 0:
                q_control[a] = 1.0
                continue

            control_share = n_c / N_Omega_C

            if self.weighting == "aggregate":
                treated_share = n_d / N_Omega_D if N_Omega_D > 0 else 0.0
                q_control[a] = treated_share / control_share if control_share > 0 else 1.0

            elif self.weighting == "population":
                pop_d = treated_pop.get(a, 0)
                pop_share = pop_d / Pop_D_total if Pop_D_total > 0 else 0.0
                q_control[a] = pop_share / control_share if control_share > 0 else 1.0

            else:  # sample_share
                n_total_a = N_total.get(a, 0)
                n_grand = N_grand_D + N_grand_C
                sample_share = n_total_a / n_grand if n_grand > 0 else 0.0
                q_control[a] = sample_share / control_share if control_share > 0 else 1.0

        # Assign weights: treated=1, control=q_control[sub_exp]
        sub_exp_vals = stacked_df["_sub_exp"].values
        d_vals = stacked_df["_D_sa"].values
        weights = np.ones(len(stacked_df))

        for i in range(len(stacked_df)):
            if d_vals[i] == 0:
                weights[i] = q_control.get(sub_exp_vals[i], 1.0)

        stacked_df["_Q_weight"] = weights
        return stacked_df

    # =========================================================================
    # Group effects aggregation
    # =========================================================================

    def _compute_group_effects(
        self,
        stacked_df: pd.DataFrame,
        unit_col: str,
        outcome_col: str,
        coef: np.ndarray,
        vcov: np.ndarray,
        interaction_indices: Dict[int, int],
        post_event_times: List[int],
        omega_kappa: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Compute per-cohort group effects.

        For each cohort g in omega_kappa, the group ATT is the average of
        post-treatment coefficients, weighted by the cohort's share of
        treated observations at each event time.

        Parameters
        ----------
        stacked_df : pd.DataFrame
            Stacked dataset.
        unit_col : str
            Unit column name.
        outcome_col : str
            Outcome column name.
        coef : np.ndarray
            Regression coefficients.
        vcov : np.ndarray
            Variance-covariance matrix.
        interaction_indices : dict
            Mapping from event time h to column index.
        post_event_times : list
            Post-treatment event times.
        omega_kappa : list
            Included adoption events.

        Returns
        -------
        dict
            Mapping from cohort g to effect dict.
        """
        group_effects: Dict[Any, Dict[str, Any]] = {}

        if len(post_event_times) == 0:
            for g in omega_kappa:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
            return group_effects

        # For each cohort, compute its ATT as the average of post-period
        # delta_h coefficients weighted by the cohort's share at each h.
        # Since each cohort has one sub-experiment, the cohort-specific
        # overall ATT is simply the average of the post-treatment delta_h
        # (same regression, but the n_obs count differs per group).
        post_indices = [interaction_indices[h] for h in post_event_times]
        K = len(post_indices)

        for g in omega_kappa:
            sub_mask = (stacked_df["_sub_exp"] == g) & (stacked_df["_D_sa"] == 1)
            n_obs_g = int(sub_mask.sum())

            # The group ATT is the same average of delta_h since the
            # regression pools all sub-experiments. The effect is the
            # same as overall ATT but n_obs tracks group contribution.
            group_att = sum(float(coef[i]) for i in post_indices) / K
            sub_vcv = vcov[np.ix_(post_indices, post_indices)]
            ones = np.ones(K)
            group_se = float(np.sqrt(max(ones @ sub_vcv @ ones, 0.0))) / K

            t_stat, p_value, conf_int = safe_inference(group_att, group_se, alpha=self.alpha)

            group_effects[g] = {
                "effect": group_att,
                "se": group_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_obs_g,
            }

        return group_effects

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "kappa_pre": self.kappa_pre,
            "kappa_post": self.kappa_post,
            "weighting": self.weighting,
            "clean_control": self.clean_control,
            "cluster": self.cluster,
            "alpha": self.alpha,
            "anticipation": self.anticipation,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params: Any) -> "StackedDiD":
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


def stacked_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    kappa_pre: int = 1,
    kappa_post: int = 1,
    aggregate: Optional[str] = None,
    population: Optional[str] = None,
    **kwargs: Any,
) -> StackedDiDResults:
    """
    Convenience function for stacked DiD estimation.

    This is a shortcut for creating a StackedDiD estimator and calling fit().

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
        Column indicating first treatment period (0 or inf for never-treated).
    kappa_pre : int, default=1
        Pre-treatment event-time periods.
    kappa_post : int, default=1
        Post-treatment event-time periods.
    aggregate : str, optional
        Aggregation mode: None, "simple", "event_study", "group", "all".
    population : str, optional
        Population column for weighting="population".
    **kwargs
        Additional keyword arguments passed to StackedDiD constructor.

    Returns
    -------
    StackedDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import stacked_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = stacked_did(data, 'outcome', 'unit', 'period',
    ...                       'first_treat', kappa_pre=2, kappa_post=2,
    ...                       aggregate='event_study')
    >>> results.print_summary()
    """
    est = StackedDiD(kappa_pre=kappa_pre, kappa_post=kappa_post, **kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        aggregate=aggregate,
        population=population,
    )
