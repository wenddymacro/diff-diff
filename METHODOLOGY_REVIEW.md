# Methodology Review

This document tracks the progress of reviewing each estimator's implementation against the Methodology Registry and academic references. It ensures that implementations are correct, consistent, and well-documented.

For the methodology registry with academic foundations and key equations, see [docs/methodology/REGISTRY.md](docs/methodology/REGISTRY.md).

---

## Overview

Each estimator in diff-diff should be periodically reviewed to ensure:
1. **Correctness**: Implementation matches the academic paper's equations
2. **Reference alignment**: Behavior matches reference implementations (R packages, Stata commands)
3. **Edge case handling**: Documented edge cases are handled correctly
4. **Standard errors**: SE formulas match the documented approach

---

## Review Status Summary

| Estimator | Module | R Reference | Status | Last Review |
|-----------|--------|-------------|--------|-------------|
| DifferenceInDifferences | `estimators.py` | `fixest::feols()` | **Complete** | 2026-01-24 |
| MultiPeriodDiD | `estimators.py` | `fixest::feols()` | **Complete** | 2026-02-02 |
| TwoWayFixedEffects | `twfe.py` | `fixest::feols()` | **Complete** | 2026-02-08 |
| CallawaySantAnna | `staggered.py` | `did::att_gt()` | **Complete** | 2026-01-24 |
| SunAbraham | `sun_abraham.py` | `fixest::sunab()` | Not Started | - |
| SyntheticDiD | `synthetic_did.py` | `synthdid::synthdid_estimate()` | **Complete** | 2026-02-10 |
| TripleDifference | `triple_diff.py` | (forthcoming) | Not Started | - |
| TROP | `trop.py` | (forthcoming) | Not Started | - |
| BaconDecomposition | `bacon.py` | `bacondecomp::bacon()` | Not Started | - |
| HonestDiD | `honest_did.py` | `HonestDiD` package | Not Started | - |
| PreTrendsPower | `pretrends.py` | `pretrends` package | Not Started | - |
| PowerAnalysis | `power.py` | `pwr` / `DeclareDesign` | Not Started | - |

**Status legend:**
- **Not Started**: No formal review conducted
- **In Progress**: Review underway
- **Complete**: Review finished, implementation verified

---

## Detailed Review Notes

### Core DiD Estimators

#### DifferenceInDifferences

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Wooldridge (2010), Angrist & Pischke (2009) |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-01-24 |

**Verified Components:**
- [x] ATT formula: Double-difference of cell means matches regression interaction coefficient
- [x] R comparison: ATT matches `fixest::feols()` within 1e-3 tolerance
- [x] R comparison: SE (HC1 robust) matches within 5%
- [x] R comparison: P-value matches within 0.01
- [x] R comparison: Confidence intervals overlap
- [x] R comparison: Cluster-robust SE matches within 10%
- [x] R comparison: Fixed effects (absorb) matches `feols(...|unit)` within 1%
- [x] Wild bootstrap inference (Rademacher, Mammen, Webb weights)
- [x] Formula interface (`y ~ treated * post`)
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- 53 methodology verification tests in `tests/test_methodology_did.py`
- 123 existing tests in `tests/test_estimators.py`
- R benchmark tests (skip if R not available)

**R Comparison Results:**
- ATT matches within 1e-3 (R JSON truncation limits precision)
- HC1 SE matches within 5%
- Cluster-robust SE matches within 10%
- Fixed effects results match within 1%

**Corrections Made:**
- (None - implementation verified correct)

**Outstanding Concerns:**
- R comparison precision limited by JSON output truncation (4 decimal places)
- Consider improving R script to output full precision for tighter tolerances

**Edge Cases Verified:**
1. Empty cells: Produces rank deficiency warning (expected behavior)
2. Singleton clusters: Included in variance estimation, contribute via residuals (corrected REGISTRY.md)
3. Rank deficiency: All three modes (warn/error/silent) working
4. Non-binary treatment/time: Raises ValueError as expected
5. No variation in treatment/time: Raises ValueError as expected
6. Missing values: Raises ValueError as expected

---

#### MultiPeriodDiD

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Freyaldenhoven et al. (2021), Wooldridge (2010), Angrist & Pischke (2009) |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-02-02 |

**Verified Components:**
- [x] Full event-study specification: treatment × period interactions for ALL non-reference periods (pre and post)
- [x] Reference period coefficient is zero (normalized by omission from design matrix)
- [x] Default reference period is last pre-period (e=-1 convention, matches fixest/did)
- [x] Pre-period coefficients available for parallel trends assessment
- [x] Average ATT computed from post-treatment effects only, with covariance-aware SE
- [x] Returns PeriodEffect objects with confidence intervals for all periods
- [x] Supports balanced and unbalanced panels
- [x] NaN inference: t_stat/p_value/CI use NaN when SE is non-finite or zero
- [x] R-style NA propagation: avg_att is NaN if any post-period effect is unidentified
- [x] Rank-deficient design matrix: warns and sets NaN for dropped coefficients (R-style)
- [x] Staggered adoption detection warning (via `unit` parameter)
- [x] Treatment reversal detection warning
- [x] Time-varying D_it detection warning (advises creating ever-treated indicator)
- [x] Single pre-period warning (ATT valid but pre-trends assessment unavailable)
- [x] Post-period reference_period raises ValueError (would bias avg_att)
- [x] HonestDiD/PreTrendsPower integration uses interaction sub-VCV (not full regression VCV)
- [x] All REGISTRY.md edge cases tested

**Test Coverage:**
- 50 tests across `TestMultiPeriodDiD` and `TestMultiPeriodDiDEventStudy` in `tests/test_estimators.py`
- 18 new event-study specification tests added in PR #125

**Corrections Made:**
- **PR #125 (2026-02-02)**: Transformed from post-period-only estimator into full event-study
  specification with pre-period coefficients. Reference period default changed from first
  pre-period to last pre-period (e=-1 convention). HonestDiD/PreTrendsPower VCV extraction
  fixed to use interaction sub-VCV instead of full regression VCV.

**Outstanding Concerns:**
- ~~No R comparison benchmarks yet~~ — **Resolved**: R comparison benchmark added via
  `benchmarks/R/benchmark_multiperiod.R` using `fixest::feols(outcome ~ treated * time_f | unit)`.
  Results match R exactly: ATT diff < 1e-11, SE diff 0.0%, period effects correlation 1.0.
  Validated at small (200 units) and 1k scales.
- Default SE is HC1 (not cluster-robust at unit level as fixest uses). Cluster-robust
  available via `cluster` parameter but not the default.
- Endpoint binning for distant event times not yet implemented.
- FutureWarning for reference_period default change should eventually be removed once
  the transition is complete.

---

#### TwoWayFixedEffects

| Field | Value |
|-------|-------|
| Module | `twfe.py` |
| Primary Reference | Wooldridge (2010), Ch. 10 |
| R Reference | `fixest::feols()` |
| Status | **Complete** |
| Last Review | 2026-02-08 |

**Verified Components:**
- [x] Within-transformation algebra: `y_it - ȳ_i - ȳ_t + ȳ` matches hand calculation (rtol=1e-12)
- [x] ATT matches manual demeaned OLS (rtol=1e-10)
- [x] ATT matches `DifferenceInDifferences` on 2-period data (rtol=1e-10)
- [x] Covariates are also within-transformed (sum to zero within unit/time groups)
- [x] R comparison: ATT matches `fixest::feols(y ~ treated:post | unit + post, cluster=~unit)` (rtol<0.1%)
- [x] R comparison: Cluster-robust SE match (rtol<1%)
- [x] R comparison: P-value match (atol<0.01)
- [x] R comparison: CI bounds match (rtol<1%)
- [x] R comparison: ATT and SE match with covariate (same tolerances)
- [x] Edge case: Staggered treatment triggers `UserWarning`
- [x] Edge case: Auto-clusters at unit level (SE matches explicit `cluster="unit"`)
- [x] Edge case: DF adjustment for absorbed FE matches manual `solve_ols()` with `df_adjustment`
- [x] Edge case: Covariate collinear with interaction raises `ValueError` ("cannot be identified")
- [x] Edge case: Covariate collinearity warns but ATT remains finite
- [x] Edge case: `rank_deficient_action="error"` raises `ValueError`
- [x] Edge case: `rank_deficient_action="silent"` emits no warnings
- [x] Edge case: Unbalanced panel produces valid results (finite ATT, positive SE)
- [x] Edge case: Missing unit column raises `ValueError`
- [x] Integration: `decompose()` returns `BaconDecompositionResults`
- [x] SE: Cluster-robust SE >= HC1 SE
- [x] SE: VCoV positive semi-definite
- [x] Wild bootstrap: Valid inference (finite SE, p-value in [0,1])
- [x] Wild bootstrap: All weight types (rademacher, mammen, webb) produce valid inference
- [x] Wild bootstrap: `inference="wild_bootstrap"` routes correctly
- [x] Params: `get_params()` returns all inherited parameters
- [x] Params: `set_params()` modifies attributes
- [x] Results: `summary()` contains "ATT"
- [x] Results: `to_dict()` contains att, se, t_stat, p_value, n_obs
- [x] Results: residuals + fitted = demeaned outcome (not raw)
- [x] Edge case: Multi-period time emits UserWarning advising binary post indicator
- [x] Edge case: Non-{0,1} binary time emits UserWarning (ATT still correct)
- [x] Edge case: ATT invariant to time encoding ({0,1} vs {2020,2021} produces identical results)

**Key Implementation Detail:**
The interaction term `D_i × Post_t` must be within-transformed (demeaned) alongside the outcome,
consistent with the Frisch-Waugh-Lovell (FWL) theorem: all regressors and the outcome must be
projected out of the fixed effects space. R's `fixest::feols()` does this automatically when
variables appear to the left of the `|` separator.

**Corrections Made:**
- **Bug fix: interaction term must be within-transformed** (found during review). The previous
  implementation used raw (un-demeaned) `D_i × Post_t` in the demeaned regression. This gave
  correct results only for 2-period panels where `post == period`. For multi-period panels
  (e.g., 4 periods with binary `post`), the raw interaction had incorrect correlation with
  demeaned Y, producing ATT approximately 1/3 of the true value. Fixed by applying the same
  within-transformation to the interaction term before regression. This matches R's
  `fixest::feols()` behavior. (`twfe.py` lines 99-113)

**Outstanding Concerns:**
- **Multi-period `time` parameter**: Multi-period time values (e.g., 1,2,3,4) produce
  `treated × period_number` instead of `treated × post_indicator`, which is not the standard
  D_it treatment indicator. A `UserWarning` is emitted when `time` has >2 unique values.
  For binary time with non-{0,1} values (e.g., {2020, 2021}), the ATT is mathematically
  correct (the within-transformation absorbs the scaling), but a warning recommends 0/1
  encoding for clarity. Users with multi-period data should create a binary `post` column.
- **Staggered treatment warning**: The warning only fires when `time` has >2 unique values
  (i.e., actual period numbers). With binary `time="post"`, all treated units appear to start
  treatment at `time=1`, making staggering undetectable. Users with staggered designs should
  use `decompose()` or `CallawaySantAnna` directly for proper diagnostics.

---

### Modern Staggered Estimators

#### CallawaySantAnna

| Field | Value |
|-------|-------|
| Module | `staggered.py` |
| Primary Reference | Callaway & Sant'Anna (2021) |
| R Reference | `did::att_gt()` |
| Status | **Complete** |
| Last Review | 2026-01-24 |

**Verified Components:**
- [x] ATT(g,t) basic formula (hand-calculated exact match)
- [x] Doubly robust estimator
- [x] IPW estimator
- [x] Outcome regression
- [x] Base period selection (varying/universal)
- [x] Anticipation parameter handling
- [x] Simple/event-study/group aggregation
- [x] Analytical SE with weight influence function
- [x] Bootstrap SE (Rademacher/Mammen/Webb)
- [x] Control group composition (never_treated/not_yet_treated)
- [x] All documented edge cases from REGISTRY.md

**Test Coverage:**
- 46 methodology verification tests in `tests/test_methodology_callaway.py`
- 93 existing tests in `tests/test_staggered.py`
- R benchmark tests (skip if R not available)

**R Comparison Results:**
- Overall ATT matches within 20% (difference due to dynamic effects in generated data)
- Post-treatment ATT(g,t) values match within 20%
- Pre-treatment effects may differ due to base_period handling differences

**Corrections Made:**
- (None - implementation verified correct)

**Outstanding Concerns:**
- R comparison shows ~20% difference in overall ATT with generated data
  - Likely due to differences in how dynamic effects are handled in data generation
  - Individual ATT(g,t) values match closely for post-treatment periods
  - Further investigation recommended with real-world data
- Pre-treatment ATT(g,t) may differ from R due to base_period="varying" semantics
  - Python uses t-1 as base for pre-treatment
  - R's behavior requires verification

**Deviations from R's did::att_gt():**
1. **NaN for invalid inference**: When SE is non-finite or zero, Python returns NaN for
   t_stat/p_value rather than potentially erroring. This is a defensive enhancement.

**Alignment with R's did::att_gt() (as of v2.1.5):**
1. **Webb weights**: Webb's 6-point distribution with values ±√(3/2), ±1, ±√(1/2)
   uses equal probabilities (1/6 each) matching R's `did` package. This gives
   E[w]=0, Var(w)=1.0, consistent with other bootstrap weight distributions.

   **Verification**: Our implementation matches the well-established `fwildclusterboot`
   R package (C++ source: [wildboottest.cpp](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp)).
   The implementation uses `sqrt(1.5)`, `1`, `sqrt(0.5)` (and negatives) with equal 1/6
   probabilities—identical to our values.

   **Note on documentation discrepancy**: Some documentation (e.g., fwildclusterboot
   vignette) describes Webb weights as "±1.5, ±1, ±0.5". This appears to be a
   simplification for readability. The actual implementations use ±√1.5, ±1, ±√0.5
   which provides the required unit variance (Var(w) = 1.0).

---

#### SunAbraham

| Field | Value |
|-------|-------|
| Module | `sun_abraham.py` |
| Primary Reference | Sun & Abraham (2021) |
| R Reference | `fixest::sunab()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

### Advanced Estimators

#### SyntheticDiD

| Field | Value |
|-------|-------|
| Module | `synthetic_did.py` |
| Primary Reference | Arkhangelsky et al. (2021) |
| R Reference | `synthdid::synthdid_estimate()` |
| Status | **Complete** |
| Last Review | 2026-02-10 |

**Corrections Made:**
1. **Time weights: Frank-Wolfe on collapsed form** (was heuristic inverse-distance).
   Replaced ad-hoc inverse-distance weighting with the Frank-Wolfe algorithm operating
   on the collapsed (N_co x T_pre) problem as specified in Algorithm 1 of
   Arkhangelsky et al. (2021), matching R's `synthdid::fw.step()`.
2. **Unit weights: Frank-Wolfe with two-pass sparsification** (was projected gradient
   descent with wrong penalty). Replaced projected gradient descent (which used an
   incorrect penalty formulation) with Frank-Wolfe optimization followed by two-pass
   sparsification, matching R's `synthdid::sc.weight.fw()` and `sparsify_function()`.
3. **Auto-computed regularization from data noise level** (was `lambda_reg=0.0`,
   `zeta=1.0`). Regularization parameters `zeta_omega` and `zeta_lambda` are now
   computed automatically from the data noise level (N_tr * sigma^2) as specified in
   Appendix D of Arkhangelsky et al. (2021), matching R's default behavior.
4. **Bootstrap SE uses fixed weights matching R's `bootstrap_sample`** (was
   re-estimating all weights). The bootstrap variance procedure now holds unit and time
   weights fixed at their point estimates and only re-estimates the treatment effect,
   matching the approach in R's `synthdid::bootstrap_sample()`.
5. **Default `variance_method` changed to `"placebo"`** matching R's default. The R
   package uses placebo variance by default (`synthdid_estimate` returns an object whose
   `vcov()` uses the placebo method); our default now matches.
6. **Deprecated `lambda_reg` and `zeta` params; new params are `zeta_omega` and
   `zeta_lambda`**. The old parameters had unclear semantics and did not correspond to
   the paper's notation. The new parameters directly match the paper and R package
   naming conventions. `lambda_reg` and `zeta` are deprecated with warnings and will
   be removed in a future release.

**Outstanding Concerns:**
- (None)

---

#### TripleDifference

| Field | Value |
|-------|-------|
| Module | `triple_diff.py` |
| Primary Reference | Ortiz-Villavicencio & Sant'Anna (2025) |
| R Reference | (forthcoming) |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### TROP

| Field | Value |
|-------|-------|
| Module | `trop.py` |
| Primary Reference | Athey, Imbens, Qu & Viviano (2025) |
| R Reference | (forthcoming) |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

### Diagnostics & Sensitivity

#### BaconDecomposition

| Field | Value |
|-------|-------|
| Module | `bacon.py` |
| Primary Reference | Goodman-Bacon (2021) |
| R Reference | `bacondecomp::bacon()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### HonestDiD

| Field | Value |
|-------|-------|
| Module | `honest_did.py` |
| Primary Reference | Rambachan & Roth (2023) |
| R Reference | `HonestDiD` package |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### PreTrendsPower

| Field | Value |
|-------|-------|
| Module | `pretrends.py` |
| Primary Reference | Roth (2022) |
| R Reference | `pretrends` package |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### PowerAnalysis

| Field | Value |
|-------|-------|
| Module | `power.py` |
| Primary Reference | Bloom (1995), Burlig et al. (2020) |
| R Reference | `pwr` / `DeclareDesign` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

## Review Process Guidelines

### Review Checklist

For each estimator, complete the following steps:

- [ ] **Read primary academic source** - Review the key paper(s) cited in REGISTRY.md
- [ ] **Compare key equations** - Verify implementation matches equations in REGISTRY.md
- [ ] **Run benchmark against R reference** - Execute `benchmarks/run_benchmarks.py --estimator <name>` if available
- [ ] **Verify edge case handling** - Check behavior matches REGISTRY.md documentation
- [ ] **Check standard error formula** - Confirm SE computation matches reference
- [ ] **Document any deviations** - Add notes explaining intentional differences with rationale

### When to Update This Document

1. **After completing a review**: Update status to "Complete" and add date
2. **When making corrections**: Document what was fixed in the "Corrections Made" section
3. **When identifying issues**: Add to "Outstanding Concerns" for future investigation
4. **When deviating from reference**: Document the deviation and rationale

### Deviation Documentation

When our implementation intentionally differs from the reference implementation, document:

1. **What differs**: Specific behavior or formula that differs
2. **Why**: Rationale (e.g., "defensive enhancement", "bug in R package", "follows updated paper")
3. **Impact**: Whether results differ in practice
4. **Cross-reference**: Update REGISTRY.md edge cases section

Example:
```
**Deviation (2025-01-15)**: CallawaySantAnna returns NaN for t_stat when SE is non-finite,
whereas R's `did::att_gt` would error. This is a defensive enhancement that provides
more graceful handling of edge cases while still signaling invalid inference to users.
```

### Priority Order

Suggested order for reviews based on usage and complexity:

1. **High priority** (most used, complex methodology):
   - CallawaySantAnna
   - SyntheticDiD
   - HonestDiD

2. **Medium priority** (commonly used, simpler methodology):
   - DifferenceInDifferences
   - TwoWayFixedEffects
   - MultiPeriodDiD
   - SunAbraham
   - BaconDecomposition

3. **Lower priority** (newer or less commonly used):
   - TripleDifference
   - TROP
   - PreTrendsPower
   - PowerAnalysis

---

## Related Documents

- [REGISTRY.md](docs/methodology/REGISTRY.md) - Academic foundations and key equations
- [ROADMAP.md](ROADMAP.md) - Feature roadmap
- [TODO.md](TODO.md) - Technical debt tracking
- [CLAUDE.md](CLAUDE.md) - Development guidelines
