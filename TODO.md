# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:779-785` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:568-587` | Low | Rarely needed |

## Code Quality

### Large Module Files

Target: < 1000 lines per module for maintainability.

| File | Lines | Action |
|------|-------|--------|
| ~~`staggered.py`~~ | ~~2301~~ 1120 | ✅ Split into staggered.py, staggered_bootstrap.py, staggered_aggregation.py, staggered_results.py |
| ~~`prep.py`~~ | ~~1993~~ 1241 | ✅ Split: DGP functions moved to `prep_dgp.py` (777 lines) |
| `trop.py` | 2904 | **Needs split** -- nearly 3x the 1000-line target |
| `imputation.py` | 2480 | **Needs split** -- results, bootstrap, aggregation like staggered |
| `two_stage.py` | 2209 | **Needs split** -- same pattern as imputation |
| `utils.py` | 1879 | **Needs split** -- legacy placebo functions could move to diagnostics |
| `visualization.py` | 1678 | Monitor -- growing but cohesive |
| `linalg.py` | 1537 | Monitor -- unified backend, splitting would hurt cohesion |
| `honest_did.py` | 1511 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `triple_diff.py` | 1322 | Acceptable |
| `sun_abraham.py` | 1227 | Acceptable |
| `estimators.py` | 1161 | Acceptable |
| `pretrends.py` | 1104 | Acceptable |

### ~~NaN Handling for Undefined t-statistics~~ -- DONE

All 7 t_stat locations fixed (diagnostics.py, sun_abraham.py, triple_diff.py) -- all now use `np.nan` or `np.isfinite()` guards. Fixed in PR #118 and follow-up PRs.

**Remaining nuance**: `diagnostics.py:785` still has `se = ... else 0.0` for the SE variable itself (not t_stat). The downstream t_stat line correctly returns `np.nan`, so inference is safe, but the SE value of 0.0 is technically incorrect for an undefined SE.

### Migrate Existing Inference Call Sites to `safe_inference()`

`safe_inference()` was added to `diff_diff/utils.py` to compute t_stat, p_value, and CI together with a NaN gate at the top. It is now the prescribed pattern for all new code (see CLAUDE.md design pattern #7). However, ~26 existing inline inference computations across 12 files have **not** been migrated yet.

**Files with inline inference to migrate:**

| File | Approx. Locations |
|------|-------------------|
| `estimators.py` | 2 (lines 1038, 1089) |
| `sun_abraham.py` | 4 (lines 621, 644, 661, 905) |
| `staggered.py` | 6 (lines 696, 725, 763, 777, 792, 806) |
| `staggered_aggregation.py` | 2 (lines 407, 479) |
| `triple_diff.py` | 1 (line 601) |
| `imputation.py` | 2 (lines 1806, 1927) |
| `two_stage.py` | 2 (lines 1325, 1431) |
| `diagnostics.py` | 2 (lines 665, 786) |
| `synthetic_did.py` | 1 (line 426) |
| `trop.py` | 2 (lines 1474, 2054) |
| `utils.py` | 1 (line 641) |
| `linalg.py` | 1 (line 1310) |

**How to find them:**
```bash
grep -En "(t_stat|overall_t)\s*=\s*[^#]*/|\[.t_stat.\]\s*=\s*[^#]*/" diff_diff/*.py | grep -v "def safe_inference"
```

**Note**: This command has one false positive (`utils.py:178`, inside the `safe_inference()` body) and misses multi-line expressions (e.g., `sun_abraham.py:660-661`). The table above is the authoritative list.

**Migration pattern:**
```python
# Before (inline, error-prone)
t_stat = effect / se if se > 0 else 0.0
p_value = compute_p_value(t_stat)
ci = compute_confidence_interval(effect, se, alpha)

# After (NaN-safe, consistent)
from diff_diff.utils import safe_inference
t_stat, p_value, ci = safe_inference(effect, se, alpha=alpha, df=df)
```

**Priority**: Medium — the NaN-handling table above covers the worst cases (those using `0.0`). The remaining sites may use partial guards but should still be migrated for consistency and to prevent regressions.

---

### Tech Debt from Code Reviews

Deferred items from PR reviews that were not addressed before merge.

#### Methodology/Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| TwoStageDiD & ImputationDiD bootstrap hardcodes Rademacher only; no `bootstrap_weights` parameter unlike CallawaySantAnna | `two_stage.py:1860`, `imputation.py:2363` | #156, #141 | Medium |
| TwoStageDiD GMM score logic duplicated between analytic/bootstrap with inconsistent NaN/overflow handling | `two_stage.py:1454-1784` | #156 | Medium |
| ImputationDiD weight construction duplicated between aggregation and bootstrap (drift risk) -- has explicit code comment acknowledging duplication | `imputation.py:1777-1786`, `imputation.py:2216-2221` | #141 | Medium |
| ImputationDiD dense `(A0'A0).toarray()` scales O((U+T+K)^2), OOM risk on large panels | `imputation.py:1564` | #141 | Medium |

#### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| TwoStageDiD per-column `.toarray()` in loop for cluster scores | `two_stage.py:1766-1767` | #156 | Medium |
| ImputationDiD event-study SEs recompute full conservative variance per horizon (should cache A0/A1 factorization) | `imputation.py:1772-1804` | #141 | Low |
| Legacy `compute_placebo_effects` uses deprecated projected-gradient weights (marked deprecated, users directed to `SyntheticDiD`) | `utils.py:1689-1691` | #145 | Low |
| Rust faer SVD ndarray-to-faer conversion overhead (minimal vs SVD cost) | `rust/src/linalg.rs:67` | #115 | Low |

#### Testing/Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Tutorial notebooks not executed in CI | `docs/tutorials/*.ipynb` | #159 | Low |
| TwoStageDiD `test_nan_propagation` is a no-op (only `pass`) | `tests/test_two_stage.py:643-652` | #156 | Low |
| ImputationDiD bootstrap + covariate path untested | `tests/test_imputation.py` | #141 | Low |
| TROP `n_bootstrap >= 2` validation missing (can yield 0/NaN SE silently) | `trop.py:462` | #124 | Low |
| SunAbraham deprecated `min_pre_periods`/`min_post_periods` still in `fit()` docstring | `sun_abraham.py:458-487` | #153 | Low |
| R comparison tests spawn separate `Rscript` per test (slow CI) | `tests/test_methodology_twfe.py:294` | #139 | Low |
| Rust TROP bootstrap SE returns 0.0 instead of NaN for <2 samples | `rust/src/trop.rs:1038-1054` | #115 | Low |

---

### Standard Error Consistency

Different estimators compute SEs differently. Consider unified interface.

| Estimator | Default SE Type |
|-----------|-----------------|
| DifferenceInDifferences | HC1 or cluster-robust |
| TwoWayFixedEffects | Always cluster-robust (unit level) |
| CallawaySantAnna | Simple difference-in-means SE |
| SyntheticDiD | Bootstrap or placebo-based |

**Action**: Consider adding `se_type` parameter for consistency across estimators.

### Type Annotations

Pyright reports 282 type errors. Most are false positives from numpy/pandas type stubs.

| Category | Count | Notes |
|----------|-------|-------|
| reportArgumentType | 94 | numpy/pandas stub mismatches |
| reportAttributeAccessIssue | 89 | Union types (results classes) |
| reportReturnType | 21 | Return type mismatches |
| reportOperatorIssue | 16 | Operators on incompatible types |
| Others | 62 | Various minor issues |

**Genuine issues to fix (low priority):**
- [ ] Optional handling in `estimators.py:291,297,308` - None checks needed
- [ ] Union type narrowing in `visualization.py:325-345` - results classes
- [ ] numpy floating conversion in `diagnostics.py:669-673`

**Note:** Most errors are false positives from imprecise type stubs. Mypy config in pyproject.toml already handles these via `disable_error_code`.

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- [x] `bootstrap_weight_type` in `CallawaySantAnna` (`staggered.py`)
  - Deprecated in favor of `bootstrap_weights` parameter
  - ✅ Deprecation warning updated to say "removed in v3.0"
  - ✅ README.md and tutorial 02 updated to use `bootstrap_weights`
  - Remove in next major version (v3.0)

---

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
- [ ] Support for CallawaySantAnnaResults (currently only MultiPeriodDiDResults)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## RuntimeWarnings in Linear Algebra Operations

### Apple Silicon M4 BLAS Bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by `np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3. The warnings appear for matrices with ≥260 rows but **do not affect result correctness** — coefficients and fitted values are valid (no NaN/Inf), and the design matrices are full rank.

**Root cause**: Apple's BLAS SME (Scalable Matrix Extension) kernels corrupt the floating-point status register, causing spurious FPE signals. Tracked in [numpy#28687](https://github.com/numpy/numpy/issues/28687) and [numpy#29820](https://github.com/numpy/numpy/issues/29820). Fixed in numpy ≥ 2.3 via [PR #29223](https://github.com/numpy/numpy/pull/29223).

**Not reproducible** on M3, Intel, or Linux.

- [ ] `linalg.py:162` - Warnings in fitted value computation (`X @ coefficients`)
  - Caused by M4 BLAS bug, not extreme coefficient values
  - Seen in test_prep.py during treatment effect recovery tests (n > 260)
- [ ] `triple_diff.py:307,323` - Warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases (separate from BLAS bug)

### Fix Plan (follow-up PR)

Replace `@` operator with `np.dot()` at affected call sites. `np.dot()` bypasses the ufunc FPE dispatch layer and produces identical results with zero spurious warnings on M4.

**Affected files and lines:**
- `linalg.py`: 332, 690, 704, 829, 1463
- `staggered.py`: 78, 85, 102, 860, 1024-1025
- `triple_diff.py`: 301, 307, 323
- `utils.py`: 515
- `imputation.py`: 1253, 1301, 1602, 1662
- `trop.py`: 1098

**Regression test:** Assert no RuntimeWarnings from `solve_ols()` with n ≥ 500 on all platforms.

**Long-term:** Revert to `@` operator when numpy ≥ 2.3 becomes the minimum supported version.

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

### QR+SVD Redundancy in Rank Detection

**Background**: The current `solve_ols()` implementation performs both QR (for rank detection) and SVD (for solving) decompositions on rank-deficient matrices. This is technically redundant since SVD can determine rank directly.

**Current approach** (R-style, chosen for robustness):
1. QR with pivoting for rank detection (`_detect_rank_deficiency()`)
2. scipy's `lstsq` with 'gelsd' driver (SVD-based) for solving

**Why we use QR for rank detection**:
- QR with pivoting provides the canonical ordering of linearly dependent columns
- R's `lm()` uses this approach for consistent dropped-column reporting
- Ensures consistent column dropping across runs (SVD column selection can vary)

**Potential optimization** (future work):
- Skip QR when `rank_deficient_action="silent"` since we don't need column names
- Use SVD rank directly in the Rust backend (already implemented)
- Add `skip_rank_check` parameter for hot paths where matrix is known to be full-rank (implemented in v2.2.0)

**Priority**: Low - the QR overhead is minimal compared to SVD solve, and correctness is more important than micro-optimization.

### Incomplete `check_finite` Bypass

**Background**: The `solve_ols()` function accepts a `check_finite=False` parameter intended to skip NaN/Inf validation for performance in hot paths where data is known to be clean.

**Current limitation**: When `check_finite=False`, our explicit validation is skipped, but scipy's internal QR decomposition in `_detect_rank_deficiency()` still validates finite values. This means callers cannot fully bypass all finite checks.

**Impact**: Minimal - the scipy check is fast and only affects edge cases where users explicitly pass `check_finite=False` with non-finite data (which would be a bug in their code anyway).

**Potential fix** (future work):
- Pass `check_finite=False` through to scipy's QR call (requires scipy >= 1.9.0)
- Or skip `_detect_rank_deficiency()` entirely when `check_finite=False` and `_skip_rank_check=True`

**Priority**: Low - this is an edge case optimization that doesn't affect correctness.

