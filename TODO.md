# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:1068-1074` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:532-554` | Low | Rarely needed |

## Code Quality

### Large Module Files

Target: < 1000 lines per module for maintainability.

| File | Lines | Action |
|------|-------|--------|
| ~~`staggered.py`~~ | ~~2301~~ 1066 | ✅ Split into staggered.py, staggered_bootstrap.py, staggered_aggregation.py, staggered_results.py |
| ~~`prep.py`~~ | ~~1993~~ 1241 | ✅ Split: DGP functions moved to `prep_dgp.py` (777 lines) |
| `trop.py` | 1703 | Monitor size |
| `visualization.py` | 1627 | Acceptable but growing |
| `honest_did.py` | 1493 | Acceptable |
| `utils.py` | 1481 | Acceptable |
| `power.py` | 1350 | Acceptable |
| `triple_diff.py` | 1291 | Acceptable |
| `sun_abraham.py` | 1176 | Acceptable |
| `pretrends.py` | 1160 | Acceptable |
| `bacon.py` | 1027 | OK |

### NaN Handling for Undefined t-statistics

Several estimators return `0.0` for t-statistic when SE is 0 or undefined. This is incorrect—a t-stat of 0 implies a null effect, whereas `np.nan` correctly indicates undefined inference.

**Pattern to fix**: `t_stat = effect / se if se > 0 else 0.0` → `t_stat = effect / se if se > 0 else np.nan`

| Location | Line | Current Code |
|----------|------|--------------|
| `diagnostics.py` | 665 | `t_stat = original_att / se if se > 0 else 0.0` |
| `diagnostics.py` | 786 | `t_stat = mean_effect / se if se > 0 else 0.0` |
| `sun_abraham.py` | 603 | `overall_t = overall_att / overall_se if overall_se > 0 else 0.0` |
| `sun_abraham.py` | 626 | `overall_t = overall_att / overall_se if overall_se > 0 else 0.0` |
| `sun_abraham.py` | 643 | `eff_val / se_val if se_val > 0 else 0.0` |
| `sun_abraham.py` | 881 | `t_stat = agg_effect / agg_se if agg_se > 0 else 0.0` |
| `triple_diff.py` | 601 | `t_stat = att / se if se > 0 else 0.0` |

**Priority**: Medium - affects inference reporting in edge cases.

**Note**: CallawaySantAnna was fixed in PR #97 to use `np.nan`. These other estimators should follow the same pattern.

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
- `utils.py`: 481
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

