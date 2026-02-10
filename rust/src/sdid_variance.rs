//! SDID variance estimation acceleration.
//!
//! This module provides parallelized implementations of:
//! - Placebo-based variance estimation (Algorithm 4, Arkhangelsky et al. 2021)
//! - Bootstrap variance estimation with fixed weights
//!
//! Both functions parallelize across replications using rayon, providing
//! near-linear speedup on multi-core machines (e.g., ~8x on 8 cores).
//!
//! Reference:
//! Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
//! (2021). Synthetic Difference-in-Differences. American Economic Review,
//! 111(12), 4088-4118.

use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::weights;

/// Fisher-Yates permutation of 0..n using the provided RNG.
fn fisher_yates_permutation(rng: &mut impl rand::Rng, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        perm.swap(i, j);
    }
    perm
}

/// Extract columns from a matrix by index.
fn extract_submatrix_cols(matrix: &Array2<f64>, col_indices: &[usize]) -> Array2<f64> {
    let n_rows = matrix.nrows();
    let n_cols = col_indices.len();
    let mut result = Array2::zeros((n_rows, n_cols));
    for (new_j, &old_j) in col_indices.iter().enumerate() {
        for i in 0..n_rows {
            result[[i, new_j]] = matrix[[i, old_j]];
        }
    }
    result
}

/// Compute column means for selected columns.
fn column_means(matrix: &Array2<f64>, col_indices: &[usize]) -> Array1<f64> {
    let n_rows = matrix.nrows();
    let n_cols = col_indices.len();
    if n_cols == 0 {
        return Array1::zeros(n_rows);
    }
    let mut means = Array1::zeros(n_rows);
    for &j in col_indices {
        for i in 0..n_rows {
            means[i] += matrix[[i, j]];
        }
    }
    means /= n_cols as f64;
    means
}

/// Compute the SDID estimator (ATT) from pre/post control/treated data and weights.
///
/// Matches `compute_sdid_estimator` in Python's `utils.py:1587-1604`:
///
///     τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
///         - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)
fn sdid_estimator_internal(
    y_pre_control: &Array2<f64>,
    y_post_control: &Array2<f64>,
    y_pre_treated_mean: &Array1<f64>,
    y_post_treated_mean: &Array1<f64>,
    unit_weights: &Array1<f64>,
    time_weights: &Array1<f64>,
) -> f64 {
    // Weighted pre-treatment averages
    // time_weights @ Y_pre_control -> (n_control,)
    let weighted_pre_control = time_weights.dot(&y_pre_control.view());
    // time_weights @ Y_pre_treated_mean -> scalar
    let weighted_pre_treated = time_weights.dot(y_pre_treated_mean);

    // Post-treatment averages
    let mean_post_control = y_post_control.mean_axis(Axis(0)).unwrap();
    let mean_post_treated = y_post_treated_mean.mean().unwrap();

    // DiD for treated: post - weighted pre
    let did_treated = mean_post_treated - weighted_pre_treated;

    // Weighted DiD for controls: sum over j of omega_j * (post_j - weighted_pre_j)
    let diff_control = &mean_post_control - &weighted_pre_control;
    let did_control = unit_weights.dot(&diff_control);

    // SDID estimator
    did_treated - did_control
}

/// Compute placebo-based variance for SDID in parallel.
///
/// Implements Algorithm 4 from Arkhangelsky et al. (2021), matching R's
/// `synthdid::vcov(method = "placebo")`:
///
/// 1. Randomly permute control indices
/// 2. Split into pseudo-controls and pseudo-treated
/// 3. Re-estimate both omega and lambda on the permuted data
/// 4. Compute SDID estimate with re-estimated weights
/// 5. Repeat `replications` times (in parallel)
/// 6. SE = sqrt((r-1)/r) * sd(estimates)
///
/// # Arguments
/// * `y_pre_control` - Control outcomes in pre-treatment periods (n_pre, n_control)
/// * `y_post_control` - Control outcomes in post-treatment periods (n_post, n_control)
/// * `y_pre_treated_mean` - Mean treated outcomes in pre-treatment (n_pre,)
/// * `y_post_treated_mean` - Mean treated outcomes in post-treatment (n_post,)
/// * `n_treated` - Number of treated units in the original estimation
/// * `zeta_omega` - Regularization parameter for unit weights
/// * `zeta_lambda` - Regularization parameter for time weights
/// * `min_decrease` - Convergence threshold for Frank-Wolfe
/// * `intercept` - Column-center if true (default: true)
/// * `max_iter_pre_sparsify` - Iterations for first pass (default: 100)
/// * `max_iter` - Iterations for second pass (default: 10000)
/// * `replications` - Number of placebo replications
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// (se, placebo_estimates) where se is the standard error and
/// placebo_estimates is the array of successful placebo effects.
#[pyfunction]
#[pyo3(signature = (y_pre_control, y_post_control, y_pre_treated_mean, y_post_treated_mean,
                     n_treated, zeta_omega, zeta_lambda, min_decrease,
                     intercept=true, max_iter_pre_sparsify=100, max_iter=10000,
                     replications=200, seed=42))]
#[allow(clippy::too_many_arguments)]
pub fn placebo_variance_sdid<'py>(
    py: Python<'py>,
    y_pre_control: PyReadonlyArray2<'py, f64>,
    y_post_control: PyReadonlyArray2<'py, f64>,
    y_pre_treated_mean: PyReadonlyArray1<'py, f64>,
    y_post_treated_mean: PyReadonlyArray1<'py, f64>,
    n_treated: usize,
    zeta_omega: f64,
    zeta_lambda: f64,
    min_decrease: f64,
    intercept: bool,
    max_iter_pre_sparsify: usize,
    max_iter: usize,
    replications: usize,
    seed: u64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    // Convert to owned arrays for Send across threads
    let y_pre_c = y_pre_control.as_array().to_owned();
    let y_post_c = y_post_control.as_array().to_owned();
    let _y_pre_t_mean = y_pre_treated_mean.as_array().to_owned();
    let _y_post_t_mean = y_post_treated_mean.as_array().to_owned();

    let n_control = y_pre_c.ncols();

    // Check if we have enough controls for the split
    let n_pseudo_control = n_control.saturating_sub(n_treated);
    if n_pseudo_control < 1 {
        let empty = Array1::<f64>::zeros(0);
        return Ok((0.0, empty.to_pyarray_bound(py)));
    }

    // Parallel loop over replications
    let placebo_estimates: Vec<f64> = (0..replications)
        .into_par_iter()
        .filter_map(|b| {
            use rand::prelude::*;
            use rand_xoshiro::Xoshiro256PlusPlus;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(b as u64));

            // Generate random permutation of control indices
            let perm = fisher_yates_permutation(&mut rng, n_control);

            // Split into pseudo-controls and pseudo-treated
            let pseudo_control_idx = &perm[..n_pseudo_control];
            let pseudo_treated_idx = &perm[n_pseudo_control..];

            // Extract submatrices
            let y_pre_pseudo_control = extract_submatrix_cols(&y_pre_c, pseudo_control_idx);
            let y_post_pseudo_control = extract_submatrix_cols(&y_post_c, pseudo_control_idx);
            let y_pre_pseudo_treated_mean = column_means(&y_pre_c, pseudo_treated_idx);
            let y_post_pseudo_treated_mean = column_means(&y_post_c, pseudo_treated_idx);

            // Re-estimate unit weights on permuted data
            let pseudo_omega = weights::compute_sdid_unit_weights_internal(
                &y_pre_pseudo_control.view(),
                &y_pre_pseudo_treated_mean.view(),
                zeta_omega,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            );

            // Re-estimate time weights on permuted data
            let pseudo_lambda = weights::compute_time_weights_internal(
                &y_pre_pseudo_control.view(),
                &y_post_pseudo_control.view(),
                zeta_lambda,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            );

            // Compute placebo SDID estimate
            let tau = sdid_estimator_internal(
                &y_pre_pseudo_control,
                &y_post_pseudo_control,
                &y_pre_pseudo_treated_mean,
                &y_post_pseudo_treated_mean,
                &pseudo_omega,
                &pseudo_lambda,
            );

            if tau.is_finite() {
                Some(tau)
            } else {
                None
            }
        })
        .collect();

    let n_successful = placebo_estimates.len();

    // Compute SE: sqrt((r-1)/r) * sd(estimates, ddof=1) matching R's formula
    let se = if n_successful < 2 {
        0.0
    } else {
        let n = n_successful as f64;
        let mean = placebo_estimates.iter().sum::<f64>() / n;
        let variance = placebo_estimates
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        ((n - 1.0) / n).sqrt() * variance.sqrt()
    };

    let estimates_arr = Array1::from_vec(placebo_estimates);
    Ok((se, estimates_arr.to_pyarray_bound(py)))
}

/// Compute bootstrap variance for SDID in parallel.
///
/// Resamples all units (control + treated) with replacement, renormalizes
/// original unit weights for the resampled controls, and computes the
/// SDID estimator with **fixed** weights (no re-estimation).
///
/// This matches R's `synthdid::vcov(method="bootstrap")`.
///
/// # Arguments
/// * `y_pre_control` - Control outcomes in pre-treatment (n_pre, n_control)
/// * `y_post_control` - Control outcomes in post-treatment (n_post, n_control)
/// * `y_pre_treated` - Treated outcomes in pre-treatment (n_pre, n_treated)
/// * `y_post_treated` - Treated outcomes in post-treatment (n_post, n_treated)
/// * `unit_weights` - Original unit weights (n_control,)
/// * `time_weights` - Original time weights (n_pre,)
/// * `n_bootstrap` - Number of bootstrap iterations
/// * `seed` - Random seed
///
/// # Returns
/// (se, bootstrap_estimates, n_failed)
#[pyfunction]
#[pyo3(signature = (y_pre_control, y_post_control, y_pre_treated, y_post_treated,
                     unit_weights, time_weights, n_bootstrap, seed))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_variance_sdid<'py>(
    py: Python<'py>,
    y_pre_control: PyReadonlyArray2<'py, f64>,
    y_post_control: PyReadonlyArray2<'py, f64>,
    y_pre_treated: PyReadonlyArray2<'py, f64>,
    y_post_treated: PyReadonlyArray2<'py, f64>,
    unit_weights: PyReadonlyArray1<'py, f64>,
    time_weights: PyReadonlyArray1<'py, f64>,
    n_bootstrap: usize,
    seed: u64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, usize)> {
    // Convert to owned arrays for Send across threads
    let y_pre_c = y_pre_control.as_array().to_owned();
    let y_post_c = y_post_control.as_array().to_owned();
    let y_pre_t = y_pre_treated.as_array().to_owned();
    let y_post_t = y_post_treated.as_array().to_owned();
    let omega = unit_weights.as_array().to_owned();
    let lambda = time_weights.as_array().to_owned();

    let n_control = y_pre_c.ncols();
    let n_treated = y_pre_t.ncols();
    let n_total = n_control + n_treated;
    let n_pre = y_pre_c.nrows();

    // Build full panel: (n_pre+n_post, n_control+n_treated)
    let n_post = y_post_c.nrows();
    let n_times = n_pre + n_post;
    let mut y_full = Array2::zeros((n_times, n_total));
    for t in 0..n_pre {
        for j in 0..n_control {
            y_full[[t, j]] = y_pre_c[[t, j]];
        }
        for j in 0..n_treated {
            y_full[[t, n_control + j]] = y_pre_t[[t, j]];
        }
    }
    for t in 0..n_post {
        for j in 0..n_control {
            y_full[[n_pre + t, j]] = y_post_c[[t, j]];
        }
        for j in 0..n_treated {
            y_full[[n_pre + t, n_control + j]] = y_post_t[[t, j]];
        }
    }

    // Parallel bootstrap loop
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            use rand::prelude::*;
            use rand_xoshiro::Xoshiro256PlusPlus;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(b as u64));

            // Resample all units with replacement
            let boot_idx: Vec<usize> = (0..n_total)
                .map(|_| rng.gen_range(0..n_total))
                .collect();

            // Identify control vs treated in resampled set
            let mut boot_control_idx = Vec::new();
            let mut boot_treated_idx = Vec::new();
            for &idx in &boot_idx {
                if idx < n_control {
                    boot_control_idx.push(idx);
                } else {
                    boot_treated_idx.push(idx);
                }
            }

            // Skip degenerate samples
            if boot_control_idx.is_empty() || boot_treated_idx.is_empty() {
                return None;
            }

            // Renormalize unit weights for resampled controls
            let boot_omega_raw: Array1<f64> =
                Array1::from_iter(boot_control_idx.iter().map(|&i| omega[i]));
            let boot_omega = weights::sum_normalize_internal(&boot_omega_raw);

            let n_boot_control = boot_control_idx.len();
            let n_boot_treated = boot_treated_idx.len();

            // Extract resampled submatrices
            let mut y_boot_pre_c = Array2::zeros((n_pre, n_boot_control));
            let mut y_boot_post_c = Array2::zeros((n_post, n_boot_control));
            for (new_j, &old_j) in boot_control_idx.iter().enumerate() {
                for t in 0..n_pre {
                    y_boot_pre_c[[t, new_j]] = y_full[[t, old_j]];
                }
                for t in 0..n_post {
                    y_boot_post_c[[t, new_j]] = y_full[[n_pre + t, old_j]];
                }
            }

            // Compute treated means for resampled treated units
            let mut y_boot_pre_t_mean = Array1::zeros(n_pre);
            let mut y_boot_post_t_mean = Array1::zeros(n_post);
            for &old_j in &boot_treated_idx {
                for t in 0..n_pre {
                    y_boot_pre_t_mean[t] += y_full[[t, old_j]];
                }
                for t in 0..n_post {
                    y_boot_post_t_mean[t] += y_full[[n_pre + t, old_j]];
                }
            }
            y_boot_pre_t_mean /= n_boot_treated as f64;
            y_boot_post_t_mean /= n_boot_treated as f64;

            // Compute ATT with FIXED time_weights and renormalized omega
            let tau = sdid_estimator_internal(
                &y_boot_pre_c,
                &y_boot_post_c,
                &y_boot_pre_t_mean,
                &y_boot_post_t_mean,
                &boot_omega,
                &lambda,
            );

            if tau.is_finite() {
                Some(tau)
            } else {
                None
            }
        })
        .collect();

    let n_successful = bootstrap_estimates.len();
    let n_failed = n_bootstrap - n_successful;

    // Compute SE: std(estimates, ddof=1) — NO sqrt((r-1)/r) for bootstrap
    let se = if n_successful < 2 {
        0.0
    } else {
        let n = n_successful as f64;
        let mean = bootstrap_estimates.iter().sum::<f64>() / n;
        let variance = bootstrap_estimates
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    };

    let estimates_arr = Array1::from_vec(bootstrap_estimates);
    Ok((se, estimates_arr.to_pyarray_bound(py), n_failed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fisher_yates_permutation_length() {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let perm = fisher_yates_permutation(&mut rng, 10);
        assert_eq!(perm.len(), 10);
        // All elements should be present
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_fisher_yates_permutation_different_seeds() {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(43);
        let perm1 = fisher_yates_permutation(&mut rng1, 20);
        let perm2 = fisher_yates_permutation(&mut rng2, 20);
        assert_ne!(perm1, perm2);
    }

    #[test]
    fn test_extract_submatrix_cols() {
        let m = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let sub = extract_submatrix_cols(&m, &[0, 2]);
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert!((sub[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((sub[[0, 1]] - 3.0).abs() < 1e-10);
        assert!((sub[[1, 0]] - 5.0).abs() < 1e-10);
        assert!((sub[[1, 1]] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_means() {
        let m = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let means = column_means(&m, &[1, 3]);
        // Mean of columns 1 and 3:
        // Row 0: (2+4)/2 = 3
        // Row 1: (6+8)/2 = 7
        assert!((means[0] - 3.0).abs() < 1e-10);
        assert!((means[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdid_estimator_internal_zero_weights() {
        // When all time weights are on last pre-period and unit weights are uniform,
        // should reduce to a simpler DiD
        let y_pre_c = array![[1.0, 2.0], [3.0, 4.0]]; // (2 pre, 2 control)
        let y_post_c = array![[5.0, 6.0]]; // (1 post, 2 control)
        let y_pre_t = array![2.0, 4.0]; // (2 pre,) means
        let y_post_t = array![10.0]; // (1 post,) means

        let omega = array![0.5, 0.5]; // uniform
        let lambda = array![0.0, 1.0]; // all weight on last pre-period

        let tau = sdid_estimator_internal(
            &y_pre_c, &y_post_c, &y_pre_t, &y_post_t, &omega, &lambda,
        );

        // Manual computation:
        // weighted_pre_control = [0, 1] @ [[1,2],[3,4]] = [3, 4]
        // weighted_pre_treated = [0, 1] @ [2, 4] = 4
        // mean_post_control = [5, 6]
        // mean_post_treated = 10
        // did_treated = 10 - 4 = 6
        // did_control = [0.5, 0.5] @ ([5,6] - [3,4]) = [0.5, 0.5] @ [2, 2] = 2
        // tau = 6 - 2 = 4
        assert!((tau - 4.0).abs() < 1e-10, "Expected 4.0, got {}", tau);
    }
}
