use num_complex::{Complex32, Complex64};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::f64::consts::PI;

const QUARTER_PI: f64 = PI / 4.0;
const QUARTER_PI_F32: f32 = std::f32::consts::PI / 4.0;
const STAGE8_NOISE_SCALE: f32 = 0.5;

struct RowData {
    valid_cols: Vec<usize>,
    cpx: Vec<Complex64>,
    bp: Vec<f64>,
    weighting: Vec<f64>,
    wb: Vec<f64>,
    den_lin: f64,
    bperp_range: f64,
    denom: f64,
    n_col: usize,
}

struct RowDataSingle {
    valid_cols: Vec<usize>,
    cpx: Vec<Complex32>,
    bp: Vec<f32>,
    weighting: Vec<f32>,
    wb: Vec<f32>,
    den_lin: f32,
    bperp_range: f32,
    denom: f32,
    n_col: usize,
}

struct RefinedRow {
    k: f64,
    c: f64,
    coh: f64,
    residual: Vec<Complex32>,
}

struct Stage7Outputs {
    k_ps_uw: Vec<f64>,
    c_ps_uw: Vec<f32>,
    ph_scla: Vec<f32>,
    ifg_vcm: Vec<f64>,
    mean_v: Vec<f32>,
    m: Vec<f32>,
    ph_ramp: Vec<f64>,
}

struct Stage7RowResult {
    k: f64,
    c: f32,
    ph_scla_row: Vec<f32>,
    mean_v: f32,
    intercept: f32,
    slope: f32,
}

fn build_pool(thread_count: usize) -> PyResult<Option<ThreadPool>> {
    if thread_count <= 1 {
        return Ok(None);
    }
    ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map(Some)
        .map_err(|err| PyValueError::new_err(format!("failed to build stage-2 thread pool: {err}")))
}

fn trial_values(n_trial_wraps: f64) -> Vec<f64> {
    let trial_n = (8.0 * n_trial_wraps).ceil() as i64;
    (-trial_n..=trial_n).map(|value| value as f64).collect()
}

fn parse_indices(values: &[i64], upper_bound: usize, label: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        if value < 0 {
            return Err(PyValueError::new_err(format!("{label} entries must be non-negative")));
        }
        let idx = value as usize;
        if idx >= upper_bound {
            return Err(PyValueError::new_err(format!("{label} entry {idx} exceeds width {upper_bound}")));
        }
        out.push(idx);
    }
    Ok(out)
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
        for row in (pivot_col + 1)..n {
            let value_abs = matrix[row * n + pivot_col].abs();
            if value_abs > pivot_abs {
                pivot_row = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= 1.0e-12 {
            return None;
        }
        if pivot_row != pivot_col {
            for col in 0..n {
                matrix.swap(pivot_col * n + col, pivot_row * n + col);
            }
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = matrix[pivot_col * n + pivot_col];
        for row in (pivot_col + 1)..n {
            let factor = matrix[row * n + pivot_col] / pivot;
            if factor == 0.0 {
                continue;
            }
            matrix[row * n + pivot_col] = 0.0;
            for col in (pivot_col + 1)..n {
                matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    let mut out = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..n {
            value -= matrix[row * n + col] * out[col];
        }
        let diag = matrix[row * n + row];
        if diag.abs() <= 1.0e-12 {
            return None;
        }
        out[row] = value / diag;
    }
    Some(out)
}

fn invert_small_matrix(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; n * n];
    for col in 0..n {
        let mut basis = vec![0.0; n];
        basis[col] = 1.0;
        let solution = solve_linear_system(matrix.to_vec(), basis, n)?;
        for row in 0..n {
            inverse[row * n + col] = solution[row];
        }
    }
    Some(inverse)
}

fn invert_small_matrix_with_jitter(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut jitter = 0.0_f64;
    loop {
        let mut adjusted = matrix.to_vec();
        for diag in 0..n {
            adjusted[diag * n + diag] += jitter;
        }
        if let Some(inverse) = invert_small_matrix(&adjusted, n) {
            return inverse;
        }
        jitter = if jitter == 0.0 { 1.0e-10 } else { jitter * 10.0 };
        if jitter > 1.0e-3 {
            let mut identity = vec![0.0; n * n];
            for diag in 0..n {
                identity[diag * n + diag] = 1.0;
            }
            return identity;
        }
    }
}

fn design_gram(design: &[f64], n_obs: usize, n_coeff: usize) -> Vec<f64> {
    let mut gram = vec![0.0; n_coeff * n_coeff];
    for row in 0..n_obs {
        let row_slice = &design[row * n_coeff..(row + 1) * n_coeff];
        for left in 0..n_coeff {
            for right in 0..n_coeff {
                gram[left * n_coeff + right] += row_slice[left] * row_slice[right];
            }
        }
    }
    gram
}

fn design_rhs(design: &[f64], y: &[f64], n_obs: usize, n_coeff: usize) -> Vec<f64> {
    let mut rhs = vec![0.0; n_coeff];
    for row in 0..n_obs {
        let y_value = y[row];
        let row_slice = &design[row * n_coeff..(row + 1) * n_coeff];
        for coeff_ix in 0..n_coeff {
            rhs[coeff_ix] += row_slice[coeff_ix] * y_value;
        }
    }
    rhs
}

fn mat_vec(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for row in 0..n {
        let mut accum = 0.0;
        for col in 0..n {
            accum += matrix[row * n + col] * vector[col];
        }
        out[row] = accum;
    }
    out
}

fn stage7_outputs(
    ph_proc: &[f64],
    ph_mean_v: &[f64],
    bperp_mat: &[f64],
    n_ps: usize,
    n_ifg: usize,
    unwrap_ix: &[usize],
    solve_ix: &[usize],
    day: &[f64],
    master_ix: usize,
    ifg_std: &[f64],
    threads: usize,
) -> PyResult<Stage7Outputs> {
    if unwrap_ix.len() < 2 {
        return Err(PyValueError::new_err("stage7_scla requires at least two unwrap indices"));
    }
    if solve_ix.len() < 2 {
        return Err(PyValueError::new_err("stage7_scla requires at least two solve indices"));
    }
    if master_ix == 0 || master_ix > n_ifg {
        return Err(PyValueError::new_err("master_ix must be 1-based within the interferogram width"));
    }
    if day.len() != n_ifg || ifg_std.len() != n_ifg {
        return Err(PyValueError::new_err("day and ifg_std must match the interferogram width"));
    }

    let unwrap_obs = unwrap_ix.len() - 1;
    let coest_mean_vel = unwrap_ix.len() >= 4;
    let seq_coeff = if coest_mean_vel { 3 } else { 2 };
    let mut mean_bperp = vec![0.0; unwrap_obs];
    for obs_ix in 0..unwrap_obs {
        let left_col = unwrap_ix[obs_ix];
        let right_col = unwrap_ix[obs_ix + 1];
        let mut accum = 0.0;
        for row_ix in 0..n_ps {
            accum += bperp_mat[row_ix * n_ifg + right_col] - bperp_mat[row_ix * n_ifg + left_col];
        }
        mean_bperp[obs_ix] = accum / n_ps as f64;
    }

    let mut design_seq = vec![0.0; unwrap_obs * seq_coeff];
    for obs_ix in 0..unwrap_obs {
        let day_diff = day[unwrap_ix[obs_ix + 1]] - day[unwrap_ix[obs_ix]];
        let row_offset = obs_ix * seq_coeff;
        design_seq[row_offset] = 1.0;
        design_seq[row_offset + 1] = mean_bperp[obs_ix];
        if coest_mean_vel {
            design_seq[row_offset + 2] = day_diff;
        }
    }
    let inv_seq = invert_small_matrix_with_jitter(&design_gram(&design_seq, unwrap_obs, seq_coeff), seq_coeff);

    let master_zero = day[master_ix - 1];
    let mut solve_design = Vec::new();
    let mut solve_scales = Vec::new();
    let inv_c = if coest_mean_vel {
        solve_design = vec![0.0; solve_ix.len() * 2];
        solve_scales = vec![1.0; solve_ix.len()];
        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
            let scale = if ifg_std[ifg_ix] > 0.0 {
                ifg_std[ifg_ix] * PI / 180.0
            } else {
                1.0
            };
            solve_scales[obs_ix] = scale;
            let row_offset = obs_ix * 2;
            solve_design[row_offset] = 1.0 / scale;
            solve_design[row_offset + 1] = (day[ifg_ix] - master_zero) / scale;
        }
        Some(invert_small_matrix_with_jitter(
            &design_gram(&solve_design, solve_ix.len(), 2),
            2,
        ))
    } else {
        None
    };

    let time_diff: Vec<f64> = day.iter().map(|&value| value - master_zero).collect();
    let weights_mv: Vec<f64> = ifg_std
        .iter()
        .map(|&std| {
            if std > 0.0 {
                1.0 / ((std * PI / 180.0) * (std * PI / 180.0))
            } else {
                0.0
            }
        })
        .collect();
    let s0: f64 = weights_mv.iter().sum();
    let s1: f64 = weights_mv.iter().zip(time_diff.iter()).map(|(&w, &t)| w * t).sum();
    let s2: f64 = weights_mv
        .iter()
        .zip(time_diff.iter())
        .map(|(&w, &t)| w * t * t)
        .sum();
    let det = s0 * s2 - s1 * s1;

    let ifg_var: Vec<f64> = ifg_std.iter().map(|&std| (std * PI / 180.0) * (std * PI / 180.0)).collect();
    let mut ifg_vcm = vec![0.0; n_ifg * n_ifg];
    for diag_ix in 0..n_ifg {
        ifg_vcm[diag_ix * n_ifg + diag_ix] = ifg_var[diag_ix];
    }

    let pool = build_pool(threads)?;
    let row_results = {
        let compute = || {
            (0..n_ps)
                .into_par_iter()
                .map(|row_ix| {
                    let row_offset = row_ix * n_ifg;
                    let mut ph_seq = vec![0.0; unwrap_obs];
                    for obs_ix in 0..unwrap_obs {
                        ph_seq[obs_ix] =
                            ph_proc[row_offset + unwrap_ix[obs_ix + 1]] - ph_proc[row_offset + unwrap_ix[obs_ix]];
                    }
                    let coeff_seq = mat_vec(&inv_seq, &design_rhs(&design_seq, &ph_seq, unwrap_obs, seq_coeff), seq_coeff);
                    let k = coeff_seq[1];

                    let mut ph_scla_row = vec![0.0_f32; n_ifg];
                    for ifg_ix in 0..n_ifg {
                        ph_scla_row[ifg_ix] = (k * bperp_mat[row_offset + ifg_ix]) as f32;
                    }

                    let c = if let Some(inv_c_ref) = inv_c.as_ref() {
                        let mut resid_weighted = vec![0.0; solve_ix.len()];
                        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
                            let resid = ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                            resid_weighted[obs_ix] = resid / solve_scales[obs_ix];
                        }
                        mat_vec(inv_c_ref, &design_rhs(&solve_design, &resid_weighted, solve_ix.len(), 2), 2)[0] as f32
                    } else {
                        let mut accum = 0.0;
                        for &ifg_ix in solve_ix {
                            accum += ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                        }
                        (accum / solve_ix.len() as f64) as f32
                    };

                    let mut wy0 = 0.0;
                    let mut wy1 = 0.0;
                    for ifg_ix in 0..n_ifg {
                        let y = ph_mean_v[row_offset + ifg_ix];
                        let weight = weights_mv[ifg_ix];
                        wy0 += y * weight;
                        wy1 += y * weight * time_diff[ifg_ix];
                    }
                    let (intercept, slope) = if det.abs() <= 1.0e-12 {
                        let base = if s0 != 0.0 { wy0 / s0 } else { 0.0 };
                        (base as f32, 0.0_f32)
                    } else {
                        (
                            ((wy0 * s2 - wy1 * s1) / det) as f32,
                            ((wy1 * s0 - wy0 * s1) / det) as f32,
                        )
                    };

                    Stage7RowResult {
                        k,
                        c,
                        ph_scla_row,
                        mean_v: slope,
                        intercept,
                        slope,
                    }
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_ps)
                .map(|row_ix| {
                    let row_offset = row_ix * n_ifg;
                    let mut ph_seq = vec![0.0; unwrap_obs];
                    for obs_ix in 0..unwrap_obs {
                        ph_seq[obs_ix] =
                            ph_proc[row_offset + unwrap_ix[obs_ix + 1]] - ph_proc[row_offset + unwrap_ix[obs_ix]];
                    }
                    let coeff_seq = mat_vec(&inv_seq, &design_rhs(&design_seq, &ph_seq, unwrap_obs, seq_coeff), seq_coeff);
                    let k = coeff_seq[1];

                    let mut ph_scla_row = vec![0.0_f32; n_ifg];
                    for ifg_ix in 0..n_ifg {
                        ph_scla_row[ifg_ix] = (k * bperp_mat[row_offset + ifg_ix]) as f32;
                    }

                    let c = if let Some(inv_c_ref) = inv_c.as_ref() {
                        let mut resid_weighted = vec![0.0; solve_ix.len()];
                        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
                            let resid = ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                            resid_weighted[obs_ix] = resid / solve_scales[obs_ix];
                        }
                        mat_vec(inv_c_ref, &design_rhs(&solve_design, &resid_weighted, solve_ix.len(), 2), 2)[0] as f32
                    } else {
                        let mut accum = 0.0;
                        for &ifg_ix in solve_ix {
                            accum += ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                        }
                        (accum / solve_ix.len() as f64) as f32
                    };

                    let mut wy0 = 0.0;
                    let mut wy1 = 0.0;
                    for ifg_ix in 0..n_ifg {
                        let y = ph_mean_v[row_offset + ifg_ix];
                        let weight = weights_mv[ifg_ix];
                        wy0 += y * weight;
                        wy1 += y * weight * time_diff[ifg_ix];
                    }
                    let (intercept, slope) = if det.abs() <= 1.0e-12 {
                        let base = if s0 != 0.0 { wy0 / s0 } else { 0.0 };
                        (base as f32, 0.0_f32)
                    } else {
                        (
                            ((wy0 * s2 - wy1 * s1) / det) as f32,
                            ((wy1 * s0 - wy0 * s1) / det) as f32,
                        )
                    };

                    Stage7RowResult {
                        k,
                        c,
                        ph_scla_row,
                        mean_v: slope,
                        intercept,
                        slope,
                    }
                })
                .collect::<Vec<_>>(),
        }
    };

    let k_ps_uw: Vec<f64> = row_results.iter().map(|row| row.k).collect();
    let c_ps_uw: Vec<f32> = row_results.iter().map(|row| row.c).collect();
    let mean_v: Vec<f32> = row_results.iter().map(|row| row.mean_v).collect();
    let mut ph_scla = vec![0.0_f32; n_ps * n_ifg];
    let mut m = vec![0.0_f32; 2 * n_ps];
    for (row_ix, row) in row_results.iter().enumerate() {
        ph_scla[row_ix * n_ifg..(row_ix + 1) * n_ifg].copy_from_slice(&row.ph_scla_row);
        m[row_ix] = row.intercept;
        m[n_ps + row_ix] = row.slope;
    }

    Ok(Stage7Outputs {
        k_ps_uw,
        c_ps_uw,
        ph_scla,
        ifg_vcm,
        mean_v,
        m,
        ph_ramp: vec![0.0; n_ps * n_ifg],
    })
}

fn argmax_first(values: &[f64]) -> usize {
    let mut best_ix = 0usize;
    let mut best_value = values.first().copied().unwrap_or(f64::NEG_INFINITY);
    for (idx, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_ix = idx;
            best_value = value;
        }
    }
    best_ix
}

const STAGE2_TOPOFIT_NEAR_MAX_COH_TOL: f64 = 5.0e-3;

fn near_max_trial_indices(coh_trial: &[f64]) -> Vec<usize> {
    if coh_trial.len() <= 1 {
        return vec![0];
    }

    let mut local_max = vec![false; coh_trial.len()];
    local_max[0] = coh_trial[0] >= coh_trial[1];
    local_max[coh_trial.len() - 1] = coh_trial[coh_trial.len() - 1] >= coh_trial[coh_trial.len() - 2];
    if coh_trial.len() > 2 {
        for idx in 1..coh_trial.len() - 1 {
            local_max[idx] = coh_trial[idx] >= coh_trial[idx - 1] && coh_trial[idx] >= coh_trial[idx + 1];
        }
    }

    let max_coh = coh_trial.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut candidate_ix = local_max
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_local_max)| {
            if is_local_max && coh_trial[idx] >= max_coh - STAGE2_TOPOFIT_NEAR_MAX_COH_TOL {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if candidate_ix.is_empty() {
        candidate_ix.push(argmax_first(coh_trial));
    }
    candidate_ix
}

fn select_candidate(candidate_ix: &[usize], candidate_coh: &[f64], refined_coh: &[f64], trial_count: usize) -> usize {
    if candidate_ix.is_empty() {
        return 0;
    }

    let coarse_best_local = argmax_first(candidate_coh);
    let coarse_best_trial_ix = candidate_ix[coarse_best_local];
    if candidate_ix.len() == 1 {
        return coarse_best_trial_ix;
    }

    let endpoint_symmetric =
        candidate_ix.len() == 2 && candidate_ix[0] == 0 && candidate_ix[candidate_ix.len() - 1] == trial_count - 1;
    if endpoint_symmetric {
        return coarse_best_trial_ix;
    }

    candidate_ix[argmax_first(refined_coh)]
}

fn collect_row(cpx_row: &[Complex64], bp_row: &[f64]) -> RowData {
    let mut valid_cols = Vec::with_capacity(cpx_row.len());
    let mut cpx = Vec::with_capacity(cpx_row.len());
    let mut bp = Vec::with_capacity(cpx_row.len());
    let mut weighting = Vec::with_capacity(cpx_row.len());
    let mut wb = Vec::with_capacity(cpx_row.len());

    let mut denom = 0.0;
    let mut den_lin = 0.0;
    let mut bperp_min = 0.0;
    let mut bperp_max = 0.0;
    let mut first_valid = true;

    for (col, (&value, &bp_value)) in cpx_row.iter().zip(bp_row.iter()).enumerate() {
        if value == Complex64::new(0.0, 0.0) {
            continue;
        }
        if first_valid {
            bperp_min = bp_value;
            bperp_max = bp_value;
            first_valid = false;
        } else {
            bperp_min = bperp_min.min(bp_value);
            bperp_max = bperp_max.max(bp_value);
        }

        let weight = value.norm();
        let wb_value = weight * bp_value;

        valid_cols.push(col);
        cpx.push(value);
        bp.push(bp_value);
        weighting.push(weight);
        wb.push(wb_value);
        denom += weight;
        den_lin += wb_value * wb_value;
    }

    if denom == 0.0 {
        denom = 1.0;
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }

    let mut bperp_range = if valid_cols.is_empty() { 1.0 } else { bperp_max - bperp_min };
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    RowData {
        valid_cols,
        cpx,
        bp,
        weighting,
        wb,
        den_lin,
        bperp_range,
        denom,
        n_col: cpx_row.len(),
    }
}

fn collect_row_single(cpx_row: &[Complex32], bp_row: &[f32]) -> RowDataSingle {
    let mut valid_cols = Vec::with_capacity(cpx_row.len());
    let mut cpx = Vec::with_capacity(cpx_row.len());
    let mut bp = Vec::with_capacity(cpx_row.len());
    let mut weighting = Vec::with_capacity(cpx_row.len());
    let mut wb = Vec::with_capacity(cpx_row.len());

    let mut denom = 0.0_f32;
    let mut den_lin = 0.0_f32;
    let mut bperp_min = 0.0_f32;
    let mut bperp_max = 0.0_f32;
    let mut first_valid = true;

    for (col, (&value, &bp_value)) in cpx_row.iter().zip(bp_row.iter()).enumerate() {
        if value == Complex32::new(0.0, 0.0) {
            continue;
        }
        if first_valid {
            bperp_min = bp_value;
            bperp_max = bp_value;
            first_valid = false;
        } else {
            bperp_min = bperp_min.min(bp_value);
            bperp_max = bperp_max.max(bp_value);
        }

        let weight = value.norm();
        let wb_value = weight * bp_value;

        valid_cols.push(col);
        cpx.push(value);
        bp.push(bp_value);
        weighting.push(weight);
        wb.push(wb_value);
        denom += weight;
        den_lin += wb_value * wb_value;
    }

    if denom == 0.0 {
        denom = 1.0;
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }

    let mut bperp_range = if valid_cols.is_empty() { 1.0 } else { bperp_max - bperp_min };
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    RowDataSingle {
        valid_cols,
        cpx,
        bp,
        weighting,
        wb,
        den_lin,
        bperp_range,
        denom,
        n_col: cpx_row.len(),
    }
}

fn coherence_trials_generic(row: &RowData, trial_mult: &[f64]) -> Vec<f64> {
    let mut coh_trial = vec![0.0; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for idx in 0..row.cpx.len() {
            let phase = (row.bp[idx] / row.bperp_range) * QUARTER_PI * trial_value;
            let (sn, cs) = phase.sin_cos();
            let ph_re = row.cpx[idx].re;
            let ph_im = row.cpx[idx].im;
            sum_re += (ph_re * cs) + (ph_im * sn);
            sum_im += (ph_im * cs) - (ph_re * sn);
        }
        coh_trial[trial_ix] = sum_re.hypot(sum_im) / row.denom;
    }
    coh_trial
}

fn coherence_trials_generic_single(row: &RowDataSingle, trial_mult: &[f32]) -> Vec<f32> {
    let mut coh_trial = vec![0.0_f32; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut phaser_sum = Complex32::new(0.0, 0.0);
        for idx in 0..row.cpx.len() {
            let phase = (row.bp[idx] / row.bperp_range) * QUARTER_PI_F32 * trial_value;
            let (sn, cs) = phase.sin_cos();
            phaser_sum += row.cpx[idx] * Complex32::new(cs, -sn);
        }
        coh_trial[trial_ix] = phaser_sum.norm() / row.denom;
    }
    coh_trial
}

fn coherence_trials_row_invariant(row: &RowData, basis: &[Complex64], trial_count: usize) -> Vec<f64> {
    let mut coh_trial = vec![0.0; trial_count];
    for trial_ix in 0..trial_count {
        let basis_row = &basis[trial_ix * row.n_col..(trial_ix + 1) * row.n_col];
        let mut phaser_sum = Complex64::new(0.0, 0.0);
        for (idx, &col) in row.valid_cols.iter().enumerate() {
            phaser_sum += row.cpx[idx] * basis_row[col];
        }
        coh_trial[trial_ix] = phaser_sum.norm() / row.denom;
    }
    coh_trial
}

fn refine_candidate(row: &RowData, coarse_k0: f64, store_phase: bool) -> RefinedRow {
    let mut offset = Complex64::new(0.0, 0.0);
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        offset += row.cpx[idx] * Complex64::new(cs, -sn);
    }

    let offset_conj = offset.conj();
    let mut mopt_num = 0.0;
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex64::new(cs, -sn);
        let angle = (res * offset_conj).arg();
        mopt_num += row.wb[idx] * (row.weighting[idx] * angle);
    }

    let k = coarse_k0 + (mopt_num / row.den_lin);
    let mut mean_phase_residual = Complex64::new(0.0, 0.0);
    let mut denom2 = 0.0;
    let mut residual = if store_phase {
        vec![Complex32::new(0.0, 0.0); row.n_col]
    } else {
        Vec::new()
    };

    for (idx, &col) in row.valid_cols.iter().enumerate() {
        let phase = k * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex64::new(cs, -sn);
        mean_phase_residual += res;
        denom2 += res.norm();
        if store_phase {
            residual[col] = Complex32::new(res.re as f32, res.im as f32);
        }
    }

    if denom2 == 0.0 {
        denom2 = 1.0;
    }

    RefinedRow {
        k,
        c: mean_phase_residual.arg(),
        coh: mean_phase_residual.norm() / denom2,
        residual,
    }
}

fn refine_candidate_single(row: &RowDataSingle, coarse_k0: f32, store_phase: bool) -> RefinedRow {
    let mut offset = Complex32::new(0.0, 0.0);
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        offset += row.cpx[idx] * Complex32::new(cs, -sn);
    }

    let offset_conj = offset.conj();
    let mut mopt_num = 0.0_f32;
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex32::new(cs, -sn);
        let angle = (res * offset_conj).arg();
        mopt_num += row.wb[idx] * (row.weighting[idx] * angle);
    }

    let k = coarse_k0 + (mopt_num / row.den_lin);
    let mut mean_phase_residual = Complex32::new(0.0, 0.0);
    let mut denom2 = 0.0_f32;
    let mut residual = if store_phase {
        vec![Complex32::new(0.0, 0.0); row.n_col]
    } else {
        Vec::new()
    };

    for (idx, &col) in row.valid_cols.iter().enumerate() {
        let phase = k * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex32::new(cs, -sn);
        mean_phase_residual += res;
        denom2 += res.norm();
        if store_phase {
            residual[col] = res;
        }
    }

    if denom2 == 0.0 {
        denom2 = 1.0;
    }

    RefinedRow {
        k: k as f64,
        c: mean_phase_residual.arg() as f64,
        coh: (mean_phase_residual.norm() / denom2) as f64,
        residual,
    }
}

fn solve_row_generic(cpx_row: &[Complex64], bp_row: &[f64], trial_mult: &[f64], store_phase: bool) -> RefinedRow {
    let row = collect_row(cpx_row, bp_row);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_generic(&row, trial_mult);
    solve_row_from_trials(&row, trial_mult, &coh_trial, store_phase)
}

fn solve_row_generic_single(cpx_row: &[Complex32], bp_row: &[f32], trial_mult: &[f32], store_phase: bool) -> RefinedRow {
    let row = collect_row_single(cpx_row, bp_row);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_generic_single(&row, trial_mult);
    let coh_trial_f64: Vec<f64> = coh_trial.iter().map(|&value| value as f64).collect();
    let candidate_ix = near_max_trial_indices(&coh_trial_f64);
    if candidate_ix.len() == 1 {
        let coarse_k0 = QUARTER_PI_F32 / row.bperp_range * trial_mult[candidate_ix[0]];
        return refine_candidate_single(&row, coarse_k0, store_phase);
    }

    let mut refined = Vec::with_capacity(candidate_ix.len());
    let mut candidate_coh = Vec::with_capacity(candidate_ix.len());
    for &trial_ix in &candidate_ix {
        let coarse_k0 = QUARTER_PI_F32 / row.bperp_range * trial_mult[trial_ix];
        refined.push(refine_candidate_single(&row, coarse_k0, store_phase));
        candidate_coh.push(coh_trial_f64[trial_ix]);
    }
    let refined_coh = refined.iter().map(|row| row.coh).collect::<Vec<_>>();
    let selected_trial_ix = select_candidate(&candidate_ix, &candidate_coh, &refined_coh, trial_mult.len());
    let selected_local_ix = candidate_ix
        .iter()
        .position(|&trial_ix| trial_ix == selected_trial_ix)
        .unwrap_or(0);
    refined.remove(selected_local_ix)
}

fn solve_row_row_invariant(
    cpx_row: &[Complex64],
    bp_vec: &[f64],
    trial_mult: &[f64],
    basis: &[Complex64],
    store_phase: bool,
) -> RefinedRow {
    let row = collect_row(cpx_row, bp_vec);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_row_invariant(&row, basis, trial_mult.len());
    solve_row_from_trials(&row, trial_mult, &coh_trial, store_phase)
}

fn solve_row_from_trials(row: &RowData, trial_mult: &[f64], coh_trial: &[f64], store_phase: bool) -> RefinedRow {
    let candidate_ix = near_max_trial_indices(coh_trial);
    if candidate_ix.len() == 1 {
        let coarse_k0 = QUARTER_PI / row.bperp_range * trial_mult[candidate_ix[0]];
        return refine_candidate(row, coarse_k0, store_phase);
    }

    let mut refined = Vec::with_capacity(candidate_ix.len());
    let mut candidate_coh = Vec::with_capacity(candidate_ix.len());
    for &trial_ix in &candidate_ix {
        let coarse_k0 = QUARTER_PI / row.bperp_range * trial_mult[trial_ix];
        refined.push(refine_candidate(row, coarse_k0, store_phase));
        candidate_coh.push(coh_trial[trial_ix]);
    }
    let refined_coh = refined.iter().map(|row| row.coh).collect::<Vec<_>>();
    let selected_trial_ix = select_candidate(&candidate_ix, &candidate_coh, &refined_coh, trial_mult.len());
    let selected_local_ix = candidate_ix
        .iter()
        .position(|&trial_ix| trial_ix == selected_trial_ix)
        .unwrap_or(0);
    refined.remove(selected_local_ix)
}

fn row_invariant_basis(bp_vec: &[f64], trial_mult: &[f64]) -> Vec<Complex64> {
    let mut bperp_min = bp_vec[0];
    let mut bperp_max = bp_vec[0];
    for &value in bp_vec.iter().skip(1) {
        bperp_min = bperp_min.min(value);
        bperp_max = bperp_max.max(value);
    }
    let mut bperp_range = bperp_max - bperp_min;
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    let mut basis = vec![Complex64::new(0.0, 0.0); trial_mult.len() * bp_vec.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let basis_row = &mut basis[trial_ix * bp_vec.len()..(trial_ix + 1) * bp_vec.len()];
        for (col, &bp_value) in bp_vec.iter().enumerate() {
            let phase = (bp_value / bperp_range) * QUARTER_PI * trial_value;
            let (sn, cs) = phase.sin_cos();
            basis_row[col] = Complex64::new(cs, -sn);
        }
    }
    basis
}

#[pyfunction]
fn accumulate_weighted_grid<'py>(
    py: Python<'py>,
    ph_weight: PyReadonlyArray2<Complex32>,
    grid_lin: PyReadonlyArray1<i64>,
    n_i: usize,
    n_j: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray3<Complex32>>> {
    let ph_view = ph_weight.as_array();
    let grid_view = grid_lin.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph_weight must be a 2-D matrix"));
    }
    if grid_view.len() != ph_view.shape()[0] {
        return Err(PyValueError::new_err("grid_lin length must match ph_weight row count"));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_weight must be C-contiguous"))?;
    let grid_slice = grid_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_lin must be contiguous"))?;
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let grid_size = n_i * n_j;
    let pool = build_pool(threads)?;

    let columns = py.detach(move || {
        let compute = || {
            (0..n_ifg)
                .into_par_iter()
                .map(|ifg_ix| {
                    let mut column = vec![Complex32::new(0.0, 0.0); grid_size];
                    for row_ix in 0..n_ps {
                        let grid_ix = grid_slice[row_ix];
                        if grid_ix >= 0 && (grid_ix as usize) < grid_size {
                            column[grid_ix as usize] += ph_slice[row_ix * n_ifg + ifg_ix];
                        }
                    }
                    column
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_ifg)
                .map(|ifg_ix| {
                    let mut column = vec![Complex32::new(0.0, 0.0); grid_size];
                    for row_ix in 0..n_ps {
                        let grid_ix = grid_slice[row_ix];
                        if grid_ix >= 0 && (grid_ix as usize) < grid_size {
                            column[grid_ix as usize] += ph_slice[row_ix * n_ifg + ifg_ix];
                        }
                    }
                    column
                })
                .collect::<Vec<_>>(),
        }
    });

    let mut out = vec![Complex32::new(0.0, 0.0); grid_size * n_ifg];
    for ifg_ix in 0..n_ifg {
        for grid_ix in 0..grid_size {
            out[grid_ix * n_ifg + ifg_ix] = columns[ifg_ix][grid_ix];
        }
    }

    let array = Array3::from_shape_vec((n_i, n_j, n_ifg), out)
        .map_err(|err| PyValueError::new_err(format!("failed to build grid output: {err}")))?;
    Ok(array.into_pyarray(py))
}

fn ps_topofit_batch_generic_f64_impl<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp: PyReadonlyArray2<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp.as_array();
    if cpx_view.ndim() != 2 || bp_view.ndim() != 2 || cpx_view.shape() != bp_view.shape() {
        return Err(PyValueError::new_err(
            "ps_topofit batch expects cpxphase and bperp with matching 2-D shapes",
        ));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    let trial_mult = trial_values(n_trial_wraps);
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual)
        .map_err(|err| PyValueError::new_err(format!("failed to build topofit residual output: {err}")))?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_batch_generic_f32<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex32>,
    bperp: PyReadonlyArray2<f32>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp.as_array();
    if cpx_view.ndim() != 2 || bp_view.ndim() != 2 || cpx_view.shape() != bp_view.shape() {
        return Err(PyValueError::new_err(
            "ps_topofit batch expects cpxphase and bperp with matching 2-D shapes",
        ));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    let trial_mult: Vec<f32> = trial_values(n_trial_wraps).into_iter().map(|value| value as f32).collect();
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic_single(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic_single(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual)
        .map_err(|err| PyValueError::new_err(format!("failed to build topofit residual output: {err}")))?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_batch_generic<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp: PyReadonlyArray2<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    ps_topofit_batch_generic_f64_impl(py, cpxphase, bperp, n_trial_wraps, threads)
}

#[pyfunction]
fn ps_topofit_batch_row_invariant<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp_vec: PyReadonlyArray1<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp_vec.as_array();
    if cpx_view.ndim() != 2 {
        return Err(PyValueError::new_err("cpxphase must be a 2-D matrix"));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp vector must be contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    if bp_view.len() != n_col {
        return Err(PyValueError::new_err("row-invariant bperp vector length must match cpxphase width"));
    }

    let trial_mult = trial_values(n_trial_wraps);
    let basis = row_invariant_basis(bp_slice, &trial_mult);
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual)
        .map_err(|err| PyValueError::new_err(format!("failed to build row-invariant residual output: {err}")))?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_coh_row_invariant<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp_vec: PyReadonlyArray1<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp_vec.as_array();
    if cpx_view.ndim() != 2 {
        return Err(PyValueError::new_err("cpxphase must be a 2-D matrix"));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp vector must be contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    if bp_view.len() != n_col {
        return Err(PyValueError::new_err("row-invariant bperp vector length must match cpxphase width"));
    }

    let trial_mult = trial_values(n_trial_wraps);
    let basis = row_invariant_basis(bp_slice, &trial_mult);
    let pool = build_pool(threads)?;

    let coh_values = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, false).coh
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, false).coh
                })
                .collect::<Vec<_>>(),
        }
    });

    Ok(Array1::from_vec(coh_values).into_pyarray(py))
}

#[pyfunction]
fn histogram_with_centers<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    centers: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let value_view = values.as_array();
    let center_view = centers.as_array();
    let value_slice = value_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("values must be contiguous"))?;
    let center_slice = center_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("centers must be contiguous"))?;
    let mut out = vec![0.0_f64; center_slice.len()];

    if center_slice.is_empty() {
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }
    if center_slice.len() == 1 {
        out[0] = value_slice.len() as f64;
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }

    let diffs: Vec<f64> = center_slice.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let max_abs_center = center_slice
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let equal_spacing = diffs.iter().all(|&diff| {
        (diff - diffs[0]).abs() <= f64::EPSILON * (1.0_f64).max(max_abs_center)
    });
    if equal_spacing {
        let d = if center_slice.len() < 3 {
            1.0_f64
        } else {
            (center_slice[center_slice.len() - 1] - center_slice[0]) / ((center_slice.len() - 1) as f64)
        };
        let cutoff0 = (center_slice[0] + center_slice[1]) / 2.0;
        let max_bin = (center_slice.len() - 1) as f64;
        for &value in value_slice {
            if !value.is_finite() {
                continue;
            }
            let assignment = 1.0 + ((value - cutoff0) / d).ceil().clamp(0.0, max_bin);
            out[(assignment as usize) - 1] += 1.0;
        }
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }

    let mids: Vec<f64> = center_slice.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0).collect();
    for &value in value_slice {
        if !value.is_finite() {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = center_slice.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if mids[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[lo] += 1.0;
    }

    Ok(Array1::from_vec(out).into_pyarray(py))
}

fn wrap_phase(value: f64) -> f64 {
    value.sin().atan2(value.cos())
}

fn weighted_affine_fit_rows(time_diff: &[f64], y: &[f64], n_row: usize, n_col: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut intercept = vec![0.0_f64; n_row];
    let mut slope = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return (intercept, slope);
    }

    let s0: f64 = w.iter().copied().sum();
    let s1: f64 = w.iter().zip(time_diff.iter()).map(|(&wi, &ti)| wi * ti).sum();
    let s2: f64 = w
        .iter()
        .zip(time_diff.iter())
        .map(|(&wi, &ti)| wi * ti * ti)
        .sum();
    let det = s0 * s2 - s1 * s1;
    if det == 0.0 {
        if s0 != 0.0 {
            for row_ix in 0..n_row {
                let mut base = 0.0_f64;
                for col_ix in 0..n_col {
                    base += y[row_ix * n_col + col_ix] * w[col_ix];
                }
                intercept[row_ix] = base / s0;
            }
        }
        return (intercept, slope);
    }

    for row_ix in 0..n_row {
        let mut wy0 = 0.0_f64;
        let mut wy1 = 0.0_f64;
        for col_ix in 0..n_col {
            let value = y[row_ix * n_col + col_ix];
            let weight = w[col_ix];
            wy0 += value * weight;
            wy1 += value * weight * time_diff[col_ix];
        }
        intercept[row_ix] = (wy0 * s2 - wy1 * s1) / det;
        slope[row_ix] = (wy1 * s0 - wy0 * s1) / det;
    }
    (intercept, slope)
}

fn weighted_slope_fit_rows_real(x: &[f64], y: &[f64], n_row: usize, n_col: usize, w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }

    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_infinite() { Some(idx) } else { None })
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row_ix in 0..n_row {
            let mut num = 0.0_f64;
            for &col_ix in &inf_idx {
                num += y[row_ix * n_col + col_ix] * x[col_ix];
            }
            out[row_ix] = num / den;
        }
        return out;
    }

    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_finite() && value > 0.0 { Some(idx) } else { None })
        .collect();
    if pos_idx.is_empty() {
        return out;
    }

    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row_ix in 0..n_row {
        let mut num = 0.0_f64;
        for &col_ix in &pos_idx {
            num += y[row_ix * n_col + col_ix] * w[col_ix] * x[col_ix];
        }
        out[row_ix] = num / den;
    }
    out
}

fn weighted_slope_fit_rows_complex(
    x: &[f64],
    y: &[Complex64],
    n_row: usize,
    n_col: usize,
    w: &[f64],
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }

    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_infinite() { Some(idx) } else { None })
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row_ix in 0..n_row {
            let mut num = Complex64::new(0.0, 0.0);
            for &col_ix in &inf_idx {
                num += y[row_ix * n_col + col_ix] * x[col_ix];
            }
            out[row_ix] = num / den;
        }
        return out;
    }

    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value.is_finite() && value > 0.0 { Some(idx) } else { None })
        .collect();
    if pos_idx.is_empty() {
        return out;
    }

    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row_ix in 0..n_row {
        let mut num = Complex64::new(0.0, 0.0);
        for &col_ix in &pos_idx {
            num += y[row_ix * n_col + col_ix] * (w[col_ix] * x[col_ix]);
        }
        out[row_ix] = num / den;
    }
    out
}

fn variance_cols_real(data: &[f64], n_row: usize, n_col: usize, ddof: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col_ix in 0..n_col {
        let mut mean = 0.0_f64;
        for row_ix in 0..n_row {
            mean += data[row_ix * n_col + col_ix];
        }
        mean /= n_row as f64;
        let mut accum = 0.0_f64;
        for row_ix in 0..n_row {
            let delta = data[row_ix * n_col + col_ix] - mean;
            accum += delta * delta;
        }
        out[col_ix] = accum / denom as f64;
    }
    out
}

fn variance_cols_complex(data: &[Complex64], n_row: usize, n_col: usize, ddof: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col_ix in 0..n_col {
        let mut mean = Complex64::new(0.0, 0.0);
        for row_ix in 0..n_row {
            mean += data[row_ix * n_col + col_ix];
        }
        mean /= n_row as f64;
        let mut accum = 0.0_f64;
        for row_ix in 0..n_row {
            let delta = data[row_ix * n_col + col_ix] - mean;
            accum += delta.norm_sqr();
        }
        out[col_ix] = accum / denom as f64;
    }
    out
}

fn std_max_rows_real(data: &[f64], n_row: usize, n_col: usize, ddof: usize) -> (Vec<f64>, Vec<f64>) {
    let mut std = vec![0.0_f64; n_row];
    let mut max_abs = vec![0.0_f64; n_row];
    if n_row == 0 || n_col == 0 {
        return (std, max_abs);
    }
    let denom = n_col.saturating_sub(ddof);
    for row_ix in 0..n_row {
        let row = &data[row_ix * n_col..(row_ix + 1) * n_col];
        let mean = row.iter().copied().sum::<f64>() / n_col as f64;
        let mut accum = 0.0_f64;
        let mut max_value = 0.0_f64;
        for &value in row {
            let delta = value - mean;
            accum += delta * delta;
            max_value = max_value.max(value.abs());
        }
        std[row_ix] = if denom == 0 { 0.0 } else { (accum / denom as f64).sqrt() };
        max_abs[row_ix] = max_value;
    }
    (std, max_abs)
}

fn stage4_edge_stats_outputs(
    ph_slice: &[Complex64],
    n_node: usize,
    n_ifg: usize,
    edge_a: &[usize],
    edge_b: &[usize],
    bperp: &[f64],
    day: &[f64],
    time_win: f64,
    small_baseline: bool,
    threads: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n_edge = edge_a.len();
    let mut ps_std = vec![f64::INFINITY; n_node];
    let mut ps_max = vec![f64::INFINITY; n_node];
    if n_edge == 0 || n_ifg == 0 {
        return Ok((ps_std, ps_max));
    }

    let pool = build_pool(threads)?;
    let mut dph_space = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
    match &pool {
        Some(pool) => pool.install(|| {
            dph_space
                .par_chunks_mut(n_ifg)
                .enumerate()
                .for_each(|(edge_ix, row)| {
                    let a_ix = edge_a[edge_ix];
                    let b_ix = edge_b[edge_ix];
                    for ifg_ix in 0..n_ifg {
                        row[ifg_ix] = ph_slice[b_ix * n_ifg + ifg_ix] * ph_slice[a_ix * n_ifg + ifg_ix].conj();
                    }
                });
        }),
        None => {
            for edge_ix in 0..n_edge {
                let a_ix = edge_a[edge_ix];
                let b_ix = edge_b[edge_ix];
                let row = &mut dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                for ifg_ix in 0..n_ifg {
                    row[ifg_ix] = ph_slice[b_ix * n_ifg + ifg_ix] * ph_slice[a_ix * n_ifg + ifg_ix].conj();
                }
            }
        }
    }

    let (edge_std, edge_max) = if !small_baseline {
        if day.len() != n_ifg {
            return Err(PyValueError::new_err("stage4_edge_stats day length must match phase width"));
        }
        let time_win_f = time_win.max(1.0e-6);
        let mut time_diff_all = vec![0.0_f64; n_ifg * n_ifg];
        let mut weight_all = vec![0.0_f64; n_ifg * n_ifg];
        for row_ix in 0..n_ifg {
            let mut weight_sum = 0.0_f64;
            for col_ix in 0..n_ifg {
                let diff = day[row_ix] - day[col_ix];
                time_diff_all[row_ix * n_ifg + col_ix] = diff;
                let weight = (-(diff * diff) / (2.0 * time_win_f * time_win_f)).exp();
                weight_all[row_ix * n_ifg + col_ix] = weight;
                weight_sum += weight;
            }
            if weight_sum <= 0.0 {
                let fill = 1.0 / n_ifg as f64;
                for col_ix in 0..n_ifg {
                    weight_all[row_ix * n_ifg + col_ix] = fill;
                }
            } else {
                for col_ix in 0..n_ifg {
                    weight_all[row_ix * n_ifg + col_ix] /= weight_sum;
                }
            }
        }

        let mut dph_smooth0 = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
        match &pool {
            Some(pool) => pool.install(|| {
                dph_smooth0
                    .par_chunks_mut(n_ifg)
                    .enumerate()
                    .for_each(|(edge_ix, row)| {
                        let source = &dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                        for out_ix in 0..n_ifg {
                            let weights = &weight_all[out_ix * n_ifg..(out_ix + 1) * n_ifg];
                            let mut accum = Complex64::new(0.0, 0.0);
                            for src_ix in 0..n_ifg {
                                accum += source[src_ix] * weights[src_ix];
                            }
                            row[out_ix] = accum;
                        }
                    });
            }),
            None => {
                for edge_ix in 0..n_edge {
                    let source = &dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                    let row = &mut dph_smooth0[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                    for out_ix in 0..n_ifg {
                        let weights = &weight_all[out_ix * n_ifg..(out_ix + 1) * n_ifg];
                        let mut accum = Complex64::new(0.0, 0.0);
                        for src_ix in 0..n_ifg {
                            accum += source[src_ix] * weights[src_ix];
                        }
                        row[out_ix] = accum;
                    }
                }
            }
        }

        let mut dph_smooth2 = dph_smooth0.clone();
        for edge_ix in 0..n_edge {
            for ifg_ix in 0..n_ifg {
                let diag = weight_all[ifg_ix * n_ifg + ifg_ix];
                dph_smooth2[edge_ix * n_ifg + ifg_ix] -= dph_space[edge_ix * n_ifg + ifg_ix] * diag;
            }
        }

        let columns = match &pool {
            Some(pool) => pool.install(|| {
                (0..n_ifg)
                    .into_par_iter()
                    .map(|ifg_ix| {
                        let time_diff = &time_diff_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                        let weight = &weight_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                        let mut dph_mean_adj = vec![0.0_f64; n_edge * n_ifg];
                        let mut dph_mean = vec![Complex64::new(0.0, 0.0); n_edge];
                        for edge_ix in 0..n_edge {
                            let mean = dph_smooth0[edge_ix * n_ifg + ifg_ix];
                            dph_mean[edge_ix] = mean;
                            let mean_conj = mean.conj();
                            for col_ix in 0..n_ifg {
                                dph_mean_adj[edge_ix * n_ifg + col_ix] =
                                    (dph_space[edge_ix * n_ifg + col_ix] * mean_conj).arg();
                            }
                        }
                        let (m0, m1) = weighted_affine_fit_rows(time_diff, &dph_mean_adj, n_edge, n_ifg, weight);
                        let mut dph_mean_adj2 = vec![0.0_f64; n_edge * n_ifg];
                        for edge_ix in 0..n_edge {
                            for col_ix in 0..n_ifg {
                                let detrended =
                                    dph_mean_adj[edge_ix * n_ifg + col_ix] - (m0[edge_ix] + m1[edge_ix] * time_diff[col_ix]);
                                dph_mean_adj2[edge_ix * n_ifg + col_ix] = wrap_phase(detrended);
                            }
                        }
                        let (m20, _) = weighted_affine_fit_rows(time_diff, &dph_mean_adj2, n_edge, n_ifg, weight);
                        let mut column = vec![Complex64::new(0.0, 0.0); n_edge];
                        for edge_ix in 0..n_edge {
                            column[edge_ix] = dph_mean[edge_ix] * Complex64::from_polar(1.0, m0[edge_ix] + m20[edge_ix]);
                        }
                        column
                    })
                    .collect::<Vec<_>>()
            }),
            None => {
                let mut cols = Vec::with_capacity(n_ifg);
                for ifg_ix in 0..n_ifg {
                    let time_diff = &time_diff_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                    let weight = &weight_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                    let mut dph_mean_adj = vec![0.0_f64; n_edge * n_ifg];
                    let mut dph_mean = vec![Complex64::new(0.0, 0.0); n_edge];
                    for edge_ix in 0..n_edge {
                        let mean = dph_smooth0[edge_ix * n_ifg + ifg_ix];
                        dph_mean[edge_ix] = mean;
                        let mean_conj = mean.conj();
                        for col_ix in 0..n_ifg {
                            dph_mean_adj[edge_ix * n_ifg + col_ix] =
                                (dph_space[edge_ix * n_ifg + col_ix] * mean_conj).arg();
                        }
                    }
                    let (m0, m1) = weighted_affine_fit_rows(time_diff, &dph_mean_adj, n_edge, n_ifg, weight);
                    let mut dph_mean_adj2 = vec![0.0_f64; n_edge * n_ifg];
                    for edge_ix in 0..n_edge {
                        for col_ix in 0..n_ifg {
                            let detrended =
                                dph_mean_adj[edge_ix * n_ifg + col_ix] - (m0[edge_ix] + m1[edge_ix] * time_diff[col_ix]);
                            dph_mean_adj2[edge_ix * n_ifg + col_ix] = wrap_phase(detrended);
                        }
                    }
                    let (m20, _) = weighted_affine_fit_rows(time_diff, &dph_mean_adj2, n_edge, n_ifg, weight);
                    let mut column = vec![Complex64::new(0.0, 0.0); n_edge];
                    for edge_ix in 0..n_edge {
                        column[edge_ix] = dph_mean[edge_ix] * Complex64::from_polar(1.0, m0[edge_ix] + m20[edge_ix]);
                    }
                    cols.push(column);
                }
                cols
            }
        };

        let mut dph_smooth = dph_smooth0;
        for ifg_ix in 0..n_ifg {
            for edge_ix in 0..n_edge {
                dph_smooth[edge_ix * n_ifg + ifg_ix] = columns[ifg_ix][edge_ix];
            }
        }

        let mut dph_noise = vec![0.0_f64; n_edge * n_ifg];
        let mut dph_noise2 = vec![0.0_f64; n_edge * n_ifg];
        for edge_ix in 0..n_edge {
            for ifg_ix in 0..n_ifg {
                dph_noise[edge_ix * n_ifg + ifg_ix] =
                    (dph_space[edge_ix * n_ifg + ifg_ix] * dph_smooth[edge_ix * n_ifg + ifg_ix].conj()).arg();
                dph_noise2[edge_ix * n_ifg + ifg_ix] =
                    (dph_space[edge_ix * n_ifg + ifg_ix] * dph_smooth2[edge_ix * n_ifg + ifg_ix].conj()).arg();
            }
        }

        let ddof_var = if n_edge > 1 { 1 } else { 0 };
        let ifg_var = variance_cols_real(&dph_noise2, n_edge, n_ifg, ddof_var);
        let w_ifg: Vec<f64> = ifg_var
            .iter()
            .map(|&value| if value == 0.0 { f64::INFINITY } else { 1.0 / value })
            .collect();
        let k_edge = weighted_slope_fit_rows_real(bperp, &dph_noise, n_edge, n_ifg, &w_ifg);
        for edge_ix in 0..n_edge {
            let slope = k_edge[edge_ix];
            for ifg_ix in 0..n_ifg {
                dph_noise[edge_ix * n_ifg + ifg_ix] -= slope * bperp[ifg_ix];
            }
        }
        let ddof = if n_ifg > 1 { 1 } else { 0 };
        std_max_rows_real(&dph_noise, n_edge, n_ifg, ddof)
    } else {
        let ddof_var = if n_edge > 1 { 1 } else { 0 };
        let ifg_var = variance_cols_complex(&dph_space, n_edge, n_ifg, ddof_var);
        let w_ifg: Vec<f64> = ifg_var
            .iter()
            .map(|&value| if value == 0.0 { f64::INFINITY } else { 1.0 / value })
            .collect();
        let k_edge = weighted_slope_fit_rows_complex(bperp, &dph_space, n_edge, n_ifg, &w_ifg);
        let mut ang = vec![0.0_f64; n_edge * n_ifg];
        for edge_ix in 0..n_edge {
            let slope = k_edge[edge_ix];
            for ifg_ix in 0..n_ifg {
                ang[edge_ix * n_ifg + ifg_ix] = (dph_space[edge_ix * n_ifg + ifg_ix] - slope * bperp[ifg_ix]).arg();
            }
        }
        let ddof = if n_ifg > 1 { 1 } else { 0 };
        std_max_rows_real(&ang, n_edge, n_ifg, ddof)
    };

    for edge_ix in 0..n_edge {
        let a_ix = edge_a[edge_ix];
        let b_ix = edge_b[edge_ix];
        ps_std[a_ix] = ps_std[a_ix].min(edge_std[edge_ix]);
        ps_std[b_ix] = ps_std[b_ix].min(edge_std[edge_ix]);
        ps_max[a_ix] = ps_max[a_ix].min(edge_max[edge_ix]);
        ps_max[b_ix] = ps_max[b_ix].min(edge_max[edge_ix]);
    }
    Ok((ps_std, ps_max))
}

#[pyfunction(signature = (ph_weed, node_a, node_b, bperp, day, time_win, small_baseline, threads = 0))]
fn stage4_edge_stats<'py>(
    py: Python<'py>,
    ph_weed: PyReadonlyArray2<Complex64>,
    node_a: PyReadonlyArray1<i64>,
    node_b: PyReadonlyArray1<i64>,
    bperp: PyReadonlyArray1<f64>,
    day: PyReadonlyArray1<f64>,
    time_win: f64,
    small_baseline: bool,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = ph_weed.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph_weed must be a 2-D matrix"));
    }
    let node_a_view = node_a.as_array();
    let node_b_view = node_b.as_array();
    if node_a_view.len() != node_b_view.len() {
        return Err(PyValueError::new_err("node_a and node_b must have matching lengths"));
    }
    let bperp_view = bperp.as_array();
    let day_view = day.as_array();

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_weed must be C-contiguous"))?;
    let node_a_slice = node_a_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_a must be contiguous"))?;
    let node_b_slice = node_b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_b must be contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be contiguous"))?;
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;

    let n_node = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if bperp_slice.len() != n_ifg {
        return Err(PyValueError::new_err("stage4_edge_stats bperp length must match phase width"));
    }
    if !small_baseline && day_slice.len() != n_ifg {
        return Err(PyValueError::new_err("stage4_edge_stats day length must match phase width"));
    }
    let edge_a = parse_indices(node_a_slice, n_node, "node_a")?;
    let edge_b = parse_indices(node_b_slice, n_node, "node_b")?;
    let (ps_std, ps_max) = py.detach(move || {
        stage4_edge_stats_outputs(
            ph_slice,
            n_node,
            n_ifg,
            &edge_a,
            &edge_b,
            bperp_slice,
            day_slice,
            time_win,
            small_baseline,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item("ps_std", Array1::from_vec(ps_std).into_pyarray(py))?;
    dict.set_item("ps_max", Array1::from_vec(ps_max).into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction(signature = (ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std, threads = 0))]
fn stage7_scla_parity<'py>(
    py: Python<'py>,
    ph_proc: PyReadonlyArray2<f64>,
    ph_mean_v: PyReadonlyArray2<f64>,
    bperp_mat: PyReadonlyArray2<f64>,
    unwrap_ix: PyReadonlyArray1<i64>,
    solve_ix: PyReadonlyArray1<i64>,
    day: PyReadonlyArray1<f64>,
    master_ix: usize,
    ifg_std: PyReadonlyArray1<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_proc_view = ph_proc.as_array();
    let ph_mean_v_view = ph_mean_v.as_array();
    let bperp_view = bperp_mat.as_array();
    if ph_proc_view.ndim() != 2 || ph_mean_v_view.ndim() != 2 || bperp_view.ndim() != 2 {
        return Err(PyValueError::new_err("stage7_scla_parity expects 2-D ph_proc, ph_mean_v, and bperp_mat"));
    }
    if ph_proc_view.shape() != ph_mean_v_view.shape() || ph_proc_view.shape() != bperp_view.shape() {
        return Err(PyValueError::new_err(
            "stage7_scla_parity expects ph_proc, ph_mean_v, and bperp_mat with matching shapes",
        ));
    }

    let ph_proc_slice = ph_proc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_proc must be C-contiguous"))?;
    let ph_mean_v_slice = ph_mean_v_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_mean_v must be C-contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp_mat must be C-contiguous"))?;
    let unwrap_view = unwrap_ix.as_array();
    let unwrap_slice = unwrap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("unwrap_ix must be contiguous"))?;
    let solve_view = solve_ix.as_array();
    let solve_slice = solve_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("solve_ix must be contiguous"))?;
    let day_view = day.as_array();
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;
    let ifg_std_view = ifg_std.as_array();
    let ifg_std_slice = ifg_std_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ifg_std must be contiguous"))?;

    let n_ps = ph_proc_view.shape()[0];
    let n_ifg = ph_proc_view.shape()[1];
    let unwrap_idx = parse_indices(unwrap_slice, n_ifg, "unwrap_ix")?;
    let solve_idx = parse_indices(solve_slice, n_ifg, "solve_ix")?;
    let outputs = py.detach(move || {
        stage7_outputs(
            ph_proc_slice,
            ph_mean_v_slice,
            bperp_slice,
            n_ps,
            n_ifg,
            &unwrap_idx,
            &solve_idx,
            day_slice,
            master_ix,
            ifg_std_slice,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item("K_ps_uw", Array1::from_vec(outputs.k_ps_uw).into_pyarray(py))?;
    dict.set_item("C_ps_uw", Array1::from_vec(outputs.c_ps_uw).into_pyarray(py))?;
    dict.set_item(
        "ph_scla",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_scla)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 ph_scla output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ifg_vcm",
        Array2::from_shape_vec((n_ifg, n_ifg), outputs.ifg_vcm)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 ifg_vcm output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item("mean_v", Array1::from_vec(outputs.mean_v).into_pyarray(py))?;
    dict.set_item(
        "m",
        Array2::from_shape_vec((2, n_ps), outputs.m)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 mean-velocity output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ph_ramp",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_ramp)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 ph_ramp output: {err}")))?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pyfunction(signature = (ph_uw, bperp_mat, no_master, day, master_ix, chunk_ps = 0, threads = 0))]
fn stage7_scla<'py>(
    py: Python<'py>,
    ph_uw: PyReadonlyArray2<f32>,
    bperp_mat: PyReadonlyArray2<f32>,
    no_master: PyReadonlyArray1<bool>,
    day: PyReadonlyArray1<f64>,
    master_ix: usize,
    chunk_ps: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = ph_uw.as_array();
    let bperp_view = bperp_mat.as_array();
    let no_master_view = no_master.as_array();
    if ph_view.ndim() != 2 || bperp_view.ndim() != 2 {
        return Err(PyValueError::new_err("stage7_scla expects 2-D ph_uw and bperp_mat"));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    if no_master_view.len() != n_ifg || day.as_array().len() != n_ifg {
        return Err(PyValueError::new_err("stage7_scla no_master/day length must match ph_uw width"));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_uw must be C-contiguous"))?;
    let bperp_slice = bperp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp_mat must be C-contiguous"))?;
    let no_master_slice = no_master_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("no_master must be contiguous"))?;
    let day_view = day.as_array();
    let day_slice = day_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("day must be contiguous"))?;
    let _ = chunk_ps;

    let mut unwrap_idx = Vec::new();
    for (ifg_ix, &keep) in no_master_slice.iter().enumerate() {
        if keep {
            unwrap_idx.push(ifg_ix);
        }
    }
    let solve_idx = unwrap_idx.clone();
    let ph_proc64: Vec<f64> = ph_slice.iter().map(|&value| value as f64).collect();
    let bperp64: Vec<f64> = bperp_slice.iter().map(|&value| value as f64).collect();
    let ifg_std = vec![1.0_f64; n_ifg];

    let outputs = py.detach(move || {
        stage7_outputs(
            &ph_proc64,
            &ph_proc64,
            &bperp64,
            n_ps,
            n_ifg,
            &unwrap_idx,
            &solve_idx,
            day_slice,
            master_ix,
            &ifg_std,
            threads,
        )
    })?;

    let dict = PyDict::new(py);
    dict.set_item("K_ps_uw", Array1::from_vec(outputs.k_ps_uw).into_pyarray(py))?;
    dict.set_item("C_ps_uw", Array1::from_vec(outputs.c_ps_uw).into_pyarray(py))?;
    dict.set_item(
        "ph_scla",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_scla)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 shim ph_scla output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ifg_vcm",
        Array2::from_shape_vec((n_ifg, n_ifg), outputs.ifg_vcm)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 shim ifg_vcm output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item("mean_v", Array1::from_vec(outputs.mean_v).into_pyarray(py))?;
    dict.set_item(
        "m",
        Array2::from_shape_vec((2, n_ps), outputs.m)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 shim mean-velocity output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ph_ramp",
        Array2::from_shape_vec((n_ps, n_ifg), outputs.ph_ramp)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage7 shim ph_ramp output: {err}")))?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pyfunction(signature = (uw_ph, node_a, node_b, chunk_edges = 0, threads = 0))]
fn stage8_edge_noise<'py>(
    py: Python<'py>,
    uw_ph: PyReadonlyArray2<Complex32>,
    node_a: PyReadonlyArray1<i64>,
    node_b: PyReadonlyArray1<i64>,
    chunk_edges: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph_view = uw_ph.as_array();
    let node_a_view = node_a.as_array();
    let node_b_view = node_b.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("uw_ph must be a 2-D matrix"));
    }
    if node_a_view.len() != node_b_view.len() {
        return Err(PyValueError::new_err("node_a and node_b must have matching lengths"));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("uw_ph must be C-contiguous"))?;
    let node_a_slice = node_a_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_a must be contiguous"))?;
    let node_b_slice = node_b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("node_b must be contiguous"))?;
    let n_node = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let edge_a = parse_indices(node_a_slice, n_node, "node_a")?;
    let edge_b = parse_indices(node_b_slice, n_node, "node_b")?;
    let n_edge = edge_a.len();
    let _ = chunk_edges;
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_edge)
                .into_par_iter()
                .map(|edge_ix| {
                    let a_ix = edge_a[edge_ix];
                    let b_ix = edge_b[edge_ix];
                    let mut dph_space = vec![0.0_f32; n_ifg];
                    let mut sum = 0.0_f64;
                    for ifg_ix in 0..n_ifg {
                        let left = ph_slice[a_ix * n_ifg + ifg_ix];
                        let right = ph_slice[b_ix * n_ifg + ifg_ix];
                        let phase = (right * left.conj()).arg();
                        dph_space[ifg_ix] = phase;
                        sum += phase as f64;
                    }
                    let mean = if n_ifg == 0 { 0.0_f32 } else { (sum / n_ifg as f64) as f32 };
                    let dph_noise: Vec<f32> =
                        dph_space.iter().map(|&value| (value - mean) * STAGE8_NOISE_SCALE).collect();
                    (dph_noise, dph_space)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_edge)
                .map(|edge_ix| {
                    let a_ix = edge_a[edge_ix];
                    let b_ix = edge_b[edge_ix];
                    let mut dph_space = vec![0.0_f32; n_ifg];
                    let mut sum = 0.0_f64;
                    for ifg_ix in 0..n_ifg {
                        let left = ph_slice[a_ix * n_ifg + ifg_ix];
                        let right = ph_slice[b_ix * n_ifg + ifg_ix];
                        let phase = (right * left.conj()).arg();
                        dph_space[ifg_ix] = phase;
                        sum += phase as f64;
                    }
                    let mean = if n_ifg == 0 { 0.0_f32 } else { (sum / n_ifg as f64) as f32 };
                    let dph_noise: Vec<f32> =
                        dph_space.iter().map(|&value| (value - mean) * STAGE8_NOISE_SCALE).collect();
                    (dph_noise, dph_space)
                })
                .collect::<Vec<_>>(),
        }
    });

    let mut dph_noise = vec![0.0_f32; n_edge * n_ifg];
    let mut dph_space_uw = vec![0.0_f32; n_edge * n_ifg];
    for (edge_ix, (noise_row, space_row)) in rows.into_iter().enumerate() {
        dph_noise[edge_ix * n_ifg..(edge_ix + 1) * n_ifg].copy_from_slice(&noise_row);
        dph_space_uw[edge_ix * n_ifg..(edge_ix + 1) * n_ifg].copy_from_slice(&space_row);
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "dph_noise",
        Array2::from_shape_vec((n_edge, n_ifg), dph_noise)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage8 dph_noise output: {err}")))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "dph_space_uw",
        Array2::from_shape_vec((n_edge, n_ifg), dph_space_uw)
            .map_err(|err| PyValueError::new_err(format!("failed to build stage8 dph_space_uw output: {err}")))?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pymodule]
fn _stage2_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accumulate_weighted_grid, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_generic, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_generic_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_row_invariant, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_coh_row_invariant, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_with_centers, m)?)?;
    m.add_function(wrap_pyfunction!(stage4_edge_stats, m)?)?;
    m.add_function(wrap_pyfunction!(stage7_scla, m)?)?;
    m.add_function(wrap_pyfunction!(stage7_scla_parity, m)?)?;
    m.add_function(wrap_pyfunction!(stage8_edge_noise, m)?)?;
    Ok(())
}
