use crate::CoreError;
use num_complex::{Complex32, Complex64};
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use rustfft::FftPlanner;
use std::path::{Path, PathBuf};

const DEFAULT_GRID_SIZE: f64 = 50.0;
const DEFAULT_CLAP_WIN: f64 = 32.0;
const DEFAULT_CLAP_LOW_PASS_WAVELENGTH: f64 = 800.0;
const DEFAULT_CLAP_ALPHA: f64 = 1.0;
const DEFAULT_CLAP_BETA: f64 = 0.3;
const DEFAULT_MAX_TOPO_ERR: f64 = 15.0;
const DEFAULT_LAMBDA_M: f64 = 0.0555;
const DEFAULT_MEAN_INCIDENCE: f64 = 23.0_f64.to_radians();
const COH_BIN_COUNT: usize = 100;
const COH_BIN_START: f64 = 0.005;
const COH_BIN_STEP: f64 = 0.01;
const STAGE2_TOPOFIT_NEAR_MAX_COH_TOL: f64 = 2.0e-4;

#[derive(Clone, Copy, Debug)]
struct Stage2Options {
    grid_size: f64,
    clap_win: f64,
    clap_low_pass_wavelength: f64,
    clap_alpha: f64,
    clap_beta: f64,
    max_topo_err: f64,
    lambda_m: f64,
}

impl Default for Stage2Options {
    fn default() -> Self {
        Self {
            grid_size: DEFAULT_GRID_SIZE,
            clap_win: DEFAULT_CLAP_WIN,
            clap_low_pass_wavelength: DEFAULT_CLAP_LOW_PASS_WAVELENGTH,
            clap_alpha: DEFAULT_CLAP_ALPHA,
            clap_beta: DEFAULT_CLAP_BETA,
            max_topo_err: DEFAULT_MAX_TOPO_ERR,
            lambda_m: DEFAULT_LAMBDA_M,
        }
    }
}

#[derive(Clone, Debug)]
struct Stage2Parms {
    small_baseline_flag: String,
    filter_weighting: String,
    gamma_change_convergence: f64,
    gamma_max_iterations: usize,
}

impl Default for Stage2Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            filter_weighting: "P-square".to_string(),
            gamma_change_convergence: 1.0e-4,
            gamma_max_iterations: 25,
        }
    }
}

#[derive(Clone, Debug)]
struct Stage2Prepared {
    n_ps: usize,
    n_ifg: usize,
    ph_nm: Vec<Complex32>,
    amp: Vec<f32>,
    bperp_mat: Option<Matrix<f64>>,
    row_invariant_bperp: bool,
    row_bperp_nm: Vec<f64>,
    grid_ij: Matrix<f32>,
    grid_lin: Vec<usize>,
    n_i: usize,
    n_j: usize,
    d_a: Vec<f64>,
    low_pass: Matrix<f64>,
    coh_bins: Vec<f64>,
    n_trial_wraps: f64,
    grid_size: f64,
    clap_window: usize,
}

#[derive(Clone, Debug)]
struct TopofitRow {
    k: f64,
    c: f64,
    coh: f64,
    residual: Vec<Complex32>,
}

pub fn run_stage2_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let parms_mat = resolve_file_optional(patch_dir, "parms.mat")
        .and_then(|path| MatData::read(path).ok());
    let parms = load_stage2_parms(parms_mat.as_ref());
    let options = load_stage2_options(parms_mat.as_ref());
    let prepared = prepare_stage2_inputs(patch_dir, &parms, &options)?;

    let mut weighting = prepared
        .d_a
        .iter()
        .map(|&value| if value != 0.0 { 1.0 / value } else { 0.0 })
        .collect::<Vec<_>>();
    let mut gamma_change_save = 0.0;
    let mut coh_ps_save = vec![0.0; prepared.n_ps];
    let mut k_ps = vec![0.0; prepared.n_ps];
    let mut c_ps = vec![0.0; prepared.n_ps];
    let mut coh_ps = vec![0.0; prepared.n_ps];
    let mut n_opt = vec![0.0; prepared.n_ps];
    let mut ph_res = vec![0.0_f32; prepared.n_ps * prepared.n_ifg];
    let mut ph_patch = vec![Complex32::new(0.0, 0.0); prepared.n_ps * prepared.n_ifg];
    let mut ph_grid = vec![Complex32::new(0.0, 0.0); prepared.n_i * prepared.n_j * prepared.n_ifg];
    let mut ph_filt = vec![Complex32::new(0.0, 0.0); prepared.n_i * prepared.n_j * prepared.n_ifg];
    let mut ph_weight = vec![Complex32::new(0.0, 0.0); prepared.n_ps * prepared.n_ifg];
    let nr_base = vec![1.0; prepared.coh_bins.len()];
    let mut nr_scaled_last = nr_base.clone();
    let nr_max_nz_ix = prepared.coh_bins.len() as f64;

    let mut i_loop = 1usize;
    loop {
        fill_phase_weight(&prepared, &k_ps, &weighting, &mut ph_weight)?;
        accumulate_grid(&ph_weight, &prepared.grid_lin, prepared.n_i, prepared.n_j, prepared.n_ifg, &mut ph_grid);
        clap_filter_grid_stack(
            &ph_grid,
            prepared.n_i,
            prepared.n_j,
            prepared.n_ifg,
            prepared.clap_window,
            options.clap_alpha,
            options.clap_beta,
            &prepared.low_pass.values,
            &mut ph_filt,
        );
        extract_patch_phase(&prepared, &ph_filt, &mut ph_patch);

        k_ps.fill(f64::NAN);
        c_ps.fill(0.0);
        coh_ps.fill(0.0);
        n_opt.fill(0.0);
        ph_res.fill(0.0);
        for row in 0..prepared.n_ps {
            let row_start = row * prepared.n_ifg;
            let row_end = row_start + prepared.n_ifg;
            let mut psdph = vec![Complex64::new(0.0, 0.0); prepared.n_ifg];
            let mut valid = false;
            for col in 0..prepared.n_ifg {
                let patch_value = ph_patch[row_start + col].conj();
                let ph_value = prepared.ph_nm[row_start + col];
                let value = Complex64::new(patch_value.re as f64, patch_value.im as f64)
                    * Complex64::new(ph_value.re as f64, ph_value.im as f64);
                if value != Complex64::new(0.0, 0.0) {
                    valid = true;
                }
                psdph[col] = value;
            }
            if !valid {
                continue;
            }
            let bperp_row = if prepared.row_invariant_bperp {
                prepared.row_bperp_nm.as_slice()
            } else {
                let Some(mat) = prepared.bperp_mat.as_ref() else {
                    return stage2_err("bp1.bperp_mat is required for non-invariant stage-2 baselines");
                };
                &mat.values[row_start..row_end]
            };
            let row_fit = topofit_row(&psdph, bperp_row, prepared.n_trial_wraps);
            k_ps[row] = row_fit.k;
            c_ps[row] = row_fit.c;
            coh_ps[row] = row_fit.coh;
            n_opt[row] = 1.0;
            for (col, value) in row_fit.residual.iter().enumerate() {
                ph_res[row_start + col] = value.arg();
            }
        }

        let gamma_change_rms = rms_difference(&coh_ps, &coh_ps_save);
        let gamma_change_change = gamma_change_rms - gamma_change_save;
        gamma_change_save = gamma_change_rms;
        coh_ps_save.clone_from(&coh_ps);

        let should_stop = gamma_change_change.abs() < parms.gamma_change_convergence
            || i_loop >= parms.gamma_max_iterations.max(1);
        if !should_stop {
            if parms.filter_weighting.eq_ignore_ascii_case("P-square") {
                let na = hist_with_centers(&coh_ps, &prepared.coh_bins);
                let low_coh_thresh = if parms.small_baseline_flag.eq_ignore_ascii_case("y") { 15 } else { 31 };
                let denom: f64 = nr_base.iter().take(low_coh_thresh).sum();
                let scale = if denom > 0.0 {
                    na.iter().take(low_coh_thresh).sum::<f64>() / denom
                } else {
                    1.0
                };
                nr_scaled_last = nr_base.iter().map(|value| value * scale).collect();
                weighting = psquare_weighting(&nr_scaled_last, &na, low_coh_thresh, nr_max_nz_ix, &coh_ps);
            } else {
                weighting = snr_weighting(&prepared, &ph_res);
            }
            i_loop += 1;
        }

        if should_stop {
            write_pm1(
                patch_dir,
                &prepared,
                &k_ps,
                &c_ps,
                &coh_ps,
                &n_opt,
                &ph_res,
                &ph_patch,
                &ph_grid,
                &ph_weight,
                &nr_scaled_last,
                nr_max_nz_ix,
                &coh_ps_save,
                gamma_change_save,
                i_loop,
            )?;
            break;
        }
    }

    Ok(format!("Stage 2 computed coherence for {} candidates in {i_loop} iterations", prepared.n_ps))
}

fn prepare_stage2_inputs(
    patch_dir: &Path,
    parms: &Stage2Parms,
    options: &Stage2Options,
) -> Result<Stage2Prepared, CoreError> {
    let ps = MatData::read(patch_dir.join("ps1.mat"))
        .map_err(|err| stage2_err_owned(format!("unable to read ps1.mat: {err}")))?;
    let ph = MatData::read(patch_dir.join("ph1.mat"))
        .map_err(|err| stage2_err_owned(format!("unable to read ph1.mat: {err}")))?;
    let n_ps = scalar_from_mat(&ps, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage2_err("ps1.mat missing valid n_ps");
    }
    let ph_full = ps_complex_matrix(&ph, "ph", n_ps, "ph1.ph")?;
    let n_ifg_full = ph_full.cols;
    let master_ix = scalar_from_mat(&ps, "master_ix", 1.0).round() as usize;
    if master_ix == 0 || master_ix > n_ifg_full {
        return stage2_err(format!("ps1.master_ix must be 1-based within ph1 width {n_ifg_full}; got {master_ix}"));
    }
    let bperp_full = vector_f64(&ps, "bperp", "ps1.bperp")?;
    if bperp_full.len() != n_ifg_full {
        return stage2_err(format!(
            "ps1.bperp has length {} but ph1.ph has {n_ifg_full} interferograms",
            bperp_full.len()
        ));
    }

    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    let no_master: Vec<usize> = (0..n_ifg_full).filter(|&ix| small_baseline || ix != master_ix - 1).collect();
    let n_ifg = no_master.len();
    let mut ph_nm = Vec::with_capacity(n_ps * n_ifg);
    let mut amp = Vec::with_capacity(n_ps * n_ifg);
    for row in 0..n_ps {
        for &col in &no_master {
            let (re, im) = ph_full.values[row * n_ifg_full + col];
            let value = Complex32::new(re, im);
            let mag = value.norm();
            let safe_mag = if mag == 0.0 { 1.0 } else { mag };
            amp.push(safe_mag);
            ph_nm.push(if safe_mag != 0.0 { value / safe_mag } else { Complex32::new(0.0, 0.0) });
        }
    }
    let bperp_nm: Vec<f64> = no_master.iter().map(|&ix| bperp_full[ix]).collect();
    let bperp_mat = load_bperp_mat(patch_dir, n_ps, n_ifg_full, n_ifg, &no_master, small_baseline, &bperp_nm)?;
    let row_invariant_bperp = bperp_rows_are_invariant(bperp_mat.as_ref());
    let row_bperp_nm = if row_invariant_bperp {
        bperp_mat
            .as_ref()
            .and_then(|matrix| (matrix.rows > 0).then(|| matrix.values[..matrix.cols].to_vec()))
            .unwrap_or_else(|| bperp_nm.clone())
    } else {
        bperp_nm.clone()
    };

    let d_a = load_da(patch_dir, n_ps)?;
    let xy = ps_dim_f32(&ps, "xy", n_ps, 3, "ps1.xy")?;
    let grid_ij = stage2_grid_indices(&xy, options.grid_size);
    let n_i = grid_ij.values.iter().step_by(2).fold(1usize, |acc, &value| acc.max(value as usize));
    let n_j = grid_ij.values.iter().skip(1).step_by(2).fold(1usize, |acc, &value| acc.max(value as usize));
    let mut grid_lin = Vec::with_capacity(n_ps);
    for row in 0..n_ps {
        let i = grid_ij.values[row * 2] as usize - 1;
        let j = grid_ij.values[row * 2 + 1] as usize - 1;
        grid_lin.push(i * n_j + j);
    }

    let low_pass = build_low_pass(options);
    let coh_bins = (0..COH_BIN_COUNT)
        .map(|ix| COH_BIN_START + COH_BIN_STEP * ix as f64)
        .collect::<Vec<_>>();
    let mean_incidence = stage2_trial_wrap_mean_incidence(patch_dir, &ps);
    let rho = 830_000.0;
    let max_k = options.max_topo_err / (options.lambda_m * rho * mean_incidence.sin() / (4.0 * std::f64::consts::PI));
    let (min_bp, max_bp) = bperp_nm
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), &value| {
            (min_v.min(value), max_v.max(value))
        });
    let n_trial_wraps = ((max_bp - min_bp) * max_k / (2.0 * std::f64::consts::PI)) as f64;

    Ok(Stage2Prepared {
        n_ps,
        n_ifg,
        ph_nm,
        amp,
        bperp_mat,
        row_invariant_bperp,
        row_bperp_nm,
        grid_ij,
        grid_lin,
        n_i,
        n_j,
        d_a,
        low_pass,
        coh_bins,
        n_trial_wraps,
        grid_size: options.grid_size,
        clap_window: (options.clap_win * 0.75).round().max(1.0) as usize,
    })
}

fn load_bperp_mat(
    patch_dir: &Path,
    n_ps: usize,
    n_ifg_full: usize,
    n_ifg: usize,
    no_master: &[usize],
    small_baseline: bool,
    bperp_nm: &[f64],
) -> Result<Option<Matrix<f64>>, CoreError> {
    let path = patch_dir.join("bp1.mat");
    if !path.exists() {
        if small_baseline {
            let mut values = Vec::with_capacity(n_ps * n_ifg);
            for _ in 0..n_ps {
                values.extend_from_slice(bperp_nm);
            }
            return Ok(Some(Matrix {
                name: "bperp_mat".to_string(),
                rows: n_ps,
                cols: n_ifg,
                values,
            }));
        }
        return Ok(None);
    }
    let bp = MatData::read(path).map_err(|err| stage2_err_owned(format!("unable to read bp1.mat: {err}")))?;
    let source = ps_matrix_f64(&bp, "bperp_mat", n_ps, "bp1.bperp_mat")?;
    if source.cols == n_ifg {
        return Ok(Some(source));
    }
    if !small_baseline && source.cols == n_ifg_full {
        let mut values = Vec::with_capacity(n_ps * n_ifg);
        for row in 0..n_ps {
            for &col in no_master {
                values.push(source.values[row * source.cols + col]);
            }
        }
        return Ok(Some(Matrix {
            name: source.name,
            rows: n_ps,
            cols: n_ifg,
            values,
        }));
    }
    stage2_err(format!(
        "bp1.bperp_mat has incompatible shape {}x{} for stage-2 ph shape {}x{}",
        source.rows, source.cols, n_ps, n_ifg
    ))
}

fn load_da(patch_dir: &Path, n_ps: usize) -> Result<Vec<f64>, CoreError> {
    let path = patch_dir.join("da1.mat");
    if !path.exists() {
        return Ok(vec![1.0; n_ps]);
    }
    let da = MatData::read(path).map_err(|err| stage2_err_owned(format!("unable to read da1.mat: {err}")))?;
    let values = optional_vector_f64(&da, "D_A").unwrap_or_else(|| vec![1.0; n_ps]);
    if values.len() == n_ps {
        Ok(values)
    } else {
        Ok(vec![1.0; n_ps])
    }
}

fn fill_phase_weight(
    prepared: &Stage2Prepared,
    k_ps: &[f64],
    weighting: &[f64],
    out: &mut [Complex32],
) -> Result<(), CoreError> {
    for row in 0..prepared.n_ps {
        for col in 0..prepared.n_ifg {
            let bp = if prepared.row_invariant_bperp {
                prepared.row_bperp_nm[col]
            } else {
                let Some(mat) = prepared.bperp_mat.as_ref() else {
                    return stage2_err("bp1.bperp_mat is required for non-invariant stage-2 baselines");
                };
                mat.values[row * prepared.n_ifg + col]
            };
            let phase = -(bp * k_ps[row]);
            let (sn, cs) = phase.sin_cos();
            let ramp = Complex64::new(cs, sn);
            let src = prepared.ph_nm[row * prepared.n_ifg + col];
            let value = Complex64::new(src.re as f64, src.im as f64) * ramp * weighting[row];
            out[row * prepared.n_ifg + col] = Complex32::new(value.re as f32, value.im as f32);
        }
    }
    Ok(())
}

fn accumulate_grid(
    ph_weight: &[Complex32],
    grid_lin: &[usize],
    n_i: usize,
    n_j: usize,
    n_ifg: usize,
    out: &mut [Complex32],
) {
    out.fill(Complex32::new(0.0, 0.0));
    let grid_cells = n_i * n_j;
    for (row, &grid_ix) in grid_lin.iter().enumerate() {
        if grid_ix >= grid_cells {
            continue;
        }
        for col in 0..n_ifg {
            out[grid_ix * n_ifg + col] += ph_weight[row * n_ifg + col];
        }
    }
}

fn clap_filter_grid_stack(
    ph_grid: &[Complex32],
    n_i: usize,
    n_j: usize,
    n_ifg: usize,
    n_win: usize,
    alpha: f64,
    beta: f64,
    low_pass: &[f64],
    out: &mut [Complex32],
) {
    out.fill(Complex32::new(0.0, 0.0));
    let n_inc = (n_win / 4).max(1);
    let n_win_i = (n_i as f64 / n_inc as f64).ceil() as isize - 3;
    let n_win_j = (n_j as f64 / n_inc as f64).ceil() as isize - 3;
    if n_win_i <= 0 || n_win_j <= 0 {
        return;
    }

    let n_win_ex = low_pass_dim(low_pass).unwrap_or(n_win);
    let kernel = clap_filter_kernel();
    let windows = clap_windows(n_i, n_j, n_win, n_inc, n_win_i as usize, n_win_j as usize);
    let mut accum = vec![Complex64::new(0.0, 0.0); n_i * n_j * n_ifg];
    let mut planner = FftPlanner::<f64>::new();
    let fft_row = planner.plan_fft_forward(n_win_ex);
    let ifft_row = planner.plan_fft_inverse(n_win_ex);
    let fft_col = planner.plan_fft_forward(n_win_ex);
    let ifft_col = planner.plan_fft_inverse(n_win_ex);

    for window in windows {
        for ifg in 0..n_ifg {
            let mut ph_bit = vec![Complex64::new(0.0, 0.0); n_win_ex * n_win_ex];
            for local_i in 0..n_win {
                let src_i = window.i1 + local_i;
                for local_j in 0..n_win {
                    let src_j = window.j1 + local_j;
                    let value = ph_grid[(src_i * n_j + src_j) * n_ifg + ifg];
                    ph_bit[local_i * n_win_ex + local_j] = if value.re.is_nan() || value.im.is_nan() {
                        Complex64::new(0.0, 0.0)
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    };
                }
            }
            fft2_in_place(&mut ph_bit, n_win_ex, &fft_row, &fft_col);
            let h_abs = ph_bit.iter().map(|value| value.norm()).collect::<Vec<_>>();
            let h_shift = fftshift_real(&h_abs, n_win_ex, n_win_ex);
            let h_conv = convolve_same_symmetric(&h_shift, n_win_ex, n_win_ex, &kernel, 7);
            let mut h_smooth = ifftshift_real(&h_conv, n_win_ex, n_win_ex);
            let mean_h = median(&mut h_smooth.clone());
            if mean_h != 0.0 {
                for value in &mut h_smooth {
                    *value /= mean_h;
                }
            }
            for (ix, value) in h_smooth.iter_mut().enumerate() {
                *value = value.powf(alpha) - 1.0;
                if *value < 0.0 {
                    *value = 0.0;
                }
                *value = *value * beta + low_pass[ix];
                ph_bit[ix] *= *value;
            }
            ifft2_in_place(&mut ph_bit, n_win_ex, &ifft_row, &ifft_col);
            let inv_scale = 1.0 / (n_win_ex * n_win_ex) as f64;
            for local_i in 0..n_win {
                let dst_i = window.i1 + local_i;
                for local_j in 0..n_win {
                    let dst_j = window.j1 + local_j;
                    let weight = window.weight[local_i * n_win + local_j];
                    let value = ph_bit[local_i * n_win_ex + local_j] * (weight * inv_scale);
                    accum[(dst_i * n_j + dst_j) * n_ifg + ifg] += value;
                }
            }
        }
    }

    for (dst, src) in out.iter_mut().zip(accum.iter()) {
        *dst = Complex32::new(src.re as f32, src.im as f32);
    }
}

#[derive(Clone, Debug)]
struct ClapWindow {
    i1: usize,
    j1: usize,
    weight: Vec<f64>,
}

fn clap_windows(
    n_i: usize,
    n_j: usize,
    n_win: usize,
    n_inc: usize,
    n_win_i: usize,
    n_win_j: usize,
) -> Vec<ClapWindow> {
    let base_weight = clap_window_weight(n_win);
    let mut windows = Vec::with_capacity(n_win_i * n_win_j);
    for ix1 in 0..n_win_i {
        let mut i1 = ix1 * n_inc;
        let mut i2 = i1 + n_win;
        let mut row_shift = 0usize;
        if i2 > n_i {
            row_shift = i2 - n_i;
            i2 = n_i;
            i1 = n_i - n_win;
        }
        let _ = i2;
        for ix2 in 0..n_win_j {
            let mut j1 = ix2 * n_inc;
            let mut j2 = j1 + n_win;
            let mut col_shift = 0usize;
            if j2 > n_j {
                col_shift = j2 - n_j;
                j2 = n_j;
                j1 = n_j - n_win;
            }
            let _ = j2;
            let mut weight = vec![0.0; n_win * n_win];
            for row in 0..n_win {
                for col in 0..n_win {
                    if row < row_shift || col < col_shift {
                        weight[row * n_win + col] = 0.0;
                    } else {
                        let src_row = row - row_shift;
                        let src_col = col - col_shift;
                        weight[row * n_win + col] = base_weight[src_row * n_win + src_col];
                    }
                }
            }
            windows.push(ClapWindow { i1, j1, weight });
        }
    }
    windows
}

fn clap_window_weight(n_win: usize) -> Vec<f64> {
    let half = n_win / 2;
    let mut quadrant = vec![0.0; half * half];
    for row in 0..half {
        for col in 0..half {
            quadrant[row * half + col] = row as f64 + col as f64 + 1.0e-6;
        }
    }
    let mut top = vec![0.0; half * n_win];
    for row in 0..half {
        for col in 0..half {
            top[row * n_win + col] = quadrant[row * half + col];
            top[row * n_win + half + col] = quadrant[row * half + (half - 1 - col)];
        }
    }
    let mut out = vec![0.0; n_win * n_win];
    for row in 0..half {
        for col in 0..n_win {
            out[row * n_win + col] = top[row * n_win + col];
            out[(half + row) * n_win + col] = top[(half - 1 - row) * n_win + col];
        }
    }
    out
}

fn low_pass_dim(low_pass: &[f64]) -> Option<usize> {
    let dim = (low_pass.len() as f64).sqrt() as usize;
    (dim > 0 && dim * dim == low_pass.len()).then_some(dim)
}

fn clap_filter_kernel() -> Vec<f64> {
    let alpha = 2.5;
    let std = (7.0 - 1.0) / (2.0 * alpha);
    let mut g = [0.0; 7];
    for (ix, value) in g.iter_mut().enumerate() {
        let x = ix as f64 - 3.0;
        *value = (-0.5 * (x / std) * (x / std)).exp();
    }
    let mut kernel = vec![0.0; 49];
    for row in 0..7 {
        for col in 0..7 {
            kernel[row * 7 + col] = g[row] * g[col];
        }
    }
    kernel
}

fn fft2_in_place(
    values: &mut [Complex64],
    n: usize,
    fft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    fft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    for row in 0..n {
        fft_row.process(&mut values[row * n..(row + 1) * n]);
    }
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        for row in 0..n {
            scratch[row] = values[row * n + col];
        }
        fft_col.process(&mut scratch);
        for row in 0..n {
            values[row * n + col] = scratch[row];
        }
    }
}

fn ifft2_in_place(
    values: &mut [Complex64],
    n: usize,
    ifft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    ifft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
) {
    for row in 0..n {
        ifft_row.process(&mut values[row * n..(row + 1) * n]);
    }
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        for row in 0..n {
            scratch[row] = values[row * n + col];
        }
        ifft_col.process(&mut scratch);
        for row in 0..n {
            values[row * n + col] = scratch[row];
        }
    }
}

fn fftshift_real(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    shift_real(values, rows, cols, rows / 2, cols / 2)
}

fn ifftshift_real(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    shift_real(values, rows, cols, rows.div_ceil(2), cols.div_ceil(2))
}

fn shift_real(values: &[f64], rows: usize, cols: usize, row_shift: usize, col_shift: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            let src_row = (row + row_shift) % rows;
            let src_col = (col + col_shift) % cols;
            out[row * cols + col] = values[src_row * cols + src_col];
        }
    }
    out
}

fn convolve_same_symmetric(values: &[f64], rows: usize, cols: usize, kernel: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    let radius = k / 2;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for kr in 0..k {
                let Some(src_row) = row.checked_add(kr).and_then(|v| v.checked_sub(radius)) else {
                    continue;
                };
                if src_row >= rows {
                    continue;
                }
                for kc in 0..k {
                    let Some(src_col) = col.checked_add(kc).and_then(|v| v.checked_sub(radius)) else {
                        continue;
                    };
                    if src_col >= cols {
                        continue;
                    }
                    sum += values[src_row * cols + src_col] * kernel[kr * k + kc];
                }
            }
            out[row * cols + col] = sum;
        }
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn extract_patch_phase(prepared: &Stage2Prepared, ph_filt: &[Complex32], out: &mut [Complex32]) {
    for row in 0..prepared.n_ps {
        let grid_ix = prepared.grid_lin[row];
        for col in 0..prepared.n_ifg {
            out[row * prepared.n_ifg + col] = ph_filt[grid_ix * prepared.n_ifg + col];
        }
    }
    normalize_complex_unit_magnitude(out);
}

fn normalize_complex_unit_magnitude(values: &mut [Complex32]) {
    for value in values {
        let mag = value.norm();
        if mag != 0.0 {
            *value /= mag;
        }
    }
}

fn topofit_row(cpx: &[Complex64], bperp: &[f64], n_trial_wraps: f64) -> TopofitRow {
    let trial_mult = trial_values(n_trial_wraps);
    let valid = cpx
        .iter()
        .zip(bperp.iter())
        .enumerate()
        .filter_map(|(ix, (&value, &bp))| (value != Complex64::new(0.0, 0.0)).then_some((ix, value, bp)))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return TopofitRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx.len()],
        };
    }
    let denom: f64 = valid.iter().map(|(_, value, _)| value.norm()).sum::<f64>().max(1.0);
    let min_bp = valid.iter().map(|(_, _, bp)| *bp).fold(f64::INFINITY, f64::min);
    let max_bp = valid.iter().map(|(_, _, bp)| *bp).fold(f64::NEG_INFINITY, f64::max);
    let bperp_range = (max_bp - min_bp).max(1.0);
    let mut coh_trial = vec![0.0; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (_, value, bp) in &valid {
            let phase = (bp / bperp_range) * (std::f64::consts::PI / 4.0) * trial_value;
            let (sn, cs) = phase.sin_cos();
            sum += *value * Complex64::new(cs, -sn);
        }
        coh_trial[trial_ix] = sum.norm() / denom;
    }
    let candidate_ix = near_max_trial_indices(&coh_trial);
    let mut refined = Vec::with_capacity(candidate_ix.len());
    let mut candidate_coh = Vec::with_capacity(candidate_ix.len());
    for &trial_ix in &candidate_ix {
        let coarse_k0 = (std::f64::consts::PI / 4.0) / bperp_range * trial_mult[trial_ix];
        refined.push(refine_candidate(&valid, cpx.len(), coarse_k0));
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

fn refine_candidate(valid: &[(usize, Complex64, f64)], n_col: usize, coarse_k0: f64) -> TopofitRow {
    let mut offset = Complex64::new(0.0, 0.0);
    for (_, value, bp) in valid {
        let phase = coarse_k0 * bp;
        let (sn, cs) = phase.sin_cos();
        offset += *value * Complex64::new(cs, -sn);
    }
    let offset_conj = offset.conj();
    let mut mopt_num = 0.0;
    let mut den_lin = 0.0;
    for (_, value, bp) in valid {
        let weight = value.norm();
        let wb = weight * bp;
        den_lin += wb * wb;
        let phase = coarse_k0 * bp;
        let (sn, cs) = phase.sin_cos();
        let res = *value * Complex64::new(cs, -sn);
        mopt_num += wb * (weight * (res * offset_conj).arg());
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }
    let k = coarse_k0 + mopt_num / den_lin;
    let mut mean_phase_residual = Complex64::new(0.0, 0.0);
    let mut denom = 0.0;
    let mut residual = vec![Complex32::new(0.0, 0.0); n_col];
    for (col, value, bp) in valid {
        let phase = k * bp;
        let (sn, cs) = phase.sin_cos();
        let res = *value * Complex64::new(cs, -sn);
        mean_phase_residual += res;
        denom += res.norm();
        residual[*col] = Complex32::new(res.re as f32, res.im as f32);
    }
    if denom == 0.0 {
        denom = 1.0;
    }
    TopofitRow {
        k,
        c: mean_phase_residual.arg(),
        coh: mean_phase_residual.norm() / denom,
        residual,
    }
}

fn trial_values(n_trial_wraps: f64) -> Vec<f64> {
    let trial_n = (8.0 * n_trial_wraps).ceil() as i64;
    (-trial_n..=trial_n).map(|value| value as f64).collect()
}

fn near_max_trial_indices(coh_trial: &[f64]) -> Vec<usize> {
    if coh_trial.len() <= 1 {
        return vec![0];
    }
    let mut local_max = vec![false; coh_trial.len()];
    local_max[0] = coh_trial[0] >= coh_trial[1];
    local_max[coh_trial.len() - 1] = coh_trial[coh_trial.len() - 1] >= coh_trial[coh_trial.len() - 2];
    for idx in 1..coh_trial.len() - 1 {
        local_max[idx] = coh_trial[idx] >= coh_trial[idx - 1] && coh_trial[idx] >= coh_trial[idx + 1];
    }
    let max_coh = coh_trial.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = local_max
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_local_max)| {
            (is_local_max && coh_trial[idx] >= max_coh - STAGE2_TOPOFIT_NEAR_MAX_COH_TOL).then_some(idx)
        })
        .collect::<Vec<_>>();
    if out.is_empty() {
        out.push(argmax_first(coh_trial));
    }
    out
}

fn select_candidate(candidate_ix: &[usize], candidate_coh: &[f64], refined_coh: &[f64], trial_count: usize) -> usize {
    let coarse_best = candidate_ix[argmax_first(candidate_coh)];
    if candidate_ix.len() == 1 {
        return coarse_best;
    }
    if candidate_ix.len() == 2 && candidate_ix[0] == 0 && candidate_ix[1] == trial_count - 1 {
        return coarse_best;
    }
    candidate_ix[argmax_first(refined_coh)]
}

fn argmax_first(values: &[f64]) -> usize {
    let mut best_ix = 0;
    let mut best_value = values.first().copied().unwrap_or(f64::NEG_INFINITY);
    for (ix, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_ix = ix;
        }
    }
    best_ix
}

fn hist_with_centers(values: &[f64], centers: &[f64]) -> Vec<f64> {
    if centers.is_empty() {
        return Vec::new();
    }
    if centers.len() == 1 {
        return vec![values.len() as f64];
    }
    let mids = centers.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0).collect::<Vec<_>>();
    let mut out = vec![0.0; centers.len()];
    for &value in values {
        let ix = mids.partition_point(|&mid| mid < value).min(centers.len() - 1);
        out[ix] += 1.0;
    }
    out
}

fn psquare_weighting(nr: &[f64], na: &[f64], low_coh_thresh: usize, nr_max_nz_ix: f64, coh_ps: &[f64]) -> Vec<f64> {
    let mut prand = vec![0.0; nr.len()];
    for ix in 0..nr.len() {
        let denom = if na[ix] == 0.0 { 1.0 } else { na[ix] };
        prand[ix] = (nr[ix] / denom).min(1.0);
    }
    for ix in 0..low_coh_thresh.min(prand.len()) {
        prand[ix] = 1.0;
    }
    for ix in (nr_max_nz_ix as usize).min(prand.len())..prand.len() {
        prand[ix] = 0.0;
    }
    coh_ps
        .iter()
        .map(|&coh| {
            let ix = ((coh * 1000.0).round() as usize / 10).min(prand.len().saturating_sub(1));
            let p = prand[ix];
            (1.0 - p) * (1.0 - p)
        })
        .collect()
}

fn snr_weighting(prepared: &Stage2Prepared, ph_res: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; prepared.n_ps];
    for row in 0..prepared.n_ps {
        let mut g = 0.0;
        let mut amp2 = 0.0;
        for col in 0..prepared.n_ifg {
            let ix = row * prepared.n_ifg + col;
            let amp = prepared.amp[ix] as f64;
            g += amp * (ph_res[ix] as f64).cos();
            amp2 += amp * amp;
        }
        g /= prepared.n_ifg.max(1) as f64;
        amp2 /= prepared.n_ifg.max(1) as f64;
        let sigma_n = (0.5 * (amp2 - g * g)).sqrt();
        if sigma_n != 0.0 {
            out[row] = g / sigma_n;
        }
    }
    out
}

fn write_pm1(
    patch_dir: &Path,
    prepared: &Stage2Prepared,
    k_ps: &[f64],
    c_ps: &[f64],
    coh_ps: &[f64],
    n_opt: &[f64],
    ph_res: &[f32],
    ph_patch: &[Complex32],
    ph_grid: &[Complex32],
    ph_weight: &[Complex32],
    nr: &[f64],
    nr_max_nz_ix: f64,
    coh_ps_save: &[f64],
    gamma_change_save: f64,
    i_loop: usize,
) -> Result<(), CoreError> {
    let mut mat = MatFile::new(patch_dir.join("pm1.mat"));
    mat.add_f64_col_vector("K_ps", k_ps.to_vec())?;
    mat.add_f64_col_vector("C_ps", c_ps.to_vec())?;
    mat.add_f64_col_vector("coh_ps", coh_ps.to_vec())?;
    mat.add_f64_col_vector("N_opt", n_opt.to_vec())?;
    mat.add_f32_matrix("ph_res", prepared.n_ps, prepared.n_ifg, ph_res.to_vec())?;
    mat.add_complex_f32_matrix("ph_patch", prepared.n_ps, prepared.n_ifg, complex32_pairs(ph_patch))?;
    mat.add_f64_scalar("step_number", 1.0)?;
    mat.add_complex_f32_matrix("ph_grid", prepared.n_i, prepared.n_j * prepared.n_ifg, complex32_pairs(ph_grid))?;
    mat.add_f32_scalar("n_trial_wraps", prepared.n_trial_wraps as f32)?;
    mat.add_f32_matrix("grid_ij", prepared.grid_ij.rows, prepared.grid_ij.cols, prepared.grid_ij.values.clone())?;
    mat.add_f64_scalar("grid_size", prepared.grid_size)?;
    mat.add_f64_matrix(
        "low_pass",
        prepared.low_pass.rows,
        prepared.low_pass.cols,
        prepared.low_pass.values.clone(),
    )?;
    mat.add_f64_scalar("i_loop", i_loop as f64)?;
    mat.add_complex_f32_matrix("ph_weight", prepared.n_ps, prepared.n_ifg, complex32_pairs(ph_weight))?;
    mat.add_f64_row_vector("Nr", nr.to_vec())?;
    mat.add_f64_scalar("Nr_max_nz_ix", nr_max_nz_ix)?;
    mat.add_f64_row_vector("coh_bins", prepared.coh_bins.clone())?;
    mat.add_f64_col_vector("coh_ps_save", coh_ps_save.to_vec())?;
    mat.add_f64_scalar("gamma_change_save", gamma_change_save)?;
    mat.write()?;
    Ok(())
}

fn complex32_pairs(values: &[Complex32]) -> Vec<(f32, f32)> {
    values.iter().map(|value| (value.re, value.im)).collect()
}

fn stage2_grid_indices(xy: &Matrix<f32>, grid_size: f64) -> Matrix<f32> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    for row in 0..xy.rows {
        min_x = min_x.min(xy.values[row * xy.cols + 1]);
        min_y = min_y.min(xy.values[row * xy.cols + 2]);
    }
    let scale = grid_size as f32;
    let eps = 1.0e-6_f32;
    let mut values = vec![0.0_f32; xy.rows * 2];
    let mut max_i = 1_i32;
    let mut max_j = 1_i32;
    for row in 0..xy.rows {
        let i = ((xy.values[row * xy.cols + 2] - min_y + eps) / scale).ceil().max(1.0) as i32;
        let j = ((xy.values[row * xy.cols + 1] - min_x + eps) / scale).ceil().max(1.0) as i32;
        values[row * 2] = i as f32;
        values[row * 2 + 1] = j as f32;
        max_i = max_i.max(i);
        max_j = max_j.max(j);
    }
    if max_i > 1 || max_j > 1 {
        for row in 0..xy.rows {
            if max_i > 1 && values[row * 2] as i32 == max_i {
                values[row * 2] = (max_i - 1) as f32;
            }
            if max_j > 1 && values[row * 2 + 1] as i32 == max_j {
                values[row * 2 + 1] = (max_j - 1) as f32;
            }
        }
    }
    Matrix {
        name: "grid_ij".to_string(),
        rows: xy.rows,
        cols: 2,
        values,
    }
}

fn build_low_pass(options: &Stage2Options) -> Matrix<f64> {
    let n_win = options.clap_win.round().max(1.0) as usize;
    let freq0 = 1.0 / options.clap_low_pass_wavelength;
    let mut butter = Vec::with_capacity(n_win);
    for ix in 0..n_win {
        let freq_i = (ix as f64 - n_win as f64 / 2.0) / (options.grid_size * n_win as f64);
        butter.push(1.0 / (1.0 + (freq_i / freq0).powi(10)));
    }
    let mut raw = vec![0.0; n_win * n_win];
    for row in 0..n_win {
        for col in 0..n_win {
            raw[row * n_win + col] = butter[row] * butter[col];
        }
    }
    let mut shifted = vec![0.0; raw.len()];
    let row_shift = n_win / 2;
    let col_shift = n_win / 2;
    for row in 0..n_win {
        for col in 0..n_win {
            let src_row = (row + row_shift) % n_win;
            let src_col = (col + col_shift) % n_win;
            shifted[row * n_win + col] = raw[src_row * n_win + src_col];
        }
    }
    Matrix {
        name: "low_pass".to_string(),
        rows: n_win,
        cols: n_win,
        values: shifted,
    }
}

fn rms_difference(left: &[f64], right: &[f64]) -> f64 {
    let denom = left.len().max(1) as f64;
    (left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| {
            let diff = l - r;
            diff * diff
        })
        .sum::<f64>()
        / denom)
        .sqrt()
}

fn load_stage2_options(mat: Option<&MatData>) -> Stage2Options {
    let mut options = Stage2Options::default();
    if let Some(mat) = mat {
        options.grid_size = scalar_from_mat_default(mat, "filter_grid_size", options.grid_size);
        options.clap_win = scalar_from_mat_default(mat, "clap_win", options.clap_win);
        options.clap_low_pass_wavelength =
            scalar_from_mat_default(mat, "clap_low_pass_wavelength", options.clap_low_pass_wavelength);
        options.clap_alpha = scalar_from_mat_default(mat, "clap_alpha", options.clap_alpha);
        options.clap_beta = scalar_from_mat_default(mat, "clap_beta", options.clap_beta);
        options.max_topo_err = scalar_from_mat_default(mat, "max_topo_err", options.max_topo_err);
        options.lambda_m = scalar_from_mat_default(mat, "lambda", options.lambda_m);
    }
    options
}

fn load_stage2_parms(mat: Option<&MatData>) -> Stage2Parms {
    let mut parms = Stage2Parms::default();
    if let Some(mat) = mat {
        parms.small_baseline_flag = text_from_mat(mat, "small_baseline_flag", &parms.small_baseline_flag);
        parms.filter_weighting = text_from_mat(mat, "filter_weighting", &parms.filter_weighting);
        parms.gamma_change_convergence =
            scalar_from_mat_default(mat, "gamma_change_convergence", parms.gamma_change_convergence);
        parms.gamma_max_iterations =
            scalar_from_mat_default(mat, "gamma_max_iterations", parms.gamma_max_iterations as f64).round() as usize;
    }
    parms
}

fn stage2_trial_wrap_mean_incidence(patch_dir: &Path, ps: &MatData) -> f64 {
    for (filename, varname, offset) in [("inc1.mat", "inc", 0.0), ("la1.mat", "la", 0.052)] {
        let path = patch_dir.join(filename);
        if !path.exists() {
            continue;
        }
        if let Ok(mat) = MatData::read(path) {
            if let Some(values) = optional_vector_f64(&mat, varname) {
                let valid = values
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite() && (*value != 0.0 || varname == "la"))
                    .collect::<Vec<_>>();
                if !valid.is_empty() {
                    return valid.iter().sum::<f64>() / valid.len() as f64 + offset;
                }
            }
        }
    }
    scalar_from_mat_default(ps, "mean_incidence", DEFAULT_MEAN_INCIDENCE)
}

fn scalar_from_mat(mat: &MatData, name: &str, default: f64) -> f64 {
    scalar_from_mat_default(mat, name, default)
}

fn scalar_from_mat_default(mat: &MatData, name: &str, default: f64) -> f64 {
    optional_vector_f64(mat, name)
        .and_then(|values| values.into_iter().next())
        .unwrap_or(default)
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
}

fn vector_f64(mat: &MatData, name: &str, label: &str) -> Result<Vec<f64>, CoreError> {
    optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 2,
        message: format!("{label} is missing"),
    })
}

fn ps_matrix_f64(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 2,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    orient_matrix_f64(source, n_ps, label)
}

fn ps_dim_f32(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
    let source = mat.get_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 2,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        let mut values = Vec::with_capacity(source.values.len());
        for row in 0..source.cols {
            for col in 0..source.rows {
                values.push(source.values[col * source.cols + row]);
            }
        }
        return Ok(Matrix {
            name: source.name,
            rows: source.cols,
            cols: source.rows,
            values,
        });
    }
    stage2_err(format!(
        "{label} has incompatible shape {}x{}; expected {n_ps}x{n_dim}",
        source.rows, source.cols
    ))
}

fn ps_complex_matrix(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat.get_complex_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 2,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        let mut values = Vec::with_capacity(source.values.len());
        for row in 0..source.cols {
            for col in 0..source.rows {
                values.push(source.values[col * source.cols + row]);
            }
        }
        return Ok(ComplexMatrixF32 {
            name: source.name,
            rows: source.cols,
            cols: source.rows,
            values,
        });
    }
    stage2_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn orient_matrix_f64(source: Matrix<f64>, n_ps: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        let mut values = Vec::with_capacity(source.values.len());
        for row in 0..source.cols {
            for col in 0..source.rows {
                values.push(source.values[col * source.cols + row]);
            }
        }
        return Ok(Matrix {
            name: source.name,
            rows: source.cols,
            cols: source.rows,
            values,
        });
    }
    stage2_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn text_from_mat(mat: &MatData, name: &str, default: &str) -> String {
    let Some(values) = optional_vector_f64(mat, name) else {
        return default.to_string();
    };
    let text = values
        .into_iter()
        .filter_map(|value| char::from_u32(value.round() as u32))
        .filter(|&ch| ch != '\0')
        .collect::<String>()
        .trim()
        .to_string();
    if text.is_empty() { default.to_string() } else { text }
}

fn bperp_rows_are_invariant(bperp_mat: Option<&Matrix<f64>>) -> bool {
    let Some(mat) = bperp_mat else {
        return true;
    };
    if mat.rows <= 1 {
        return true;
    }
    for row in 1..mat.rows {
        for col in 0..mat.cols {
            if mat.values[row * mat.cols + col] != mat.values[col] {
                return false;
            }
        }
    }
    true
}

fn resolve_file_optional(patch_dir: &Path, filename: &str) -> Option<PathBuf> {
    [
        patch_dir.join(filename),
        patch_dir.parent().map(|parent| parent.join(filename)).unwrap_or_default(),
        patch_dir
            .parent()
            .and_then(|parent| parent.parent())
            .map(|parent| parent.join(filename))
            .unwrap_or_default(),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn stage2_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage2_err_owned(message.into()))
}

fn stage2_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 2, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_stage2_matches_python_empty_clap_fixture_and_is_faster() {
        let root = temp_root("stage2-native");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage2_fixture(&python_root, false);
        create_stage2_fixture(&rust_root, false);

        let python_start = Instant::now();
        run_python_stage2(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage2_native(rust_root.join("PATCH_1")).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let summary = compare_fixture_artifacts(
            2,
            "patch",
            "synthetic_stage2_empty_clap",
            &python_root,
            &rust_root,
            &[ArtifactComparisonSpec::new(
                "PATCH_1/pm1.mat",
                [
                    "K_ps",
                    "C_ps",
                    "coh_ps",
                    "N_opt",
                    "ph_res",
                    "ph_patch",
                    "step_number",
                    "n_trial_wraps",
                    "grid_ij",
                    "grid_size",
                    "i_loop",
                    "coh_bins",
                    "coh_ps_save",
                    "gamma_change_save",
                ],
            )],
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 2 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 2 should beat Python/native-kernel path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incompatible_bp1_shape_returns_stage2_error() {
        let root = temp_root("stage2-bad-bp");
        create_stage2_fixture(&root, true);
        let err = run_stage2_native(root.join("PATCH_1")).unwrap_err().to_string();
        assert!(err.contains("stage 2 native implementation error"));
        assert!(err.contains("bp1.bperp_mat has incompatible shape"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn coverage_reports_stage2_native_after_parity_certification() {
        let coverage = crate::processing_chain_coverage(2, 2).unwrap();
        assert_eq!(coverage.len(), 1);
        assert!(coverage[0].native_stage);
    }

    fn create_stage2_fixture(root: &Path, bad_bp_shape: bool) {
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        let n_ps = 3;
        let n_ifg = 3;
        let mut ps = MatFile::new(patch.join("ps1.mat"));
        ps.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps.add_f64_scalar("n_ifg", n_ifg as f64).unwrap();
        ps.add_f64_scalar("n_image", n_ifg as f64).unwrap();
        ps.add_f64_scalar("master_ix", 1.0).unwrap();
        ps.add_f64_row_vector("bperp", vec![0.0, 15.0, 30.0]).unwrap();
        ps.add_f64_scalar("mean_incidence", DEFAULT_MEAN_INCIDENCE).unwrap();
        ps.add_f32_matrix(
            "xy",
            n_ps,
            3,
            vec![
                1.0, 0.0, 0.0,
                2.0, 5.0, 5.0,
                3.0, 10.0, 10.0,
            ],
        )
        .unwrap();
        ps.write().unwrap();

        let mut ph = MatFile::new(patch.join("ph1.mat"));
        ph.add_complex_f32_matrix(
            "ph",
            n_ps,
            n_ifg,
            vec![
                (1.0, 0.0), (0.8, 0.2), (0.6, 0.4),
                (1.0, 0.0), (0.7, 0.3), (0.5, 0.5),
                (1.0, 0.0), (0.9, 0.1), (0.4, 0.6),
            ],
        )
        .unwrap();
        ph.write().unwrap();

        let mut bp = MatFile::new(patch.join("bp1.mat"));
        if bad_bp_shape {
            bp.add_f64_matrix("bperp_mat", n_ps, 1, vec![10.0, 20.0, 30.0]).unwrap();
        } else {
            bp.add_f64_matrix(
                "bperp_mat",
                n_ps,
                2,
                vec![15.0, 30.0, 15.0, 30.0, 15.0, 30.0],
            )
            .unwrap();
        }
        bp.write().unwrap();

        let mut da = MatFile::new(patch.join("da1.mat"));
        da.add_f64_row_vector("D_A", vec![1.0; n_ps]).unwrap();
        da.write().unwrap();

        let mut parms = MatFile::new(patch.join("parms.mat"));
        parms.add_f64_scalar("gamma_max_iterations", 1.0).unwrap();
        parms.add_f64_scalar("filter_grid_size", 50.0).unwrap();
        parms.write().unwrap();
    }

    fn run_python_stage2(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage2_estimate_gamma; stage2_estimate_gamma(Path(sys.argv[1]) / 'PATCH_1', kernel_backend='native')";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage2 failed: {}\nstdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        root
    }
}
