use crate::CoreError;
use num_complex::{Complex32, Complex64};
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const HDF5_SIGNATURE_SCAN_BYTES: usize = 1024 * 1024;
const STAGE2_RANDOM_SEED: u32 = 2005;
#[cfg(not(test))]
const STAGE2_RANDOM_COUNT: usize = 300_000;
#[cfg(test)]
const STAGE2_RANDOM_COUNT: usize = 10_000;
// MATLAB interp([1, Prand], 10) delegates to the Signal Processing Toolbox
// interpolation FIR. Stage 2 only uses factor 10, so keep the audited taps
// fixed instead of approximating them with a different low-pass designer.
const STAGE2_INTERP10_TAPS: [f64; 79] = [
    -7.92925708923623813e-05,
    -1.73293349766172333e-04,
    -3.00248431516961407e-04,
    -4.48605550383722462e-04,
    -5.95835452295412349e-04,
    -7.08759204842826507e-04,
    -7.46231412934915406e-04,
    -6.64375214040318490e-04,
    -4.24121369705182660e-04,
    -3.01053961794528480e-07,
    6.08939001185030554e-04,
    1.37399755086099230e-03,
    2.22872001918483081e-03,
    3.06999955141492101e-03,
    3.76333143346069959e-03,
    4.15488007985555935e-03,
    4.08970016389962471e-03,
    3.43472424182160452e-03,
    2.10413493213892910e-03,
    8.39421246326618742e-05,
    -2.54786253685349538e-03,
    -5.60918261221970971e-03,
    -8.81278723412589476e-03,
    -1.17783753398490003e-02,
    -1.40582499688596801e-02,
    -1.51753768689126753e-02,
    -1.46708218626883501e-02,
    -1.21559904304523667e-02,
    -7.36396963079587880e-03,
    -1.93792008491816548e-04,
    9.25827682864269544e-03,
    2.06852593184593814e-02,
    3.35768644427044682e-02,
    4.72485589714582085e-02,
    6.08905923193367796e-02,
    7.36328154139928304e-02,
    8.46190871379102622e-02,
    9.30836882813813049e-02,
    9.84216619119140101e-02,
    1.00245459871706619e-01,
    9.84216619119134828e-02,
    9.30836882813811800e-02,
    8.46190871379124271e-02,
    7.36328154139975766e-02,
    6.08905923193391457e-02,
    4.72485589714637388e-02,
    3.35768644427109145e-02,
    2.06852593184574732e-02,
    9.25827682864464527e-03,
    -1.93792008489998774e-04,
    -7.36396963079247527e-03,
    -1.21559904304541604e-02,
    -1.46708218626876996e-02,
    -1.51753768689126579e-02,
    -1.40582499688608267e-02,
    -1.17783753398508096e-02,
    -8.81278723413165405e-03,
    -5.60918261221661496e-03,
    -2.54786253685826761e-03,
    8.39421246353382272e-05,
    2.10413493213411785e-03,
    3.43472424182329154e-03,
    4.08970016390024574e-03,
    4.15488007985538154e-03,
    3.76333143346146634e-03,
    3.06999955141576842e-03,
    2.22872001918783102e-03,
    1.37399755086213202e-03,
    6.08939001188346478e-04,
    -3.01053965413812172e-07,
    -4.24121369703597340e-04,
    -6.64375214039929912e-04,
    -7.46231412935480926e-04,
    -7.08759204842192465e-04,
    -5.95835452295058465e-04,
    -4.48605550383040716e-04,
    -3.00248431517695195e-04,
    -1.73293349768269614e-04,
    -7.92925708931989381e-05,
];

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
    nr_base: Vec<f64>,
    nr_max_nz_ix: f64,
    n_trial_wraps: f64,
    trial_values: Vec<f64>,
    grid_size: f64,
    clap_window: usize,
    low_coh_thresh: usize,
}

#[derive(Clone, Debug)]
struct TopofitRow {
    k: f64,
    c: f64,
    coh: f64,
    residual: Vec<Complex32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stage2RuntimeOptions {
    pub native_threads: usize,
}

pub fn run_stage2_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    run_stage2_native_with_options(patch_dir, Stage2RuntimeOptions::default())
}

pub fn run_stage2_native_with_threads(
    patch_dir: impl AsRef<Path>,
    native_threads: usize,
) -> Result<String, CoreError> {
    run_stage2_native_with_options(patch_dir, Stage2RuntimeOptions { native_threads })
}

pub fn run_stage2_native_with_options(
    patch_dir: impl AsRef<Path>,
    runtime: Stage2RuntimeOptions,
) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    if runtime.native_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(runtime.native_threads)
            .build()
            .map_err(|err| {
                stage2_err_owned(format!("unable to initialize stage-2 thread pool: {err}"))
            })?;
        return pool.install(|| run_stage2_native_inner(patch_dir));
    }
    run_stage2_native_inner(patch_dir)
}

fn run_stage2_native_inner(patch_dir: &Path) -> Result<String, CoreError> {
    let parms_source = Stage2ParmSource::from_patch(patch_dir);
    let parms = load_stage2_parms(&parms_source);
    let options = load_stage2_options(&parms_source);
    let prepared = prepare_stage2_inputs(patch_dir, &parms, &options)?;
    eprintln!(
        "stage2 native setup: n_ps={} n_ifg={} grid={}x{} clap_window={} trial_values={} max_iterations={} threads={}",
        prepared.n_ps,
        prepared.n_ifg,
        prepared.n_i,
        prepared.n_j,
        prepared.clap_window,
        prepared.trial_values.len(),
        parms.gamma_max_iterations,
        rayon::current_num_threads(),
    );

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
    let mut nr_scaled_last = prepared.nr_base.clone();
    let debug_pm_iterations = std::env::var_os("PYSTAMPS_STAGE2_NATIVE_DEBUG_PM").is_some();

    let mut i_loop = 1usize;
    loop {
        let iteration = i_loop;
        let iteration_start = Instant::now();
        let phase_start = Instant::now();
        fill_phase_weight(&prepared, &k_ps, &weighting, &mut ph_weight)?;
        let fill_elapsed = phase_start.elapsed();
        let phase_start = Instant::now();
        accumulate_grid(
            &ph_weight,
            &prepared.grid_lin,
            prepared.n_i,
            prepared.n_j,
            prepared.n_ifg,
            &mut ph_grid,
        );
        let grid_elapsed = phase_start.elapsed();
        let phase_start = Instant::now();
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
        let clap_elapsed = phase_start.elapsed();
        let phase_start = Instant::now();
        extract_patch_phase(&prepared, &ph_filt, &mut ph_patch);
        let extract_elapsed = phase_start.elapsed();

        let phase_start = Instant::now();
        fit_stage2_rows(
            &prepared,
            &ph_patch,
            &mut k_ps,
            &mut c_ps,
            &mut coh_ps,
            &mut n_opt,
            &mut ph_res,
        )?;
        let topofit_elapsed = phase_start.elapsed();

        let gamma_change_rms = rms_difference(&coh_ps, &coh_ps_save);
        let gamma_change_change = gamma_change_rms - gamma_change_save;
        gamma_change_save = gamma_change_rms;
        coh_ps_save.clone_from(&coh_ps);

        let should_stop = gamma_change_change.abs() < parms.gamma_change_convergence
            || i_loop >= parms.gamma_max_iterations.max(1);
        if should_stop {
            eprintln!(
                "stage2 native iteration {iteration} complete: fill={fill_elapsed:?} grid={grid_elapsed:?} clap={clap_elapsed:?} extract={extract_elapsed:?} topofit={topofit_elapsed:?} total={:?} gamma_change={gamma_change_save:.6e} gamma_delta={gamma_change_change:.6e} stop=true",
                iteration_start.elapsed(),
            );
            write_pm1(
                patch_dir,
                "pm1.mat",
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
                prepared.nr_max_nz_ix,
                &coh_ps_save,
                gamma_change_save,
                iteration,
            )?;
            if debug_pm_iterations {
                write_pm1(
                    patch_dir,
                    &format!("pm1_iter_{iteration:02}.mat"),
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
                    prepared.nr_max_nz_ix,
                    &coh_ps_save,
                    gamma_change_save,
                    iteration,
                )?;
            }
            break;
        }
        eprintln!(
            "stage2 native iteration {iteration} complete: fill={fill_elapsed:?} grid={grid_elapsed:?} clap={clap_elapsed:?} extract={extract_elapsed:?} topofit={topofit_elapsed:?} total={:?} gamma_change={gamma_change_save:.6e} gamma_delta={gamma_change_change:.6e} stop=false",
            iteration_start.elapsed(),
        );

        if parms.filter_weighting.eq_ignore_ascii_case("P-square") {
            let na = hist_with_centers(&coh_ps, &prepared.coh_bins);
            let denom: f64 = nr_scaled_last.iter().take(prepared.low_coh_thresh).sum();
            let scale = if denom > 0.0 {
                na.iter().take(prepared.low_coh_thresh).sum::<f64>() / denom
            } else {
                1.0
            };
            for value in &mut nr_scaled_last {
                *value *= scale;
            }
            weighting = psquare_weighting(
                &nr_scaled_last,
                &na,
                prepared.low_coh_thresh,
                prepared.nr_max_nz_ix,
                &coh_ps,
            );
        } else {
            weighting = snr_weighting(&prepared, &ph_res);
        }
        if debug_pm_iterations {
            write_pm1(
                patch_dir,
                &format!("pm1_iter_{iteration:02}.mat"),
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
                prepared.nr_max_nz_ix,
                &coh_ps_save,
                gamma_change_save,
                iteration,
            )?;
        }
        i_loop += 1;
    }

    Ok(format!(
        "Stage 2 computed coherence for {} candidates in {i_loop} iterations using {} native threads",
        prepared.n_ps,
        rayon::current_num_threads()
    ))
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
        return stage2_err(format!(
            "ps1.master_ix must be 1-based within ph1 width {n_ifg_full}; got {master_ix}"
        ));
    }
    let bperp_full = vector_f64(&ps, "bperp", "ps1.bperp")?
        .into_iter()
        .map(|value| value as f32 as f64)
        .collect::<Vec<_>>();
    if bperp_full.len() != n_ifg_full {
        return stage2_err(format!(
            "ps1.bperp has length {} but ph1.ph has {n_ifg_full} interferograms",
            bperp_full.len()
        ));
    }

    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    let no_master: Vec<usize> = (0..n_ifg_full)
        .filter(|&ix| small_baseline || ix != master_ix - 1)
        .collect();
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
            ph_nm.push(if safe_mag != 0.0 {
                value / safe_mag
            } else {
                Complex32::new(0.0, 0.0)
            });
        }
    }
    let bperp_nm: Vec<f64> = no_master.iter().map(|&ix| bperp_full[ix]).collect();
    let bperp_mat = load_bperp_mat(
        patch_dir,
        n_ps,
        n_ifg_full,
        n_ifg,
        &no_master,
        small_baseline,
        &bperp_nm,
    )?;
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
    let n_i = grid_ij
        .values
        .iter()
        .step_by(2)
        .fold(1usize, |acc, &value| acc.max(value as usize));
    let n_j = grid_ij
        .values
        .iter()
        .skip(1)
        .step_by(2)
        .fold(1usize, |acc, &value| acc.max(value as usize));
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
    let max_k = options.max_topo_err
        / (options.lambda_m * rho * mean_incidence.sin() / (4.0 * std::f64::consts::PI));
    let (min_bp, max_bp) = bperp_nm.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_v, max_v), &value| (min_v.min(value), max_v.max(value)),
    );
    let n_trial_wraps = ((max_bp - min_bp) * max_k / (2.0 * std::f64::consts::PI)) as f64;
    let trial_values = trial_values(n_trial_wraps);
    let low_coh_thresh = if parms.small_baseline_flag.eq_ignore_ascii_case("y") {
        15
    } else {
        31
    };
    let (nr_base, nr_max_nz_ix) = random_coherence_histogram(
        &bperp_nm,
        n_trial_wraps,
        &trial_values,
        &coh_bins,
        parms,
        &ps,
        n_ifg,
    )?;

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
        nr_base,
        nr_max_nz_ix,
        n_trial_wraps,
        trial_values,
        grid_size: options.grid_size,
        clap_window: (options.clap_win * 0.75).round().max(1.0) as usize,
        low_coh_thresh,
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
    let bp = MatData::read(path)
        .map_err(|err| stage2_err_owned(format!("unable to read bp1.mat: {err}")))?;
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
    let da = MatData::read(path)
        .map_err(|err| stage2_err_owned(format!("unable to read da1.mat: {err}")))?;
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
    out.par_chunks_mut(prepared.n_ifg).enumerate().try_for_each(
        |(row, out_row)| -> Result<(), CoreError> {
            for col in 0..prepared.n_ifg {
                let bp = if prepared.row_invariant_bperp {
                    prepared.row_bperp_nm[col]
                } else {
                    let Some(mat) = prepared.bperp_mat.as_ref() else {
                        return stage2_err(
                            "bp1.bperp_mat is required for non-invariant stage-2 baselines",
                        );
                    };
                    mat.values[row * prepared.n_ifg + col]
                };
                let phase = -((bp as f32) * (k_ps[row] as f32));
                let (sn, cs) = phase.sin_cos();
                let ramp = Complex32::new(cs, sn);
                let src = prepared.ph_nm[row * prepared.n_ifg + col];
                out_row[col] = src * ramp * (weighting[row] as f32);
            }
            Ok(())
        },
    )
}

fn fit_stage2_rows(
    prepared: &Stage2Prepared,
    ph_patch: &[Complex32],
    k_ps: &mut [f64],
    c_ps: &mut [f64],
    coh_ps: &mut [f64],
    n_opt: &mut [f64],
    ph_res: &mut [f32],
) -> Result<(), CoreError> {
    k_ps.par_iter_mut()
        .zip(c_ps.par_iter_mut())
        .zip(coh_ps.par_iter_mut())
        .zip(n_opt.par_iter_mut())
        .zip(ph_res.par_chunks_mut(prepared.n_ifg))
        .enumerate()
        .try_for_each(
            |(row, ((((k_out, c_out), coh_out), n_opt_out), ph_res_row))| -> Result<(), CoreError> {
                *k_out = f64::NAN;
                *c_out = 0.0;
                *coh_out = 0.0;
                *n_opt_out = 0.0;
                ph_res_row.fill(0.0);

                let row_start = row * prepared.n_ifg;
                let row_end = row_start + prepared.n_ifg;
                let mut psdph = vec![Complex32::new(0.0, 0.0); prepared.n_ifg];
                let mut valid = true;
                for col in 0..prepared.n_ifg {
                    let patch_value = ph_patch[row_start + col].conj();
                    let ph_value = prepared.ph_nm[row_start + col];
                    let value32 = patch_value * ph_value;
                    if value32 == Complex32::new(0.0, 0.0) {
                        valid = false;
                    }
                    psdph[col] = value32;
                }
                if !valid {
                    return Ok(());
                }

                let bperp_row = if prepared.row_invariant_bperp {
                    prepared.row_bperp_nm.as_slice()
                } else {
                    let Some(mat) = prepared.bperp_mat.as_ref() else {
                        return stage2_err(
                            "bp1.bperp_mat is required for non-invariant stage-2 baselines",
                        );
                    };
                    &mat.values[row_start..row_end]
                };
                let row_fit = topofit_row(&psdph, bperp_row, &prepared.trial_values);
                *k_out = row_fit.k;
                *c_out = row_fit.c;
                *coh_out = row_fit.coh;
                *n_opt_out = 1.0;
                for (col, value) in row_fit.residual.iter().enumerate() {
                    ph_res_row[col] = value.arg();
                }
                Ok(())
            },
        )
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
    let kernel = clap_filter_kernel_1d();
    let window_weight = clap_window_weight(n_win);
    let windows = clap_windows(n_i, n_j, n_win, n_inc, n_win_i as usize, n_win_j as usize);
    let grid_cells = n_i * n_j;
    let alpha_is_one = (alpha - 1.0).abs() <= f64::EPSILON;
    eprintln!(
        "stage2 native clap start: ifg={} windows={} window={} fft_dim={}",
        n_ifg,
        windows.len(),
        n_win,
        n_win_ex
    );
    let completed_ifg = AtomicUsize::new(0);
    let filtered_by_ifg = (0..n_ifg)
        .into_par_iter()
        .map(|ifg| {
            let ph_grid_ifg = extract_ifg_grid(ph_grid, grid_cells, n_ifg, ifg);
            let filtered = clap_filter_one_ifg(
                &ph_grid_ifg,
                n_i,
                n_j,
                n_win,
                n_win_ex,
                alpha,
                alpha_is_one,
                beta,
                low_pass,
                &kernel,
                &window_weight,
                &windows,
            );
            let done = completed_ifg.fetch_add(1, Ordering::Relaxed) + 1;
            if done == n_ifg || done % 8 == 0 {
                eprintln!("stage2 native clap progress: {done}/{n_ifg} ifg");
            }
            filtered
        })
        .collect::<Vec<_>>();

    out.par_chunks_mut(n_ifg)
        .enumerate()
        .for_each(|(cell, out_cell)| {
            for ifg in 0..n_ifg {
                out_cell[ifg] = filtered_by_ifg[ifg][cell];
            }
        });
}

fn extract_ifg_grid(
    ph_grid: &[Complex32],
    grid_cells: usize,
    n_ifg: usize,
    ifg: usize,
) -> Vec<Complex32> {
    let mut out = Vec::with_capacity(grid_cells);
    for cell in 0..grid_cells {
        out.push(ph_grid[cell * n_ifg + ifg]);
    }
    out
}

fn clap_filter_one_ifg(
    ph_grid: &[Complex32],
    n_i: usize,
    n_j: usize,
    n_win: usize,
    n_win_ex: usize,
    alpha: f64,
    alpha_is_one: bool,
    beta: f64,
    low_pass: &[f64],
    kernel: &[f64],
    window_weight: &[f64],
    windows: &[ClapWindow],
) -> Vec<Complex32> {
    let mut accum = vec![Complex64::new(0.0, 0.0); n_i * n_j];
    let mut scratch = ClapScratch::new(n_win_ex);
    let mut planner = FftPlanner::<f64>::new();
    let fft_row = planner.plan_fft_forward(n_win_ex);
    let ifft_row = planner.plan_fft_inverse(n_win_ex);
    let fft_col = planner.plan_fft_forward(n_win_ex);
    let ifft_col = planner.plan_fft_inverse(n_win_ex);
    let inv_scale = 1.0 / (n_win_ex * n_win_ex) as f64;

    for window in windows {
        scratch.ph_bit.fill(Complex64::new(0.0, 0.0));
        for local_i in 0..n_win {
            let src_i = window.i1 + local_i;
            for local_j in 0..n_win {
                let src_j = window.j1 + local_j;
                let value = ph_grid[src_i * n_j + src_j];
                scratch.ph_bit[local_i * n_win_ex + local_j] =
                    if value.re.is_nan() || value.im.is_nan() {
                        Complex64::new(0.0, 0.0)
                    } else {
                        Complex64::new(value.re as f64, value.im as f64)
                    };
            }
        }
        fft2_in_place(
            &mut scratch.ph_bit,
            n_win_ex,
            &fft_row,
            &fft_col,
            &mut scratch.fft_scratch,
        );
        for (ix, value) in scratch.ph_bit.iter().enumerate() {
            scratch.h_abs[ix] = value.norm();
        }
        fftshift_real_into(&scratch.h_abs, n_win_ex, n_win_ex, &mut scratch.h_shift);
        convolve_same_separable(
            &scratch.h_shift,
            n_win_ex,
            n_win_ex,
            kernel,
            &mut scratch.h_conv_tmp,
            &mut scratch.h_filter,
        );
        ifftshift_real_into(&scratch.h_filter, n_win_ex, n_win_ex, &mut scratch.h_shift);
        let mean_h = median_from_copy(&scratch.h_shift, &mut scratch.h_median);
        for ix in 0..scratch.h_shift.len() {
            let mut value = scratch.h_shift[ix];
            if mean_h != 0.0 {
                value /= mean_h;
            }
            value = if alpha_is_one {
                value - 1.0
            } else {
                value.powf(alpha) - 1.0
            };
            if value < 0.0 {
                value = 0.0;
            }
            scratch.ph_bit[ix] *= value * beta + low_pass[ix];
        }
        ifft2_in_place(
            &mut scratch.ph_bit,
            n_win_ex,
            &ifft_row,
            &ifft_col,
            &mut scratch.fft_scratch,
        );
        for local_i in 0..n_win {
            let dst_i = window.i1 + local_i;
            for local_j in 0..n_win {
                let dst_j = window.j1 + local_j;
                let weight = clap_window_weight_at(window_weight, n_win, window, local_i, local_j);
                if weight == 0.0 {
                    continue;
                }
                let value = scratch.ph_bit[local_i * n_win_ex + local_j] * (weight * inv_scale);
                accum[dst_i * n_j + dst_j] += value;
            }
        }
    }

    accum
        .into_iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

struct ClapScratch {
    ph_bit: Vec<Complex64>,
    fft_scratch: Vec<Complex64>,
    h_abs: Vec<f64>,
    h_shift: Vec<f64>,
    h_conv_tmp: Vec<f64>,
    h_filter: Vec<f64>,
    h_median: Vec<f64>,
}

impl ClapScratch {
    fn new(n: usize) -> Self {
        let n2 = n * n;
        Self {
            ph_bit: vec![Complex64::new(0.0, 0.0); n2],
            fft_scratch: vec![Complex64::new(0.0, 0.0); n],
            h_abs: vec![0.0; n2],
            h_shift: vec![0.0; n2],
            h_conv_tmp: vec![0.0; n2],
            h_filter: vec![0.0; n2],
            h_median: vec![0.0; n2],
        }
    }
}

#[derive(Clone, Debug)]
struct ClapWindow {
    i1: usize,
    j1: usize,
    row_shift: usize,
    col_shift: usize,
}

fn clap_windows(
    n_i: usize,
    n_j: usize,
    n_win: usize,
    n_inc: usize,
    n_win_i: usize,
    n_win_j: usize,
) -> Vec<ClapWindow> {
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
            windows.push(ClapWindow {
                i1,
                j1,
                row_shift,
                col_shift,
            });
        }
    }
    windows
}

#[inline]
fn clap_window_weight_at(
    base_weight: &[f64],
    n_win: usize,
    window: &ClapWindow,
    row: usize,
    col: usize,
) -> f64 {
    if row < window.row_shift || col < window.col_shift {
        0.0
    } else {
        let src_row = row - window.row_shift;
        let src_col = col - window.col_shift;
        base_weight[src_row * n_win + src_col]
    }
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

fn clap_filter_kernel_1d() -> Vec<f64> {
    let alpha = 2.5;
    let std = (7.0 - 1.0) / (2.0 * alpha);
    let mut g = [0.0; 7];
    for (ix, value) in g.iter_mut().enumerate() {
        let x = ix as f64 - 3.0;
        *value = (-0.5 * (x / std) * (x / std)).exp();
    }
    g.to_vec()
}

fn fft2_in_place(
    values: &mut [Complex64],
    n: usize,
    fft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    fft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    scratch: &mut [Complex64],
) {
    for row in 0..n {
        fft_row.process(&mut values[row * n..(row + 1) * n]);
    }
    for col in 0..n {
        for row in 0..n {
            scratch[row] = values[row * n + col];
        }
        fft_col.process(scratch);
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
    scratch: &mut [Complex64],
) {
    for row in 0..n {
        ifft_row.process(&mut values[row * n..(row + 1) * n]);
    }
    for col in 0..n {
        for row in 0..n {
            scratch[row] = values[row * n + col];
        }
        ifft_col.process(scratch);
        for row in 0..n {
            values[row * n + col] = scratch[row];
        }
    }
}

fn fftshift_real_into(values: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    shift_real_into(values, rows, cols, rows / 2, cols / 2, out)
}

fn ifftshift_real_into(values: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    shift_real_into(values, rows, cols, rows.div_ceil(2), cols.div_ceil(2), out)
}

fn shift_real_into(
    values: &[f64],
    rows: usize,
    cols: usize,
    row_shift: usize,
    col_shift: usize,
    out: &mut [f64],
) {
    for row in 0..rows {
        for col in 0..cols {
            let src_row = (row + row_shift) % rows;
            let src_col = (col + col_shift) % cols;
            out[row * cols + col] = values[src_row * cols + src_col];
        }
    }
}

fn convolve_same_separable(
    values: &[f64],
    rows: usize,
    cols: usize,
    kernel: &[f64],
    tmp: &mut [f64],
    out: &mut [f64],
) {
    let k = kernel.len();
    let radius = k / 2;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for kc in 0..k {
                let Some(src_col) = col
                    .checked_add(kc)
                    .and_then(|value| value.checked_sub(radius))
                else {
                    continue;
                };
                if src_col < cols {
                    sum += values[row * cols + src_col] * kernel[kc];
                }
            }
            tmp[row * cols + col] = sum;
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for kr in 0..k {
                let Some(src_row) = row
                    .checked_add(kr)
                    .and_then(|value| value.checked_sub(radius))
                else {
                    continue;
                };
                if src_row < rows {
                    sum += tmp[src_row * cols + col] * kernel[kr];
                }
            }
            out[row * cols + col] = sum;
        }
    }
}

fn median_from_copy(values: &[f64], scratch: &mut [f64]) -> f64 {
    scratch.copy_from_slice(values);
    median(scratch)
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
    out.par_chunks_mut(prepared.n_ifg)
        .enumerate()
        .for_each(|(row, out_row)| {
            let grid_ix = prepared.grid_lin[row];
            for col in 0..prepared.n_ifg {
                out_row[col] = ph_filt[grid_ix * prepared.n_ifg + col];
            }
        });
    normalize_complex_unit_magnitude(out);
}

fn normalize_complex_unit_magnitude(values: &mut [Complex32]) {
    values.par_iter_mut().for_each(|value| {
        let mag = value.norm();
        if mag != 0.0 {
            *value /= mag;
        }
    });
}

fn topofit_row(cpx: &[Complex32], bperp: &[f64], trial_mult: &[f64]) -> TopofitRow {
    let valid = cpx
        .iter()
        .zip(bperp.iter())
        .enumerate()
        .filter_map(|(ix, (&value, &bp))| {
            (value != Complex32::new(0.0, 0.0)).then_some((ix, value, bp as f32))
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return TopofitRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx.len()],
        };
    }
    let mut denom: f32 = 0.0;
    for (_, value, _) in &valid {
        denom += value.norm();
    }
    if denom == 0.0 {
        denom = 1.0;
    }
    let min_bp = valid
        .iter()
        .map(|(_, _, bp)| *bp)
        .fold(f32::INFINITY, f32::min);
    let max_bp = valid
        .iter()
        .map(|(_, _, bp)| *bp)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut bperp_range = max_bp - min_bp;
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }
    let mut coh_trial = vec![0.0_f32; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut sum = Complex32::new(0.0, 0.0);
        for (_, value, bp) in &valid {
            let phase = (bp / bperp_range) * (std::f32::consts::PI / 4.0) * (trial_value as f32);
            let (sn, cs) = phase.sin_cos();
            sum += *value * Complex32::new(cs, -sn);
        }
        coh_trial[trial_ix] = sum.norm() / denom;
    }
    let trial_ix = argmax_first_f32(&coh_trial);
    let coarse_k0 =
        ((std::f32::consts::PI / 4.0) / bperp_range * (trial_mult[trial_ix] as f32)) as f64;
    refine_candidate(&valid, cpx.len(), coarse_k0)
}

fn refine_candidate(valid: &[(usize, Complex32, f32)], n_col: usize, coarse_k0: f64) -> TopofitRow {
    let mut k0 = coarse_k0 as f32;
    let mut offset = Complex32::new(0.0, 0.0);
    for (_, value, bp) in valid {
        let phase = k0 * bp;
        let (sn, cs) = phase.sin_cos();
        offset += *value * Complex32::new(cs, -sn);
    }
    let offset_conj = offset.conj();
    let mut mopt_num = 0.0;
    let mut den_lin = 0.0;
    for (_, value, bp) in valid {
        let weight = value.norm() as f64;
        let wb = weight * (*bp as f64);
        den_lin += wb * wb;
        let phase = k0 * bp;
        let (sn, cs) = phase.sin_cos();
        let res = *value * Complex32::new(cs, -sn);
        mopt_num += wb * (weight * (res * offset_conj).arg() as f64);
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }
    k0 = (k0 as f64 + mopt_num / den_lin) as f32;
    let mut mean_phase_residual = Complex32::new(0.0, 0.0);
    let mut denom = 0.0_f32;
    let mut residual = vec![Complex32::new(0.0, 0.0); n_col];
    for (col, value, bp) in valid {
        let phase = k0 * bp;
        let (sn, cs) = phase.sin_cos();
        let res = *value * Complex32::new(cs, -sn);
        mean_phase_residual += res;
        denom += res.norm();
        residual[*col] = res;
    }
    if denom == 0.0 {
        denom = 1.0;
    }
    TopofitRow {
        k: k0 as f64,
        c: mean_phase_residual.arg() as f64,
        coh: (mean_phase_residual.norm() / denom) as f64,
        residual,
    }
}

fn trial_values(n_trial_wraps: f64) -> Vec<f64> {
    let trial_n = (8.0 * n_trial_wraps).ceil() as i64;
    (-trial_n..=trial_n).map(|value| value as f64).collect()
}

fn random_coherence_histogram(
    bperp_nm: &[f64],
    _n_trial_wraps: f64,
    trial_values: &[f64],
    coh_bins: &[f64],
    parms: &Stage2Parms,
    ps: &MatData,
    n_ifg: usize,
) -> Result<(Vec<f64>, f64), CoreError> {
    let mut rng = MatlabV5UniformRng::new(STAGE2_RANDOM_SEED);
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    let nr = if small_baseline {
        let (image_a, image_b, n_image) = small_baseline_random_indices(ps, n_ifg)?;
        let mut image_phase = rng.uniform_flat(STAGE2_RANDOM_COUNT * n_image);
        for value in &mut image_phase {
            *value *= 2.0 * std::f64::consts::PI;
        }
        histogram_random_rows(
            coh_bins,
            STAGE2_RANDOM_COUNT,
            |row, ifg| {
                image_phase[row + image_b[ifg] * STAGE2_RANDOM_COUNT]
                    - image_phase[row + image_a[ifg] * STAGE2_RANDOM_COUNT]
            },
            bperp_nm,
            trial_values,
        )
    } else {
        let mut ifg_phase = rng.uniform_flat(STAGE2_RANDOM_COUNT * n_ifg);
        for value in &mut ifg_phase {
            *value *= 2.0 * std::f64::consts::PI;
        }
        histogram_random_rows(
            coh_bins,
            STAGE2_RANDOM_COUNT,
            |row, ifg| ifg_phase[row + ifg * STAGE2_RANDOM_COUNT],
            bperp_nm,
            trial_values,
        )
    };
    let nr_max_nz_ix = nr
        .iter()
        .rposition(|&value| value > 0.0)
        .map(|ix| (ix + 1) as f64)
        .unwrap_or(1.0);
    Ok((nr, nr_max_nz_ix))
}

fn small_baseline_random_indices(
    ps: &MatData,
    n_ifg: usize,
) -> Result<(Vec<usize>, Vec<usize>, usize), CoreError> {
    let source = ps
        .get_f64_matrix("ifgday_ix")
        .map_err(|err| CoreError::NativeStage {
            stage: 2,
            message: format!("ps1.ifgday_ix is missing or invalid: {err}"),
        })?;
    let matrix = if source.rows == n_ifg && source.cols == 2 {
        source
    } else if source.rows == 2 && source.cols == n_ifg {
        let mut values = Vec::with_capacity(source.values.len());
        for row in 0..source.cols {
            for col in 0..source.rows {
                values.push(source.values[col * source.cols + row]);
            }
        }
        Matrix {
            name: source.name,
            rows: source.cols,
            cols: source.rows,
            values,
        }
    } else {
        return stage2_err(format!(
            "ps1.ifgday_ix has incompatible shape {}x{} for n_ifg={n_ifg}",
            source.rows, source.cols
        ));
    };
    let mut image_a = Vec::with_capacity(n_ifg);
    let mut image_b = Vec::with_capacity(n_ifg);
    let mut n_image = 0usize;
    for row in 0..n_ifg {
        let a = matrix.values[row * 2].round() as isize;
        let b = matrix.values[row * 2 + 1].round() as isize;
        if a <= 0 || b <= 0 {
            return stage2_err("ps1.ifgday_ix must contain positive one-based image ids");
        }
        n_image = n_image.max(a as usize).max(b as usize);
        image_a.push(a as usize - 1);
        image_b.push(b as usize - 1);
    }
    Ok((image_a, image_b, n_image))
}

fn histogram_random_rows<F>(
    coh_bins: &[f64],
    n_row: usize,
    phase_at: F,
    bperp: &[f64],
    trial_values: &[f64],
) -> Vec<f64>
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    (0..n_row)
        .into_par_iter()
        .fold(
            || vec![0.0; coh_bins.len()],
            |mut hist, row| {
                let coh = topofit_phase_row_coh(bperp.len(), bperp, trial_values, |ifg| {
                    phase_at(row, ifg)
                });
                add_hist_value(&mut hist, coh_bins, coh);
                hist
            },
        )
        .reduce(
            || vec![0.0; coh_bins.len()],
            |mut left, right| {
                for (left_value, right_value) in left.iter_mut().zip(right) {
                    *left_value += right_value;
                }
                left
            },
        )
}

fn topofit_phase_row_coh<F>(
    n_col: usize,
    bperp: &[f64],
    trial_values: &[f64],
    mut phase_at: F,
) -> f64
where
    F: FnMut(usize) -> f64,
{
    if n_col == 0 {
        return f64::NAN;
    }
    let min_bp = bperp.iter().copied().fold(f64::INFINITY, f64::min);
    let max_bp = bperp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut bperp_range = max_bp - min_bp;
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }
    let denom = n_col as f64;
    let mut coh_trial = vec![0.0; trial_values.len()];
    for (trial_ix, &trial_value) in trial_values.iter().enumerate() {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for (ifg, &bp) in bperp.iter().enumerate().take(n_col) {
            let phase =
                phase_at(ifg) - (bp / bperp_range) * (std::f64::consts::PI / 4.0) * trial_value;
            let (sn, cs) = phase.sin_cos();
            sum_re += cs;
            sum_im += sn;
        }
        coh_trial[trial_ix] = sum_re.hypot(sum_im) / denom;
    }
    let trial_ix = argmax_first(&coh_trial);
    let coarse_k0 = (std::f64::consts::PI / 4.0) / bperp_range * trial_values[trial_ix];
    refine_phase_candidate_coh(n_col, bperp, coarse_k0, |ifg| phase_at(ifg))
}

fn refine_phase_candidate_coh<F>(
    n_col: usize,
    bperp: &[f64],
    coarse_k0: f64,
    mut phase_at: F,
) -> f64
where
    F: FnMut(usize) -> f64,
{
    let mut offset = Complex64::new(0.0, 0.0);
    for (ifg, &bp) in bperp.iter().enumerate().take(n_col) {
        let phase = phase_at(ifg) - coarse_k0 * bp;
        let (sn, cs) = phase.sin_cos();
        offset += Complex64::new(cs, sn);
    }
    let offset_conj = offset.conj();
    let mut mopt_num = 0.0;
    let mut den_lin = 0.0;
    for (ifg, &bp) in bperp.iter().enumerate().take(n_col) {
        den_lin += bp * bp;
        let phase = phase_at(ifg) - coarse_k0 * bp;
        let (sn, cs) = phase.sin_cos();
        mopt_num += bp * (Complex64::new(cs, sn) * offset_conj).arg();
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }
    let k = coarse_k0 + mopt_num / den_lin;
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for (ifg, &bp) in bperp.iter().enumerate().take(n_col) {
        let phase = phase_at(ifg) - k * bp;
        let (sn, cs) = phase.sin_cos();
        sum_re += cs;
        sum_im += sn;
    }
    sum_re.hypot(sum_im) / n_col.max(1) as f64
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

fn argmax_first_f32(values: &[f32]) -> usize {
    let mut best_ix = 0;
    let mut best_value = values.first().copied().unwrap_or(f32::NEG_INFINITY);
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
        return vec![values.iter().filter(|value| value.is_finite()).count() as f64];
    }
    let equal_spacing = histogram_centers_equal_spacing(centers);
    let mids = (!equal_spacing).then(|| {
        centers
            .windows(2)
            .map(|pair| (pair[0] + pair[1]) / 2.0)
            .collect::<Vec<_>>()
    });
    let mut out = vec![0.0; centers.len()];
    for &value in values {
        if let Some(ix) = histogram_center_index(value, centers, equal_spacing, mids.as_deref()) {
            out[ix] += 1.0;
        }
    }
    out
}

fn add_hist_value(out: &mut [f64], centers: &[f64], value: f64) {
    if centers.is_empty() {
        return;
    }
    if centers.len() == 1 {
        if value.is_finite() {
            out[0] += 1.0;
        }
        return;
    }
    if let Some(ix) = histogram_center_index(
        value,
        centers,
        histogram_centers_equal_spacing(centers),
        None,
    ) {
        out[ix] += 1.0;
    }
}

fn histogram_center_index(
    value: f64,
    centers: &[f64],
    equal_spacing: bool,
    mids: Option<&[f64]>,
) -> Option<usize> {
    if !value.is_finite() || centers.is_empty() {
        return None;
    }
    if centers.len() == 1 {
        return Some(0);
    }
    if equal_spacing {
        let d = if centers.len() < 3 {
            1.0
        } else {
            (centers[centers.len() - 1] - centers[0]) / (centers.len() - 1) as f64
        };
        if d == 0.0 {
            return Some(0);
        }
        let cutoff0 = (centers[0] + centers[1]) / 2.0;
        let assignment = ((value - cutoff0) / d)
            .ceil()
            .clamp(0.0, (centers.len() - 1) as f64);
        return Some(assignment as usize);
    }
    let owned_mids;
    let mids = if let Some(mids) = mids {
        mids
    } else {
        owned_mids = centers
            .windows(2)
            .map(|pair| (pair[0] + pair[1]) / 2.0)
            .collect::<Vec<_>>();
        &owned_mids
    };
    Some(
        mids.partition_point(|&mid| mid < value)
            .min(centers.len() - 1),
    )
}

fn histogram_centers_equal_spacing(centers: &[f64]) -> bool {
    if centers.len() < 2 {
        return true;
    }
    let diffs = centers
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let reference = diffs[0];
    let max_center = centers
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let tol = f64::EPSILON * max_center.max(1.0);
    diffs.iter().all(|&diff| (diff - reference).abs() <= tol)
}

fn psquare_weighting(
    nr: &[f64],
    na: &[f64],
    low_coh_thresh: usize,
    nr_max_nz_ix: f64,
    coh_ps: &[f64],
) -> Vec<f64> {
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
    let win = gausswin(7, 2.5);
    let win_sum = win.iter().sum::<f64>();
    let mut padded = vec![1.0; 7];
    padded.extend_from_slice(&prand);
    let filtered = lfilter(&win, &padded);
    for (dst, src) in prand.iter_mut().zip(filtered.iter().skip(7)) {
        *dst = *src / win_sum;
    }
    let mut interp_input = Vec::with_capacity(prand.len() + 1);
    interp_input.push(1.0);
    interp_input.extend_from_slice(&prand);
    let mut prand_hi = matlab_interp(&interp_input, 10);
    let keep = prand_hi.len().saturating_sub(9);
    prand_hi.truncate(keep);
    coh_ps
        .iter()
        .map(|&coh| {
            let ix = round_half_away_from_zero(coh * 1000.0)
                .clamp(0.0, prand_hi.len().saturating_sub(1) as f64) as usize;
            let p = prand_hi[ix];
            (1.0 - p) * (1.0 - p)
        })
        .collect()
}

fn gausswin(n: usize, alpha: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || alpha <= 0.0 {
        return vec![1.0; n];
    }
    let std = (n as f64 - 1.0) / (2.0 * alpha);
    (0..n)
        .map(|ix| {
            let x = ix as f64 - (n as f64 - 1.0) / 2.0;
            (-0.5 * (x / std) * (x / std)).exp()
        })
        .collect()
}

fn lfilter(kernel: &[f64], values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for ix in 0..values.len() {
        let mut sum = 0.0;
        for (k_ix, &coef) in kernel.iter().enumerate() {
            if ix >= k_ix {
                sum += coef * values[ix - k_ix];
            }
        }
        out[ix] = sum;
    }
    out
}

fn matlab_interp(values: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 || values.is_empty() {
        return values.to_vec();
    }
    if factor == 10 {
        return interp_with_taps(values, factor, &STAGE2_INTERP10_TAPS, 39);
    }
    let n = 4usize;
    let mut expanded = vec![0.0; values.len() * factor + factor * n];
    for (ix, &value) in values.iter().enumerate() {
        expanded[ix * factor] = value;
    }
    let taps = firwin_hamming_lowpass(2 * factor * n + 1, 1.0 / factor as f64);
    let filtered = lfilter(&taps, &expanded);
    filtered
        .into_iter()
        .skip(factor * n)
        .map(|value| value * factor as f64)
        .collect()
}

fn interp_with_taps(values: &[f64], factor: usize, taps: &[f64], delay: usize) -> Vec<f64> {
    let mut expanded = vec![0.0; values.len() * factor + delay];
    for (ix, &value) in values.iter().enumerate() {
        expanded[ix * factor] = value;
    }
    let filtered = lfilter(taps, &expanded);
    filtered
        .into_iter()
        .skip(delay)
        .take(values.len() * factor)
        .map(|value| value * factor as f64)
        .collect()
}

fn firwin_hamming_lowpass(numtaps: usize, cutoff: f64) -> Vec<f64> {
    let alpha = (numtaps as f64 - 1.0) / 2.0;
    let mut taps = Vec::with_capacity(numtaps);
    for ix in 0..numtaps {
        let m = ix as f64 - alpha;
        let hamming =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * ix as f64 / (numtaps as f64 - 1.0)).cos();
        taps.push(cutoff * sinc(cutoff * m) * hamming);
    }
    let sum = taps.iter().sum::<f64>();
    if sum != 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

fn sinc(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else {
        let arg = std::f64::consts::PI * value;
        arg.sin() / arg
    }
}

fn round_half_away_from_zero(value: f64) -> f64 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    }
}

struct MatlabV5UniformRng {
    index: usize,
    borrow: f64,
    j: u32,
    state: [f64; 32],
}

impl MatlabV5UniformRng {
    const ULP: f64 = 1.110_223_024_625_156_5e-16;
    const MASK52: u64 = (1_u64 << 52) - 1;

    fn new(seed: u32) -> Self {
        let j = if seed == 0 { 1_u32 << 31 } else { seed };
        let state = Self::randsetup(j);
        Self {
            index: 0,
            borrow: 0.0,
            j,
            state,
        }
    }

    fn uniform_flat(&mut self, size: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            let mut value = self.state[(self.index + 20) & 31]
                - self.state[(self.index + 5) & 31]
                - self.borrow;
            if value < 0.0 {
                value += 1.0;
                self.borrow = Self::ULP;
            } else {
                self.borrow = 0.0;
            }
            self.state[self.index] = value;
            self.index = (self.index + 1) & 31;
            out.push(self.randbits(value));
        }
        out
    }

    fn randsetup(seed: u32) -> [f64; 32] {
        let mut state = [0.0; 32];
        let mut j = seed;
        for value in &mut state {
            let mut x = 0_u64;
            for _ in 0..53 {
                j = Self::randint32(j);
                x = (x << 1) | (((j >> 19) & 1) as u64);
            }
            *value = (x as f64) * Self::ULP;
        }
        state
    }

    fn randint32(mut value: u32) -> u32 {
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        value
    }

    fn randbits(&mut self, value: f64) -> f64 {
        let jlo = self.j;
        let jhi = Self::randint32(jlo);
        self.j = jhi;
        let mask = (((jhi as u64) << 32) & Self::MASK52) ^ jlo as u64;
        let (mantissa, exp) = frexp_mantissa53(value);
        ((mantissa ^ mask) as f64) * 2.0_f64.powi(exp - 53)
    }
}

fn frexp_mantissa53(value: f64) -> (u64, i32) {
    if value == 0.0 {
        return (0, 0);
    }
    let bits = value.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & ((1_u64 << 52) - 1);
    if exp_bits == 0 {
        let leading = 63 - mantissa_bits.leading_zeros() as i32;
        let exp = leading - 1073;
        let mantissa = mantissa_bits << (52 - leading);
        (mantissa, exp)
    } else {
        let exp = exp_bits - 1022;
        ((1_u64 << 52) | mantissa_bits, exp)
    }
}

fn snr_weighting(prepared: &Stage2Prepared, ph_res: &[f32]) -> Vec<f64> {
    (0..prepared.n_ps)
        .into_par_iter()
        .map(|row| {
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
                g / sigma_n
            } else {
                0.0
            }
        })
        .collect()
}

fn write_pm1(
    patch_dir: &Path,
    filename: &str,
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
    let mut mat = MatFile::new(patch_dir.join(filename));
    mat.add_f64_col_vector("K_ps", k_ps.to_vec())?;
    mat.add_f64_col_vector("C_ps", c_ps.to_vec())?;
    mat.add_f64_col_vector("coh_ps", coh_ps.to_vec())?;
    mat.add_f64_col_vector("N_opt", n_opt.to_vec())?;
    mat.add_f32_matrix("ph_res", prepared.n_ps, prepared.n_ifg, ph_res.to_vec())?;
    mat.add_complex_f32_matrix(
        "ph_patch",
        prepared.n_ps,
        prepared.n_ifg,
        complex32_pairs(ph_patch),
    )?;
    mat.add_f64_scalar("step_number", 1.0)?;
    mat.add_complex_f32_array3(
        "ph_grid",
        prepared.n_i,
        prepared.n_j,
        prepared.n_ifg,
        complex32_pairs(ph_grid),
    )?;
    mat.add_f32_scalar("n_trial_wraps", prepared.n_trial_wraps as f32)?;
    mat.add_f32_matrix(
        "grid_ij",
        prepared.grid_ij.rows,
        prepared.grid_ij.cols,
        prepared.grid_ij.values.clone(),
    )?;
    mat.add_f64_scalar("grid_size", prepared.grid_size)?;
    mat.add_f64_matrix(
        "low_pass",
        prepared.low_pass.rows,
        prepared.low_pass.cols,
        prepared.low_pass.values.clone(),
    )?;
    mat.add_f64_scalar("i_loop", i_loop as f64)?;
    mat.add_complex_f32_matrix(
        "ph_weight",
        prepared.n_ps,
        prepared.n_ifg,
        complex32_pairs(ph_weight),
    )?;
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
        let i = ((xy.values[row * xy.cols + 2] - min_y + eps) / scale)
            .ceil()
            .max(1.0) as i32;
        let j = ((xy.values[row * xy.cols + 1] - min_x + eps) / scale)
            .ceil()
            .max(1.0) as i32;
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

#[derive(Clone, Debug, Default)]
struct Stage2ParmSource {
    path: Option<PathBuf>,
    mat: Option<MatData>,
}

impl Stage2ParmSource {
    fn from_patch(patch_dir: &Path) -> Self {
        let path = resolve_file_optional(patch_dir, "parms.mat");
        let mat = path.as_ref().and_then(|path| MatData::read(path).ok());
        Self { path, mat }
    }

    fn scalar(&self, name: &str, default: f64) -> f64 {
        self.mat
            .as_ref()
            .and_then(|mat| optional_vector_f64(mat, name))
            .and_then(|values| values.into_iter().next())
            .or_else(|| {
                self.path
                    .as_ref()
                    .and_then(|path| read_hdf5_scalar_f64(path, name).ok())
            })
            .unwrap_or(default)
    }

    fn text(&self, name: &str, default: &str) -> String {
        let text = self
            .mat
            .as_ref()
            .map(|mat| text_from_mat(mat, name, ""))
            .filter(|value| !value.is_empty())
            .or_else(|| {
                self.path
                    .as_ref()
                    .and_then(|path| read_hdf5_text(path, name).ok())
            })
            .unwrap_or_else(|| default.to_string());
        if text.is_empty() {
            default.to_string()
        } else {
            text
        }
    }
}

fn load_stage2_options(source: &Stage2ParmSource) -> Stage2Options {
    let mut options = Stage2Options::default();
    options.grid_size = source.scalar("filter_grid_size", options.grid_size);
    options.clap_win = source.scalar("clap_win", options.clap_win);
    options.clap_low_pass_wavelength =
        source.scalar("clap_low_pass_wavelength", options.clap_low_pass_wavelength);
    options.clap_alpha = source.scalar("clap_alpha", options.clap_alpha);
    options.clap_beta = source.scalar("clap_beta", options.clap_beta);
    options.max_topo_err = source.scalar("max_topo_err", options.max_topo_err);
    options.lambda_m = source.scalar("lambda", options.lambda_m);
    options
}

fn load_stage2_parms(source: &Stage2ParmSource) -> Stage2Parms {
    let mut parms = Stage2Parms::default();
    parms.small_baseline_flag = source.text("small_baseline_flag", &parms.small_baseline_flag);
    parms.filter_weighting = source.text("filter_weighting", &parms.filter_weighting);
    parms.gamma_change_convergence =
        source.scalar("gamma_change_convergence", parms.gamma_change_convergence);
    parms.gamma_max_iterations = source
        .scalar("gamma_max_iterations", parms.gamma_max_iterations as f64)
        .round() as usize;
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
    optional_vector_f64(ps, "mean_incidence")
        .and_then(|values| values.into_iter().next())
        .map(|value| value + 0.052)
        .unwrap_or(DEFAULT_MEAN_INCIDENCE)
}

fn read_hdf5_scalar_f64(path: &Path, variable: &str) -> Result<f64, String> {
    match read_hdf5_scalar_f64_direct(path, variable) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock(path, offset, |temp_path| {
                read_hdf5_scalar_f64_direct(temp_path, variable)
            })
            .map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_scalar_f64_direct(path: &Path, variable: &str) -> Result<f64, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset.read_raw::<f64>().map_err(|err| err.to_string())?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| format!("{variable} has no scalar values"))
}

fn read_hdf5_text(path: &Path, variable: &str) -> Result<String, String> {
    match read_hdf5_text_direct(path, variable) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock(path, offset, |temp_path| {
                read_hdf5_text_direct(temp_path, variable)
            })
            .map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_text_direct(path: &Path, variable: &str) -> Result<String, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset.read_raw::<u16>().map_err(|err| err.to_string())?;
    let text = values
        .into_iter()
        .filter_map(|value| char::from_u32(value as u32))
        .filter(|&ch| ch != '\0')
        .collect::<String>()
        .trim()
        .to_string();
    if text.is_empty() {
        Err(format!("{variable} has empty text"))
    } else {
        Ok(text)
    }
}

fn read_hdf5_from_userblock<T, F>(path: &Path, offset: usize, read_direct: F) -> Result<T, String>
where
    F: FnOnce(&Path) -> Result<T, String>,
{
    let temp_path = std::env::temp_dir().join(format!(
        "pystamps-stage2-hdf5-{}-{}.h5",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| err.to_string())?
            .as_nanos()
    ));
    let mut input = fs::File::open(path).map_err(|err| err.to_string())?;
    input
        .seek(SeekFrom::Start(offset as u64))
        .map_err(|err| err.to_string())?;
    {
        let mut output = fs::File::create(&temp_path).map_err(|err| err.to_string())?;
        std::io::copy(&mut input, &mut output).map_err(|err| err.to_string())?;
        output.flush().map_err(|err| err.to_string())?;
    }
    let result = read_direct(&temp_path);
    let _ = fs::remove_file(&temp_path);
    result
}

fn find_hdf5_signature_offset(path: &Path) -> Result<usize, String> {
    let mut file = fs::File::open(path).map_err(|err| err.to_string())?;
    let mut buffer = vec![0_u8; HDF5_SIGNATURE_SCAN_BYTES];
    let read_len = file.read(&mut buffer).map_err(|err| err.to_string())?;
    buffer.truncate(read_len);
    buffer
        .windows(HDF5_SIGNATURE.len())
        .position(|window| window == HDF5_SIGNATURE)
        .ok_or_else(|| "HDF5 signature not found".to_string())
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

fn ps_matrix_f64(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<Matrix<f64>, CoreError> {
    let source = mat
        .get_f64_matrix(name)
        .map_err(|err| CoreError::NativeStage {
            stage: 2,
            message: format!("{label} is missing or invalid: {err}"),
        })?;
    orient_matrix_f64(source, n_ps, label)
}

fn ps_dim_f32(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    n_dim: usize,
    label: &str,
) -> Result<Matrix<f32>, CoreError> {
    let source = mat
        .get_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
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
    let source = mat
        .get_complex_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
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

fn orient_matrix_f64(
    source: Matrix<f64>,
    n_ps: usize,
    label: &str,
) -> Result<Matrix<f64>, CoreError> {
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
    if text.is_empty() {
        default.to_string()
    } else {
        text
    }
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
        patch_dir
            .parent()
            .map(|parent| parent.join(filename))
            .unwrap_or_default(),
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
    fn matlab_v5_uniform_rng_matches_python_reference() {
        let mut rng = MatlabV5UniformRng::new(2005);
        let values = rng.uniform_flat(1_000_001);
        let expected = [
            (0, 0.092958990583191195),
            (1, 0.37373775840752277),
            (2, 0.44877057937117765),
            (9, 0.574897127600281),
            (10, 0.067322284274237226),
            (31, 0.43152622807181651),
            (32, 0.88306283014597708),
            (33, 0.074287163034898504),
            (1000, 0.35920165460545467),
            (99_999, 0.020806164093088213),
            (1_000_000, 0.4917673475125352),
        ];
        for (ix, expected) in expected {
            let observed = values[ix];
            assert!((observed - expected).abs() <= 1.0e-15);
        }
    }

    #[test]
    fn psquare_weighting_matches_python_smoothing_reference() {
        let nr = (0..100).map(|ix| ix as f64).collect::<Vec<_>>();
        let na = (100..200).map(|ix| ix as f64).collect::<Vec<_>>();
        let coh = vec![0.0, 0.0049, 0.005, 0.3144, 0.3145, 0.9999, 1.1];

        let weighting = psquare_weighting(&nr, &na, 31, 45.0, &coh);

        let expected = [
            1.8308507589276466e-06,
            0.011860147976383777,
            0.011860147976383777,
            4.288734335628797e-06,
            8.21030522409814e-06,
            1.0,
            1.0,
        ];
        for (observed, expected) in weighting.iter().zip(expected) {
            assert!(
                (observed - expected).abs() <= 1.0e-12,
                "observed={observed} expected={expected}"
            );
        }
    }

    #[test]
    fn histogram_with_centers_matches_equal_spacing_octave_rule() {
        let centers = vec![0.005, 0.015, 0.025, 0.035];
        let values = vec![0.01, 0.02, 0.03, f64::NAN];

        let observed = hist_with_centers(&values, &centers);

        assert_eq!(observed, vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn random_coherence_histogram_matches_python_reference() {
        let root = temp_root("stage2-random-hist");
        let ps_path = root.join("ps1.mat");
        let mut ps_mat = MatFile::new(&ps_path);
        ps_mat.add_f64_scalar("n_ps", 1.0).unwrap();
        ps_mat.write().unwrap();
        let ps = MatData::read(&ps_path).unwrap();
        let bperp = vec![-20.0, -5.0, 0.0, 13.0, 27.0];
        let n_trial_wraps = 0.5;
        let coh_bins = (0..COH_BIN_COUNT)
            .map(|ix| COH_BIN_START + COH_BIN_STEP * ix as f64)
            .collect::<Vec<_>>();

        let (nr, nr_max_nz_ix) = random_coherence_histogram(
            &bperp,
            n_trial_wraps,
            &trial_values(n_trial_wraps),
            &coh_bins,
            &Stage2Parms::default(),
            &ps,
            bperp.len(),
        )
        .unwrap();

        let expected = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 4.0, 5.0,
            0.0, 2.0, 7.0, 5.0, 14.0, 14.0, 15.0, 10.0, 29.0, 21.0, 26.0, 29.0, 44.0, 42.0, 57.0,
            54.0, 63.0, 71.0, 68.0, 71.0, 87.0, 87.0, 76.0, 98.0, 111.0, 100.0, 137.0, 123.0,
            139.0, 154.0, 148.0, 149.0, 162.0, 195.0, 212.0, 202.0, 200.0, 203.0, 228.0, 209.0,
            222.0, 227.0, 223.0, 239.0, 201.0, 199.0, 231.0, 203.0, 192.0, 184.0, 187.0, 188.0,
            185.0, 180.0, 183.0, 186.0, 163.0, 152.0, 147.0, 155.0, 158.0, 171.0, 169.0, 155.0,
            139.0, 140.0, 144.0, 126.0, 125.0, 125.0, 115.0, 110.0, 116.0, 104.0, 95.0, 75.0, 74.0,
            85.0, 73.0, 56.0, 61.0, 37.0, 26.0,
        ];
        assert_eq!(nr, expected);
        assert_eq!(nr_max_nz_ix, 100.0);
        fs::remove_dir_all(root).unwrap();
    }

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
        let err = run_stage2_native(root.join("PATCH_1"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("stage 2 native implementation error"));
        assert!(err.contains("bp1.bperp_mat has incompatible shape"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_stage2_thread_pool_writes_pm1() {
        let root = temp_root("stage2-threaded");
        create_stage2_fixture(&root, false);

        let details = run_stage2_native_with_threads(root.join("PATCH_1"), 2).unwrap();

        assert!(details.contains("using 2 native threads"));
        assert!(root.join("PATCH_1/pm1.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage2_reads_hdf5_parms_when_v5_reader_cannot() {
        let root = temp_root("stage2-hdf5-parms");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        let raw_hdf5 = patch.join("parms-raw.h5");
        let h5 = rust_hdf5::H5File::create(&raw_hdf5).unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("lambda")
            .unwrap()
            .write_raw(&[0.05546576_f64])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("gamma_max_iterations")
            .unwrap()
            .write_raw(&[7.0_f64])
            .unwrap();
        h5.new_dataset::<u16>()
            .shape([1, 1])
            .create("small_baseline_flag")
            .unwrap()
            .write_raw(&['n' as u16])
            .unwrap();
        h5.new_dataset::<u16>()
            .shape([8, 1])
            .create("filter_weighting")
            .unwrap()
            .write_raw(&"P-square".chars().map(|ch| ch as u16).collect::<Vec<_>>())
            .unwrap();
        h5.close().unwrap();

        let mut matlab_hdf5 = fs::File::create(patch.join("parms.mat")).unwrap();
        matlab_hdf5.write_all(&vec![b' '; 512]).unwrap();
        matlab_hdf5
            .write_all(&fs::read(&raw_hdf5).unwrap())
            .unwrap();
        fs::remove_file(raw_hdf5).unwrap();

        let source = Stage2ParmSource::from_patch(&patch);
        let options = load_stage2_options(&source);
        let parms = load_stage2_parms(&source);

        assert_eq!(options.lambda_m, 0.05546576);
        assert_eq!(parms.gamma_max_iterations, 7);
        assert_eq!(parms.small_baseline_flag, "n");
        assert_eq!(parms.filter_weighting, "P-square");
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
        ps.add_f64_row_vector("bperp", vec![0.0, 15.0, 30.0])
            .unwrap();
        ps.add_f64_scalar("mean_incidence", DEFAULT_MEAN_INCIDENCE)
            .unwrap();
        ps.add_f32_matrix(
            "xy",
            n_ps,
            3,
            vec![1.0, 0.0, 0.0, 2.0, 5.0, 5.0, 3.0, 10.0, 10.0],
        )
        .unwrap();
        ps.write().unwrap();

        let mut ph = MatFile::new(patch.join("ph1.mat"));
        ph.add_complex_f32_matrix(
            "ph",
            n_ps,
            n_ifg,
            vec![
                (1.0, 0.0),
                (0.8, 0.2),
                (0.6, 0.4),
                (1.0, 0.0),
                (0.7, 0.3),
                (0.5, 0.5),
                (1.0, 0.0),
                (0.9, 0.1),
                (0.4, 0.6),
            ],
        )
        .unwrap();
        ph.write().unwrap();

        let mut bp = MatFile::new(patch.join("bp1.mat"));
        if bad_bp_shape {
            bp.add_f64_matrix("bperp_mat", n_ps, 1, vec![10.0, 20.0, 30.0])
                .unwrap();
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

        let mut la = MatFile::new(patch.join("la1.mat"));
        la.add_f64_row_vector("la", vec![DEFAULT_MEAN_INCIDENCE; n_ps])
            .unwrap();
        la.write().unwrap();

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
