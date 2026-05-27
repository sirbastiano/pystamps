use crate::CoreError;
use pystamps_mat::{
    read_hdf5_f32_dataset_raw, read_hdf5_f32_datasets_raw, ComplexMatrixF32, MatData, MatFile,
    Matrix,
};
use rayon::prelude::*;
use rust_hdf5::H5File;
use std::collections::BTreeSet;
use std::env;
use std::path::Path;
use std::time::Instant;

const PYSTAMPS_ROW_MAJOR_ATTR: &str = "PY_STAMPS_row_major";

#[derive(Clone, Debug)]
struct Stage8Parms {
    small_baseline_flag: String,
    unwrap_method: String,
    unwrap_la_error_flag: String,
    unwrap_spatial_cost_func_flag: String,
    drop_ifg_index: Vec<i64>,
    ref_lon: Vec<f64>,
    ref_lat: Vec<f64>,
    ref_radius: f64,
    max_topo_err: f64,
    lambda_m: f64,
    unwrap_time_win: f64,
}

impl Default for Stage8Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            unwrap_method: "3D".to_string(),
            unwrap_la_error_flag: "y".to_string(),
            unwrap_spatial_cost_func_flag: "n".to_string(),
            drop_ifg_index: Vec::new(),
            ref_lon: vec![f64::NEG_INFINITY, f64::INFINITY],
            ref_lat: vec![f64::NEG_INFINITY, f64::INFINITY],
            ref_radius: f64::INFINITY,
            max_topo_err: 15.0,
            lambda_m: 0.0555,
            unwrap_time_win: 36.0,
        }
    }
}

#[derive(Clone, Debug)]
struct Stage8EdgeOutput {
    dph_noise: Matrix<f32>,
    dph_space_uw: Matrix<f32>,
}

#[derive(Clone, Debug)]
enum PhaseMatrixF32 {
    PsByIfg(Matrix<f32>),
    IfgByPs(Matrix<f32>),
}

#[derive(Clone, Debug)]
struct Stage8SpaceInputs {
    uw_ph: ComplexMatrixF32,
    edges: Vec<(usize, usize)>,
    day_use: Vec<f64>,
    bperp_use: Vec<f64>,
    n_trial_wraps: f64,
}

pub fn run_stage8_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let timing_enabled = stage8_timing_enabled();
    let total_start = Instant::now();
    let mut last_timing = total_start;
    let ps2 = read_mat_stage8_selected(
        dataset_root,
        "ps2.mat",
        &[
            "n_ps",
            "n_ifg",
            "master_ix",
            "day",
            "lonlat",
            "xy",
            "bperp",
            "mean_range",
            "mean_incidence",
        ],
    )?;
    ensure_exists(dataset_root, "phuw2.mat", "stage-6 unwrap output")?;
    ensure_exists(dataset_root, "scla2.mat", "stage-7 SCLA output")?;
    ensure_exists(dataset_root, "uw_grid.mat", "stage-6 grid output")?;
    ensure_exists(
        dataset_root,
        "uw_interp.mat",
        "stage-6 interpolation output",
    )?;
    let parms = load_stage8_parms(dataset_root);
    validate_supported_stage8_mode(&parms)?;

    let n_ps = scalar_from_mat(&ps2, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage8_err("ps2.mat missing valid n_ps");
    }
    let n_ifg = scalar_from_mat(&ps2, "n_ifg", 0.0).round() as usize;
    if n_ifg == 0 {
        return stage8_err("ps2.mat missing valid n_ifg");
    }
    let master_ix = scalar_from_mat(&ps2, "master_ix", 1.0).round() as usize;
    if master_ix == 0 || master_ix > n_ifg {
        return stage8_err(format!(
            "ps2.master_ix must be 1-based within n_ifg={n_ifg}; got {master_ix}"
        ));
    }

    let (mean_v_result, space_output_result) = rayon::join(
        || stage8_mean_velocity_payload(dataset_root, &ps2, &parms, n_ps, n_ifg, master_ix),
        || {
            let space_inputs =
                load_stage8_space_inputs(dataset_root, &ps2, &parms, n_ifg, master_ix)?;
            let edge_count = space_inputs.edges.len();
            let output = stage8_active_single_master_space_time(
                &space_inputs.uw_ph,
                &space_inputs.edges,
                &space_inputs.day_use,
                &space_inputs.bperp_use,
                parms.unwrap_time_win,
                space_inputs.n_trial_wraps,
            )?;
            Ok::<(Stage8EdgeOutput, usize), CoreError>((output, edge_count))
        },
    );
    let mean_v = mean_v_result?;
    let (output, edge_count) = space_output_result?;
    log_stage8_timing(
        timing_enabled,
        "compute_mean_velocity_and_space_time",
        &mut last_timing,
    );
    let (mean_v_write, space_time_write) = rayon::join(
        || write_mean_v(dataset_root, mean_v),
        || write_uw_space_time(dataset_root, output, n_ifg, master_ix, &ps2),
    );
    mean_v_write?;
    space_time_write?;
    log_stage8_timing(timing_enabled, "write_outputs", &mut last_timing);
    if timing_enabled {
        eprintln!(
            "stage8_timing total {:.6}",
            total_start.elapsed().as_secs_f64()
        );
    }

    Ok(format!(
        "Stage 8 produced mean velocity and space-time noise model for {} arcs",
        edge_count
    ))
}

fn load_stage8_space_inputs(
    dataset_root: &Path,
    ps2: &MatData,
    parms: &Stage8Parms,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Stage8SpaceInputs, CoreError> {
    let uw_grid = read_mat_stage8_selected(dataset_root, "uw_grid.mat", &["n_ps", "ph"])?;
    let n_grid_ps = scalar_from_mat(&uw_grid, "n_ps", 0.0).round() as usize;
    if n_grid_ps == 0 {
        return stage8_err("uw_grid.mat missing valid n_ps");
    }
    let uw_ph = complex_ps_matrix(&uw_grid, "ph", n_grid_ps, "uw_grid.ph")?;
    let uw_interp = read_mat_stage8_selected(dataset_root, "uw_interp.mat", &["edgs"])?;
    let edges = edge_table(&uw_interp, "edgs", n_grid_ps)?;
    let day = ps_vector_f64(ps2, "day", n_ifg, "ps2.day")?;
    let bperp = ps_vector_f64(ps2, "bperp", n_ifg, "ps2.bperp")?;
    let unwrap_ifg: Vec<usize> = (1..=n_ifg).filter(|ix| *ix != master_ix).collect();
    if uw_ph.cols != unwrap_ifg.len() {
        return stage8_err(format!(
            "uw_grid.ph has {} columns but single-master Stage 8 expects {} non-master interferograms",
            uw_ph.cols,
            unwrap_ifg.len()
        ));
    }
    let day_use = unwrap_ifg
        .iter()
        .map(|&ix| day[ix - 1] - day[master_ix - 1])
        .collect::<Vec<_>>();
    let bperp_use = unwrap_ifg
        .iter()
        .map(|&ix| bperp[ix - 1])
        .collect::<Vec<_>>();
    let n_trial_wraps = stage8_n_trial_wraps(ps2, parms, &bperp);
    Ok(Stage8SpaceInputs {
        uw_ph,
        edges,
        day_use,
        bperp_use,
        n_trial_wraps,
    })
}

fn validate_supported_stage8_mode(parms: &Stage8Parms) -> Result<(), CoreError> {
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    let unwrap_upper = parms.unwrap_method.to_uppercase();
    let effective_unwrap = if !small_baseline && matches!(unwrap_upper.as_str(), "3D" | "3D_NEW") {
        "3D_FULL"
    } else {
        unwrap_upper.as_str()
    };
    let la_flag = parms.unwrap_la_error_flag.eq_ignore_ascii_case("y");
    let scf_flag = parms
        .unwrap_spatial_cost_func_flag
        .eq_ignore_ascii_case("y");
    if small_baseline || effective_unwrap != "3D_FULL" || !la_flag || scf_flag {
        return stage8_err(
            "Stage 8 native path currently supports only single-master unwrap_method=3D_FULL \
             with unwrap_la_error_flag='y' and unwrap_spatial_cost_func_flag='n'",
        );
    }
    Ok(())
}

fn stage8_timing_enabled() -> bool {
    env::var("PYSTAMPS_STAGE8_TIMINGS").is_ok_and(|value| value == "1")
}

fn log_stage8_timing(enabled: bool, label: &str, last: &mut Instant) {
    if !enabled {
        return;
    }
    let now = Instant::now();
    eprintln!(
        "stage8_timing {label} {:.6}",
        now.duration_since(*last).as_secs_f64()
    );
    *last = now;
}

fn stage8_mean_velocity_payload(
    dataset_root: &Path,
    ps2: &MatData,
    parms: &Stage8Parms,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Matrix<f32>, CoreError> {
    let timing_enabled = stage8_timing_enabled();
    let mut last_timing = Instant::now();
    let ifgstd = read_mat_stage8_selected(dataset_root, "ifgstd2.mat", &["ifg_std"])?;
    log_stage8_timing(
        timing_enabled,
        "mean_velocity_read_ifgstd",
        &mut last_timing,
    );
    let (ph_uw_result, scla_result) = rayon::join(
        || read_stage8_phase_matrix(dataset_root, "phuw2.mat", "ph_uw", n_ps, n_ifg),
        || read_stage8_scla_inputs(dataset_root, n_ps, n_ifg),
    );
    let ph_uw = ph_uw_result?;
    let (ph_scla, c_ps_uw) = scla_result?;
    log_stage8_timing(
        timing_enabled,
        "mean_velocity_read_phase_inputs",
        &mut last_timing,
    );

    let day = ps_vector_f64(ps2, "day", n_ifg, "ps2.day")?;
    let ifg_std = ps_vector_f64(&ifgstd, "ifg_std", n_ifg, "ifgstd2.ifg_std")?;
    let drop_set: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    let unwrap_ix: Vec<usize> = (1..=n_ifg)
        .filter(|ix| !drop_set.contains(&(*ix as i64)) && *ix != master_ix)
        .map(|ix| ix - 1)
        .collect();
    if unwrap_ix.is_empty() {
        return stage8_err(
            "stage-8 mean velocity export requires at least one non-master interferogram",
        );
    }

    let ref_ix = select_reference_ps(ps2, parms, n_ps)?;
    let mut ph_use = stage8_phase_use(&ph_uw, &ph_scla, n_ps, n_ifg, &unwrap_ix);
    log_stage8_timing(timing_enabled, "mean_velocity_phase_use", &mut last_timing);
    deramp_unwrapped_phase_in_place(ps2, &mut ph_use, n_ps, unwrap_ix.len())?;
    log_stage8_timing(timing_enabled, "mean_velocity_deramp", &mut last_timing);
    subtract_c_ps_uw(&mut ph_use, n_ps, unwrap_ix.len(), &c_ps_uw);
    center_values_to_reference(&mut ph_use, n_ps, unwrap_ix.len(), &ref_ix);
    log_stage8_timing(timing_enabled, "mean_velocity_reference", &mut last_timing);

    let mut design = Vec::with_capacity(unwrap_ix.len() * 2);
    let mut weights = Vec::with_capacity(unwrap_ix.len());
    let master_day = day[master_ix - 1];
    for &src_col in &unwrap_ix {
        design.push(1.0);
        design.push(day[src_col] - master_day);
        let variance = (ifg_std[src_col] * std::f64::consts::PI / 180.0).powi(2);
        weights.push(if variance > 0.0 { 1.0 / variance } else { 0.0 });
    }
    let mut values = vec![0.0f32; 2 * n_ps];
    fit_stage8_mean_velocity(
        &design,
        &weights,
        &ph_use,
        n_ps,
        unwrap_ix.len(),
        &mut values,
    )?;
    log_stage8_timing(timing_enabled, "mean_velocity_fit", &mut last_timing);
    Ok(Matrix {
        name: "m".to_string(),
        rows: 2,
        cols: n_ps,
        values,
    })
}

fn read_stage8_phase_matrix(
    dataset_root: &Path,
    filename: &str,
    variable: &str,
    n_ps: usize,
    n_ifg: usize,
) -> Result<PhaseMatrixF32, CoreError> {
    let path = dataset_root.join(filename);
    if let Ok(raw) = read_hdf5_f32_dataset_raw(&path, variable) {
        if raw.rows == n_ifg && raw.cols == n_ps {
            return Ok(PhaseMatrixF32::IfgByPs(raw));
        }
    }

    let mat = read_mat_stage8_selected(dataset_root, filename, &[variable])?;
    let matrix = ps_matrix_f32(&mat, variable, n_ps, &format!("{filename}.{variable}"))?;
    if matrix.cols != n_ifg {
        return stage8_err(format!(
            "{filename}.{variable} must match ps2.n_ifg={n_ifg} for stage-8 mean velocity export; got {} columns",
            matrix.cols
        ));
    }
    Ok(PhaseMatrixF32::PsByIfg(matrix))
}

fn read_stage8_scla_inputs(
    dataset_root: &Path,
    n_ps: usize,
    n_ifg: usize,
) -> Result<(PhaseMatrixF32, Vec<f32>), CoreError> {
    let path = dataset_root.join("scla2.mat");
    if let Ok(mut raw) = read_hdf5_f32_datasets_raw(&path, &["ph_scla", "C_ps_uw"]) {
        if let (Some(ph_scla), Some(c_ps_uw)) = (raw.remove("ph_scla"), raw.remove("C_ps_uw")) {
            if ph_scla.rows == n_ifg && ph_scla.cols == n_ps && c_ps_uw.values.len() == n_ps {
                return Ok((PhaseMatrixF32::IfgByPs(ph_scla), c_ps_uw.values));
            }
        }
    }

    let ph_scla = read_stage8_phase_matrix(dataset_root, "scla2.mat", "ph_scla", n_ps, n_ifg)?;
    let c_ps_uw = read_stage8_c_ps_uw(dataset_root, n_ps)?;
    Ok((ph_scla, c_ps_uw))
}

fn read_stage8_c_ps_uw(dataset_root: &Path, n_ps: usize) -> Result<Vec<f32>, CoreError> {
    let path = dataset_root.join("scla2.mat");
    if let Ok(raw) = read_hdf5_f32_dataset_raw(&path, "C_ps_uw") {
        if raw.values.len() == n_ps {
            return Ok(raw.values);
        }
    }
    let Ok(mat) = read_mat_stage8_selected(dataset_root, "scla2.mat", &["C_ps_uw"]) else {
        return Ok(vec![0.0; n_ps]);
    };
    match ps_vector_f64(&mat, "C_ps_uw", n_ps, "scla2.C_ps_uw") {
        Ok(values) => Ok(values.into_iter().map(|value| value as f32).collect()),
        Err(_) => Ok(vec![0.0; n_ps]),
    }
}

fn stage8_phase_use(
    ph_uw: &PhaseMatrixF32,
    ph_scla: &PhaseMatrixF32,
    n_ps: usize,
    n_ifg: usize,
    unwrap_ix: &[usize],
) -> Vec<f64> {
    let cols = unwrap_ix.len();
    let mut ph_use = vec![0.0; n_ps * cols];
    match (ph_uw, ph_scla) {
        (PhaseMatrixF32::IfgByPs(uw), PhaseMatrixF32::IfgByPs(scla)) => {
            ph_use
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, out_row)| {
                    for (out_col, &src_col) in unwrap_ix.iter().enumerate() {
                        let source = src_col * n_ps + row;
                        out_row[out_col] = uw.values[source] as f64 - scla.values[source] as f64;
                    }
                });
        }
        (PhaseMatrixF32::PsByIfg(uw), PhaseMatrixF32::PsByIfg(scla)) => {
            ph_use
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let source_row = row * n_ifg;
                    for (out_col, &src_col) in unwrap_ix.iter().enumerate() {
                        let source = source_row + src_col;
                        out_row[out_col] = uw.values[source] as f64 - scla.values[source] as f64;
                    }
                });
        }
        (PhaseMatrixF32::IfgByPs(uw), PhaseMatrixF32::PsByIfg(scla)) => {
            ph_use
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let scla_source_row = row * n_ifg;
                    for (out_col, &src_col) in unwrap_ix.iter().enumerate() {
                        out_row[out_col] = uw.values[src_col * n_ps + row] as f64
                            - scla.values[scla_source_row + src_col] as f64;
                    }
                });
        }
        (PhaseMatrixF32::PsByIfg(uw), PhaseMatrixF32::IfgByPs(scla)) => {
            ph_use
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let uw_source_row = row * n_ifg;
                    for (out_col, &src_col) in unwrap_ix.iter().enumerate() {
                        out_row[out_col] = uw.values[uw_source_row + src_col] as f64
                            - scla.values[src_col * n_ps + row] as f64;
                    }
                });
        }
    }
    ph_use
}

fn deramp_unwrapped_phase_in_place(
    ps2: &MatData,
    values: &mut [f64],
    rows: usize,
    cols: usize,
) -> Result<(), CoreError> {
    if values.len() != rows * cols {
        return stage8_err("stage8 deramp input has inconsistent dimensions");
    }
    let xy = ps_dim_f64(ps2, "xy", rows, 3, "ps2.xy")?;
    let mut design = vec![0.0; rows * 3];
    for row in 0..rows {
        design[row * 3] = xy.values[row * 3 + 1] / 1000.0;
        design[row * 3 + 1] = xy.values[row * 3 + 2] / 1000.0;
        design[row * 3 + 2] = 1.0;
    }
    let mut normal = vec![0.0; 9];
    for row in 0..rows {
        for i in 0..3 {
            let xi = design[row * 3 + i];
            for j in 0..3 {
                normal[i * 3 + j] += xi * design[row * 3 + j];
            }
        }
    }
    let rhs_by_col = values
        .par_chunks(cols)
        .zip(design.par_chunks(3))
        .fold(
            || vec![0.0; cols * 3],
            |mut acc, (value_row, design_row)| {
                for col in 0..cols {
                    let y = value_row[col];
                    for i in 0..3 {
                        acc[col * 3 + i] += design_row[i] * y;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0; cols * 3],
            |mut left, right| {
                for (left_value, right_value) in left.iter_mut().zip(right) {
                    *left_value += right_value;
                }
                left
            },
        );
    let mut coeffs = vec![0.0; cols * 3];
    for col in 0..cols {
        let rhs = rhs_by_col[col * 3..col * 3 + 3].to_vec();
        let coeff = solve_linear(normal.clone(), rhs, 3)?;
        coeffs[col * 3..col * 3 + 3].copy_from_slice(&coeff);
    }
    values
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, value_row)| {
            let design_row = &design[row * 3..row * 3 + 3];
            for col in 0..cols {
                let coeff = &coeffs[col * 3..col * 3 + 3];
                let ramp =
                    design_row[0] * coeff[0] + design_row[1] * coeff[1] + design_row[2] * coeff[2];
                value_row[col] -= ramp;
            }
        });
    Ok(())
}

fn subtract_c_ps_uw(values: &mut [f64], rows: usize, cols: usize, c_ps_uw: &[f32]) {
    debug_assert_eq!(values.len(), rows * cols);
    if c_ps_uw.len() != rows {
        return;
    }
    values
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, value_row)| {
            let correction = c_ps_uw[row] as f64;
            for value in value_row {
                *value -= correction;
            }
        });
}

#[derive(Clone, Copy, Debug, Default)]
struct Complex64Lite {
    re: f64,
    im: f64,
}

impl Complex64Lite {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn scale(self, value: f64) -> Self {
        Self::new(self.re * value, self.im * value)
    }

    fn unit_or_zero(self) -> Self {
        let abs = self.abs();
        if abs == 0.0 {
            Self::default()
        } else {
            self.scale(1.0 / abs)
        }
    }
}

fn stage8_active_single_master_space_time(
    uw_ph: &ComplexMatrixF32,
    edges: &[(usize, usize)],
    day: &[f64],
    bperp: &[f64],
    time_win: f64,
    n_trial_wraps: f64,
) -> Result<Stage8EdgeOutput, CoreError> {
    let timing_enabled = stage8_timing_enabled();
    let mut last_timing = Instant::now();
    let n_edge = edges.len();
    let n_ifg = uw_ph.cols;
    if day.len() != n_ifg || bperp.len() != n_ifg {
        return stage8_err("Stage 8 space-time day/bperp vectors must match uw_grid.ph columns");
    }
    let dph_space = stage8_edge_phasors(uw_ph, edges);
    log_stage8_timing(timing_enabled, "space_time_edge_phasors", &mut last_timing);
    let la_model = Stage8LaModel::new(day, bperp, n_trial_wraps);
    let k = estimate_stage8_la_error(&dph_space, n_edge, n_ifg, &la_model);
    log_stage8_timing(timing_enabled, "space_time_la_error", &mut last_timing);
    let smooth_model = Stage8SmoothModel::new(day, time_win);
    let mut dph_noise_rows = vec![0.0f32; n_edge * n_ifg];
    let mut dph_space_uw_rows = vec![0.0f32; n_edge * n_ifg];
    dph_noise_rows
        .par_chunks_mut(n_ifg)
        .zip(dph_space_uw_rows.par_chunks_mut(n_ifg))
        .enumerate()
        .for_each_init(
            || Stage8SmoothScratch::new(n_ifg),
            |scratch, (edge_ix, (noise_row, space_uw_row))| {
                let row = &dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                smooth_stage8_edge_row(
                    row,
                    k[edge_ix],
                    bperp,
                    &smooth_model,
                    scratch,
                    noise_row,
                    space_uw_row,
                );
            },
        );
    log_stage8_timing(timing_enabled, "space_time_smoothing", &mut last_timing);
    Ok(Stage8EdgeOutput {
        dph_noise: Matrix {
            name: "dph_noise".to_string(),
            rows: n_edge,
            cols: n_ifg,
            values: dph_noise_rows,
        },
        dph_space_uw: Matrix {
            name: "dph_space_uw".to_string(),
            rows: n_edge,
            cols: n_ifg,
            values: dph_space_uw_rows,
        },
    })
}

fn stage8_edge_phasors(uw_ph: &ComplexMatrixF32, edges: &[(usize, usize)]) -> Vec<Complex64Lite> {
    let n_ifg = uw_ph.cols;
    let mut values = vec![Complex64Lite::default(); edges.len() * n_ifg];
    values
        .par_chunks_mut(n_ifg)
        .enumerate()
        .for_each(|(edge_ix, row)| {
            let (a_ix, b_ix) = edges[edge_ix];
            for (ifg_ix, out) in row.iter_mut().enumerate() {
                let left = uw_ph.values[a_ix * n_ifg + ifg_ix];
                let right = uw_ph.values[b_ix * n_ifg + ifg_ix];
                let real = right.0 * left.0 + right.1 * left.1;
                let imag = right.1 * left.0 - right.0 * left.1;
                *out = Complex64Lite::new(real as f64, imag as f64).unit_or_zero();
            }
        });
    values
}

#[derive(Clone, Debug)]
struct Stage8LaModel {
    insert_ix: usize,
    use_ix: Vec<usize>,
    bperp_diff: Vec<f64>,
    bperp_range: f64,
    trial_mult: Vec<f64>,
    trial_phase: Vec<Complex64Lite>,
}

impl Stage8LaModel {
    fn new(day: &[f64], bperp: &[f64], n_trial_wraps: f64) -> Self {
        let insert_ix = single_master_insert_ix(day);
        let mut bperp_master = Vec::with_capacity(bperp.len() + 1);
        bperp_master.extend_from_slice(&bperp[..insert_ix.min(bperp.len())]);
        bperp_master.push(0.0);
        if insert_ix < bperp.len() {
            bperp_master.extend_from_slice(&bperp[insert_ix..]);
        }
        let bperp_diff_all = bperp_master
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        let bperp_range_orig = vector_range(bperp);
        let bperp_range = vector_range(&bperp_diff_all);
        let mut n_trial_wraps_sub = n_trial_wraps;
        if bperp_range_orig != 0.0 {
            n_trial_wraps_sub *= bperp_range / bperp_range_orig;
        }
        let trial_bound = (8.0 * n_trial_wraps_sub).ceil().max(0.0) as i64;
        let trial_mult = (-trial_bound..=trial_bound)
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        let mut use_ix = Vec::new();
        let mut bperp_diff = Vec::new();
        for (ix, &diff) in bperp_diff_all.iter().enumerate() {
            if diff != 0.0 {
                use_ix.push(ix);
                bperp_diff.push(diff);
            }
        }
        let safe_range = bperp_range.abs().max(1e-12);
        let mut trial_phase = Vec::with_capacity(bperp_diff.len() * trial_mult.len());
        for &diff in &bperp_diff {
            let base = diff / safe_range * std::f64::consts::PI / 4.0;
            for &trial in &trial_mult {
                let phase = -base * trial;
                trial_phase.push(Complex64Lite::new(phase.cos(), phase.sin()));
            }
        }
        Self {
            insert_ix,
            use_ix,
            bperp_diff,
            bperp_range,
            trial_mult,
            trial_phase,
        }
    }

    fn trial(&self, diff_ix: usize, trial_ix: usize) -> Complex64Lite {
        self.trial_phase[diff_ix * self.trial_mult.len() + trial_ix]
    }
}

#[derive(Clone, Debug)]
struct Stage8SmoothModel {
    cols: usize,
    close_master_ix: Vec<usize>,
    per_ifg: Vec<Stage8SmoothWeights>,
}

#[derive(Clone, Debug)]
struct Stage8SmoothWeights {
    time_diff: Vec<f64>,
    weight: Vec<f64>,
    weighted_time: Vec<f64>,
    s0: f64,
    s1: f64,
    s2: f64,
    det: f64,
}

#[derive(Clone, Debug)]
struct Stage8SmoothScratch {
    adjusted: Vec<Complex64Lite>,
    angle: Vec<f64>,
    smooth_angle: Vec<f64>,
    smooth_uw: Vec<f32>,
}

impl Stage8SmoothScratch {
    fn new(cols: usize) -> Self {
        Self {
            adjusted: vec![Complex64Lite::default(); cols],
            angle: vec![0.0; cols],
            smooth_angle: vec![0.0; cols],
            smooth_uw: vec![0.0; cols],
        }
    }

    fn resize(&mut self, cols: usize) {
        self.adjusted.resize(cols, Complex64Lite::default());
        self.angle.resize(cols, 0.0);
        self.smooth_angle.resize(cols, 0.0);
        self.smooth_uw.resize(cols, 0.0);
    }
}

impl Stage8SmoothModel {
    fn new(day: &[f64], time_win: f64) -> Self {
        let cols = day.len();
        let time_win = time_win.max(1e-6);
        let per_ifg = (0..cols)
            .map(|col| {
                let mut time_diff = Vec::with_capacity(cols);
                let mut weight = Vec::with_capacity(cols);
                for &day_value in day {
                    let diff = day[col] - day_value;
                    time_diff.push(diff);
                    weight.push((-(diff * diff) / (2.0 * time_win * time_win)).exp());
                }
                let sum: f64 = weight.iter().sum();
                let denom = sum.max(1e-12);
                for value in &mut weight {
                    *value /= denom;
                }
                let mut weighted_time = Vec::with_capacity(cols);
                let mut s0 = 0.0;
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                for ix in 0..cols {
                    let w = weight[ix];
                    let t = time_diff[ix];
                    weighted_time.push(w * t);
                    s0 += w;
                    s1 += w * t;
                    s2 += w * t * t;
                }
                Stage8SmoothWeights {
                    time_diff,
                    weight,
                    weighted_time,
                    s0,
                    s1,
                    s2,
                    det: s0 * s2 - s1 * s1,
                }
            })
            .collect();
        Self {
            cols,
            close_master_ix: single_master_close_master_ix(day),
            per_ifg,
        }
    }
}

fn estimate_stage8_la_error(
    dph_space: &[Complex64Lite],
    rows: usize,
    cols: usize,
    model: &Stage8LaModel,
) -> Vec<f32> {
    if rows == 0 || cols == 0 || model.trial_mult.is_empty() || model.bperp_diff.is_empty() {
        return vec![0.0; rows];
    }
    dph_space
        .par_chunks(cols)
        .map(|row| estimate_stage8_la_error_row(row, model))
        .collect()
}

fn estimate_stage8_la_error_row(row: &[Complex64Lite], model: &Stage8LaModel) -> f32 {
    let mut temp = Vec::with_capacity(row.len() + 1);
    let mean_abs = if row.is_empty() {
        0.0
    } else {
        row.iter().map(|value| value.abs()).sum::<f64>() / row.len() as f64
    };
    temp.extend_from_slice(&row[..model.insert_ix.min(row.len())]);
    temp.push(Complex64Lite::new(mean_abs, 0.0));
    if model.insert_ix < row.len() {
        temp.extend_from_slice(&row[model.insert_ix..]);
    }

    let mut cpxphase_all = Vec::with_capacity(row.len());
    for pair in temp.windows(2) {
        cpxphase_all.push(pair[1].mul(pair[0].conj()).unit_or_zero());
    }
    let cpxphase = model
        .use_ix
        .iter()
        .filter_map(|&ix| cpxphase_all.get(ix).copied())
        .collect::<Vec<_>>();
    let denom: f64 = cpxphase.iter().map(|value| value.abs()).sum();
    if denom == 0.0 {
        return 0.0;
    }

    let n_trials = model.trial_mult.len();
    let mut phaser_sum = vec![Complex64Lite::default(); n_trials];
    for (diff_ix, &phase) in cpxphase.iter().enumerate() {
        for (trial_ix, sum) in phaser_sum.iter_mut().enumerate() {
            let shifted = phase.mul(model.trial(diff_ix, trial_ix));
            sum.re += shifted.re;
            sum.im += shifted.im;
        }
    }
    let coh_trial = phaser_sum
        .iter()
        .map(|value| value.abs() / denom)
        .collect::<Vec<_>>();
    if coh_trial.is_empty() {
        return 0.0;
    }
    let mut coh_max_ix = 0;
    let mut coh_max = coh_trial[0];
    for (ix, &value) in coh_trial.iter().enumerate().skip(1) {
        if value > coh_max {
            coh_max = value;
            coh_max_ix = ix;
        }
    }
    let mut peak_start_ix = 0;
    for ix in 0..coh_max_ix {
        if coh_trial[ix + 1] - coh_trial[ix] < 0.0 {
            peak_start_ix = ix + 1;
        }
    }
    let mut peak_end_ix = n_trials - 1;
    for ix in coh_max_ix..n_trials.saturating_sub(1) {
        if coh_trial[ix + 1] - coh_trial[ix] > 0.0 {
            peak_end_ix = ix;
            break;
        }
    }
    let mut next_max = 0.0;
    for (ix, &value) in coh_trial.iter().enumerate() {
        if ix < peak_start_ix || ix > peak_end_ix {
            if value > next_max {
                next_max = value;
            }
        }
    }
    if coh_max - next_max <= 0.1 {
        return 0.0;
    }

    let safe_range = model.bperp_range.abs().max(1e-12);
    let k0 = (std::f64::consts::PI / 4.0 / safe_range) * model.trial_mult[coh_max_ix];
    let mut offset_phase = Complex64Lite::default();
    let mut resphase = Vec::with_capacity(cpxphase.len());
    for (&phase, &diff) in cpxphase.iter().zip(model.bperp_diff.iter()) {
        let correction = complex_exp(-k0 * diff);
        let value = phase.mul(correction);
        offset_phase.re += value.re;
        offset_phase.im += value.im;
        resphase.push(value);
    }
    let offset_conj = offset_phase.conj();
    let mut den = 0.0;
    let mut num = 0.0;
    for ((&phase, &diff), &value) in cpxphase
        .iter()
        .zip(model.bperp_diff.iter())
        .zip(resphase.iter())
    {
        let weight = phase.abs();
        let resphase_angle = value.mul(offset_conj).arg();
        let weighted_diff = weight * diff;
        den += weighted_diff * weighted_diff;
        num += weighted_diff * (weight * resphase_angle);
    }
    let mopt = if den != 0.0 { num / den } else { 0.0 };
    let k = k0 + mopt;
    let mut phase_residual_sum = Complex64Lite::default();
    let mut phase_residual_abs = 0.0;
    let mut any = false;
    for (&phase, &diff) in cpxphase.iter().zip(model.bperp_diff.iter()) {
        if phase.re != 0.0 || phase.im != 0.0 {
            any = true;
        }
        let value = phase.mul(complex_exp(-k * diff));
        phase_residual_sum.re += value.re;
        phase_residual_sum.im += value.im;
        phase_residual_abs += value.abs();
    }
    let coh = if any && phase_residual_abs != 0.0 {
        phase_residual_sum.abs() / phase_residual_abs
    } else {
        0.0
    };
    if coh < 0.31 {
        0.0
    } else {
        k as f32
    }
}

fn smooth_stage8_edge_row(
    row: &[Complex64Lite],
    k: f32,
    bperp: &[f64],
    model: &Stage8SmoothModel,
    scratch: &mut Stage8SmoothScratch,
    noise_row: &mut [f32],
    space_uw_row: &mut [f32],
) {
    let cols = model.cols;
    debug_assert_eq!(row.len(), cols);
    scratch.resize(cols);
    let adjusted = &mut scratch.adjusted;
    let angle = &mut scratch.angle;
    for col in 0..cols {
        let correction = complex_exp_f32ish(-(k as f64) * bperp[col]);
        let value = row[col].mul(correction).unit_or_zero();
        adjusted[col] = value;
        angle[col] = value.arg();
    }

    let smooth_angle = &mut scratch.smooth_angle;
    for col in 0..cols {
        let weights = &model.per_ifg[col];
        let mut mean = Complex64Lite::default();
        for (&value, &weight) in adjusted.iter().zip(weights.weight.iter()) {
            mean.re += value.re * weight;
            mean.im += value.im * weight;
        }
        let mean_angle = mean.arg();
        let mut wy0 = 0.0;
        let mut wy1 = 0.0;
        for ix in 0..cols {
            let mut adjusted_angle = wrap_to_pi(angle[ix] - mean_angle);
            if (adjusted_angle + std::f64::consts::PI).abs() <= 2e-7 && weights.time_diff[ix] > 0.0
            {
                adjusted_angle = std::f64::consts::PI;
            }
            wy0 += adjusted_angle * weights.weight[ix];
            wy1 += adjusted_angle * weights.weighted_time[ix];
        }
        let intercept = if weights.det == 0.0 {
            if weights.s0 != 0.0 {
                wy0 / weights.s0
            } else {
                0.0
            }
        } else {
            (wy0 * weights.s2 - wy1 * weights.s1) / weights.det
        };
        smooth_angle[col] = mean.mul(complex_exp(intercept)).arg();
        noise_row[col] = wrap_to_pi(angle[col] - smooth_angle[col]) as f32;
    }

    let smooth_uw = &mut scratch.smooth_uw;
    smooth_uw.fill(0.0);
    if cols > 0 {
        smooth_uw[0] = wrap_to_pi(smooth_angle[0]) as f32;
        for col in 1..cols {
            smooth_uw[col] =
                smooth_uw[col - 1] + wrap_to_pi(smooth_angle[col] - smooth_angle[col - 1]) as f32;
        }
        let close_mean = if model.close_master_ix.is_empty() {
            0.0
        } else {
            model
                .close_master_ix
                .iter()
                .map(|&ix| smooth_uw[ix] as f64)
                .sum::<f64>()
                / model.close_master_ix.len() as f64
        };
        let close_offset = close_mean - wrap_to_pi(close_mean);
        for value in smooth_uw.iter_mut() {
            *value -= close_offset as f32;
        }
    }

    let bad_noise = row_std_f32(noise_row) > 1.2;
    for col in 0..cols {
        if bad_noise {
            noise_row[col] = f32::NAN;
            space_uw_row[col] = f32::NAN;
        } else {
            space_uw_row[col] = smooth_uw[col] + noise_row[col] + (k as f64 * bperp[col]) as f32;
        }
    }
}

fn complex_exp(phase: f64) -> Complex64Lite {
    Complex64Lite::new(phase.cos(), phase.sin())
}

fn complex_exp_f32ish(phase: f64) -> Complex64Lite {
    Complex64Lite::new(phase.cos() as f32 as f64, phase.sin() as f32 as f64)
}

fn wrap_to_pi(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

fn row_std_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let count = values.len() as f64;
    let mean = values.iter().map(|&value| value as f64).sum::<f64>() / count;
    let sumsq = values
        .iter()
        .map(|&value| {
            let diff = value as f64 - mean;
            diff * diff
        })
        .sum::<f64>();
    let denom = if values.len() > 1 { count - 1.0 } else { count };
    (sumsq / denom).sqrt() as f32
}

fn single_master_close_master_ix(day: &[f64]) -> Vec<usize> {
    if day.is_empty() {
        return Vec::new();
    }
    let mut best = None;
    for (ix, &value) in day.iter().enumerate() {
        if value > 0.0 {
            match best {
                Some((_, best_value)) if value >= best_value => {}
                _ => best = Some((ix, value)),
            }
        }
    }
    let insert_ix = best.map(|(ix, _)| ix).unwrap_or(day.len() - 1);
    if insert_ix > 0 {
        vec![insert_ix - 1, insert_ix]
    } else {
        vec![insert_ix]
    }
}

fn single_master_insert_ix(day: &[f64]) -> usize {
    single_master_close_master_ix(day)
        .last()
        .copied()
        .unwrap_or(0)
}

fn vector_range(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for &value in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }
    max_value - min_value
}

fn stage8_n_trial_wraps(ps2: &MatData, parms: &Stage8Parms, bperp: &[f64]) -> f64 {
    let mean_range = scalar_from_mat(ps2, "mean_range", 830_000.0);
    let mean_incidence = scalar_from_mat(ps2, "mean_incidence", 23.0_f64.to_radians());
    let denominator =
        parms.lambda_m * mean_range * mean_incidence.sin() / (4.0 * std::f64::consts::PI);
    if denominator == 0.0 {
        return 0.0;
    }
    let max_k = parms.max_topo_err / denominator;
    vector_range(bperp) * max_k / (2.0 * std::f64::consts::PI)
}

fn write_mean_v(dataset_root: &Path, m: Matrix<f32>) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("mean_v.mat"));
    mat.add_f32_matrix("m", m.rows, m.cols, m.values)?;
    mat.write()?;
    Ok(())
}

fn write_uw_space_time(
    dataset_root: &Path,
    output: Stage8EdgeOutput,
    n_ifg: usize,
    master_ix: usize,
    ps2: &MatData,
) -> Result<(), CoreError> {
    let day_len = optional_vector_f64(ps2, "day")
        .map(|values| values.len())
        .unwrap_or(n_ifg);
    let unwrap_ifg: Vec<usize> = (1..=n_ifg).filter(|ix| *ix != master_ix).collect();
    let mut g = vec![0.0f64; unwrap_ifg.len() * day_len];
    for (row, &ifg_ix) in unwrap_ifg.iter().enumerate() {
        if master_ix <= day_len {
            g[row * day_len + master_ix - 1] = -1.0;
        }
        if ifg_ix <= day_len {
            g[row * day_len + ifg_ix - 1] = 1.0;
        }
    }

    let noise_rows = output.dph_noise.rows;
    let noise_cols = output.dph_noise.cols;
    let space_rows = output.dph_space_uw.rows;
    let space_cols = output.dph_space_uw.cols;
    let path = dataset_root.join("uw_space_time.mat");
    let file = H5File::create(&path)
        .map_err(|err| stage8_err_owned(format!("unable to create {}: {err}", path.display())))?;
    write_stage8_hdf5_f64_matrix(&file, "G", unwrap_ifg.len(), day_len, &g)?;
    write_stage8_hdf5_f32_matrix(
        &file,
        "dph_noise",
        noise_rows,
        noise_cols,
        &output.dph_noise.values,
    )?;
    write_stage8_hdf5_f32_matrix(
        &file,
        "dph_space_uw",
        space_rows,
        space_cols,
        &output.dph_space_uw.values,
    )?;
    write_stage8_hdf5_sparse_empty(&file, "spread", noise_rows, noise_cols)?;
    write_stage8_hdf5_f64_matrix(&file, "ifreq_ij", 0, 0, &[])?;
    write_stage8_hdf5_f64_matrix(&file, "jfreq_ij", 0, 0, &[])?;
    write_stage8_hdf5_f64_matrix(&file, "shaky_ix", 0, 0, &[])?;
    write_stage8_hdf5_f64_matrix(&file, "predef_ix", 0, 0, &[])?;
    file.close()
        .map_err(|err| stage8_err_owned(format!("unable to close {}: {err}", path.display())))?;
    Ok(())
}

fn write_stage8_hdf5_f32_matrix(
    file: &H5File,
    name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> Result<(), CoreError> {
    let dataset = file
        .new_dataset::<f32>()
        .shape([rows, cols])
        .create(name)
        .map_err(|err| stage8_err_owned(format!("unable to create HDF5 dataset {name}: {err}")))?;
    mark_stage8_hdf5_row_major(&dataset, name)?;
    if !values.is_empty() {
        dataset.write_raw(values).map_err(|err| {
            stage8_err_owned(format!("unable to write HDF5 dataset {name}: {err}"))
        })?;
    }
    Ok(())
}

fn write_stage8_hdf5_f64_matrix(
    file: &H5File,
    name: &str,
    rows: usize,
    cols: usize,
    values: &[f64],
) -> Result<(), CoreError> {
    let dataset = file
        .new_dataset::<f64>()
        .shape([rows, cols])
        .create(name)
        .map_err(|err| stage8_err_owned(format!("unable to create HDF5 dataset {name}: {err}")))?;
    mark_stage8_hdf5_row_major(&dataset, name)?;
    if !values.is_empty() {
        dataset.write_raw(values).map_err(|err| {
            stage8_err_owned(format!("unable to write HDF5 dataset {name}: {err}"))
        })?;
    }
    Ok(())
}

fn write_stage8_hdf5_i32_vector(
    group: &rust_hdf5::H5Group,
    name: &str,
    values: &[i32],
) -> Result<(), CoreError> {
    let dataset = group
        .new_dataset::<i32>()
        .shape([values.len()])
        .create(name)
        .map_err(|err| {
            stage8_err_owned(format!(
                "unable to create HDF5 dataset spread/{name}: {err}"
            ))
        })?;
    if !values.is_empty() {
        dataset.write_raw(values).map_err(|err| {
            stage8_err_owned(format!("unable to write HDF5 dataset spread/{name}: {err}"))
        })?;
    }
    Ok(())
}

fn write_stage8_hdf5_f64_vector(
    group: &rust_hdf5::H5Group,
    name: &str,
    values: &[f64],
) -> Result<(), CoreError> {
    let dataset = group
        .new_dataset::<f64>()
        .shape([values.len()])
        .create(name)
        .map_err(|err| {
            stage8_err_owned(format!(
                "unable to create HDF5 dataset spread/{name}: {err}"
            ))
        })?;
    if !values.is_empty() {
        dataset.write_raw(values).map_err(|err| {
            stage8_err_owned(format!("unable to write HDF5 dataset spread/{name}: {err}"))
        })?;
    }
    Ok(())
}

fn write_stage8_hdf5_u64_vector(
    group: &rust_hdf5::H5Group,
    name: &str,
    values: &[u64],
) -> Result<(), CoreError> {
    let dataset = group
        .new_dataset::<u64>()
        .shape([values.len()])
        .create(name)
        .map_err(|err| {
            stage8_err_owned(format!(
                "unable to create HDF5 dataset spread/{name}: {err}"
            ))
        })?;
    if !values.is_empty() {
        dataset.write_raw(values).map_err(|err| {
            stage8_err_owned(format!("unable to write HDF5 dataset spread/{name}: {err}"))
        })?;
    }
    Ok(())
}

fn write_stage8_hdf5_sparse_empty(
    file: &H5File,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(), CoreError> {
    let group = file
        .create_group(name)
        .map_err(|err| stage8_err_owned(format!("unable to create HDF5 group {name}: {err}")))?;
    write_stage8_hdf5_f64_vector(&group, "data", &[])?;
    write_stage8_hdf5_i32_vector(&group, "ir", &[])?;
    write_stage8_hdf5_i32_vector(&group, "jc", &vec![0; cols + 1])?;
    write_stage8_hdf5_u64_vector(&group, "shape", &[rows as u64, cols as u64])?;
    Ok(())
}

fn mark_stage8_hdf5_row_major(dataset: &rust_hdf5::H5Dataset, name: &str) -> Result<(), CoreError> {
    let attr = dataset
        .new_attr::<u8>()
        .shape(())
        .create(PYSTAMPS_ROW_MAJOR_ATTR)
        .map_err(|err| {
            stage8_err_owned(format!(
                "unable to create HDF5 row-major attribute for {name}: {err}"
            ))
        })?;
    attr.write_numeric(&1u8).map_err(|err| {
        stage8_err_owned(format!(
            "unable to write HDF5 row-major attribute for {name}: {err}"
        ))
    })
}

fn edge_table(mat: &MatData, name: &str, n_nodes: usize) -> Result<Vec<(usize, usize)>, CoreError> {
    let source = mat
        .get_f64_matrix(name)
        .map_err(|err| stage8_err_owned(format!("uw_interp.{name} is invalid: {err}")))?;
    if source.cols != 3 {
        return stage8_err(format!(
            "uw_interp.{name} must be an Nx3 edge table with 1-based node columns 2 and 3; got {}x{}",
            source.rows, source.cols
        ));
    }
    let mut edges = Vec::with_capacity(source.rows);
    for row in 0..source.rows {
        let a = source.values[row * source.cols + 1].round() as i64;
        let b = source.values[row * source.cols + 2].round() as i64;
        if a <= 0 || b <= 0 || a as usize > n_nodes || b as usize > n_nodes || a == b {
            return stage8_err(format!(
                "uw_interp.{name} row {} has malformed 1-based edge nodes ({a}, {b}) for n_ps={n_nodes}",
                row + 1
            ));
        }
        edges.push((a as usize - 1, b as usize - 1));
    }
    Ok(edges)
}

fn ensure_exists(dataset_root: &Path, filename: &str, label: &str) -> Result<(), CoreError> {
    if dataset_root.join(filename).exists() {
        Ok(())
    } else {
        stage8_err(format!(
            "Missing required artifact: {filename} ({label}) before stage 8"
        ))
    }
}

fn center_values_to_reference(values: &mut [f64], rows: usize, cols: usize, ref_ix: &[usize]) {
    if ref_ix.is_empty() {
        return;
    }
    debug_assert_eq!(values.len(), rows * cols);
    let mut means = if ref_ix.len() == rows {
        values
            .par_chunks(cols)
            .fold(
                || vec![0.0; cols],
                |mut acc, value_row| {
                    for col in 0..cols {
                        acc[col] += value_row[col];
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; cols],
                |mut left, right| {
                    for (left_value, right_value) in left.iter_mut().zip(right) {
                        *left_value += right_value;
                    }
                    left
                },
            )
    } else {
        let mut means = vec![0.0; cols];
        for &row in ref_ix {
            let value_row = &values[row * cols..row * cols + cols];
            for col in 0..cols {
                means[col] += value_row[col];
            }
        }
        means
    };
    for mean in &mut means {
        *mean /= ref_ix.len() as f64;
    }
    values.par_chunks_mut(cols).for_each(|value_row| {
        for col in 0..cols {
            value_row[col] -= means[col];
        }
    });
}

fn select_reference_ps(
    ps2: &MatData,
    parms: &Stage8Parms,
    n_ps: usize,
) -> Result<Vec<usize>, CoreError> {
    let lonlat = ps_dim_f64(ps2, "lonlat", n_ps, 2, "ps2.lonlat")?;
    if parms.ref_radius == f64::NEG_INFINITY {
        return Ok(Vec::new());
    }
    let lon_min = parms.ref_lon.first().copied().unwrap_or(f64::NEG_INFINITY);
    let lon_max = parms.ref_lon.get(1).copied().unwrap_or(f64::INFINITY);
    let lat_min = parms.ref_lat.first().copied().unwrap_or(f64::NEG_INFINITY);
    let lat_max = parms.ref_lat.get(1).copied().unwrap_or(f64::INFINITY);
    let mut ref_ix = Vec::new();
    for row in 0..n_ps {
        let lon = lonlat.values[row * 2];
        let lat = lonlat.values[row * 2 + 1];
        if lon > lon_min && lon < lon_max && lat > lat_min && lat < lat_max {
            ref_ix.push(row);
        }
    }
    if ref_ix.is_empty() {
        ref_ix.extend(0..n_ps);
    }
    Ok(ref_ix)
}

fn fit_stage8_mean_velocity(
    design: &[f64],
    weights: &[f64],
    y_by_target: &[f64],
    targets: usize,
    rows: usize,
    out: &mut [f32],
) -> Result<(), CoreError> {
    if design.len() != rows * 2 || weights.len() != rows || y_by_target.len() != targets * rows {
        return stage8_err("stage8 native mean-velocity fit has inconsistent dimensions");
    }
    if out.len() != targets * 2 {
        return stage8_err("stage8 native mean-velocity output has inconsistent dimensions");
    }

    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut weighted_time = Vec::with_capacity(rows);
    for row in 0..rows {
        let weight = weights[row];
        let t = design[row * 2 + 1];
        s0 += weight;
        s1 += weight * t;
        s2 += weight * t * t;
        weighted_time.push(weight * t);
    }
    let det = s0 * s2 - s1 * s1;
    let (intercepts, slopes) = out.split_at_mut(targets);
    intercepts
        .par_iter_mut()
        .zip(slopes.par_iter_mut())
        .enumerate()
        .for_each(|(target, (intercept, slope))| {
            let y = &y_by_target[target * rows..target * rows + rows];
            let mut wy0 = 0.0;
            let mut wy1 = 0.0;
            for row in 0..rows {
                wy0 += y[row] * weights[row];
                wy1 += y[row] * weighted_time[row];
            }
            if det == 0.0 {
                *intercept = if s0 != 0.0 { (wy0 / s0) as f32 } else { 0.0 };
                *slope = 0.0;
            } else {
                *intercept = ((wy0 * s2 - wy1 * s1) / det) as f32;
                *slope = ((wy1 * s0 - wy0 * s1) / det) as f32;
            }
        });
    Ok(())
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, CoreError> {
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_abs = a[pivot * n + pivot].abs();
        for row in pivot + 1..n {
            let value = a[row * n + pivot].abs();
            if value > best_abs {
                best = row;
                best_abs = value;
            }
        }
        if best_abs <= 1e-12 {
            a[pivot * n + pivot] += 1e-10;
            best_abs = a[pivot * n + pivot].abs();
        }
        if best_abs <= 1e-12 {
            return stage8_err("stage8 native least-squares system is singular");
        }
        if best != pivot {
            for col in 0..n {
                a.swap(pivot * n + col, best * n + col);
            }
            b.swap(pivot, best);
        }
        let pivot_value = a[pivot * n + pivot];
        for row in pivot + 1..n {
            let factor = a[row * n + pivot] / pivot_value;
            a[row * n + pivot] = 0.0;
            for col in pivot + 1..n {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for col in row + 1..n {
            sum -= a[row * n + col] * x[col];
        }
        x[row] = sum / a[row * n + row];
    }
    Ok(x)
}

fn read_mat_stage8_selected(
    dataset_root: &Path,
    filename: &str,
    variables: &[&str],
) -> Result<MatData, CoreError> {
    MatData::read_selected(dataset_root.join(filename), variables)
        .map_err(|err| stage8_err_owned(format!("unable to read {filename}: {err}")))
}

fn load_stage8_parms(dataset_root: &Path) -> Stage8Parms {
    let path = dataset_root.join("parms.mat");
    if !path.exists() {
        return Stage8Parms::default();
    }
    let Ok(mat) = MatData::read(path) else {
        return Stage8Parms::default();
    };
    Stage8Parms {
        small_baseline_flag: text_from_mat(&mat, "small_baseline_flag", "n"),
        unwrap_method: text_from_mat(&mat, "unwrap_method", "3D"),
        unwrap_la_error_flag: text_from_mat(&mat, "unwrap_la_error_flag", "y"),
        unwrap_spatial_cost_func_flag: text_from_mat(&mat, "unwrap_spatial_cost_func_flag", "n"),
        drop_ifg_index: optional_vector_f64(&mat, "drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| (value > 0.0).then_some(value.round() as i64))
            .collect(),
        ref_lon: optional_vector_f64(&mat, "ref_lon")
            .unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
        ref_lat: optional_vector_f64(&mat, "ref_lat")
            .unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
        ref_radius: scalar_from_mat(&mat, "ref_radius", f64::INFINITY),
        max_topo_err: scalar_from_mat(&mat, "max_topo_err", Stage8Parms::default().max_topo_err),
        lambda_m: scalar_from_mat(&mat, "lambda", Stage8Parms::default().lambda_m),
        unwrap_time_win: scalar_from_mat(
            &mat,
            "unwrap_time_win",
            Stage8Parms::default().unwrap_time_win,
        ),
    }
}

fn scalar_from_mat(mat: &MatData, name: &str, default: f64) -> f64 {
    optional_vector_f64(mat, name)
        .and_then(|values| values.first().copied())
        .unwrap_or(default)
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
}

fn text_from_mat(mat: &MatData, name: &str, default: &str) -> String {
    let Some(values) = optional_vector_f64(mat, name) else {
        return default.to_string();
    };
    let text: String = values
        .into_iter()
        .filter_map(|value| {
            let code = value.round() as u32;
            (code != 0).then(|| char::from_u32(code)).flatten()
        })
        .collect::<String>()
        .trim()
        .to_string();
    if text.is_empty() {
        default.to_string()
    } else {
        text
    }
}

fn ps_vector_f64(
    mat: &MatData,
    name: &str,
    len: usize,
    label: &str,
) -> Result<Vec<f64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 8,
        message: format!("{label} is missing"),
    })?;
    if values.len() != len {
        return stage8_err(format!(
            "{label} has incompatible length {} for expected length {len}",
            values.len()
        ));
    }
    Ok(values)
}

fn ps_matrix_f32(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<Matrix<f32>, CoreError> {
    let source = mat
        .get_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
            stage: 8,
            message: format!("{label} is invalid: {err}"),
        })?;
    orient_matrix_f32(source, n_ps, label)
}

fn ps_dim_f64(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    n_dim: usize,
    label: &str,
) -> Result<Matrix<f64>, CoreError> {
    let source = mat
        .get_f64_matrix(name)
        .map_err(|err| CoreError::NativeStage {
            stage: 8,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f64(source));
    }
    stage8_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}, expected {n_ps}x{n_dim}",
        source.rows, source.cols
    ))
}

fn complex_ps_matrix(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat
        .get_complex_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
            stage: 8,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        return Ok(transpose_complex_f32(source));
    }
    stage8_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn orient_matrix_f32(
    source: Matrix<f32>,
    n_ps: usize,
    label: &str,
) -> Result<Matrix<f32>, CoreError> {
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        return Ok(transpose_f32(source));
    }
    stage8_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn transpose_f64(matrix: Matrix<f64>) -> Matrix<f64> {
    let mut values = vec![0.0; matrix.values.len()];
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            values[col * matrix.rows + row] = matrix.values[row * matrix.cols + col];
        }
    }
    Matrix {
        name: matrix.name,
        rows: matrix.cols,
        cols: matrix.rows,
        values,
    }
}

fn transpose_f32(matrix: Matrix<f32>) -> Matrix<f32> {
    let mut values = vec![0.0; matrix.values.len()];
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            values[col * matrix.rows + row] = matrix.values[row * matrix.cols + col];
        }
    }
    Matrix {
        name: matrix.name,
        rows: matrix.cols,
        cols: matrix.rows,
        values,
    }
}

fn transpose_complex_f32(matrix: ComplexMatrixF32) -> ComplexMatrixF32 {
    let mut values = vec![(0.0, 0.0); matrix.values.len()];
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            values[col * matrix.rows + row] = matrix.values[row * matrix.cols + col];
        }
    }
    ComplexMatrixF32 {
        name: matrix.name,
        rows: matrix.cols,
        cols: matrix.rows,
        values,
    }
}

fn stage8_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage8_err_owned(message.into()))
}

fn stage8_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 8, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;

    #[test]
    fn synthetic_stage8_matches_python_space_time_reference() {
        let root = temp_root("stage8-edge");
        let rust_root = root.join("rust");
        create_stage8_fixture(&rust_root, 3);

        run_stage8_native(&rust_root).unwrap();
        let observed = MatData::read_selected(
            rust_root.join("uw_space_time.mat"),
            &["dph_noise", "dph_space_uw"],
        )
        .unwrap();
        assert_matrix_close(
            &observed.get_f32_matrix("dph_noise").unwrap().values,
            &[
                -0.043821227,
                0.15751110,
                -0.10184394,
                -0.043821212,
                0.15751106,
                -0.10184391,
                -0.043821238,
                0.15751114,
                -0.10184396,
            ],
            1e-5,
        );
        assert_matrix_close(
            &observed.get_f32_matrix("dph_space_uw").unwrap().values,
            &[
                0.37000006, 0.38099992, 0.39200020, 0.37000006, 0.38099986, 0.39200020, 0.37000006,
                0.38100016, 0.39199996,
            ],
            1e-5,
        );

        let specs = vec![ArtifactComparisonSpec::new(
            "uw_space_time.mat",
            ["dph_noise", "dph_space_uw"],
        )];
        let summary = compare_fixture_artifacts(
            8,
            "merged",
            "synthetic_stage8_edge_graph",
            &rust_root,
            &rust_root,
            &specs,
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 8 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_edge_orientation_returns_structured_stage8_error() {
        let root = temp_root("stage8-bad-edges");
        create_stage8_fixture(&root, 3);
        let mut bad = MatFile::new(root.join("uw_interp.mat"));
        bad.add_f64_matrix("edgs", 3, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
            .unwrap();
        bad.write().unwrap();

        let err = run_stage8_native(&root).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 8);
                assert!(message.contains("uw_interp.edgs"));
                assert!(message.contains("Nx3 edge table"));
            }
            other => panic!("expected structured Stage 8 error, got {other:?}"),
        }
        assert!(!root.join("mean_v.mat").exists());
        assert!(!root.join("uw_space_time.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    fn create_stage8_fixture(root: &Path, edge_count: usize) {
        fs::create_dir_all(root).unwrap();
        let mut parms = MatFile::new(root.join("parms.mat"));
        parms
            .add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        parms
            .add_u32_matrix("unwrap_method", 1, 2, vec!['3' as u32, 'D' as u32])
            .unwrap();
        parms
            .add_u32_matrix("unwrap_la_error_flag", 1, 1, vec!['y' as u32])
            .unwrap();
        parms
            .add_u32_matrix("unwrap_spatial_cost_func_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        parms
            .add_f64_matrix("drop_ifg_index", 0, 0, Vec::new())
            .unwrap();
        parms
            .add_f64_scalar("ref_radius", f64::NEG_INFINITY)
            .unwrap();
        parms.write().unwrap();

        let n_ps = 4;
        let n_ifg = 4;
        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps2.add_f64_scalar("n_ifg", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("n_image", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("master_ix", 2.0).unwrap();
        ps2.add_f64_scalar("master_day", 20.0).unwrap();
        ps2.add_f64_row_vector("day", vec![10.0, 20.0, 30.0, 40.0])
            .unwrap();
        ps2.add_f64_row_vector("bperp", vec![-3.0, 0.0, 7.0, 14.0])
            .unwrap();
        ps2.add_f64_matrix(
            "lonlat",
            n_ps,
            2,
            vec![-118.0, 34.0, -117.9, 34.1, -117.8, 34.2, -117.7, 34.3],
        )
        .unwrap();
        ps2.add_f32_matrix(
            "xy",
            n_ps,
            3,
            vec![
                1.0, 0.0, 0.0, 2.0, 100.0, 0.0, 3.0, 0.0, 100.0, 4.0, 100.0, 100.0,
            ],
        )
        .unwrap();
        ps2.write().unwrap();

        let ph_uw = vec![
            1.0, 0.0, 4.0, 8.0, //
            -1.0, 0.0, 3.0, 9.0, //
            2.0, 0.0, 7.0, 12.0, //
            0.5, 0.0, 5.0, 10.0,
        ];
        let mut phuw2 = MatFile::new(root.join("phuw2.mat"));
        phuw2.add_f32_matrix("ph_uw", n_ps, n_ifg, ph_uw).unwrap();
        phuw2.add_f32_col_vector("msd", vec![0.0; n_ifg]).unwrap();
        phuw2.write().unwrap();

        let mut scla2 = MatFile::new(root.join("scla2.mat"));
        scla2
            .add_f32_matrix("ph_scla", n_ps, n_ifg, vec![0.0; n_ps * n_ifg])
            .unwrap();
        scla2.write().unwrap();

        let mut ifgstd = MatFile::new(root.join("ifgstd2.mat"));
        ifgstd
            .add_f64_col_vector("ifg_std", vec![1.0, 1.5, 2.0, 2.5])
            .unwrap();
        ifgstd.write().unwrap();

        let n_grid = 5;
        let n_unwrap_ifg = n_ifg - 1;
        let mut ph = Vec::with_capacity(n_grid * n_unwrap_ifg);
        for row in 0..n_grid {
            for col in 0..n_unwrap_ifg {
                let angle = row as f32 * 0.37 + col as f32 * 0.23 + (row * col) as f32 * 0.011;
                ph.push((angle.cos(), angle.sin()));
            }
        }
        let mut uw_grid = MatFile::new(root.join("uw_grid.mat"));
        uw_grid.add_f64_scalar("n_ps", n_grid as f64).unwrap();
        uw_grid
            .add_complex_f32_matrix("ph", n_grid, n_unwrap_ifg, ph)
            .unwrap();
        uw_grid.write().unwrap();

        let mut edgs = Vec::with_capacity(edge_count * 3);
        for edge_ix in 0..edge_count {
            let a = edge_ix % n_grid + 1;
            let b = (edge_ix + 1) % n_grid + 1;
            edgs.push(edge_ix as f64 + 1.0);
            edgs.push(a as f64);
            edgs.push(b as f64);
        }
        let mut uw_interp = MatFile::new(root.join("uw_interp.mat"));
        uw_interp
            .add_f64_matrix("edgs", edge_count, 3, edgs)
            .unwrap();
        uw_interp.write().unwrap();
    }

    fn assert_matrix_close(observed: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(observed.len(), expected.len());
        for (ix, (&left, &right)) in observed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "value {ix} mismatch: observed={left} expected={right}"
            );
        }
    }

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        root
    }
}
