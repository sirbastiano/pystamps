use crate::CoreError;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_COH_START: f64 = 0.005;
const DEFAULT_COH_STEP: f64 = 0.01;
const DEFAULT_COH_COUNT: usize = 100;
const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const HDF5_SIGNATURE_SCAN_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
struct Stage3Parms {
    select_method: String,
    percent_rand: f64,
    density_rand: f64,
    small_baseline_flag: String,
    drop_ifg_index: Vec<i64>,
    gamma_stdev_reject: f64,
}

impl Default for Stage3Parms {
    fn default() -> Self {
        Self {
            select_method: "PERCENT".to_string(),
            percent_rand: 1.0,
            density_rand: 1.0,
            small_baseline_flag: "n".to_string(),
            drop_ifg_index: Vec::new(),
            gamma_stdev_reject: 0.0,
        }
    }
}

pub fn run_stage3_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let pm = Stage3MatSource::read(patch_dir.join("pm1.mat"));
    let ps = Stage3MatSource::read(patch_dir.join("ps1.mat"));
    let parms = load_stage3_parms(patch_dir);

    let n_ps = ps.scalar("n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage3_err("ps1.mat missing valid n_ps");
    }

    let coh_ps = pm.ps_vector_f64("coh_ps", n_ps, "pm1.coh_ps")?;
    let mut coh_bins = pm.vector_f64("coh_bins").unwrap_or_default();
    if coh_bins.is_empty() {
        coh_bins = (0..DEFAULT_COH_COUNT)
            .map(|ix| DEFAULT_COH_START + DEFAULT_COH_STEP * ix as f64)
            .collect();
    }
    let mut nr_dist = pm.vector_f64("Nr").unwrap_or_default();
    if nr_dist.is_empty() {
        nr_dist = vec![1.0; coh_bins.len()];
    }

    let mut d_a = load_da(patch_dir, n_ps)?;
    let d_a_max = if d_a.len() >= 10_000 {
        da_bin_edges(&d_a)
    } else {
        d_a = vec![1.0; n_ps];
        vec![0.0, 1.0]
    };
    if d_a.len() != n_ps {
        return stage3_err(format!(
            "da1.D_A has incompatible length {} for n_ps={n_ps}",
            d_a.len()
        ));
    }

    let low_coh_thresh = if parms.small_baseline_flag.eq_ignore_ascii_case("y") {
        15
    } else {
        31
    };
    let method = parms.select_method.to_ascii_uppercase();
    let max_percent_rand = if method == "PERCENT" {
        parms.percent_rand
    } else {
        let xy = ps.ps_dim_f64("xy", n_ps, 3, "ps1.xy")?;
        let patch_area = patch_area_square_km(&xy);
        parms.density_rand * patch_area / (d_a_max.len().saturating_sub(1).max(1) as f64)
    };

    let (coh_thresh_all, coh_thresh_coeffs) = coh_threshold_from_dist(
        &coh_ps,
        &d_a,
        &d_a_max,
        &coh_bins,
        &nr_dist,
        low_coh_thresh,
        max_percent_rand,
        &method,
    );

    let mut ix0 = selected_indices_from_thresholds(&coh_ps, &coh_thresh_all);

    let ph_patch = pm.ps_complex_matrix("ph_patch", n_ps, "pm1.ph_patch")?;
    let ph_res = pm.ps_matrix_f32("ph_res", n_ps, "pm1.ph_res")?;
    let k_ps = pm.ps_vector_f64("K_ps", n_ps, "pm1.K_ps")?;
    let c_ps = pm.ps_vector_f64("C_ps", n_ps, "pm1.C_ps")?;

    if parms.gamma_stdev_reject > 0.0 && !ix0.is_empty() {
        let ifg_index_ix: Vec<usize> = ifg_index_for_selection(&ps, &parms)?
            .into_iter()
            .filter_map(|value| usize::try_from((value - 1.0) as i64).ok())
            .filter(|&ix| ix < ph_res.cols)
            .collect();
        if !ifg_index_ix.is_empty() {
            ix0.retain(|&row| {
                let mean = ifg_index_ix
                    .iter()
                    .map(|&col| ph_res.values[row * ph_res.cols + col] as f64)
                    .sum::<f64>()
                    / ifg_index_ix.len() as f64;
                let variance = ifg_index_ix
                    .iter()
                    .map(|&col| {
                        let diff = ph_res.values[row * ph_res.cols + col] as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / ifg_index_ix.len() as f64;
                variance.sqrt() < parms.gamma_stdev_reject
            });
        }
    }

    write_select_artifact(
        patch_dir.join("select1.mat"),
        &ix0,
        &ph_patch,
        &ph_res,
        &k_ps,
        &c_ps,
        &coh_ps,
        &coh_thresh_all,
        &coh_thresh_coeffs,
        max_percent_rand,
        parms.gamma_stdev_reject,
        &parms.small_baseline_flag,
        &ifg_index_for_selection(&ps, &parms)?,
    )?;

    Ok(format!("Stage 3 selected {} PS", ix0.len()))
}

fn load_stage3_parms(patch_dir: &Path) -> Stage3Parms {
    let Some(path) = resolve_file_optional(patch_dir, "parms.mat") else {
        return Stage3Parms::default();
    };
    let source = Stage3MatSource::read(path);
    Stage3Parms {
        select_method: source.text("select_method", "PERCENT"),
        percent_rand: source.scalar("percent_rand", 1.0),
        density_rand: source.scalar("density_rand", 1.0),
        small_baseline_flag: source.text("small_baseline_flag", "n"),
        drop_ifg_index: source
            .vector_f64("drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter(|value| value.is_finite())
            .map(|value| value.round() as i64)
            .collect(),
        gamma_stdev_reject: source.scalar("gamma_stdev_reject", 0.0),
    }
}

#[derive(Debug)]
struct Stage3MatSource {
    path: PathBuf,
    mat: Option<MatData>,
}

impl Stage3MatSource {
    fn read(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let mat = MatData::read(&path).ok();
        Self { path, mat }
    }

    fn scalar(&self, name: &str, default: f64) -> f64 {
        self.vector_f64(name)
            .and_then(|values| values.into_iter().next())
            .unwrap_or(default)
    }

    fn text(&self, name: &str, default: &str) -> String {
        let value = self
            .mat
            .as_ref()
            .and_then(|mat| text_from_mat_opt(mat, name))
            .or_else(|| read_hdf5_text(&self.path, name).ok());
        match value {
            Some(text) if !text.is_empty() => text,
            _ => default.to_string(),
        }
    }

    fn vector_f64(&self, name: &str) -> Option<Vec<f64>> {
        self.mat
            .as_ref()
            .and_then(|mat| optional_vector_f64(mat, name))
            .or_else(|| {
                read_hdf5_matrix_f64(&self.path, name)
                    .ok()
                    .map(|matrix| matrix.values)
            })
    }

    fn ps_vector_f64(&self, name: &str, n_ps: usize, label: &str) -> Result<Vec<f64>, CoreError> {
        let values = self
            .vector_f64(name)
            .ok_or_else(|| CoreError::NativeStage {
                stage: 3,
                message: format!("{label} is missing"),
            })?;
        if values.len() != n_ps {
            return stage3_err(format!(
                "{label} has incompatible length {} for n_ps={n_ps}",
                values.len()
            ));
        }
        Ok(values)
    }

    fn ps_matrix_f32(
        &self,
        name: &str,
        n_ps: usize,
        label: &str,
    ) -> Result<Matrix<f32>, CoreError> {
        if let Some(mat) = &self.mat {
            if let Ok(matrix) = ps_matrix_f32(mat, name, n_ps, label) {
                return Ok(matrix);
            }
        }
        if let Ok(matrix) = read_hdf5_matrix_f32(&self.path, name) {
            return orient_matrix_f32(matrix, n_ps, label);
        }
        stage3_err(format!("{label} is missing or invalid"))
    }

    fn ps_complex_matrix(
        &self,
        name: &str,
        n_ps: usize,
        label: &str,
    ) -> Result<ComplexMatrixF32, CoreError> {
        if let Some(mat) = &self.mat {
            if let Ok(matrix) = ps_complex_matrix(mat, name, n_ps, label) {
                return Ok(matrix);
            }
        }
        if let Ok(matrix) = read_hdf5_complex_matrix_f32(&self.path, name) {
            return orient_complex_matrix_f32(matrix, n_ps, label);
        }
        stage3_err(format!("{label} is missing or invalid"))
    }

    fn ps_dim_f64(
        &self,
        name: &str,
        n_ps: usize,
        n_dim: usize,
        label: &str,
    ) -> Result<Matrix<f64>, CoreError> {
        if let Some(mat) = &self.mat {
            if let Ok(matrix) = ps_dim_f64_from_matrix(mat, name, n_ps, n_dim) {
                return Ok(matrix);
            }
        }
        if let Ok(matrix) = read_hdf5_matrix_f64(&self.path, name) {
            return orient_ps_dim_f64(matrix, name, n_ps, n_dim, label);
        }
        stage3_err(format!(
            "{label} is missing or has incompatible shape; expected {n_ps}x{n_dim}"
        ))
    }
}

fn load_da(patch_dir: &Path, n_ps: usize) -> Result<Vec<f64>, CoreError> {
    if patch_dir.join("da1.mat").exists() {
        let da = Stage3MatSource::read(patch_dir.join("da1.mat"));
        da.vector_f64("D_A").ok_or_else(|| CoreError::NativeStage {
            stage: 3,
            message: "da1.mat missing D_A".to_string(),
        })
    } else {
        Ok(vec![1.0; n_ps])
    }
}

fn da_bin_edges(d_a: &[f64]) -> Vec<f64> {
    let mut sorted = d_a.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let bin_size = if d_a.len() >= 50_000 { 10_000 } else { 2_000 };
    let mut edges = Vec::new();
    edges.push(0.0);
    let last_interior = d_a.len().saturating_sub(bin_size);
    let mut one_based_ix = bin_size;
    while one_based_ix <= last_interior {
        edges.push(sorted[one_based_ix - 1]);
        one_based_ix += bin_size;
    }
    edges.push(*sorted.last().unwrap_or(&1.0));
    edges
}

fn coh_threshold_from_dist(
    coh_values: &[f64],
    d_a: &[f64],
    d_a_max: &[f64],
    coh_bins: &[f64],
    nr_dist: &[f64],
    low_coh_thresh: usize,
    max_percent_rand: f64,
    select_method: &str,
) -> (Vec<f64>, Vec<f64>) {
    let bin_count = d_a_max.len().saturating_sub(1);
    let mut min_coh = vec![f64::NAN; bin_count];
    let mut d_a_mean = vec![f64::NAN; bin_count];

    for i in 0..bin_count {
        let selected: Vec<usize> = d_a
            .iter()
            .enumerate()
            .filter_map(|(ix, &value)| {
                (value > d_a_max[i] && value <= d_a_max[i + 1]).then_some(ix)
            })
            .collect();
        if selected.is_empty() {
            continue;
        }
        let coh_chunk: Vec<f64> = selected
            .iter()
            .filter_map(|&ix| {
                let value = coh_values[ix];
                (value.is_finite() && value != 0.0).then_some(value)
            })
            .collect();
        if coh_chunk.is_empty() {
            continue;
        }
        d_a_mean[i] = selected.iter().map(|&ix| d_a[ix]).sum::<f64>() / selected.len() as f64;
        let na = hist_with_centers(&coh_chunk, coh_bins);
        let low_cut = low_coh_thresh.min(na.len()).min(nr_dist.len());
        let denom = nr_dist.iter().take(low_cut).sum::<f64>();
        let scale = if denom > 0.0 {
            na.iter().take(low_cut).sum::<f64>() / denom
        } else {
            1.0
        };
        let nr: Vec<f64> = nr_dist.iter().map(|value| value * scale).collect();
        let mut na_safe = na.clone();
        for value in &mut na_safe {
            if *value == 0.0 {
                *value = 1.0;
            }
        }

        let mut nr_cum = vec![0.0; nr.len()];
        let mut na_cum = vec![0.0; na_safe.len()];
        let mut nr_sum = 0.0;
        let mut na_sum = 0.0;
        for ix in (0..nr.len()).rev() {
            nr_sum += nr[ix];
            nr_cum[ix] = nr_sum;
        }
        for ix in (0..na_safe.len()).rev() {
            na_sum += na_safe[ix];
            na_cum[ix] = na_sum;
        }
        let percent_rand: Vec<f64> = if select_method == "PERCENT" {
            nr_cum
                .iter()
                .zip(na_cum.iter())
                .map(|(&nr_value, &na_value)| nr_value / na_value * 100.0)
                .collect()
        } else {
            nr_cum
        };
        let Some(min_ok) = percent_rand
            .iter()
            .position(|&value| value < max_percent_rand)
        else {
            min_coh[i] = 1.0;
            continue;
        };
        let min_ok_1b = min_ok + 1;
        let min_fit_ix = min_ok_1b as isize - 3;
        if min_fit_ix <= 0 {
            continue;
        }
        let max_fit_ix = (min_ok_1b + 2).min(100);
        let start = (min_fit_ix - 1) as usize;
        let end = max_fit_ix.min(percent_rand.len());
        let xs = &percent_rand[start..end];
        if xs.len() < 4 {
            continue;
        }
        let ys: Vec<f64> = ((min_fit_ix as usize)..=(start + xs.len()))
            .map(|value| value as f64 * 0.01)
            .collect();
        min_coh[i] = polyfit_eval_centered(xs, &ys, 3, max_percent_rand);
    }

    let valid: Vec<usize> = min_coh
        .iter()
        .zip(d_a_mean.iter())
        .enumerate()
        .filter_map(|(ix, (&coh, &mean))| (!coh.is_nan() && !mean.is_nan()).then_some(ix))
        .collect();
    let (mut threshold, coeffs) = if valid.is_empty() {
        (vec![0.3; coh_values.len()], Vec::new())
    } else if valid.len() == 1 {
        (vec![min_coh[valid[0]]; coh_values.len()], Vec::new())
    } else {
        let xs: Vec<f64> = valid.iter().map(|&ix| d_a_mean[ix]).collect();
        let ys: Vec<f64> = valid.iter().map(|&ix| min_coh[ix]).collect();
        let (slope, intercept) = linear_fit(&xs, &ys);
        if slope > 0.0 {
            (
                d_a.iter().map(|&value| slope * value + intercept).collect(),
                vec![slope, intercept],
            )
        } else {
            (vec![slope * 0.35 + intercept; coh_values.len()], Vec::new())
        }
    };
    for value in &mut threshold {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    (threshold, coeffs)
}

fn selected_indices_from_thresholds(coh_ps: &[f64], coh_thresh_all: &[f64]) -> Vec<usize> {
    coh_ps
        .iter()
        .zip(coh_thresh_all.iter())
        .enumerate()
        .filter_map(|(ix, (&coh, &threshold))| (coh > threshold).then_some(ix))
        .collect()
}

fn hist_with_centers(values: &[f64], centers: &[f64]) -> Vec<f64> {
    if centers.is_empty() {
        return Vec::new();
    }
    if centers.len() == 1 {
        return vec![values.len() as f64];
    }
    let mids: Vec<f64> = centers
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect();
    let mut counts = vec![0.0; centers.len()];
    for &value in values {
        let ix = mids
            .partition_point(|&mid| mid < value)
            .min(centers.len() - 1);
        counts[ix] += 1.0;
    }
    counts
}

fn polyfit_eval_centered(x: &[f64], y: &[f64], degree: usize, x_eval: f64) -> f64 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let variance = if x.len() > 1 {
        x.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
    } else {
        1.0
    };
    let std = if variance.is_finite() && variance > 0.0 {
        variance.sqrt()
    } else {
        1.0
    };
    let scaled: Vec<f64> = x.iter().map(|value| (value - mean) / std).collect();
    let coeffs = least_squares_poly(&scaled, y, degree);
    let x0 = (x_eval - mean) / std;
    coeffs
        .iter()
        .enumerate()
        .map(|(power, &coeff)| coeff * x0.powi(power as i32))
        .sum()
}

fn least_squares_poly(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = degree + 1;
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];
    for (&x_value, &y_value) in x.iter().zip(y.iter()) {
        let mut powers = vec![1.0; n];
        for power in 1..n {
            powers[power] = powers[power - 1] * x_value;
        }
        for row in 0..n {
            aty[row] += powers[row] * y_value;
            for col in 0..n {
                ata[row][col] += powers[row] * powers[col];
            }
        }
    }
    solve_linear_system(ata, aty).unwrap_or_else(|| vec![f64::NAN; n])
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let max_row = (pivot..n)
            .max_by(|&left, &right| a[left][pivot].abs().total_cmp(&a[right][pivot].abs()))?;
        if a[max_row][pivot].abs() <= f64::EPSILON {
            return None;
        }
        a.swap(pivot, max_row);
        b.swap(pivot, max_row);
        let pivot_value = a[pivot][pivot];
        for col in pivot..n {
            a[pivot][col] /= pivot_value;
        }
        b[pivot] /= pivot_value;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for col in pivot..n {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let den = x.iter().map(|value| (value - mean_x).powi(2)).sum::<f64>();
    if den == 0.0 {
        return (0.0, mean_y);
    }
    let num = x
        .iter()
        .zip(y.iter())
        .map(|(&x_value, &y_value)| (x_value - mean_x) * (y_value - mean_y))
        .sum::<f64>();
    let slope = num / den;
    (slope, mean_y - slope * mean_x)
}

fn write_select_artifact(
    path: PathBuf,
    ix0: &[usize],
    ph_patch: &ComplexMatrixF32,
    ph_res: &Matrix<f32>,
    k_ps: &[f64],
    c_ps: &[f64],
    coh_ps: &[f64],
    coh_thresh_all: &[f64],
    coh_thresh_coeffs: &[f64],
    max_percent_rand: f64,
    gamma_stdev_reject: f64,
    small_baseline_flag: &str,
    ifg_index: &[f64],
) -> Result<(), CoreError> {
    let mut mat = MatFile::new(path);
    let ix_values: Vec<f64> = ix0.iter().map(|&ix| ix as f64 + 1.0).collect();
    mat.add_f64_col_vector("ix", ix_values.clone())?;
    mat.add_u8_matrix("keep_ix", ix0.len(), 1, vec![1; ix0.len()])?;

    let mut ph_patch2 = Vec::with_capacity(ix0.len() * ph_patch.cols);
    let mut ph_res2 = Vec::with_capacity(ix0.len() * ph_res.cols);
    let mut k_ps2 = Vec::with_capacity(ix0.len());
    let mut c_ps2 = Vec::with_capacity(ix0.len());
    let mut coh_ps2 = Vec::with_capacity(ix0.len());
    let mut coh_thresh = Vec::with_capacity(ix0.len());
    for &row in ix0 {
        ph_patch2
            .extend_from_slice(&ph_patch.values[row * ph_patch.cols..(row + 1) * ph_patch.cols]);
        ph_res2.extend_from_slice(&ph_res.values[row * ph_res.cols..(row + 1) * ph_res.cols]);
        k_ps2.push(k_ps[row]);
        c_ps2.push(c_ps[row]);
        coh_ps2.push(coh_ps[row]);
        coh_thresh.push(coh_thresh_all[row]);
    }
    mat.add_complex_f32_matrix("ph_patch2", ix0.len(), ph_patch.cols, ph_patch2)?;
    mat.add_f32_matrix("ph_res2", ix0.len(), ph_res.cols, ph_res2)?;
    mat.add_f64_col_vector("K_ps2", k_ps2)?;
    mat.add_f64_col_vector("C_ps2", c_ps2)?;
    mat.add_f64_col_vector("coh_ps2", coh_ps2)?;
    mat.add_f64_col_vector("coh_thresh", coh_thresh)?;
    if coh_thresh_coeffs.is_empty() {
        mat.add_f64_matrix("coh_thresh_coeffs", 0, 0, Vec::new())?;
    } else {
        mat.add_f64_row_vector("coh_thresh_coeffs", coh_thresh_coeffs.to_vec())?;
    }
    mat.add_f64_scalar("clap_alpha", 1.0)?;
    mat.add_f64_scalar("clap_beta", 0.3)?;
    mat.add_f64_scalar("n_win", 32.0)?;
    mat.add_f32_scalar("max_percent_rand", max_percent_rand as f32)?;
    mat.add_f64_scalar("gamma_stdev_reject", gamma_stdev_reject)?;
    let small_flag: Vec<u32> = small_baseline_flag.chars().map(|ch| ch as u32).collect();
    mat.add_u32_matrix("small_baseline_flag", 1, small_flag.len(), small_flag)?;
    mat.add_f64_row_vector("ifg_index", ifg_index.to_vec())?;
    mat.write()?;
    Ok(())
}

fn ifg_index_for_selection(
    ps: &Stage3MatSource,
    parms: &Stage3Parms,
) -> Result<Vec<f64>, CoreError> {
    let n_ifg = ps.scalar("n_ifg", 0.0).round() as i64;
    let drop: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    let mut ifg: Vec<i64> = (1..=n_ifg).filter(|value| !drop.contains(value)).collect();
    if !parms.small_baseline_flag.eq_ignore_ascii_case("y") {
        let master_ix = ps.scalar("master_ix", 1.0).round() as i64;
        ifg.retain(|&value| value != master_ix);
        for value in &mut ifg {
            if *value > master_ix {
                *value -= 1;
            }
        }
    }
    Ok(ifg.into_iter().map(|value| value as f64).collect())
}

fn patch_area_square_km(xy: &Matrix<f64>) -> f64 {
    if xy.rows == 0 || xy.cols < 3 {
        return 1.0;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for row in 0..xy.rows {
        let x = xy.values[row * xy.cols + 1];
        let y = xy.values[row * xy.cols + 2];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let area = (max_x - min_x) * (max_y - min_y) / 1e6;
    if area > 0.0 {
        area
    } else {
        1.0
    }
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
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
            stage: 3,
            message: format!("{label} is missing or invalid: {err}"),
        })?;
    orient_matrix_f32(source, n_ps, label)
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
            stage: 3,
            message: format!("{label} is missing or invalid: {err}"),
        })?;
    orient_complex_matrix_f32(source, n_ps, label)
}

fn orient_complex_matrix_f32(
    source: ComplexMatrixF32,
    n_ps: usize,
    label: &str,
) -> Result<ComplexMatrixF32, CoreError> {
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
    stage3_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn ps_dim_f64_from_matrix(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    n_dim: usize,
) -> Result<Matrix<f64>, ()> {
    let source = mat.get_f64_matrix(name).map_err(|_| ())?;
    orient_ps_dim_f64(source, name, n_ps, n_dim, "").map_err(|_| ())
}

fn orient_ps_dim_f64(
    source: Matrix<f64>,
    name: &str,
    n_ps: usize,
    n_dim: usize,
    label: &str,
) -> Result<Matrix<f64>, CoreError> {
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
            name: name.to_string(),
            rows: source.cols,
            cols: source.rows,
            values,
        });
    }
    stage3_err(format!(
        "{label} has incompatible shape {}x{}; expected {n_ps}x{n_dim}",
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
    stage3_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn text_from_mat_opt(mat: &MatData, name: &str) -> Option<String> {
    let Some(values) = optional_vector_f64(mat, name) else {
        return None;
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
        None
    } else {
        Some(text)
    }
}

fn read_hdf5_matrix_f64(path: &Path, variable: &str) -> Result<Matrix<f64>, String> {
    match read_hdf5_matrix_f64_direct(path, variable) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock(path, offset, |temp_path| {
                read_hdf5_matrix_f64_direct(temp_path, variable)
            })
            .map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_matrix_f64_direct(path: &Path, variable: &str) -> Result<Matrix<f64>, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset
        .read_raw::<f64>()
        .or_else(|_| {
            dataset
                .read_raw::<f32>()
                .map(|values| values.into_iter().map(f64::from).collect())
        })
        .map_err(|err| err.to_string())?;
    let shape = dataset.shape();
    let rows = shape.first().copied().unwrap_or(1);
    let cols = if shape.len() <= 1 {
        1
    } else {
        shape[1..].iter().copied().product()
    };
    Ok(Matrix {
        name: variable.to_string(),
        rows,
        cols,
        values,
    })
}

fn read_hdf5_matrix_f32(path: &Path, variable: &str) -> Result<Matrix<f32>, String> {
    match read_hdf5_matrix_f32_direct(path, variable) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock(path, offset, |temp_path| {
                read_hdf5_matrix_f32_direct(temp_path, variable)
            })
            .map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_matrix_f32_direct(path: &Path, variable: &str) -> Result<Matrix<f32>, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset
        .read_raw::<f32>()
        .or_else(|_| {
            dataset
                .read_raw::<f64>()
                .map(|values| values.into_iter().map(|value| value as f32).collect())
        })
        .map_err(|err| err.to_string())?;
    let (rows, cols) = hdf5_matrix_shape(&dataset);
    Ok(Matrix {
        name: variable.to_string(),
        rows,
        cols,
        values,
    })
}

fn read_hdf5_complex_matrix_f32(path: &Path, variable: &str) -> Result<ComplexMatrixF32, String> {
    match read_hdf5_complex_matrix_f32_direct(path, variable) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock(path, offset, |temp_path| {
                read_hdf5_complex_matrix_f32_direct(temp_path, variable)
            })
            .map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_complex_matrix_f32_direct(
    path: &Path,
    variable: &str,
) -> Result<ComplexMatrixF32, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset
        .read_raw::<rust_hdf5::Complex32>()
        .map_err(|err| err.to_string())?
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    let (rows, cols) = hdf5_matrix_shape(&dataset);
    Ok(ComplexMatrixF32 {
        name: variable.to_string(),
        rows,
        cols,
        values,
    })
}

fn hdf5_matrix_shape(dataset: &rust_hdf5::H5Dataset) -> (usize, usize) {
    let shape = dataset.shape();
    let rows = shape.first().copied().unwrap_or(1);
    let cols = if shape.len() <= 1 {
        1
    } else {
        shape[1..].iter().copied().product()
    };
    (rows, cols)
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
        "pystamps-stage3-hdf5-{}-{}.h5",
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

fn stage3_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage3_err_owned(message.into()))
}

fn stage3_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 3, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;
    use std::io::Write;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_percent_stage3_matches_python_reference_and_is_faster() {
        let root = temp_root("stage3-percent");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage3_fixture(&python_root, "PERCENT", 8.0, 1.0);
        create_stage3_fixture(&rust_root, "PERCENT", 8.0, 1.0);

        let python_start = Instant::now();
        run_python_stage3(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage3_native(rust_root.join("PATCH_1")).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let summary = compare_fixture_artifacts(
            3,
            "patch",
            "synthetic_stage3_percent",
            &python_root,
            &rust_root,
            &[ArtifactComparisonSpec::new(
                "PATCH_1/select1.mat",
                [
                    "ix",
                    "keep_ix",
                    "ph_patch2",
                    "ph_res2",
                    "K_ps2",
                    "C_ps2",
                    "coh_ps2",
                    "coh_thresh",
                    "coh_thresh_coeffs",
                    "max_percent_rand",
                    "small_baseline_flag",
                    "ifg_index",
                ],
            )],
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 3 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 3 should beat Python path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn synthetic_density_stage3_matches_python_thresholds() {
        let root = temp_root("stage3-density");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage3_fixture(&python_root, "DENSITY", 1.0, 25_000.0);
        create_stage3_fixture(&rust_root, "DENSITY", 1.0, 25_000.0);

        run_python_stage3(&python_root);
        run_stage3_native(rust_root.join("PATCH_1")).unwrap();
        let summary = compare_fixture_artifacts(
            3,
            "patch",
            "synthetic_stage3_density",
            &python_root,
            &rust_root,
            &[ArtifactComparisonSpec::new(
                "PATCH_1/select1.mat",
                ["ix", "coh_thresh", "coh_thresh_coeffs", "max_percent_rand"],
            )],
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 3 density parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_coh_ps_returns_structured_stage3_error() {
        let root = temp_root("stage3-missing-coh");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        write_ps1(&patch, 2);
        let mut pm = MatFile::new(patch.join("pm1.mat"));
        pm.add_f64_row_vector("coh_bins", default_bins()).unwrap();
        pm.write().unwrap();

        let err = run_stage3_native(&patch).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 3);
                assert!(message.contains("pm1.coh_ps"));
            }
            other => panic!("expected structured Stage 3 error, got {other:?}"),
        }
        assert!(!patch.join("select1.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selection_rejects_threshold_ties_and_nan_candidates() {
        let selected = selected_indices_from_thresholds(
            &[0.20, 0.30, f64::NAN, 0.5001],
            &[0.20, 0.30, 0.10, 0.50],
        );
        assert_eq!(selected, vec![3]);
    }

    #[test]
    fn da_bin_edges_follow_matlab_one_based_interior_indices() {
        let edges = da_bin_edges(&(1..=50_000).map(|value| value as f64).collect::<Vec<_>>());
        assert_eq!(
            edges,
            vec![0.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0]
        );
    }

    #[test]
    fn reads_hdf5_ps1_and_parms_for_density_selection() {
        let root = temp_root("stage3-hdf5-parms");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        write_ps1_hdf5(&patch, 8);
        write_parms_hdf5(&patch);
        write_pm1(&patch, 8);

        run_stage3_native(&patch).unwrap();

        let select = MatData::read(patch.join("select1.mat")).unwrap();
        let max_percent_rand = select.get_f64_matrix("max_percent_rand").unwrap().values[0];
        let small_baseline_flag = select.get_f64_matrix("small_baseline_flag").unwrap().values;
        assert!((max_percent_rand - 1.5).abs() < f64::from(f32::EPSILON));
        assert_eq!(small_baseline_flag, vec!['n' as u32 as f64]);
        fs::remove_dir_all(root).unwrap();
    }

    fn create_stage3_fixture(
        root: &Path,
        select_method: &str,
        percent_rand: f64,
        density_rand: f64,
    ) {
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        let n_ps = 240;
        write_parms(&patch, select_method, percent_rand, density_rand);
        write_ps1(&patch, n_ps);
        write_da(&patch, n_ps);
        write_pm1(&patch, n_ps);
    }

    fn write_parms(patch: &Path, select_method: &str, percent_rand: f64, density_rand: f64) {
        let mut mat = MatFile::new(patch.join("parms.mat"));
        mat.add_u32_matrix(
            "select_method",
            1,
            select_method.len(),
            select_method.chars().map(|ch| ch as u32).collect(),
        )
        .unwrap();
        mat.add_f64_scalar("percent_rand", percent_rand).unwrap();
        mat.add_f64_scalar("density_rand", density_rand).unwrap();
        mat.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_f64_scalar("gamma_stdev_reject", 0.0).unwrap();
        mat.write().unwrap();
    }

    fn write_ps1(patch: &Path, n_ps: usize) {
        let mut xy = Vec::with_capacity(n_ps * 3);
        for ix in 0..n_ps {
            xy.push(ix as f64 + 1.0);
            xy.push((ix % 40) as f64 * 100.0);
            xy.push((ix / 40) as f64 * 100.0);
        }
        let mut mat = MatFile::new(patch.join("ps1.mat"));
        mat.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        mat.add_f64_scalar("n_ifg", 4.0).unwrap();
        mat.add_f64_scalar("master_ix", 1.0).unwrap();
        mat.add_f64_row_vector("bperp", vec![0.0, 10.0, 20.0, 30.0, 40.0])
            .unwrap();
        mat.add_f64_matrix("xy", n_ps, 3, xy).unwrap();
        mat.write().unwrap();
    }

    fn write_ps1_hdf5(patch: &Path, n_ps: usize) {
        let raw_hdf5 = patch.join("ps1-raw.h5");
        let h5 = rust_hdf5::H5File::create(&raw_hdf5).unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("n_ps")
            .unwrap()
            .write_raw(&[n_ps as f64])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("n_ifg")
            .unwrap()
            .write_raw(&[4.0_f64])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("master_ix")
            .unwrap()
            .write_raw(&[1.0_f64])
            .unwrap();
        let mut xy = Vec::with_capacity(3 * n_ps);
        for dim in 0..3 {
            for ix in 0..n_ps {
                let value = match dim {
                    0 => ix as f32 + 1.0,
                    1 => (ix % 2) as f32 * 1_000.0,
                    _ => (ix / 2) as f32 * 1_000.0,
                };
                xy.push(value);
            }
        }
        h5.new_dataset::<f32>()
            .shape([3, n_ps])
            .create("xy")
            .unwrap()
            .write_raw(&xy)
            .unwrap();
        h5.close().unwrap();
        write_matlab_hdf5_with_userblock(&raw_hdf5, &patch.join("ps1.mat"));
    }

    fn write_parms_hdf5(patch: &Path) {
        let raw_hdf5 = patch.join("parms-raw.h5");
        let h5 = rust_hdf5::H5File::create(&raw_hdf5).unwrap();
        h5.new_dataset::<u16>()
            .shape([7, 1])
            .create("select_method")
            .unwrap()
            .write_raw(&"DENSITY".chars().map(|ch| ch as u16).collect::<Vec<_>>())
            .unwrap();
        h5.new_dataset::<u16>()
            .shape([1, 1])
            .create("small_baseline_flag")
            .unwrap()
            .write_raw(&['n' as u16])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("percent_rand")
            .unwrap()
            .write_raw(&[1.0_f64])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("density_rand")
            .unwrap()
            .write_raw(&[0.5_f64])
            .unwrap();
        h5.new_dataset::<f64>()
            .shape([1, 1])
            .create("gamma_stdev_reject")
            .unwrap()
            .write_raw(&[0.0_f64])
            .unwrap();
        h5.close().unwrap();
        write_matlab_hdf5_with_userblock(&raw_hdf5, &patch.join("parms.mat"));
    }

    fn write_matlab_hdf5_with_userblock(raw_hdf5: &Path, matlab_path: &Path) {
        let mut matlab_hdf5 = fs::File::create(matlab_path).unwrap();
        matlab_hdf5.write_all(&vec![b' '; 512]).unwrap();
        matlab_hdf5.write_all(&fs::read(raw_hdf5).unwrap()).unwrap();
        fs::remove_file(raw_hdf5).unwrap();
    }

    fn write_da(patch: &Path, n_ps: usize) {
        let mut mat = MatFile::new(patch.join("da1.mat"));
        mat.add_f64_row_vector(
            "D_A",
            (0..n_ps).map(|ix| 0.2 + ix as f64 / n_ps as f64).collect(),
        )
        .unwrap();
        mat.write().unwrap();
    }

    fn write_pm1(patch: &Path, n_ps: usize) {
        let coh_bins = default_bins();
        let mut coh_ps = Vec::with_capacity(n_ps);
        let mut ph_patch = Vec::with_capacity(n_ps * 4);
        let mut ph_res = Vec::with_capacity(n_ps * 4);
        let mut k_ps = Vec::with_capacity(n_ps);
        let mut c_ps = Vec::with_capacity(n_ps);
        for row in 0..n_ps {
            coh_ps.push(0.08 + 0.9 * ((row * 37 % n_ps) as f64 / n_ps as f64));
            k_ps.push(row as f64 * 0.001);
            c_ps.push(row as f64 * 0.002);
            for col in 0..4 {
                ph_patch.push((1.0_f32, 0.0_f32));
                ph_res.push((row as f32 + col as f32) * 0.001);
            }
        }
        let mut mat = MatFile::new(patch.join("pm1.mat"));
        mat.add_f64_row_vector("coh_ps", coh_ps).unwrap();
        mat.add_f64_row_vector("coh_bins", coh_bins.clone())
            .unwrap();
        mat.add_f64_row_vector("Nr", vec![1.0; coh_bins.len()])
            .unwrap();
        mat.add_complex_f32_matrix("ph_patch", n_ps, 4, ph_patch)
            .unwrap();
        mat.add_f32_matrix("ph_res", n_ps, 4, ph_res).unwrap();
        mat.add_f64_row_vector("K_ps", k_ps).unwrap();
        mat.add_f64_row_vector("C_ps", c_ps).unwrap();
        mat.write().unwrap();
    }

    fn default_bins() -> Vec<f64> {
        (0..DEFAULT_COH_COUNT)
            .map(|ix| DEFAULT_COH_START + DEFAULT_COH_STEP * ix as f64)
            .collect()
    }

    fn run_python_stage3(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage3_select_ps; stage3_select_ps(Path(sys.argv[1]) / 'PATCH_1')";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage3 failed: {}",
            String::from_utf8_lossy(&output.stderr)
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
