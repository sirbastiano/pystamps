use crate::CoreError;
use pystamps_mat::{MatData, MatFile, Matrix};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug)]
struct Stage7Parms {
    small_baseline_flag: String,
    scla_deramp: String,
    drop_ifg_index: Vec<i64>,
    scla_drop_index: Vec<i64>,
    ref_lon: Vec<f64>,
    ref_lat: Vec<f64>,
    ref_radius: f64,
}

impl Default for Stage7Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            scla_deramp: "y".to_string(),
            drop_ifg_index: Vec::new(),
            scla_drop_index: Vec::new(),
            ref_lon: vec![f64::NEG_INFINITY, f64::INFINITY],
            ref_lat: vec![f64::NEG_INFINITY, f64::INFINITY],
            ref_radius: f64::INFINITY,
        }
    }
}

#[derive(Clone, Debug)]
struct Stage7KernelOutput {
    k_ps_uw: Vec<f64>,
    c_ps_uw: Vec<f32>,
    ph_scla: Matrix<f32>,
    ph_ramp: Matrix<f64>,
    ifg_vcm: Matrix<f64>,
}

pub fn run_stage7_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let ps2 = read_mat_stage7(dataset_root, "ps2.mat")?;
    if !dataset_root.join("phuw2.mat").exists() {
        return stage7_err(
            "Missing required artifact: phuw2.mat (stage-6 unwrap output) before stage 7",
        );
    }
    let phuw = read_mat_stage7(dataset_root, "phuw2.mat")?;
    let ifgstd = read_mat_stage7(dataset_root, "ifgstd2.mat")?;
    let parms = load_stage7_parms(dataset_root);

    let n_ps = scalar_from_mat(&ps2, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage7_err("ps2.mat missing valid n_ps");
    }
    let ph_uw = ps_matrix_f32(&phuw, "ph_uw", n_ps, "phuw2.ph_uw")?;
    let n_ifg = ph_uw.cols;
    if n_ifg == 0 {
        return stage7_err("phuw2.ph_uw must contain at least one interferogram");
    }
    let master_ix = scalar_from_mat(&ps2, "master_ix", 1.0).round() as usize;
    if master_ix == 0 || master_ix > n_ifg {
        return stage7_err(format!(
            "ps2.master_ix must be 1-based within phuw2.ph_uw columns; got {master_ix}"
        ));
    }
    let day = ps_vector_f64(&ps2, "day", n_ifg, "ps2.day")?;
    let ifg_std = ps_vector_f64(&ifgstd, "ifg_std", n_ifg, "ifgstd2.ifg_std")?;
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    let bperp_mat =
        load_or_rebuild_bperp(dataset_root, &ps2, n_ps, n_ifg, master_ix, small_baseline)?;

    let ph_raw = Matrix {
        name: "ph_raw".to_string(),
        rows: ph_uw.rows,
        cols: ph_uw.cols,
        values: ph_uw.values.iter().map(|&value| value as f64).collect(),
    };
    let (ph_deramped, ph_ramp) = if parms.scla_deramp.eq_ignore_ascii_case("y") {
        deramp_unwrapped_phase(&ps2, &ph_raw)?
    } else {
        (
            ph_raw.clone(),
            Matrix {
                name: "ph_ramp".to_string(),
                rows: 0,
                cols: 0,
                values: Vec::new(),
            },
        )
    };
    let ref_ix = select_reference_ps(&ps2, &parms, n_ps)?;
    let ph_proc = center_to_reference(&ph_deramped, &ref_ix);
    let ph_mean_v = center_to_reference(&ph_raw, &ref_ix);

    let drop_set: BTreeSet<i64> = parms
        .drop_ifg_index
        .iter()
        .chain(parms.scla_drop_index.iter())
        .copied()
        .collect();
    let (unwrap_ifg, solve_ifg) = if small_baseline {
        let unwrap: Vec<usize> = (1..=n_ifg)
            .filter(|ix| !drop_set.contains(&(*ix as i64)))
            .map(|ix| ix - 1)
            .collect();
        (unwrap.clone(), unwrap)
    } else {
        let unwrap: Vec<usize> = (1..=n_ifg)
            .filter(|ix| !drop_set.contains(&(*ix as i64)))
            .map(|ix| ix - 1)
            .collect();
        let solve: Vec<usize> = unwrap
            .iter()
            .copied()
            .filter(|ix| *ix != master_ix - 1)
            .collect();
        (unwrap, solve)
    };
    if solve_ifg.len() < 2 {
        if small_baseline {
            return stage7_err(
                "stage7 native SCLA requires at least two interferograms after drops",
            );
        }
        return stage7_err("stage7 native SCLA requires at least two non-master interferograms");
    }

    let output = stage7_scla_kernel(
        &ph_proc,
        &ph_mean_v,
        &bperp_mat,
        &unwrap_ifg,
        &solve_ifg,
        &day,
        master_ix,
        &ifg_std,
        ph_ramp,
    )?;
    write_stage7_outputs(dataset_root, &output, &bperp_mat)?;
    Ok(format!("Stage 7 estimated SCLA for {n_ps} PS"))
}

fn stage7_scla_kernel(
    ph_proc: &Matrix<f64>,
    _ph_mean_v: &Matrix<f64>,
    bperp_mat: &Matrix<f64>,
    unwrap_ix: &[usize],
    solve_ix: &[usize],
    day: &[f64],
    master_ix: usize,
    ifg_std: &[f64],
    ph_ramp: Matrix<f64>,
) -> Result<Stage7KernelOutput, CoreError> {
    if unwrap_ix.len() < 2 {
        return stage7_err("stage7 native SCLA requires at least two unwrap interferograms");
    }
    let n_ps = ph_proc.rows;
    let n_ifg = ph_proc.cols;
    let seq_count = unwrap_ix.len() - 1;
    let coest_mean_vel = unwrap_ix.len() >= 4;
    let mut ph_seq = vec![0.0; n_ps * seq_count];
    let mut bperp_seq = vec![0.0; n_ps * seq_count];
    let mut day_seq = vec![0.0; seq_count];
    for seq in 0..seq_count {
        let left = unwrap_ix[seq];
        let right = unwrap_ix[seq + 1];
        day_seq[seq] = day[right] - day[left];
        for row in 0..n_ps {
            ph_seq[row * seq_count + seq] =
                ph_proc.values[row * n_ifg + right] - ph_proc.values[row * n_ifg + left];
            bperp_seq[row * seq_count + seq] =
                bperp_mat.values[row * n_ifg + right] - bperp_mat.values[row * n_ifg + left];
        }
    }

    let mut mean_bperp = vec![0.0; seq_count];
    for seq in 0..seq_count {
        let mut sum = 0.0;
        for row in 0..n_ps {
            sum += bperp_seq[row * seq_count + seq];
        }
        mean_bperp[seq] = sum / n_ps as f64;
    }
    let design_cols = if coest_mean_vel { 3 } else { 2 };
    let mut g_seq = Vec::with_capacity(seq_count * design_cols);
    for seq in 0..seq_count {
        g_seq.push(1.0);
        g_seq.push(mean_bperp[seq]);
        if coest_mean_vel {
            g_seq.push(day_seq[seq]);
        }
    }
    let coeffs_seq = fit_shared_design(&g_seq, seq_count, design_cols, &ph_seq, n_ps, None)?;
    let mut k_ps_uw = vec![0.0; n_ps];
    for row in 0..n_ps {
        k_ps_uw[row] = coeffs_seq[row * design_cols + 1];
    }
    let mut ph_scla = vec![0.0f32; n_ps * n_ifg];
    for row in 0..n_ps {
        for col in 0..n_ifg {
            ph_scla[row * n_ifg + col] =
                (k_ps_uw[row] * bperp_mat.values[row * n_ifg + col]) as f32;
        }
    }

    let mut ifg_vcm = vec![0.0; n_ifg * n_ifg];
    let mut weights_full = vec![0.0; n_ifg];
    for col in 0..n_ifg {
        let variance = (ifg_std[col] * std::f64::consts::PI / 180.0).powi(2);
        ifg_vcm[col * n_ifg + col] = variance;
        weights_full[col] = if variance > 0.0 { 1.0 / variance } else { 0.0 };
    }

    let solve_count = solve_ix.len();
    let mut resid = vec![0.0; n_ps * solve_count];
    for row in 0..n_ps {
        for (out_col, &src_col) in solve_ix.iter().enumerate() {
            resid[row * solve_count + out_col] =
                ph_proc.values[row * n_ifg + src_col] - ph_scla[row * n_ifg + src_col] as f64;
        }
    }
    let c_ps_uw = if coest_mean_vel {
        let mut g_c = Vec::with_capacity(solve_count * 2);
        let mut weights = Vec::with_capacity(solve_count);
        for &src_col in solve_ix {
            g_c.push(1.0);
            g_c.push(day[src_col] - day[master_ix - 1]);
            weights.push(weights_full[src_col]);
        }
        let coeffs_c = fit_shared_design(&g_c, solve_count, 2, &resid, n_ps, Some(&weights))?;
        (0..n_ps).map(|row| coeffs_c[row * 2] as f32).collect()
    } else {
        (0..n_ps)
            .map(|row| {
                let start = row * solve_count;
                (resid[start..start + solve_count].iter().sum::<f64>() / solve_count as f64) as f32
            })
            .collect()
    };

    Ok(Stage7KernelOutput {
        k_ps_uw,
        c_ps_uw,
        ph_scla: Matrix {
            name: "ph_scla".to_string(),
            rows: n_ps,
            cols: n_ifg,
            values: ph_scla,
        },
        ph_ramp,
        ifg_vcm: Matrix {
            name: "ifg_vcm".to_string(),
            rows: n_ifg,
            cols: n_ifg,
            values: ifg_vcm,
        },
    })
}

fn write_stage7_outputs(
    dataset_root: &Path,
    output: &Stage7KernelOutput,
    bperp_mat: &Matrix<f64>,
) -> Result<(), CoreError> {
    let mut scla2 = MatFile::new(dataset_root.join("scla2.mat"));
    scla2.add_f32_col_vector(
        "K_ps_uw",
        output.k_ps_uw.iter().map(|&value| value as f32).collect(),
    )?;
    scla2.add_f32_col_vector("C_ps_uw", output.c_ps_uw.clone())?;
    scla2.add_f32_matrix(
        "ph_scla",
        output.ph_scla.rows,
        output.ph_scla.cols,
        output.ph_scla.values.clone(),
    )?;
    scla2.add_f64_matrix(
        "ph_ramp",
        output.ph_ramp.rows,
        output.ph_ramp.cols,
        output.ph_ramp.values.clone(),
    )?;
    scla2.add_f64_matrix(
        "ifg_vcm",
        output.ifg_vcm.rows,
        output.ifg_vcm.cols,
        output.ifg_vcm.values.clone(),
    )?;
    scla2.write()?;

    let (k_smooth, c_smooth) = smooth_scla_complete_envelope(&output.k_ps_uw, &output.c_ps_uw);
    let mut ph_scla_smooth = vec![0.0f32; bperp_mat.rows * bperp_mat.cols];
    for row in 0..bperp_mat.rows {
        for col in 0..bperp_mat.cols {
            ph_scla_smooth[row * bperp_mat.cols + col] =
                (k_smooth[row] * bperp_mat.values[row * bperp_mat.cols + col]) as f32;
        }
    }
    let mut scla_smooth2 = MatFile::new(dataset_root.join("scla_smooth2.mat"));
    scla_smooth2.add_f32_col_vector(
        "K_ps_uw",
        k_smooth.iter().map(|&value| value as f32).collect(),
    )?;
    scla_smooth2.add_f32_col_vector("C_ps_uw", c_smooth)?;
    scla_smooth2.add_f32_matrix("ph_scla", bperp_mat.rows, bperp_mat.cols, ph_scla_smooth)?;
    scla_smooth2.add_f64_matrix(
        "ph_ramp",
        output.ph_ramp.rows,
        output.ph_ramp.cols,
        output.ph_ramp.values.clone(),
    )?;
    scla_smooth2.write()?;
    Ok(())
}

fn load_or_rebuild_bperp(
    dataset_root: &Path,
    ps2: &MatData,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
    small_baseline: bool,
) -> Result<Matrix<f64>, CoreError> {
    let bp2_path = dataset_root.join("bp2.mat");
    let bp_nm = if bp2_path.exists() {
        let bp2 = read_mat_stage7(dataset_root, "bp2.mat")?;
        ps_matrix_f32(&bp2, "bperp_mat", n_ps, "bp2.bperp_mat").map(|matrix| Matrix {
            name: "bperp_mat".to_string(),
            rows: matrix.rows,
            cols: matrix.cols,
            values: matrix
                .values
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        })?
    } else {
        let bperp = ps_vector_f64(ps2, "bperp", n_ifg, "ps2.bperp")?;
        let source_cols = if small_baseline { n_ifg } else { n_ifg - 1 };
        let source = if small_baseline {
            bperp
        } else {
            bperp
                .iter()
                .enumerate()
                .filter_map(|(ix, &value)| (ix != master_ix - 1).then_some(value))
                .collect()
        };
        let mut tiled = Vec::with_capacity(n_ps * source_cols);
        for _ in 0..n_ps {
            tiled.extend_from_slice(&source);
        }
        let mut mat = MatFile::new(&bp2_path);
        mat.add_f32_matrix(
            "bperp_mat",
            n_ps,
            source_cols,
            tiled.iter().map(|&value| value as f32).collect(),
        )?;
        mat.write()?;
        Matrix {
            name: "bperp_mat".to_string(),
            rows: n_ps,
            cols: source_cols,
            values: tiled,
        }
    };
    if small_baseline {
        if bp_nm.cols != n_ifg {
            return stage7_err(format!(
                "bp2.bperp_mat has {} columns for small-baseline n_ifg={n_ifg}",
                bp_nm.cols
            ));
        }
        return Ok(bp_nm);
    }
    if bp_nm.cols == n_ifg {
        return Ok(bp_nm);
    }
    if bp_nm.cols != n_ifg - 1 {
        return stage7_err(format!(
            "bp2.bperp_mat has {} columns for single-master n_ifg={n_ifg}",
            bp_nm.cols
        ));
    }
    let mut full = Vec::with_capacity(n_ps * n_ifg);
    for row in 0..n_ps {
        let row_start = row * bp_nm.cols;
        full.extend_from_slice(&bp_nm.values[row_start..row_start + master_ix - 1]);
        full.push(0.0);
        full.extend_from_slice(&bp_nm.values[row_start + master_ix - 1..row_start + bp_nm.cols]);
    }
    Ok(Matrix {
        name: "bperp_mat".to_string(),
        rows: n_ps,
        cols: n_ifg,
        values: full,
    })
}

fn deramp_unwrapped_phase(
    ps2: &MatData,
    ph_all: &Matrix<f64>,
) -> Result<(Matrix<f64>, Matrix<f64>), CoreError> {
    let xy = ps_dim_f64(ps2, "xy", ph_all.rows, 3, "ps2.xy")?;
    let mut design = vec![0.0; ph_all.rows * 3];
    for row in 0..ph_all.rows {
        design[row * 3] = xy.values[row * 3 + 1] / 1000.0;
        design[row * 3 + 1] = xy.values[row * 3 + 2] / 1000.0;
        design[row * 3 + 2] = 1.0;
    }
    let mut ph_ramp = vec![0.0; ph_all.rows * ph_all.cols];
    let mut ph_out = ph_all.values.clone();
    for col in 0..ph_all.cols {
        let mut y = vec![0.0; ph_all.rows];
        for row in 0..ph_all.rows {
            y[row] = ph_all.values[row * ph_all.cols + col];
        }
        let coeff = fit_single_target(&design, ph_all.rows, 3, &y, None)?;
        for row in 0..ph_all.rows {
            let ramp = design[row * 3] * coeff[0] + design[row * 3 + 1] * coeff[1] + coeff[2];
            ph_ramp[row * ph_all.cols + col] = ramp;
            ph_out[row * ph_all.cols + col] -= ramp;
        }
    }
    Ok((
        Matrix {
            name: "ph_deramped".to_string(),
            rows: ph_all.rows,
            cols: ph_all.cols,
            values: ph_out,
        },
        Matrix {
            name: "ph_ramp".to_string(),
            rows: ph_all.rows,
            cols: ph_all.cols,
            values: ph_ramp,
        },
    ))
}

fn center_to_reference(ph: &Matrix<f64>, ref_ix: &[usize]) -> Matrix<f64> {
    if ref_ix.is_empty() {
        return ph.clone();
    }
    let mut centered = ph.values.clone();
    for col in 0..ph.cols {
        let mean = ref_ix
            .iter()
            .map(|&row| ph.values[row * ph.cols + col])
            .sum::<f64>()
            / ref_ix.len() as f64;
        for row in 0..ph.rows {
            centered[row * ph.cols + col] -= mean;
        }
    }
    Matrix {
        name: ph.name.clone(),
        rows: ph.rows,
        cols: ph.cols,
        values: centered,
    }
}

fn select_reference_ps(
    ps2: &MatData,
    parms: &Stage7Parms,
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

fn fit_shared_design(
    design: &[f64],
    rows: usize,
    cols: usize,
    y_by_target: &[f64],
    targets: usize,
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, CoreError> {
    let mut out = vec![0.0; targets * cols];
    for target in 0..targets {
        let y = &y_by_target[target * rows..target * rows + rows];
        let coeff = fit_single_target(design, rows, cols, y, weights)?;
        out[target * cols..target * cols + cols].copy_from_slice(&coeff);
    }
    Ok(out)
}

fn fit_single_target(
    design: &[f64],
    rows: usize,
    cols: usize,
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, CoreError> {
    let mut normal = vec![0.0; cols * cols];
    let mut rhs = vec![0.0; cols];
    for row in 0..rows {
        let weight = weights.map(|values| values[row]).unwrap_or(1.0);
        if weight == 0.0 {
            continue;
        }
        for i in 0..cols {
            let xi = design[row * cols + i];
            rhs[i] += weight * xi * y[row];
            for j in 0..cols {
                normal[i * cols + j] += weight * xi * design[row * cols + j];
            }
        }
    }
    solve_linear(normal, rhs, cols)
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
            return stage7_err("stage7 native least-squares system is singular");
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

fn smooth_scla_complete_envelope(k_ps_uw: &[f64], c_ps_uw: &[f32]) -> (Vec<f64>, Vec<f32>) {
    if k_ps_uw.len() <= 1 {
        return (k_ps_uw.to_vec(), c_ps_uw.to_vec());
    }

    let c_values: Vec<f64> = c_ps_uw.iter().map(|&value| value as f64).collect();
    let k_out = clamp_to_peer_envelope(k_ps_uw);
    let c_out = clamp_to_peer_envelope(&c_values)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    (k_out, c_out)
}

fn clamp_to_peer_envelope(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n <= 1 {
        return values.to_vec();
    }

    let mut prefix_min = vec![f64::INFINITY; n + 1];
    let mut prefix_max = vec![f64::NEG_INFINITY; n + 1];
    for ix in 0..n {
        prefix_min[ix + 1] = prefix_min[ix].min(values[ix]);
        prefix_max[ix + 1] = prefix_max[ix].max(values[ix]);
    }

    let mut suffix_min = vec![f64::INFINITY; n + 1];
    let mut suffix_max = vec![f64::NEG_INFINITY; n + 1];
    for ix in (0..n).rev() {
        suffix_min[ix] = suffix_min[ix + 1].min(values[ix]);
        suffix_max[ix] = suffix_max[ix + 1].max(values[ix]);
    }

    let mut out = values.to_vec();
    for ix in 0..n {
        let lower = prefix_min[ix].min(suffix_min[ix + 1]);
        let upper = prefix_max[ix].max(suffix_max[ix + 1]);
        if out[ix] > upper {
            out[ix] = upper;
        }
        if out[ix] < lower {
            out[ix] = lower;
        }
    }
    out
}

fn read_mat_stage7(dataset_root: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(dataset_root.join(filename))
        .map_err(|err| stage7_err_owned(format!("unable to read {filename}: {err}")))
}

fn load_stage7_parms(dataset_root: &Path) -> Stage7Parms {
    let path = dataset_root.join("parms.mat");
    if !path.exists() {
        return Stage7Parms::default();
    }
    let Ok(mat) = MatData::read(path) else {
        return Stage7Parms::default();
    };
    Stage7Parms {
        small_baseline_flag: text_from_mat(&mat, "small_baseline_flag", "n"),
        scla_deramp: text_from_mat(&mat, "scla_deramp", "y"),
        drop_ifg_index: optional_vector_f64(&mat, "drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| (value > 0.0).then_some(value.round() as i64))
            .collect(),
        scla_drop_index: optional_vector_f64(&mat, "scla_drop_index")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| (value > 0.0).then_some(value.round() as i64))
            .collect(),
        ref_lon: optional_vector_f64(&mat, "ref_lon")
            .unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
        ref_lat: optional_vector_f64(&mat, "ref_lat")
            .unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
        ref_radius: scalar_from_mat(&mat, "ref_radius", f64::INFINITY),
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
        stage: 7,
        message: format!("{label} is missing"),
    })?;
    if values.len() != len {
        return stage7_err(format!(
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
            stage: 7,
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
            stage: 7,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f64(source));
    }
    stage7_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}, expected {n_ps}x{n_dim}",
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
    stage7_err(format!(
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

fn stage7_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage7_err_owned(message.into()))
}

fn stage7_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 7, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_stage7_matches_python_reference_and_is_faster() {
        let root = temp_root("stage7-scla");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage7_fixture(&python_root);
        create_stage7_fixture(&rust_root);

        let python_start = Instant::now();
        run_python_stage7(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage7_native(&rust_root).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let specs = vec![
            ArtifactComparisonSpec::new(
                "scla2.mat",
                ["K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp", "ifg_vcm"],
            ),
            ArtifactComparisonSpec::new(
                "scla_smooth2.mat",
                ["K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp"],
            ),
        ];
        let summary = compare_fixture_artifacts(
            7,
            "merged",
            "synthetic_stage7_scla",
            &python_root,
            &rust_root,
            &specs,
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 7 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 7 should beat Python/native-kernel path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_phuw2_returns_structured_stage7_error() {
        let root = temp_root("stage7-missing-phuw");
        create_stage7_fixture(&root);
        fs::remove_file(root.join("phuw2.mat")).unwrap();

        let err = run_stage7_native(&root).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 7);
                assert!(message.contains("phuw2.mat"));
                assert!(message.contains("stage 7"));
            }
            other => panic!("expected structured Stage 7 error, got {other:?}"),
        }
        assert!(!root.join("scla2.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn complete_scla_envelope_clamps_only_unique_extrema() {
        let k = vec![10.0, 1.0, 4.0, 4.0];
        let c = vec![0.0, 5.0, 3.0, 3.0];
        let (k_smooth, c_smooth) = smooth_scla_complete_envelope(&k, &c);
        assert_eq!(k_smooth, vec![4.0, 4.0, 4.0, 4.0]);
        assert_eq!(c_smooth, vec![3.0, 3.0, 3.0, 3.0]);

        let duplicate_extrema = vec![1.0, 1.0, 10.0];
        let (k_smooth, _) = smooth_scla_complete_envelope(&duplicate_extrema, &[0.0, 0.0, 0.0]);
        assert_eq!(k_smooth, vec![1.0, 1.0, 1.0]);
    }

    fn create_stage7_fixture(root: &Path) {
        fs::create_dir_all(root).unwrap();
        let mut parms = MatFile::new(root.join("parms.mat"));
        parms
            .add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        parms
            .add_u32_matrix("scla_deramp", 1, 1, vec!['n' as u32])
            .unwrap();
        parms
            .add_f64_matrix("drop_ifg_index", 0, 0, Vec::new())
            .unwrap();
        parms
            .add_f64_matrix("scla_drop_index", 0, 0, Vec::new())
            .unwrap();
        parms
            .add_f64_scalar("ref_radius", f64::NEG_INFINITY)
            .unwrap();
        parms.write().unwrap();

        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", 3.0).unwrap();
        ps2.add_f64_scalar("n_ifg", 4.0).unwrap();
        ps2.add_f64_scalar("n_image", 4.0).unwrap();
        ps2.add_f64_scalar("master_ix", 2.0).unwrap();
        ps2.add_f64_scalar("master_day", 20.0).unwrap();
        ps2.add_f64_row_vector("day", vec![10.0, 20.0, 30.0, 40.0])
            .unwrap();
        ps2.add_f64_row_vector("bperp", vec![-3.0, 0.0, 7.0, 14.0])
            .unwrap();
        ps2.add_f64_matrix(
            "lonlat",
            3,
            2,
            vec![-118.0, 34.0, -117.9, 34.1, -117.8, 34.2],
        )
        .unwrap();
        ps2.add_f32_matrix(
            "xy",
            3,
            3,
            vec![1.0, 0.0, 0.0, 2.0, 100.0, 0.0, 3.0, 0.0, 100.0],
        )
        .unwrap();
        ps2.write().unwrap();

        let mut bp2 = MatFile::new(root.join("bp2.mat"));
        bp2.add_f32_matrix(
            "bperp_mat",
            3,
            3,
            vec![
                -3.0, 7.0, 14.0, //
                -2.0, 8.0, 13.0, //
                -4.0, 6.0, 15.0,
            ],
        )
        .unwrap();
        bp2.write().unwrap();

        let mut phuw2 = MatFile::new(root.join("phuw2.mat"));
        phuw2
            .add_f32_matrix(
                "ph_uw",
                3,
                4,
                vec![
                    1.0, 0.0, 4.0, 8.0, //
                    -1.0, 0.0, 3.0, 9.0, //
                    2.0, 0.0, 7.0, 12.0,
                ],
            )
            .unwrap();
        phuw2
            .add_f32_col_vector("msd", vec![0.0, 0.0, 0.0, 0.0])
            .unwrap();
        phuw2.write().unwrap();

        let mut ifgstd = MatFile::new(root.join("ifgstd2.mat"));
        ifgstd
            .add_f64_col_vector("ifg_std", vec![1.0, 1.5, 2.0, 2.5])
            .unwrap();
        ifgstd.write().unwrap();
    }

    fn run_python_stage7(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage7_calc_scla; stage7_calc_scla(Path(sys.argv[1]), backend='native', triangle_path='')";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage7 failed: {}\nstdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
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
