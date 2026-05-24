use crate::CoreError;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use std::collections::BTreeSet;
use std::path::Path;

const STAGE8_NOISE_SCALE: f32 = 0.5;

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
        }
    }
}

#[derive(Clone, Debug)]
struct Stage8EdgeOutput {
    dph_noise: Matrix<f32>,
    dph_space_uw: Matrix<f32>,
}

pub fn run_stage8_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let ps2 = read_mat_stage8(dataset_root, "ps2.mat")?;
    ensure_exists(dataset_root, "phuw2.mat", "stage-6 unwrap output")?;
    ensure_exists(dataset_root, "scla2.mat", "stage-7 SCLA output")?;
    ensure_exists(dataset_root, "uw_grid.mat", "stage-6 grid output")?;
    ensure_exists(dataset_root, "uw_interp.mat", "stage-6 interpolation output")?;
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
        return stage8_err(format!("ps2.master_ix must be 1-based within n_ifg={n_ifg}; got {master_ix}"));
    }

    let uw_grid = read_mat_stage8(dataset_root, "uw_grid.mat")?;
    let n_grid_ps = scalar_from_mat(&uw_grid, "n_ps", 0.0).round() as usize;
    if n_grid_ps == 0 {
        return stage8_err("uw_grid.mat missing valid n_ps");
    }
    let uw_ph = complex_ps_matrix(&uw_grid, "ph", n_grid_ps, "uw_grid.ph")?;
    let uw_interp = read_mat_stage8(dataset_root, "uw_interp.mat")?;
    let edges = edge_table(&uw_interp, "edgs", n_grid_ps)?;

    let mean_v = stage8_mean_velocity_payload(dataset_root, &ps2, &parms, n_ps, n_ifg, master_ix)?;
    write_mean_v(dataset_root, &mean_v)?;
    let output = stage8_edge_noise_kernel(&uw_ph, &edges);
    write_uw_space_time(dataset_root, &output, n_ifg, master_ix, &ps2)?;

    Ok(format!("Stage 8 produced mean velocity and space-time noise model for {} arcs", edges.len()))
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
    let scf_flag = parms.unwrap_spatial_cost_func_flag.eq_ignore_ascii_case("y");
    if small_baseline || effective_unwrap != "3D_FULL" || !la_flag || scf_flag {
        return stage8_err(
            "Stage 8 native path currently supports only single-master unwrap_method=3D_FULL \
             with unwrap_la_error_flag='y' and unwrap_spatial_cost_func_flag='n'",
        );
    }
    Ok(())
}

fn stage8_mean_velocity_payload(
    dataset_root: &Path,
    ps2: &MatData,
    parms: &Stage8Parms,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Matrix<f32>, CoreError> {
    let phuw = read_mat_stage8(dataset_root, "phuw2.mat")?;
    let scla = read_mat_stage8(dataset_root, "scla2.mat")?;
    let ifgstd = read_mat_stage8(dataset_root, "ifgstd2.mat")?;
    let ph_uw = ps_matrix_f32(&phuw, "ph_uw", n_ps, "phuw2.ph_uw")?;
    let ph_scla = ps_matrix_f32(&scla, "ph_scla", n_ps, "scla2.ph_scla")?;
    if ph_uw.cols != n_ifg || ph_scla.cols != n_ifg {
        return stage8_err("phuw2.ph_uw and scla2.ph_scla must match ps2.n_ifg for stage-8 mean velocity export");
    }

    let residual = Matrix {
        name: "ph_plot_source".to_string(),
        rows: n_ps,
        cols: n_ifg,
        values: ph_uw
            .values
            .iter()
            .zip(ph_scla.values.iter())
            .map(|(&uw, &scla)| uw as f64 - scla as f64)
            .collect(),
    };
    let (ph_plot, _ph_ramp) = deramp_unwrapped_phase(ps2, &residual)?;
    let day = ps_vector_f64(ps2, "day", n_ifg, "ps2.day")?;
    let ifg_std = ps_vector_f64(&ifgstd, "ifg_std", n_ifg, "ifgstd2.ifg_std")?;
    let drop_set: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    let unwrap_ix: Vec<usize> = (1..=n_ifg)
        .filter(|ix| !drop_set.contains(&(*ix as i64)) && *ix != master_ix)
        .map(|ix| ix - 1)
        .collect();
    if unwrap_ix.is_empty() {
        return stage8_err("stage-8 mean velocity export requires at least one non-master interferogram");
    }

    let ref_ix = select_reference_ps(ps2, parms, n_ps)?;
    let mut ph_use = vec![0.0; n_ps * unwrap_ix.len()];
    for row in 0..n_ps {
        for (out_col, &src_col) in unwrap_ix.iter().enumerate() {
            ph_use[row * unwrap_ix.len() + out_col] = ph_plot.values[row * n_ifg + src_col];
        }
    }
    center_values_to_reference(&mut ph_use, n_ps, unwrap_ix.len(), &ref_ix);

    let mut design = Vec::with_capacity(unwrap_ix.len() * 2);
    let mut weights = Vec::with_capacity(unwrap_ix.len());
    let master_day = day[master_ix - 1];
    for &src_col in &unwrap_ix {
        design.push(1.0);
        design.push(day[src_col] - master_day);
        let variance = (ifg_std[src_col] * std::f64::consts::PI / 180.0).powi(2);
        weights.push(if variance > 0.0 { 1.0 / variance } else { 0.0 });
    }
    let coeffs = fit_shared_design(&design, unwrap_ix.len(), 2, &ph_use, n_ps, Some(&weights))?;
    let mut values = vec![0.0f32; 2 * n_ps];
    for row in 0..n_ps {
        values[row] = coeffs[row * 2] as f32;
        values[n_ps + row] = coeffs[row * 2 + 1] as f32;
    }
    Ok(Matrix {
        name: "m".to_string(),
        rows: 2,
        cols: n_ps,
        values,
    })
}

fn stage8_edge_noise_kernel(uw_ph: &ComplexMatrixF32, edges: &[(usize, usize)]) -> Stage8EdgeOutput {
    let n_edge = edges.len();
    let n_ifg = uw_ph.cols;
    let mut dph_space_uw = vec![0.0f32; n_edge * n_ifg];
    let mut dph_noise = vec![0.0f32; n_edge * n_ifg];
    for (edge_ix, &(a_ix, b_ix)) in edges.iter().enumerate() {
        let mut sum = 0.0f64;
        for ifg_ix in 0..n_ifg {
            let left = uw_ph.values[a_ix * n_ifg + ifg_ix];
            let right = uw_ph.values[b_ix * n_ifg + ifg_ix];
            let phase = complex_mul_conj_arg(right, left);
            dph_space_uw[edge_ix * n_ifg + ifg_ix] = phase;
            sum += phase as f64;
        }
        let mean = if n_ifg == 0 { 0.0 } else { (sum / n_ifg as f64) as f32 };
        for ifg_ix in 0..n_ifg {
            let value = dph_space_uw[edge_ix * n_ifg + ifg_ix];
            dph_noise[edge_ix * n_ifg + ifg_ix] = (value - mean) * STAGE8_NOISE_SCALE;
        }
    }
    Stage8EdgeOutput {
        dph_noise: Matrix {
            name: "dph_noise".to_string(),
            rows: n_edge,
            cols: n_ifg,
            values: dph_noise,
        },
        dph_space_uw: Matrix {
            name: "dph_space_uw".to_string(),
            rows: n_edge,
            cols: n_ifg,
            values: dph_space_uw,
        },
    }
}

fn complex_mul_conj_arg(right: (f32, f32), left: (f32, f32)) -> f32 {
    let real = right.0 * left.0 + right.1 * left.1;
    let imag = right.1 * left.0 - right.0 * left.1;
    imag.atan2(real)
}

fn write_mean_v(dataset_root: &Path, m: &Matrix<f32>) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("mean_v.mat"));
    mat.add_f32_matrix("m", m.rows, m.cols, m.values.clone())?;
    mat.write()?;
    Ok(())
}

fn write_uw_space_time(
    dataset_root: &Path,
    output: &Stage8EdgeOutput,
    n_ifg: usize,
    master_ix: usize,
    ps2: &MatData,
) -> Result<(), CoreError> {
    let day_len = optional_vector_f64(ps2, "day").map(|values| values.len()).unwrap_or(n_ifg);
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

    let mut mat = MatFile::new(dataset_root.join("uw_space_time.mat"));
    mat.add_f64_matrix("G", unwrap_ifg.len(), day_len, g)?;
    mat.add_f32_matrix(
        "dph_noise",
        output.dph_noise.rows,
        output.dph_noise.cols,
        output.dph_noise.values.clone(),
    )?;
    mat.add_f32_matrix(
        "dph_space_uw",
        output.dph_space_uw.rows,
        output.dph_space_uw.cols,
        output.dph_space_uw.values.clone(),
    )?;
    mat.add_f64_matrix("spread", output.dph_noise.rows, output.dph_noise.cols, vec![0.0; output.dph_noise.rows * output.dph_noise.cols])?;
    mat.add_f64_matrix("ifreq_ij", 0, 0, Vec::new())?;
    mat.add_f64_matrix("jfreq_ij", 0, 0, Vec::new())?;
    mat.add_f64_matrix("shaky_ix", 0, 0, Vec::new())?;
    mat.add_f64_matrix("predef_ix", 0, 0, Vec::new())?;
    mat.write()?;
    Ok(())
}

fn edge_table(mat: &MatData, name: &str, n_nodes: usize) -> Result<Vec<(usize, usize)>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| stage8_err_owned(format!("uw_interp.{name} is invalid: {err}")))?;
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
        stage8_err(format!("Missing required artifact: {filename} ({label}) before stage 8"))
    }
}

fn deramp_unwrapped_phase(ps2: &MatData, ph_all: &Matrix<f64>) -> Result<(Matrix<f64>, Matrix<f64>), CoreError> {
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

fn center_values_to_reference(values: &mut [f64], rows: usize, cols: usize, ref_ix: &[usize]) {
    if ref_ix.is_empty() {
        return;
    }
    for col in 0..cols {
        let mean = ref_ix.iter().map(|&row| values[row * cols + col]).sum::<f64>() / ref_ix.len() as f64;
        for row in 0..rows {
            values[row * cols + col] -= mean;
        }
    }
}

fn select_reference_ps(ps2: &MatData, parms: &Stage8Parms, n_ps: usize) -> Result<Vec<usize>, CoreError> {
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

fn fit_single_target(design: &[f64], rows: usize, cols: usize, y: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>, CoreError> {
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

fn read_mat_stage8(dataset_root: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(dataset_root.join(filename)).map_err(|err| stage8_err_owned(format!("unable to read {filename}: {err}")))
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
        ref_lon: optional_vector_f64(&mat, "ref_lon").unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
        ref_lat: optional_vector_f64(&mat, "ref_lat").unwrap_or_else(|| vec![f64::NEG_INFINITY, f64::INFINITY]),
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

fn ps_vector_f64(mat: &MatData, name: &str, len: usize, label: &str) -> Result<Vec<f64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 8,
        message: format!("{label} is missing"),
    })?;
    if values.len() != len {
        return stage8_err(format!("{label} has incompatible length {} for expected length {len}", values.len()));
    }
    Ok(values)
}

fn ps_matrix_f32(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
    let source = mat.get_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 8,
        message: format!("{label} is invalid: {err}"),
    })?;
    orient_matrix_f32(source, n_ps, label)
}

fn ps_dim_f64(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| CoreError::NativeStage {
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

fn complex_ps_matrix(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat.get_complex_f32_matrix(name).map_err(|err| CoreError::NativeStage {
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

fn orient_matrix_f32(source: Matrix<f32>, n_ps: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
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
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_stage8_matches_python_native_kernel_and_is_faster() {
        let root = temp_root("stage8-edge");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage8_fixture(&python_root, 600);
        create_stage8_fixture(&rust_root, 600);

        let python_start = Instant::now();
        run_python_stage8_edge_fixture(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage8_native(&rust_root).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let specs = vec![ArtifactComparisonSpec::new("uw_space_time.mat", ["dph_noise", "dph_space_uw"])];
        let summary = compare_fixture_artifacts(
            8,
            "merged",
            "synthetic_stage8_edge_graph",
            &python_root,
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
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 8 should beat Python/native-kernel path: rust={rust_elapsed:?} python={python_elapsed:?}"
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
        parms.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32]).unwrap();
        parms.add_u32_matrix("unwrap_method", 1, 2, vec!['3' as u32, 'D' as u32]).unwrap();
        parms.add_u32_matrix("unwrap_la_error_flag", 1, 1, vec!['y' as u32]).unwrap();
        parms.add_u32_matrix("unwrap_spatial_cost_func_flag", 1, 1, vec!['n' as u32]).unwrap();
        parms.add_f64_matrix("drop_ifg_index", 0, 0, Vec::new()).unwrap();
        parms.add_f64_scalar("ref_radius", f64::NEG_INFINITY).unwrap();
        parms.write().unwrap();

        let n_ps = 4;
        let n_ifg = 4;
        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps2.add_f64_scalar("n_ifg", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("n_image", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("master_ix", 2.0).unwrap();
        ps2.add_f64_scalar("master_day", 20.0).unwrap();
        ps2.add_f64_row_vector("day", vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        ps2.add_f64_row_vector("bperp", vec![-3.0, 0.0, 7.0, 14.0]).unwrap();
        ps2.add_f64_matrix("lonlat", n_ps, 2, vec![-118.0, 34.0, -117.9, 34.1, -117.8, 34.2, -117.7, 34.3])
            .unwrap();
        ps2.add_f32_matrix(
            "xy",
            n_ps,
            3,
            vec![1.0, 0.0, 0.0, 2.0, 100.0, 0.0, 3.0, 0.0, 100.0, 4.0, 100.0, 100.0],
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
        scla2.add_f32_matrix("ph_scla", n_ps, n_ifg, vec![0.0; n_ps * n_ifg]).unwrap();
        scla2.write().unwrap();

        let mut ifgstd = MatFile::new(root.join("ifgstd2.mat"));
        ifgstd.add_f64_col_vector("ifg_std", vec![1.0, 1.5, 2.0, 2.5]).unwrap();
        ifgstd.write().unwrap();

        let n_grid = 5;
        let mut ph = Vec::with_capacity(n_grid * n_ifg);
        for row in 0..n_grid {
            for col in 0..n_ifg {
                let angle = row as f32 * 0.37 + col as f32 * 0.23 + (row * col) as f32 * 0.011;
                ph.push((angle.cos(), angle.sin()));
            }
        }
        let mut uw_grid = MatFile::new(root.join("uw_grid.mat"));
        uw_grid.add_f64_scalar("n_ps", n_grid as f64).unwrap();
        uw_grid.add_complex_f32_matrix("ph", n_grid, n_ifg, ph).unwrap();
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
        uw_interp.add_f64_matrix("edgs", edge_count, 3, edgs).unwrap();
        uw_interp.write().unwrap();
    }

    fn run_python_stage8_edge_fixture(root: &Path) {
        let script = r#"
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from pystamps.kernels import run_stage8_edge_noise_kernel

root = Path(sys.argv[1])
uw_grid = loadmat(root / "uw_grid.mat")
uw_interp = loadmat(root / "uw_interp.mat")
uw_ph = np.asarray(uw_grid["ph"], dtype=np.complex64)
edgs = np.asarray(uw_interp["edgs"], dtype=np.float64)
out = run_stage8_edge_noise_kernel(
    uw_ph,
    edgs[:, 1].astype(np.int64) - 1,
    edgs[:, 2].astype(np.int64) - 1,
    backend="native",
)
savemat(
    root / "uw_space_time.mat",
    {
        "dph_noise": out["dph_noise"],
        "dph_space_uw": out["dph_space_uw"],
    },
)
"#;
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage8 failed: {}\nstdout: {}",
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
