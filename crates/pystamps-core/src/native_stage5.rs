use crate::CoreError;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Stage5Parms {
    small_baseline_flag: String,
}

impl Default for Stage5Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
        }
    }
}

pub fn run_stage5_patch_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let ps1 = read_mat_stage5(patch_dir, "ps1.mat")?;
    let pm1 = read_mat_stage5(patch_dir, "pm1.mat")?;
    let select1 = read_mat_stage5(patch_dir, "select1.mat")?;
    let weed1 = read_mat_stage5(patch_dir, "weed1.mat")?;
    let ph1 = read_mat_stage5(patch_dir, "ph1.mat")?;
    let parms = load_stage5_parms(patch_dir);

    let n_ps1 = scalar_from_mat(&ps1, "n_ps", 0.0).round() as usize;
    if n_ps1 == 0 {
        return stage5_err("ps1.mat missing valid n_ps");
    }

    let ph1 = ps_complex_matrix(&ph1, "ph", n_ps1, "ph1.ph")?;
    let ij1 = ps_dim_f64(&ps1, "ij", n_ps1, 3, "ps1.ij")?;
    let lonlat1 = ps_dim_f64(&ps1, "lonlat", n_ps1, 2, "ps1.lonlat")?;
    let xy1 = ps_dim_f32(&ps1, "xy", n_ps1, 3, "ps1.xy")?;

    let ix = vector_i64(&select1, "ix", "select1.ix")?;
    if ix.is_empty() {
        return stage5_err("select1.mat has empty ix");
    }
    let keep_ix = bool_vector_or_default(&select1, "keep_ix", ix.len(), true);
    let ix2: Vec<i64> = ix
        .iter()
        .zip(keep_ix.iter())
        .filter_map(|(&value, &keep)| keep.then_some(value))
        .collect();
    validate_one_based_indices(&ix2, n_ps1, "select1.ix after keep_ix for weed1.mat")?;

    let ix_weed = bool_vector_exact(&weed1, "ix_weed", ix2.len());
    let mut final_ix1 = Vec::new();
    let mut kept_select_positions = Vec::new();
    let mut kept_ix2_positions = Vec::new();
    let mut ix2_pos = 0usize;
    for (select_pos, &keep) in keep_ix.iter().enumerate() {
        if !keep {
            continue;
        }
        let keep_after_weed = ix_weed.as_ref().map(|mask| mask[ix2_pos]).unwrap_or(true);
        if keep_after_weed {
            final_ix1.push(ix[select_pos]);
            kept_select_positions.push(select_pos);
            kept_ix2_positions.push(ix2_pos);
        }
        ix2_pos += 1;
    }
    validate_one_based_indices(&final_ix1, n_ps1, "weed1.mat promoted PS indices")?;
    let final_ix0: Vec<usize> = final_ix1.iter().map(|&value| (value - 1) as usize).collect();

    let master_ix = scalar_from_mat(&ps1, "master_ix", 1.0);
    let mut ps2 = MatFile::new(patch_dir.join("ps2.mat"));
    ps2.add_f32_col_vector("bperp", optional_vector_f32(&ps1, "bperp").unwrap_or_default())?;
    ps2.add_f64_col_vector("day", optional_vector_f64(&ps1, "day").unwrap_or_default())?;
    ps2.add_f64_matrix("ij", final_ix0.len(), ij1.cols, select_rows_f64(&ij1, &final_ix0))?;
    if let Some(ll0) = optional_matrix_f64(&ps1, "ll0") {
        ps2.add_f64_matrix("ll0", ll0.rows, ll0.cols, ll0.values)?;
    }
    ps2.add_f64_matrix(
        "lonlat",
        final_ix0.len(),
        lonlat1.cols,
        select_rows_f64(&lonlat1, &final_ix0),
    )?;
    ps2.add_f64_scalar("master_day", scalar_from_mat(&ps1, "master_day", 0.0))?;
    ps2.add_f64_scalar("master_ix", master_ix)?;
    ps2.add_f64_scalar("n_ifg", scalar_from_mat(&ps1, "n_ifg", ph1.cols as f64))?;
    ps2.add_f64_scalar("n_image", scalar_from_mat(&ps1, "n_image", ph1.cols as f64))?;
    ps2.add_f64_scalar("n_ps", final_ix0.len() as f64)?;
    ps2.add_f32_matrix("xy", final_ix0.len(), xy1.cols, select_rows_f32(&xy1, &final_ix0))?;
    if let Some(mean_incidence) = optional_vector_f64(&ps1, "mean_incidence").and_then(|values| values.first().copied()) {
        ps2.add_f64_scalar("mean_incidence", mean_incidence)?;
    }
    if let Some(mean_range) = optional_vector_f64(&ps1, "mean_range").and_then(|values| values.first().copied()) {
        ps2.add_f64_scalar("mean_range", mean_range)?;
    }
    ps2.write()?;

    let ph2 = select_rows_complex(&ph1, &final_ix0);
    let mut ph2_mat = MatFile::new(patch_dir.join("ph2.mat"));
    ph2_mat.add_complex_f32_matrix("ph", final_ix0.len(), ph1.cols, ph2.clone())?;
    ph2_mat.write()?;

    let k_ps2 = ps_vector_f64(&select1, "K_ps2", ix.len(), "select1.K_ps2")?;
    let c_ps2 = ps_vector_f64(&select1, "C_ps2", ix.len(), "select1.C_ps2")?;
    let coh_ps2 = ps_vector_f64(&select1, "coh_ps2", ix.len(), "select1.coh_ps2")?;
    let ph_res2 = ps_matrix_f32(&select1, "ph_res2", ix.len(), "select1.ph_res2")?;
    let ph_patch_all = ps_complex_matrix(&pm1, "ph_patch", n_ps1, "pm1.ph_patch")?;

    let k_ps = select_values_f64(&k_ps2, &kept_select_positions);
    let c_ps = select_values_f64(&c_ps2, &kept_select_positions);
    let coh_ps = select_values_f64(&coh_ps2, &kept_select_positions);
    let ph_res = select_rows_f32(&ph_res2, &kept_select_positions);
    let ph_patch2_rows: Vec<usize> = ix2.iter().map(|&value| (value - 1) as usize).collect();
    let ph_patch2 = select_rows_complex_matrix(&ph_patch_all, &ph_patch2_rows);
    let ph_patch = select_rows_complex_matrix(&ph_patch2, &kept_ix2_positions);

    let mut pm2 = MatFile::new(patch_dir.join("pm2.mat"));
    pm2.add_f64_col_vector("K_ps", k_ps.clone())?;
    pm2.add_f64_col_vector("C_ps", c_ps.clone())?;
    pm2.add_f64_col_vector("coh_ps", coh_ps)?;
    pm2.add_complex_f32_matrix(
        "ph_patch",
        final_ix0.len(),
        ph_patch_all.cols,
        ph_patch.values.clone(),
    )?;
    pm2.add_f32_matrix("ph_res", final_ix0.len(), ph_res2.cols, ph_res)?;
    pm2.write()?;

    write_psver(patch_dir)?;
    promote_optional_vector_f32(patch_dir, "hgt1.mat", "hgt2.mat", "hgt", n_ps1, &final_ix0)?;
    promote_optional_vector_f64(patch_dir, "la1.mat", "la2.mat", "la", n_ps1, &final_ix0)?;
    promote_optional_vector_f64(patch_dir, "da1.mat", "da2.mat", "D_A", n_ps1, &final_ix0)?;

    let bperp_mat2 = if patch_dir.join("bp1.mat").exists() {
        let bp1 = read_mat_stage5(patch_dir, "bp1.mat")?;
        let bperp_mat = ps_matrix_f32(&bp1, "bperp_mat", n_ps1, "bp1.bperp_mat")?;
        let selected = select_rows_f32(&bperp_mat, &final_ix0);
        let mut bp2 = MatFile::new(patch_dir.join("bp2.mat"));
        bp2.add_f32_matrix("bperp_mat", final_ix0.len(), bperp_mat.cols, selected.clone())?;
        bp2.write()?;
        Matrix {
            name: "bperp_mat".to_string(),
            rows: final_ix0.len(),
            cols: bperp_mat.cols,
            values: selected,
        }
    } else {
        Matrix {
            name: "bperp_mat".to_string(),
            rows: final_ix0.len(),
            cols: ph1.cols.saturating_sub(1).max(1),
            values: vec![0.0; final_ix0.len() * ph1.cols.saturating_sub(1).max(1)],
        }
    };

    write_rc2(
        patch_dir,
        &parms,
        &ph2,
        final_ix0.len(),
        ph1.cols,
        &k_ps,
        &c_ps,
        &bperp_mat2,
        &ph_patch.values,
        master_ix.round() as usize,
    )?;

    Ok(format!("Stage 5 promoted {} PS to version 2", final_ix0.len()))
}

fn read_mat_stage5(patch_dir: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(patch_dir.join(filename)).map_err(|err| stage5_err_owned(format!("unable to read {filename}: {err}")))
}

fn load_stage5_parms(patch_dir: &Path) -> Stage5Parms {
    let Some(path) = resolve_file_optional(patch_dir, "parms.mat") else {
        return Stage5Parms::default();
    };
    let Ok(mat) = MatData::read(path) else {
        return Stage5Parms::default();
    };
    Stage5Parms {
        small_baseline_flag: text_from_mat(&mat, "small_baseline_flag", "n"),
    }
}

fn write_psver(patch_dir: &Path) -> Result<(), CoreError> {
    let mut mat = MatFile::new(patch_dir.join("psver.mat"));
    mat.add_f64_scalar("psver", 2.0)?;
    mat.write()?;
    Ok(())
}

fn promote_optional_vector_f32(
    patch_dir: &Path,
    source_file: &str,
    dest_file: &str,
    variable: &str,
    n_ps: usize,
    final_ix0: &[usize],
) -> Result<(), CoreError> {
    if !patch_dir.join(source_file).exists() {
        return Ok(());
    }
    let source = read_mat_stage5(patch_dir, source_file)?;
    let values = ps_vector_f32(&source, variable, n_ps, &format!("{source_file}.{variable}"))?;
    let mut mat = MatFile::new(patch_dir.join(dest_file));
    mat.add_f32_col_vector(variable, select_values_f32(&values, final_ix0))?;
    mat.write()?;
    Ok(())
}

fn promote_optional_vector_f64(
    patch_dir: &Path,
    source_file: &str,
    dest_file: &str,
    variable: &str,
    n_ps: usize,
    final_ix0: &[usize],
) -> Result<(), CoreError> {
    if !patch_dir.join(source_file).exists() {
        return Ok(());
    }
    let source = read_mat_stage5(patch_dir, source_file)?;
    let values = ps_vector_f64(&source, variable, n_ps, &format!("{source_file}.{variable}"))?;
    let mut mat = MatFile::new(patch_dir.join(dest_file));
    mat.add_f64_col_vector(variable, select_values_f64(&values, final_ix0))?;
    mat.write()?;
    Ok(())
}

fn write_rc2(
    patch_dir: &Path,
    parms: &Stage5Parms,
    ph2: &[(f32, f32)],
    rows: usize,
    cols: usize,
    k_ps: &[f64],
    c_ps: &[f64],
    bperp_mat2: &Matrix<f32>,
    ph_patch: &[(f32, f32)],
    master_ix: usize,
) -> Result<(), CoreError> {
    if k_ps.len() != rows || c_ps.len() != rows {
        return stage5_err(format!(
            "select1 K/C vectors have incompatible lengths K={} C={} for promoted n_ps={rows}",
            k_ps.len(),
            c_ps.len()
        ));
    }
    let mut mat = MatFile::new(patch_dir.join("rc2.mat"));
    if parms.small_baseline_flag.eq_ignore_ascii_case("y") {
        if bperp_mat2.cols != cols {
            return stage5_err(format!(
                "bp2.bperp_mat has {} columns but small-baseline ph2 has {cols}",
                bperp_mat2.cols
            ));
        }
        let mut ph_rc = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let theta = k_ps[row] * bperp_mat2.values[row * bperp_mat2.cols + col] as f64;
                ph_rc.push(mul_exp_neg_i(ph2[row * cols + col], theta));
            }
        }
        mat.add_complex_f32_matrix("ph_rc", rows, cols, ph_rc)?;
    } else {
        if master_ix == 0 || master_ix > cols {
            return stage5_err(format!("ps2.master_ix must be 1-based within ph2 columns; got {master_ix}"));
        }
        if bperp_mat2.cols + 1 != cols {
            return stage5_err(format!(
                "bp2.bperp_mat has {} columns but single-master ph2 has {cols}",
                bperp_mat2.cols
            ));
        }
        if ph_patch.len() != rows * bperp_mat2.cols {
            return stage5_err(format!(
                "pm2.ph_patch has {} values for {rows}x{}",
                ph_patch.len(),
                bperp_mat2.cols
            ));
        }
        let mut ph_rc = Vec::with_capacity(rows * cols);
        let mut ph_reref = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let bperp = if col + 1 == master_ix {
                    0.0
                } else {
                    let src_col = if col + 1 < master_ix { col } else { col - 1 };
                    bperp_mat2.values[row * bperp_mat2.cols + src_col] as f64
                };
                let theta = k_ps[row] * bperp + c_ps[row];
                ph_rc.push(mul_exp_neg_i(ph2[row * cols + col], theta));
                if col + 1 == master_ix {
                    ph_reref.push((1.0, 0.0));
                } else {
                    let src_col = if col + 1 < master_ix { col } else { col - 1 };
                    ph_reref.push(ph_patch[row * bperp_mat2.cols + src_col]);
                }
            }
        }
        mat.add_complex_f32_matrix("ph_rc", rows, cols, ph_rc)?;
        mat.add_complex_f32_matrix("ph_reref", rows, cols, ph_reref)?;
    }
    mat.write()?;
    Ok(())
}

fn mul_exp_neg_i(value: (f32, f32), theta: f64) -> (f32, f32) {
    let (sin, cos) = theta.sin_cos();
    let real = value.0 as f64 * cos + value.1 as f64 * sin;
    let imag = value.1 as f64 * cos - value.0 as f64 * sin;
    (real as f32, imag as f32)
}

fn scalar_from_mat(mat: &MatData, name: &str, default: f64) -> f64 {
    optional_vector_f64(mat, name)
        .and_then(|values| values.into_iter().next())
        .unwrap_or(default)
}

fn optional_matrix_f64(mat: &MatData, name: &str) -> Option<Matrix<f64>> {
    mat.get_f64_matrix(name).ok()
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
}

fn optional_vector_f32(mat: &MatData, name: &str) -> Option<Vec<f32>> {
    mat.get_f32_matrix(name).ok().map(|matrix| matrix.values)
}

fn vector_i64(mat: &MatData, name: &str, label: &str) -> Result<Vec<i64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing"),
    })?;
    Ok(values
        .into_iter()
        .filter(|value| value.is_finite())
        .map(|value| value.round() as i64)
        .collect())
}

fn bool_vector_or_default(mat: &MatData, name: &str, expected_len: usize, default_value: bool) -> Vec<bool> {
    let Some(values) = optional_vector_f64(mat, name) else {
        return vec![default_value; expected_len];
    };
    if values.len() != expected_len {
        return vec![default_value; expected_len];
    }
    values.into_iter().map(|value| value != 0.0).collect()
}

fn bool_vector_exact(mat: &MatData, name: &str, expected_len: usize) -> Option<Vec<bool>> {
    let values = optional_vector_f64(mat, name)?;
    if values.len() != expected_len {
        return None;
    }
    Some(values.into_iter().map(|value| value != 0.0).collect())
}

fn ps_vector_f64(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Vec<f64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing"),
    })?;
    if values.len() != n_ps {
        return stage5_err(format!("{label} has incompatible length {} for n_ps={n_ps}", values.len()));
    }
    Ok(values)
}

fn ps_vector_f32(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Vec<f32>, CoreError> {
    let values = optional_vector_f32(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing"),
    })?;
    if values.len() != n_ps {
        return stage5_err(format!("{label} has incompatible length {} for n_ps={n_ps}", values.len()));
    }
    Ok(values)
}

fn ps_matrix_f32(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
    let source = mat.get_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    orient_matrix_f32(source, n_ps, label)
}

fn ps_dim_f64(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f64(source));
    }
    stage5_err(format!(
        "{label} has incompatible shape {}x{}; expected {n_ps}x{n_dim}",
        source.rows, source.cols
    ))
}

fn ps_dim_f32(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
    let source = mat.get_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f32(source));
    }
    stage5_err(format!(
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
        stage: 5,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        return Ok(transpose_complex(source));
    }
    stage5_err(format!(
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
    stage5_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn validate_one_based_indices(values: &[i64], n_ps: usize, label: &str) -> Result<(), CoreError> {
    for (pos, &value) in values.iter().enumerate() {
        if value < 1 || value as usize > n_ps {
            return stage5_err(format!(
                "{label} contains out-of-bounds 1-based index {value} at position {} for n_ps={n_ps}",
                pos + 1
            ));
        }
    }
    Ok(())
}

fn select_rows_f64(matrix: &Matrix<f64>, rows: &[usize]) -> Vec<f64> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
    values
}

fn select_rows_f32(matrix: &Matrix<f32>, rows: &[usize]) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
    values
}

fn select_rows_complex(matrix: &ComplexMatrixF32, rows: &[usize]) -> Vec<(f32, f32)> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
    values
}

fn select_rows_complex_matrix(matrix: &ComplexMatrixF32, rows: &[usize]) -> ComplexMatrixF32 {
    ComplexMatrixF32 {
        name: matrix.name.clone(),
        rows: rows.len(),
        cols: matrix.cols,
        values: select_rows_complex(matrix, rows),
    }
}

fn select_values_f64(values: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&row| values[row]).collect()
}

fn select_values_f32(values: &[f32], rows: &[usize]) -> Vec<f32> {
    rows.iter().map(|&row| values[row]).collect()
}

fn transpose_f64(source: Matrix<f64>) -> Matrix<f64> {
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
}

fn transpose_f32(source: Matrix<f32>) -> Matrix<f32> {
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
}

fn transpose_complex(source: ComplexMatrixF32) -> ComplexMatrixF32 {
    let mut values = Vec::with_capacity(source.values.len());
    for row in 0..source.cols {
        for col in 0..source.rows {
            values.push(source.values[col * source.cols + row]);
        }
    }
    ComplexMatrixF32 {
        name: source.name,
        rows: source.cols,
        cols: source.rows,
        values,
    }
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

fn stage5_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage5_err_owned(message.into()))
}

fn stage5_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 5, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_stage5_promotes_same_rows_and_variables_as_python_and_is_faster() {
        let root = temp_root("stage5-promotion");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage5_fixture(&python_root, "n");
        create_stage5_fixture(&rust_root, "n");

        let python_start = Instant::now();
        run_python_stage5(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage5_patch_native(rust_root.join("PATCH_1")).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let specs = vec![
            ArtifactComparisonSpec::new(
                "PATCH_1/ps2.mat",
                [
                    "bperp",
                    "day",
                    "ij",
                    "ll0",
                    "lonlat",
                    "master_day",
                    "master_ix",
                    "n_ifg",
                    "n_image",
                    "n_ps",
                    "xy",
                    "mean_incidence",
                    "mean_range",
                ],
            ),
            ArtifactComparisonSpec::new("PATCH_1/ph2.mat", ["ph"]),
            ArtifactComparisonSpec::new("PATCH_1/pm2.mat", ["K_ps", "C_ps", "coh_ps", "ph_patch", "ph_res"]),
            ArtifactComparisonSpec::new("PATCH_1/bp2.mat", ["bperp_mat"]),
            ArtifactComparisonSpec::new("PATCH_1/hgt2.mat", ["hgt"]),
            ArtifactComparisonSpec::new("PATCH_1/la2.mat", ["la"]),
            ArtifactComparisonSpec::new("PATCH_1/da2.mat", ["D_A"]),
            ArtifactComparisonSpec::new("PATCH_1/rc2.mat", ["ph_rc", "ph_reref"]),
            ArtifactComparisonSpec::new("PATCH_1/psver.mat", ["psver"]),
        ];
        let summary = compare_fixture_artifacts(
            5,
            "patch",
            "synthetic_stage5_patch",
            &python_root,
            &rust_root,
            &specs,
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 5 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 5 should beat Python path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage5_small_baseline_writes_phase_correction_without_reref() {
        let root = temp_root("stage5-small-baseline");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage5_fixture(&python_root, "y");
        create_stage5_fixture(&rust_root, "y");

        run_python_stage5(&python_root);
        run_stage5_patch_native(rust_root.join("PATCH_1")).unwrap();
        let summary = compare_fixture_artifacts(
            5,
            "patch",
            "synthetic_stage5_small_baseline",
            &python_root,
            &rust_root,
            &[ArtifactComparisonSpec::new("PATCH_1/rc2.mat", ["ph_rc"])],
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 5 small-baseline failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        let rc2 = MatData::read(rust_root.join("PATCH_1/rc2.mat")).unwrap();
        assert!(rc2.get("ph_reref").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mismatched_weed_mask_falls_back_to_stage3_kept_rows() {
        let root = temp_root("stage5-weed-mismatch");
        create_stage5_fixture(&root, "n");
        let patch = root.join("PATCH_1");
        let mut weed = MatFile::new(patch.join("weed1.mat"));
        weed.add_u8_matrix("ix_weed", 1, 1, vec![0]).unwrap();
        weed.write().unwrap();

        run_stage5_patch_native(&patch).unwrap();

        let ps2 = MatData::read(patch.join("ps2.mat")).unwrap();
        assert_eq!(scalar_from_mat(&ps2, "n_ps", 0.0), 3.0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn weeded_out_of_bounds_index_returns_structured_stage5_error() {
        let root = temp_root("stage5-oob");
        create_stage5_fixture(&root, "n");
        let patch = root.join("PATCH_1");
        let mut select = MatFile::new(patch.join("select1.mat"));
        select.add_f64_col_vector("ix", vec![1.0, 6.0]).unwrap();
        select.add_u8_matrix("keep_ix", 2, 1, vec![1, 1]).unwrap();
        select.add_f64_col_vector("K_ps2", vec![0.1, 0.2]).unwrap();
        select.add_f64_col_vector("C_ps2", vec![0.2, 0.3]).unwrap();
        select.add_f64_col_vector("coh_ps2", vec![0.8, 0.7]).unwrap();
        select.add_f32_matrix("ph_res2", 2, 3, vec![0.0; 6]).unwrap();
        select.write().unwrap();
        let mut weed = MatFile::new(patch.join("weed1.mat"));
        weed.add_u8_matrix("ix_weed", 2, 1, vec![1, 1]).unwrap();
        weed.write().unwrap();

        let err = run_stage5_patch_native(&patch).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 5);
                assert!(message.contains("out-of-bounds"));
                assert!(message.contains("weed1.mat"));
            }
            other => panic!("expected structured Stage 5 error, got {other:?}"),
        }
        assert!(!patch.join("ps2.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    fn create_stage5_fixture(root: &Path, small_baseline_flag: &str) {
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        write_parms(&patch, small_baseline_flag);
        write_ps1(&patch);
        write_ph1(&patch);
        write_pm1(&patch);
        write_select1(&patch);
        write_weed1(&patch);
        write_bp1(&patch, small_baseline_flag);
        write_optional_inputs(&patch);
    }

    fn write_parms(patch: &Path, small_baseline_flag: &str) {
        let mut mat = MatFile::new(patch.join("parms.mat"));
        mat.add_u32_matrix(
            "small_baseline_flag",
            1,
            small_baseline_flag.len(),
            small_baseline_flag.chars().map(|ch| ch as u32).collect(),
        )
        .unwrap();
        mat.write().unwrap();
    }

    fn write_ps1(patch: &Path) {
        let mut ij = Vec::new();
        let mut lonlat = Vec::new();
        let mut xy = Vec::new();
        for row in 0..5 {
            ij.extend_from_slice(&[(row + 1) as f64, (10 + row) as f64, (20 + row) as f64]);
            lonlat.extend_from_slice(&[-118.0 + row as f64 * 0.01, 34.0 + row as f64 * 0.02]);
            xy.extend_from_slice(&[(row + 1) as f32, (row as f32) * 100.0, (row as f32) * 200.0]);
        }
        let mut mat = MatFile::new(patch.join("ps1.mat"));
        mat.add_f64_scalar("n_ps", 5.0).unwrap();
        mat.add_f64_scalar("n_ifg", 4.0).unwrap();
        mat.add_f64_scalar("n_image", 4.0).unwrap();
        mat.add_f64_scalar("master_day", 738_584.0).unwrap();
        mat.add_f64_scalar("master_ix", 2.0).unwrap();
        mat.add_f64_row_vector("bperp", vec![-12.0, 0.0, 14.0, 28.0]).unwrap();
        mat.add_f64_row_vector("day", vec![738_572.0, 738_584.0, 738_596.0, 738_608.0])
            .unwrap();
        mat.add_f64_matrix("ij", 5, 3, ij).unwrap();
        mat.add_f64_matrix("lonlat", 5, 2, lonlat).unwrap();
        mat.add_f32_matrix("xy", 5, 3, xy).unwrap();
        mat.add_f64_matrix("ll0", 1, 2, vec![-118.0, 34.0]).unwrap();
        mat.add_f64_scalar("mean_incidence", 0.42).unwrap();
        mat.add_f64_scalar("mean_range", 820_000.0).unwrap();
        mat.write().unwrap();
    }

    fn write_ph1(patch: &Path) {
        let mut values = Vec::new();
        for row in 0..5 {
            for col in 0..4 {
                values.push((1.0 + row as f32 * 0.1, 0.2 + col as f32 * 0.05));
            }
        }
        let mut mat = MatFile::new(patch.join("ph1.mat"));
        mat.add_complex_f32_matrix("ph", 5, 4, values).unwrap();
        mat.write().unwrap();
    }

    fn write_pm1(patch: &Path) {
        let mut ph_patch = Vec::new();
        let mut ph_res = Vec::new();
        for row in 0..5 {
            for col in 0..3 {
                ph_patch.push((0.5 + row as f32 * 0.2, -0.25 + col as f32 * 0.1));
                ph_res.push(row as f32 * 0.01 + col as f32 * 0.02);
            }
        }
        let mut mat = MatFile::new(patch.join("pm1.mat"));
        mat.add_complex_f32_matrix("ph_patch", 5, 3, ph_patch).unwrap();
        mat.add_f32_matrix("ph_res", 5, 3, ph_res).unwrap();
        mat.add_f64_row_vector("K_ps", vec![0.01, 0.02, 0.03, 0.04, 0.05]).unwrap();
        mat.add_f64_row_vector("C_ps", vec![0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        mat.add_f64_row_vector("coh_ps", vec![0.9, 0.8, 0.7, 0.6, 0.5]).unwrap();
        mat.write().unwrap();
    }

    fn write_select1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("select1.mat"));
        mat.add_f64_col_vector("ix", vec![1.0, 3.0, 4.0, 5.0]).unwrap();
        mat.add_u8_matrix("keep_ix", 4, 1, vec![1, 1, 0, 1]).unwrap();
        mat.add_f64_col_vector("K_ps2", vec![0.11, 0.22, 0.33, 0.44]).unwrap();
        mat.add_f64_col_vector("C_ps2", vec![0.15, 0.25, 0.35, 0.45]).unwrap();
        mat.add_f64_col_vector("coh_ps2", vec![0.91, 0.82, 0.73, 0.64]).unwrap();
        mat.add_f32_matrix(
            "ph_res2",
            4,
            3,
            vec![0.01, 0.02, 0.03, 0.11, 0.12, 0.13, 0.21, 0.22, 0.23, 0.31, 0.32, 0.33],
        )
        .unwrap();
        mat.write().unwrap();
    }

    fn write_weed1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("weed1.mat"));
        mat.add_u8_matrix("ix_weed", 3, 1, vec![1, 0, 1]).unwrap();
        mat.add_u8_matrix("ix_weed2", 3, 1, vec![1, 0, 1]).unwrap();
        mat.write().unwrap();
    }

    fn write_bp1(patch: &Path, small_baseline_flag: &str) {
        let cols = if small_baseline_flag.eq_ignore_ascii_case("y") {
            4
        } else {
            3
        };
        let mut values = Vec::new();
        for row in 0..5 {
            for col in 0..cols {
                values.push(row as f32 * 10.0 + col as f32);
            }
        }
        let mut mat = MatFile::new(patch.join("bp1.mat"));
        mat.add_f32_matrix("bperp_mat", 5, cols, values).unwrap();
        mat.write().unwrap();
    }

    fn write_optional_inputs(patch: &Path) {
        let mut hgt = MatFile::new(patch.join("hgt1.mat"));
        hgt.add_f32_row_vector("hgt", vec![100.0, 110.0, 120.0, 130.0, 140.0]).unwrap();
        hgt.write().unwrap();

        let mut la = MatFile::new(patch.join("la1.mat"));
        la.add_f64_row_vector("la", vec![0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        la.write().unwrap();

        let mut da = MatFile::new(patch.join("da1.mat"));
        da.add_f64_row_vector("D_A", vec![1.0, 1.1, 1.2, 1.3, 1.4]).unwrap();
        da.write().unwrap();
    }

    fn run_python_stage5(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage5_correct_and_promote; stage5_correct_and_promote(Path(sys.argv[1]) / 'PATCH_1')";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage5 failed: {}\nstdout: {}",
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
