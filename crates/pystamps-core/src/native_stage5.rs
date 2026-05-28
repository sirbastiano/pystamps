use crate::CoreError;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const HDF5_SIGNATURE_SCAN_BYTES: usize = 1024 * 1024;
const STAGE5_PS1_VARS: &[&str] = &[
    "n_ps",
    "ij",
    "lonlat",
    "xy",
    "master_ix",
    "bperp",
    "day",
    "ll0",
    "master_day",
    "n_ifg",
    "n_image",
    "mean_incidence",
    "mean_range",
];
const STAGE5_SELECT1_VARS: &[&str] = &["ix", "keep_ix", "K_ps2", "C_ps2", "coh_ps2", "ph_res2"];

#[derive(Clone, Debug)]
struct Stage5Parms {
    small_baseline_flag: String,
    heading: f64,
}

impl Default for Stage5Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            heading: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct Stage5PatchBundle {
    ps: MatData,
    ij: Matrix<f64>,
    lonlat: Matrix<f64>,
    ph: ComplexMatrixF32,
    k_ps: Vec<f64>,
    c_ps: Vec<f64>,
    coh_ps: Vec<f64>,
    ph_patch: ComplexMatrixF32,
    ph_res: Matrix<f32>,
    ij_keys: Vec<(i64, i64)>,
    patch_bounds: Option<(i64, i64, i64, i64)>,
    bp: Option<Matrix<f32>>,
    hgt: Option<Vec<f64>>,
    la: Option<Vec<f64>>,
    rc: Option<ComplexMatrixF32>,
}

pub fn run_stage5_patch_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let ps1 = read_mat_stage5_vars(patch_dir, "ps1.mat", STAGE5_PS1_VARS)?;
    let pm1 = read_mat_stage5_vars(patch_dir, "pm1.mat", &["ph_patch"])?;
    let select1 = read_mat_stage5_vars(patch_dir, "select1.mat", STAGE5_SELECT1_VARS)?;
    let weed1 = read_mat_stage5_vars(patch_dir, "weed1.mat", &["ix_weed"])?;
    let ph1 = read_mat_stage5_vars(patch_dir, "ph1.mat", &["ph"])?;
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
    let final_ix0: Vec<usize> = final_ix1
        .iter()
        .map(|&value| (value - 1) as usize)
        .collect();

    let master_ix = scalar_from_mat(&ps1, "master_ix", 1.0);
    let mut ps2 = MatFile::new(patch_dir.join("ps2.mat"));
    ps2.add_f32_col_vector(
        "bperp",
        optional_vector_f32(&ps1, "bperp").unwrap_or_default(),
    )?;
    ps2.add_f64_col_vector("day", optional_vector_f64(&ps1, "day").unwrap_or_default())?;
    ps2.add_f64_matrix(
        "ij",
        final_ix0.len(),
        ij1.cols,
        select_rows_f64(&ij1, &final_ix0),
    )?;
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
    ps2.add_f32_matrix(
        "xy",
        final_ix0.len(),
        xy1.cols,
        select_rows_f32(&xy1, &final_ix0),
    )?;
    if let Some(mean_incidence) =
        optional_vector_f64(&ps1, "mean_incidence").and_then(|values| values.first().copied())
    {
        ps2.add_f64_scalar("mean_incidence", mean_incidence)?;
    }
    if let Some(mean_range) =
        optional_vector_f64(&ps1, "mean_range").and_then(|values| values.first().copied())
    {
        ps2.add_f64_scalar("mean_range", mean_range)?;
    }
    ps2.write()?;

    let ph2 = select_rows_complex(&ph1, &final_ix0);
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
    promote_optional_vector_f32(patch_dir, "hgt1.mat", "hgt2.mat", "hgt", n_ps1, &final_ix0)?;
    promote_optional_vector_f64(patch_dir, "la1.mat", "la2.mat", "la", n_ps1, &final_ix0)?;
    promote_optional_vector_f64(patch_dir, "da1.mat", "da2.mat", "D_A", n_ps1, &final_ix0)?;

    let bperp_mat2 = if patch_dir.join("bp1.mat").exists() {
        let bp1 = read_mat_stage5_vars(patch_dir, "bp1.mat", &["bperp_mat"])?;
        let bperp_mat = ps_matrix_f32(&bp1, "bperp_mat", n_ps1, "bp1.bperp_mat")?;
        let selected = select_rows_f32(&bperp_mat, &final_ix0);
        let mut bp2 = MatFile::new(patch_dir.join("bp2.mat"));
        bp2.add_f32_matrix(
            "bperp_mat",
            final_ix0.len(),
            bperp_mat.cols,
            selected.clone(),
        )?;
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

    let mut ph2_mat = MatFile::new(patch_dir.join("ph2.mat"));
    ph2_mat.add_complex_f32_matrix("ph", final_ix0.len(), ph1.cols, ph2)?;
    ph2_mat.write()?;

    let mut pm2 = MatFile::new(patch_dir.join("pm2.mat"));
    pm2.add_f64_col_vector("K_ps", k_ps)?;
    pm2.add_f64_col_vector("C_ps", c_ps)?;
    pm2.add_f64_col_vector("coh_ps", coh_ps)?;
    pm2.add_complex_f32_matrix(
        "ph_patch",
        final_ix0.len(),
        ph_patch_all.cols,
        ph_patch.values,
    )?;
    pm2.add_f32_matrix("ph_res", final_ix0.len(), ph_res2.cols, ph_res)?;
    pm2.write()?;

    write_psver(patch_dir)?;

    Ok(format!(
        "Stage 5 promoted {} PS to version 2",
        final_ix0.len()
    ))
}

pub fn run_stage5_merge_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let patch_dirs = discover_stage5_patch_dirs(dataset_root)?;
    if patch_dirs.is_empty() {
        return stage5_err("No patch directories found for merged stage-5 processing");
    }

    let parms = load_stage5_parms(dataset_root);
    let mut bundles = Vec::with_capacity(patch_dirs.len());
    for patch in &patch_dirs {
        bundles.push(load_stage5_patch_bundle(patch)?);
    }

    let mut ij = Vec::new();
    let mut lonlat = Vec::new();
    let mut ph2 = Vec::new();
    let mut k_ps = Vec::new();
    let mut c_ps = Vec::new();
    let mut coh_ps = Vec::new();
    let mut ph_patch = Vec::new();
    let mut ph_res = Vec::new();
    let mut bp = Vec::new();
    let mut hgt = Vec::new();
    let mut la = Vec::new();
    let mut rc = Vec::new();
    let mut has_bp = false;
    let mut has_hgt = false;
    let mut has_la = false;
    let mut has_rc = false;
    let mut remove_ix = Vec::new();
    let mut merged_index_by_key: HashMap<(i64, i64), usize> = HashMap::new();
    let mut merged_count = 0usize;
    let mut base_ps: Option<MatData> = None;
    let mut ph_cols = 0usize;
    let mut ph_patch_cols = 0usize;
    let mut ph_res_cols = 0usize;
    let mut bp_cols = 0usize;
    let mut rc_cols = 0usize;

    for bundle in &bundles {
        base_ps = Some(bundle.ps.clone());
        let (keep_patch, remove_patch_ix) = compute_patch_keep_mask(
            &bundle.ij_keys,
            &bundle.ij,
            bundle.patch_bounds,
            &merged_index_by_key,
        );
        remove_ix.extend(remove_patch_ix);
        let kept_ix: Vec<usize> = keep_patch
            .iter()
            .enumerate()
            .filter_map(|(idx, &keep)| keep.then_some(idx))
            .collect();
        if kept_ix.is_empty() {
            continue;
        }

        append_rows_f64(&mut ij, &bundle.ij, &kept_ix);
        append_rows_f64(&mut lonlat, &bundle.lonlat, &kept_ix);
        append_rows_complex(&mut ph2, &bundle.ph, &kept_ix);
        append_values_f64(&mut k_ps, &bundle.k_ps, &kept_ix);
        append_values_f64(&mut c_ps, &bundle.c_ps, &kept_ix);
        append_values_f64(&mut coh_ps, &bundle.coh_ps, &kept_ix);
        append_rows_complex(&mut ph_patch, &bundle.ph_patch, &kept_ix);
        append_rows_f32(&mut ph_res, &bundle.ph_res, &kept_ix);
        ph_cols = bundle.ph.cols;
        ph_patch_cols = bundle.ph_patch.cols;
        ph_res_cols = bundle.ph_res.cols;

        if let Some(bp_patch) = &bundle.bp {
            has_bp = true;
            bp_cols = bp_patch.cols;
            append_rows_f32(&mut bp, bp_patch, &kept_ix);
        }
        if let Some(hgt_patch) = &bundle.hgt {
            has_hgt = true;
            append_values_f64(&mut hgt, hgt_patch, &kept_ix);
        }
        if let Some(la_patch) = &bundle.la {
            has_la = true;
            append_values_f64(&mut la, la_patch, &kept_ix);
        }
        if let Some(rc_patch) = &bundle.rc {
            has_rc = true;
            rc_cols = rc_patch.cols;
            append_rows_complex(&mut rc, rc_patch, &kept_ix);
        }
        for (offset, &idx) in kept_ix.iter().enumerate() {
            merged_index_by_key
                .entry(bundle.ij_keys[idx])
                .or_insert(merged_count + offset);
        }
        merged_count += kept_ix.len();
    }

    let Some(base_ps) = base_ps else {
        return stage5_err("No patch PS data available for merge");
    };
    let initial_n_ps = ij.len() / 3;
    if initial_n_ps == 0 {
        return stage5_err("No patch PS data available for merge");
    }

    let mut active_indices: Vec<usize> = (0..initial_n_ps).collect();
    if !remove_ix.is_empty() {
        let mut keep = vec![true; initial_n_ps];
        for idx in remove_ix {
            if idx < keep.len() {
                keep[idx] = false;
            }
        }
        active_indices.retain(|&idx| keep[idx]);
    }

    active_indices = dedup_lonlat_keep_highest_coh_indices(&lonlat, &coh_ps, &active_indices);

    let active_lonlat = select_rows_plain(&lonlat, 2, &active_indices);
    let (xy_local, ll0_xy) = local_xy_from_lonlat(&active_lonlat, parms.heading)?;
    let mut sort_ix: Vec<usize> = (0..active_indices.len()).collect();
    sort_ix.sort_by(|&left, &right| {
        xy_local[left * 2 + 1]
            .total_cmp(&xy_local[right * 2 + 1])
            .then_with(|| xy_local[left * 2].total_cmp(&xy_local[right * 2]))
    });
    let xy_sorted = select_rows_plain(&xy_local, 2, &sort_ix);
    let final_indices: Vec<usize> = sort_ix.iter().map(|&pos| active_indices[pos]).collect();
    (
        ij, lonlat, ph2, k_ps, c_ps, coh_ps, ph_patch, ph_res, bp, hgt, la, rc,
    ) = apply_stage5_index_all(
        &final_indices,
        ij,
        lonlat,
        ph2,
        k_ps,
        c_ps,
        coh_ps,
        ph_patch,
        ph_res,
        bp,
        hgt,
        la,
        rc,
        ph_cols,
        ph_patch_cols,
        ph_res_cols,
        bp_cols,
        rc_cols,
    );
    let n_ps = ij.len() / 3;
    for row in 0..n_ps {
        ij[row * 3] = (row + 1) as f64;
    }
    let xy = merged_xy(&xy_sorted);
    if !has_bp {
        return stage5_err("bp2.mat is required to write merged Stage 5 outputs");
    }
    let ifg_std = merged_ifg_std(
        &parms,
        &base_ps,
        n_ps,
        ph_cols,
        ph_patch_cols,
        bp_cols,
        &ph2,
        &ph_patch,
        &bp,
        &k_ps,
        &c_ps,
    )?;

    write_merged_ps2(dataset_root, &base_ps, &ij, &lonlat, &xy, &ll0_xy)?;
    write_merged_ph2(dataset_root, n_ps, ph_cols, ph2)?;
    write_merged_pm2(
        dataset_root,
        n_ps,
        ph_patch_cols,
        ph_res_cols,
        k_ps,
        c_ps,
        coh_ps,
        ph_patch,
        ph_res,
    )?;
    write_psver(dataset_root)?;

    let mut bp2 = MatFile::new(dataset_root.join("bp2.mat"));
    bp2.add_f32_matrix("bperp_mat", n_ps, bp_cols, bp)?;
    bp2.write()?;
    if has_hgt {
        let mut hgt2 = MatFile::new(dataset_root.join("hgt2.mat"));
        hgt2.add_f64_col_vector("hgt", hgt)?;
        hgt2.write()?;
    }
    if has_la {
        let mut la2 = MatFile::new(dataset_root.join("la2.mat"));
        la2.add_f64_col_vector("la", la)?;
        la2.write()?;
    }
    if has_rc {
        let rc_payload = format_merged_rc2_payload(&rc, n_ps, rc_cols);
        let mut rc2 = MatFile::new(dataset_root.join("rc2.mat"));
        rc2.add_complex_f32_matrix("ph_rc", rc_cols, n_ps, rc_payload)?;
        rc2.write()?;
    }

    let mut ifgstd = MatFile::new(dataset_root.join("ifgstd2.mat"));
    ifgstd.add_f32_col_vector("ifg_std", ifg_std)?;
    ifgstd.write()?;

    Ok(format!(
        "Merged {} patches into {n_ps} PS records",
        patch_dirs.len()
    ))
}

fn read_mat_stage5(patch_dir: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(patch_dir.join(filename))
        .map_err(|err| stage5_err_owned(format!("unable to read {filename}: {err}")))
}

fn read_mat_stage5_vars(
    patch_dir: &Path,
    filename: &str,
    variables: &[&str],
) -> Result<MatData, CoreError> {
    MatData::read_selected(patch_dir.join(filename), variables)
        .map_err(|err| stage5_err_owned(format!("unable to read {filename}: {err}")))
}

fn discover_stage5_patch_dirs(dataset_root: &Path) -> Result<Vec<PathBuf>, CoreError> {
    let patch_list_old = read_stage5_patch_manifest(dataset_root, "patch.list_old")?;
    let patch_list = read_stage5_patch_manifest(dataset_root, "patch.list")?;
    if let (Some(old), Some(current)) = (&patch_list_old, &patch_list) {
        if old != current {
            return stage5_err(format!(
                "patch.list lists {} patch(es) but patch.list_old lists {}; restore the full patch.list before merged Stage 5",
                current.len(),
                old.len()
            ));
        }
    }
    let manifest = patch_list.or(patch_list_old);
    if let Some(names) = manifest {
        let mut discovered = Vec::with_capacity(names.len());
        for name in names {
            if !valid_stage5_patch_name(&name) {
                return stage5_err(format!("Invalid patch name in manifest: {name}"));
            }
            let path = dataset_root.join(&name);
            if !path.is_dir() {
                return stage5_err(format!(
                    "Patch manifest references missing patch directory: {name}"
                ));
            }
            discovered.push(path);
        }
        return Ok(discovered);
    }

    let mut discovered = Vec::new();
    for entry in fs::read_dir(dataset_root).map_err(|source| CoreError::ReadDataset {
        path: dataset_root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| CoreError::ReadDataset {
            path: dataset_root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("PATCH_"))
        {
            discovered.push(path);
        }
    }
    discovered.sort();
    Ok(discovered)
}

fn read_stage5_patch_manifest(
    dataset_root: &Path,
    filename: &str,
) -> Result<Option<Vec<String>>, CoreError> {
    let path = dataset_root.join(filename);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|source| CoreError::FileIo {
        path: path.clone(),
        source,
    })?;
    let names: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    Ok((!names.is_empty()).then_some(names))
}

fn valid_stage5_patch_name(name: &str) -> bool {
    name.starts_with("PATCH_") && !name.contains('/') && !name.contains('\\')
}

fn load_stage5_patch_bundle(patch: &Path) -> Result<Stage5PatchBundle, CoreError> {
    let patch_name = patch
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("PATCH")
        .to_string();
    for filename in ["ps2.mat", "ph2.mat", "pm2.mat"] {
        if !patch.join(filename).exists() {
            return stage5_err(format!(
                "Patch missing stage-5 outputs: {patch_name}/{filename}"
            ));
        }
    }

    let ps = read_mat_stage5(patch, "ps2.mat")?;
    let ph = read_mat_stage5(patch, "ph2.mat")?;
    let pm = read_mat_stage5(patch, "pm2.mat")?;
    let n_ps = scalar_from_mat(&ps, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage5_err(format!("{patch_name}/ps2.mat missing valid n_ps"));
    }

    let ij = ps_dim_f64(&ps, "ij", n_ps, 3, &format!("{patch_name}.ps2.ij"))?;
    let lonlat = ps_dim_f64(&ps, "lonlat", n_ps, 2, &format!("{patch_name}.ps2.lonlat"))?;
    let ph = ps_complex_matrix(&ph, "ph", n_ps, &format!("{patch_name}.ph2.ph"))?;
    let k_ps = ps_vector_f64(&pm, "K_ps", n_ps, &format!("{patch_name}.pm2.K_ps"))?;
    let c_ps = ps_vector_f64(&pm, "C_ps", n_ps, &format!("{patch_name}.pm2.C_ps"))?;
    let coh_ps = ps_vector_f64(&pm, "coh_ps", n_ps, &format!("{patch_name}.pm2.coh_ps"))?;
    let ph_patch = ps_complex_matrix(&pm, "ph_patch", n_ps, &format!("{patch_name}.pm2.ph_patch"))?;
    let ph_res = ps_matrix_f32(&pm, "ph_res", n_ps, &format!("{patch_name}.pm2.ph_res"))?;
    let ij_keys = ij
        .values
        .chunks_exact(3)
        .map(|row| (row[1].round() as i64, row[2].round() as i64))
        .collect();
    let patch_bounds = read_patch_bounds(patch)?;

    let bp = if patch.join("bp2.mat").exists() {
        let mat = read_mat_stage5(patch, "bp2.mat")?;
        Some(ps_matrix_f32(
            &mat,
            "bperp_mat",
            n_ps,
            &format!("{patch_name}.bp2.bperp_mat"),
        )?)
    } else {
        None
    };
    let hgt = if patch.join("hgt2.mat").exists() {
        let mat = read_mat_stage5(patch, "hgt2.mat")?;
        Some(ps_vector_f64(
            &mat,
            "hgt",
            n_ps,
            &format!("{patch_name}.hgt2.hgt"),
        )?)
    } else {
        None
    };
    let la = if patch.join("la2.mat").exists() {
        let mat = read_mat_stage5(patch, "la2.mat")?;
        Some(ps_vector_f64(
            &mat,
            "la",
            n_ps,
            &format!("{patch_name}.la2.la"),
        )?)
    } else {
        None
    };
    let rc = if patch.join("rc2.mat").exists() {
        let mat = read_mat_stage5_vars(patch, "rc2.mat", &["ph_rc", "rc"])?;
        match mat
            .get_complex_f32_matrix("ph_rc")
            .or_else(|_| mat.get_complex_f32_matrix("rc"))
        {
            Ok(rc) => Some(ps_complex_rows(
                rc,
                n_ps,
                &format!("{patch_name}.rc2.ph_rc"),
            )?),
            Err(_) => None,
        }
    } else {
        None
    };
    Ok(Stage5PatchBundle {
        ps,
        ij,
        lonlat,
        ph,
        k_ps,
        c_ps,
        coh_ps,
        ph_patch,
        ph_res,
        ij_keys,
        patch_bounds,
        bp,
        hgt,
        la,
        rc,
    })
}

fn read_patch_bounds(patch: &Path) -> Result<Option<(i64, i64, i64, i64)>, CoreError> {
    let path = patch.join("patch_noover.in");
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|source| CoreError::FileIo {
        path: path.clone(),
        source,
    })?;
    let values: Vec<i64> = text
        .split_whitespace()
        .filter_map(|value| value.parse::<i64>().ok())
        .collect();
    Ok((values.len() >= 4).then_some((values[0], values[1], values[2], values[3])))
}

fn compute_patch_keep_mask(
    ij_keys: &[(i64, i64)],
    ij: &Matrix<f64>,
    patch_bounds: Option<(i64, i64, i64, i64)>,
    merged_index_by_key: &HashMap<(i64, i64), usize>,
) -> (Vec<bool>, Vec<usize>) {
    let mut keep_patch = vec![true; ij_keys.len()];
    if let Some((row_min, row_max, col_min, col_max)) = patch_bounds {
        for (idx, row) in ij.values.chunks_exact(3).enumerate() {
            let col = row[1].round() as i64;
            let line = row[2].round() as i64;
            keep_patch[idx] = col >= col_min - 1
                && col <= col_max - 1
                && line >= row_min - 1
                && line <= row_max - 1;
        }
    }

    let mut remove_ix = Vec::new();
    for (idx, &keep) in keep_patch.iter().enumerate() {
        if keep {
            if let Some(&merged_ix) = merged_index_by_key.get(&ij_keys[idx]) {
                remove_ix.push(merged_ix);
            }
        }
    }

    for (idx, key) in ij_keys.iter().enumerate() {
        if !merged_index_by_key.contains_key(key) {
            keep_patch[idx] = true;
        }
    }
    (keep_patch, remove_ix)
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
        heading: scalar_from_mat(&mat, "heading", 0.0),
    }
}

fn write_merged_ps2(
    dataset_root: &Path,
    base_ps: &MatData,
    ij: &[f64],
    lonlat: &[f64],
    xy: &[f32],
    ll0_xy: &[f64],
) -> Result<(), CoreError> {
    let n_ps = ij.len() / 3;
    let mut ps2 = MatFile::new(dataset_root.join("ps2.mat"));
    ps2.add_f32_col_vector(
        "bperp",
        optional_vector_f32(base_ps, "bperp").unwrap_or_default(),
    )?;
    ps2.add_f64_col_vector(
        "day",
        optional_vector_f64(base_ps, "day").unwrap_or_default(),
    )?;
    ps2.add_f64_matrix("ij", n_ps, 3, ij.to_vec())?;
    if let Some(ll0) = optional_vector_f64(base_ps, "ll0") {
        ps2.add_f64_row_vector("ll0", ll0)?;
    } else {
        ps2.add_f64_row_vector("ll0", ll0_xy.to_vec())?;
    }
    ps2.add_f64_matrix("lonlat", n_ps, 2, lonlat.to_vec())?;
    ps2.add_f64_scalar("master_day", scalar_from_mat(base_ps, "master_day", 0.0))?;
    ps2.add_f64_scalar("master_ix", scalar_from_mat(base_ps, "master_ix", 1.0))?;
    ps2.add_f64_scalar("n_ifg", scalar_from_mat(base_ps, "n_ifg", 0.0))?;
    ps2.add_f64_scalar("n_image", scalar_from_mat(base_ps, "n_image", 0.0))?;
    ps2.add_f64_scalar("n_ps", n_ps as f64)?;
    ps2.add_f32_matrix("xy", n_ps, 3, xy.to_vec())?;
    if let Some(mean_incidence) =
        optional_vector_f64(base_ps, "mean_incidence").and_then(|values| values.first().copied())
    {
        ps2.add_f64_scalar("mean_incidence", mean_incidence)?;
    }
    if let Some(mean_range) =
        optional_vector_f64(base_ps, "mean_range").and_then(|values| values.first().copied())
    {
        ps2.add_f64_scalar("mean_range", mean_range)?;
    }
    ps2.write()?;
    Ok(())
}

fn write_merged_ph2(
    dataset_root: &Path,
    rows: usize,
    cols: usize,
    ph2: Vec<(f32, f32)>,
) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("ph2.mat"));
    mat.add_complex_f32_matrix("ph", rows, cols, ph2)?;
    mat.write()?;
    Ok(())
}

fn write_merged_pm2(
    dataset_root: &Path,
    rows: usize,
    ph_patch_cols: usize,
    ph_res_cols: usize,
    k_ps: Vec<f64>,
    c_ps: Vec<f64>,
    coh_ps: Vec<f64>,
    ph_patch: Vec<(f32, f32)>,
    ph_res: Vec<f32>,
) -> Result<(), CoreError> {
    let mut pm2 = MatFile::new(dataset_root.join("pm2.mat"));
    pm2.add_f64_col_vector("K_ps", k_ps)?;
    pm2.add_f64_col_vector("C_ps", c_ps)?;
    pm2.add_f64_col_vector("coh_ps", coh_ps)?;
    pm2.add_complex_f32_matrix("ph_patch", rows, ph_patch_cols, ph_patch)?;
    pm2.add_f32_matrix("ph_res", rows, ph_res_cols, ph_res)?;
    pm2.write()?;
    Ok(())
}

fn merged_ifg_std(
    parms: &Stage5Parms,
    base_ps: &MatData,
    rows: usize,
    ph_cols: usize,
    ph_patch_cols: usize,
    bp_cols: usize,
    ph2: &[(f32, f32)],
    ph_patch: &[(f32, f32)],
    bp: &[f32],
    k_ps: &[f64],
    c_ps: &[f64],
) -> Result<Vec<f32>, CoreError> {
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    if small_baseline {
        if bp_cols != ph_cols || ph_patch_cols != ph_cols {
            return stage5_err(format!(
                "small-baseline merged Stage 5 expects bp/ph_patch columns to match ph2 columns; bp={bp_cols} ph_patch={ph_patch_cols} ph={ph_cols}"
            ));
        }
    } else if bp_cols + 1 != ph_cols || ph_patch_cols + 1 != ph_cols {
        return stage5_err(format!(
            "single-master merged Stage 5 expects bp/ph_patch columns plus master to match ph2 columns; bp={bp_cols} ph_patch={ph_patch_cols} ph={ph_cols}"
        ));
    }

    let master_ix = scalar_from_mat(base_ps, "master_ix", 1.0).round() as usize;
    if !small_baseline && (master_ix == 0 || master_ix > ph_cols) {
        return stage5_err(format!(
            "ps2.master_ix must be 1-based within ph2 columns; got {master_ix}"
        ));
    }

    let sums = (0..rows)
        .into_par_iter()
        .fold(
            || vec![0.0f64; ph_cols],
            |mut local, row| {
                for col in 0..ph_cols {
                    let (patch_value, bperp) = if small_baseline {
                        (
                            ph_patch[row * ph_patch_cols + col],
                            bp[row * bp_cols + col] as f64,
                        )
                    } else if col + 1 == master_ix {
                        ((1.0, 0.0), 0.0)
                    } else {
                        let src_col = if col + 1 < master_ix { col } else { col - 1 };
                        (
                            ph_patch[row * ph_patch_cols + src_col],
                            bp[row * bp_cols + src_col] as f64,
                        )
                    };
                    let theta = if small_baseline {
                        k_ps[row] * bperp
                    } else {
                        k_ps[row] * bperp + c_ps[row]
                    };
                    let diff = mul_complex(ph2[row * ph_cols + col], conj_complex(patch_value));
                    let diff = mul_exp_neg_i(diff, theta);
                    let angle = (diff.1 as f64).atan2(diff.0 as f64);
                    local[col] += angle * angle;
                }
                local
            },
        )
        .reduce(
            || vec![0.0f64; ph_cols],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        );
    Ok(sums
        .into_iter()
        .map(|sum| ((sum / rows.max(1) as f64).sqrt() * 180.0 / std::f64::consts::PI) as f32)
        .collect())
}

fn mul_complex(left: (f32, f32), right: (f32, f32)) -> (f32, f32) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

fn conj_complex(value: (f32, f32)) -> (f32, f32) {
    (value.0, -value.1)
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
    let source_path = patch_dir.join(source_file);
    let label = format!("{source_file}.{variable}");
    let values = read_stage5_optional_vector_f32(&source_path, variable, n_ps, &label)?;
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
    let source_path = patch_dir.join(source_file);
    let label = format!("{source_file}.{variable}");
    let values = read_stage5_optional_vector_f64(&source_path, variable, n_ps, &label)?;
    let mut mat = MatFile::new(patch_dir.join(dest_file));
    mat.add_f64_col_vector(variable, select_values_f64(&values, final_ix0))?;
    mat.write()?;
    Ok(())
}

fn read_stage5_optional_vector_f32(
    path: &Path,
    variable: &str,
    n_ps: usize,
    label: &str,
) -> Result<Vec<f32>, CoreError> {
    match MatData::read_selected(path, &[variable]) {
        Ok(source) => match ps_vector_f32(&source, variable, n_ps, label) {
            Ok(values) => return Ok(values),
            Err(mat_err) => match read_hdf5_vector_f32(path, variable, n_ps) {
                Ok(values) => return Ok(values),
                Err(hdf_err) => {
                    return stage5_err(format!(
                        "{label} is missing in MAT v5 reader ({mat_err}); HDF5 fallback failed for {}: {hdf_err}",
                        path.display()
                    ))
                }
            },
        },
        Err(mat_err) => match read_hdf5_vector_f32(path, variable, n_ps) {
            Ok(values) => Ok(values),
            Err(hdf_err) => stage5_err(format!(
                "unable to read {label} from {} as MAT v5 ({mat_err}) or HDF5 ({hdf_err})",
                path.display()
            )),
        },
    }
}

fn read_stage5_optional_vector_f64(
    path: &Path,
    variable: &str,
    n_ps: usize,
    label: &str,
) -> Result<Vec<f64>, CoreError> {
    match MatData::read_selected(path, &[variable]) {
        Ok(source) => match ps_vector_f64(&source, variable, n_ps, label) {
            Ok(values) => return Ok(values),
            Err(mat_err) => match read_hdf5_vector_f64(path, variable, n_ps) {
                Ok(values) => return Ok(values),
                Err(hdf_err) => {
                    return stage5_err(format!(
                        "{label} is missing in MAT v5 reader ({mat_err}); HDF5 fallback failed for {}: {hdf_err}",
                        path.display()
                    ))
                }
            },
        },
        Err(mat_err) => match read_hdf5_vector_f64(path, variable, n_ps) {
            Ok(values) => Ok(values),
            Err(hdf_err) => stage5_err(format!(
                "unable to read {label} from {} as MAT v5 ({mat_err}) or HDF5 ({hdf_err})",
                path.display()
            )),
        },
    }
}

fn read_hdf5_vector_f32(path: &Path, variable: &str, n_ps: usize) -> Result<Vec<f32>, String> {
    read_hdf5_with_userblock(path, |candidate| {
        read_hdf5_vector_f32_direct(candidate, variable, n_ps)
    })
}

fn read_hdf5_vector_f64(path: &Path, variable: &str, n_ps: usize) -> Result<Vec<f64>, String> {
    read_hdf5_with_userblock(path, |candidate| {
        read_hdf5_vector_f64_direct(candidate, variable, n_ps)
    })
}

fn read_hdf5_with_userblock<T, F>(path: &Path, read_direct: F) -> Result<T, String>
where
    F: Fn(&Path) -> Result<T, String>,
{
    match read_direct(path) {
        Ok(values) => Ok(values),
        Err(direct_err) => {
            let offset = find_hdf5_signature_offset(path)?;
            if offset == 0 {
                return Err(direct_err);
            }
            read_hdf5_from_userblock_file(path, offset, read_direct).map_err(|userblock_err| {
                format!(
                    "{direct_err}; MATLAB HDF5 user-block fallback at offset {offset} failed: {userblock_err}"
                )
            })
        }
    }
}

fn read_hdf5_vector_f32_direct(
    path: &Path,
    variable: &str,
    n_ps: usize,
) -> Result<Vec<f32>, String> {
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
    if values.len() != n_ps {
        return Err(format!(
            "{variable} has incompatible length {} for n_ps={n_ps}",
            values.len()
        ));
    }
    Ok(values)
}

fn read_hdf5_vector_f64_direct(
    path: &Path,
    variable: &str,
    n_ps: usize,
) -> Result<Vec<f64>, String> {
    let file = rust_hdf5::H5File::open(path).map_err(|err| err.to_string())?;
    let dataset = file.dataset(variable).map_err(|err| err.to_string())?;
    let values = dataset.read_raw::<f64>().map_err(|err| err.to_string())?;
    if values.len() != n_ps {
        return Err(format!(
            "{variable} has incompatible length {} for n_ps={n_ps}",
            values.len()
        ));
    }
    Ok(values)
}

fn read_hdf5_from_userblock_file<T, F>(
    path: &Path,
    offset: usize,
    read_direct: F,
) -> Result<T, String>
where
    F: Fn(&Path) -> Result<T, String>,
{
    let temp_path = std::env::temp_dir().join(format!(
        "pystamps-stage5-hdf5-{}-{}.h5",
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
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|err| err.to_string())?;
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
        let total = rows * cols;
        let mut ph_rc = Vec::with_capacity(total);
        ph_rc
            .spare_capacity_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, dst)| {
                let row = idx / cols;
                let col = idx % cols;
                let theta = k_ps[row] * bperp_mat2.values[row * bperp_mat2.cols + col] as f64;
                dst.write(mul_exp_neg_i(ph2[row * cols + col], theta));
            });
        // Every spare slot is written exactly once by the parallel loop above.
        unsafe {
            ph_rc.set_len(total);
        }
        mat.add_complex_f32_matrix("ph_rc", rows, cols, ph_rc)?;
    } else {
        if master_ix == 0 || master_ix > cols {
            return stage5_err(format!(
                "ps2.master_ix must be 1-based within ph2 columns; got {master_ix}"
            ));
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
        let total = rows * cols;
        let mut ph_rc = Vec::with_capacity(total);
        let mut ph_reref = Vec::with_capacity(total);
        ph_rc
            .spare_capacity_mut()
            .par_iter_mut()
            .zip(ph_reref.spare_capacity_mut().par_iter_mut())
            .enumerate()
            .for_each(|(idx, (rc_dst, reref_dst))| {
                let row = idx / cols;
                let col = idx % cols;
                let bperp = if col + 1 == master_ix {
                    0.0
                } else {
                    let src_col = if col + 1 < master_ix { col } else { col - 1 };
                    bperp_mat2.values[row * bperp_mat2.cols + src_col] as f64
                };
                let theta = k_ps[row] * bperp + c_ps[row];
                rc_dst.write(mul_exp_neg_i(ph2[row * cols + col], theta));
                if col + 1 == master_ix {
                    reref_dst.write((1.0, 0.0));
                } else {
                    let src_col = if col + 1 < master_ix { col } else { col - 1 };
                    reref_dst.write(ph_patch[row * bperp_mat2.cols + src_col]);
                }
            });
        // Every spare slot is written exactly once by the parallel loop above.
        unsafe {
            ph_rc.set_len(total);
            ph_reref.set_len(total);
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

fn bool_vector_or_default(
    mat: &MatData,
    name: &str,
    expected_len: usize,
    default_value: bool,
) -> Vec<bool> {
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

fn ps_vector_f64(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<Vec<f64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing"),
    })?;
    if values.len() != n_ps {
        return stage5_err(format!(
            "{label} has incompatible length {} for n_ps={n_ps}",
            values.len()
        ));
    }
    Ok(values)
}

fn ps_vector_f32(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<Vec<f32>, CoreError> {
    let values = optional_vector_f32(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 5,
        message: format!("{label} is missing"),
    })?;
    if values.len() != n_ps {
        return stage5_err(format!(
            "{label} has incompatible length {} for n_ps={n_ps}",
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
            stage: 5,
            message: format!("{label} is missing or invalid: {err}"),
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
    let source = mat
        .get_complex_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
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
    select_rows_plain(&matrix.values, matrix.cols, rows)
}

fn select_rows_f32(matrix: &Matrix<f32>, rows: &[usize]) -> Vec<f32> {
    select_rows_plain(&matrix.values, matrix.cols, rows)
}

fn select_rows_complex(matrix: &ComplexMatrixF32, rows: &[usize]) -> Vec<(f32, f32)> {
    select_rows_plain(&matrix.values, matrix.cols, rows)
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
    select_values_plain(values, rows)
}

fn select_values_f32(values: &[f32], rows: &[usize]) -> Vec<f32> {
    select_values_plain(values, rows)
}

fn append_rows_f64(out: &mut Vec<f64>, matrix: &Matrix<f64>, rows: &[usize]) {
    for &row in rows {
        out.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
}

fn append_rows_f32(out: &mut Vec<f32>, matrix: &Matrix<f32>, rows: &[usize]) {
    for &row in rows {
        out.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
}

fn append_rows_complex(out: &mut Vec<(f32, f32)>, matrix: &ComplexMatrixF32, rows: &[usize]) {
    for &row in rows {
        out.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
}

fn append_values_f64(out: &mut Vec<f64>, values: &[f64], rows: &[usize]) {
    for &row in rows {
        out.push(values[row]);
    }
}

type Stage5MergeArrays = (
    Vec<f64>,
    Vec<f64>,
    Vec<(f32, f32)>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<(f32, f32)>,
    Vec<f32>,
    Vec<f32>,
    Vec<f64>,
    Vec<f64>,
    Vec<(f32, f32)>,
);

fn apply_stage5_index_all(
    indices: &[usize],
    ij: Vec<f64>,
    lonlat: Vec<f64>,
    ph2: Vec<(f32, f32)>,
    k_ps: Vec<f64>,
    c_ps: Vec<f64>,
    coh_ps: Vec<f64>,
    ph_patch: Vec<(f32, f32)>,
    ph_res: Vec<f32>,
    bp: Vec<f32>,
    hgt: Vec<f64>,
    la: Vec<f64>,
    rc: Vec<(f32, f32)>,
    ph_cols: usize,
    ph_patch_cols: usize,
    ph_res_cols: usize,
    bp_cols: usize,
    rc_cols: usize,
) -> Stage5MergeArrays {
    (
        select_rows_plain(&ij, 3, indices),
        select_rows_plain(&lonlat, 2, indices),
        select_rows_plain(&ph2, ph_cols, indices),
        select_values_plain(&k_ps, indices),
        select_values_plain(&c_ps, indices),
        select_values_plain(&coh_ps, indices),
        select_rows_plain(&ph_patch, ph_patch_cols, indices),
        select_rows_plain(&ph_res, ph_res_cols, indices),
        select_rows_plain(&bp, bp_cols, indices),
        select_values_plain(&hgt, indices),
        select_values_plain(&la, indices),
        select_rows_plain(&rc, rc_cols, indices),
    )
}

fn select_rows_plain<T: Copy + Send + Sync>(values: &[T], cols: usize, rows: &[usize]) -> Vec<T> {
    if cols == 0 || values.is_empty() {
        return Vec::new();
    }
    let total = rows.len() * cols;
    if total == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(total);
    let spare = out.spare_capacity_mut();
    spare
        .par_chunks_mut(cols)
        .zip(rows.par_iter())
        .for_each(|(chunk, &row)| {
            for (dst, &value) in chunk.iter_mut().zip(&values[row * cols..(row + 1) * cols]) {
                dst.write(value);
            }
        });
    // Every spare slot is written exactly once by the chunk loop above.
    unsafe {
        out.set_len(total);
    }
    out
}

fn select_values_plain<T: Copy + Send + Sync>(values: &[T], rows: &[usize]) -> Vec<T> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(rows.len());
    let spare = out.spare_capacity_mut();
    spare
        .par_iter_mut()
        .zip(rows.par_iter())
        .for_each(|(dst, &row)| {
            dst.write(values[row]);
        });
    // Every spare slot is written exactly once by the parallel zip above.
    unsafe {
        out.set_len(rows.len());
    }
    out
}

fn dedup_lonlat_keep_highest_coh_indices(
    lonlat: &[f64],
    coh_ps: &[f64],
    indices: &[usize],
) -> Vec<usize> {
    let mut best_by_key: HashMap<(u64, u64), (usize, f64)> = HashMap::new();
    for &row in indices {
        let key = (lonlat[row * 2].to_bits(), lonlat[row * 2 + 1].to_bits());
        let coh = coh_ps[row];
        match best_by_key.get_mut(&key) {
            Some((best_idx, best_coh)) if coh > *best_coh => {
                *best_idx = row;
                *best_coh = coh;
            }
            Some(_) => {}
            None => {
                best_by_key.insert(key, (row, coh));
            }
        }
    }
    indices
        .iter()
        .copied()
        .filter(|&row| {
            let key = (lonlat[row * 2].to_bits(), lonlat[row * 2 + 1].to_bits());
            best_by_key
                .get(&key)
                .map_or(true, |&(best_idx, _)| row == best_idx)
        })
        .collect()
}

fn ps_complex_rows(
    source: ComplexMatrixF32,
    n_ps: usize,
    label: &str,
) -> Result<ComplexMatrixF32, CoreError> {
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

fn format_merged_rc2_payload(rc: &[(f32, f32)], rows: usize, cols: usize) -> Vec<(f32, f32)> {
    if rows == 0 || cols == 0 || rc.is_empty() {
        return Vec::new();
    }
    let mut out = vec![(0.0, 0.0); rc.len()];
    out.par_iter_mut().enumerate().for_each(|(dst, out)| {
        let col = dst / rows;
        let row = dst % rows;
        let mut value = rc[row * cols + col];
        let amp = (value.0 * value.0 + value.1 * value.1).sqrt();
        if amp != 0.0 {
            value.0 /= amp;
            value.1 /= amp;
        }
        *out = value;
    });
    out
}

fn merged_xy(xy_local: &[f64]) -> Vec<f32> {
    let rows = xy_local.len() / 2;
    let mut xy = Vec::with_capacity(rows * 3);
    for row in 0..rows {
        xy.push((row + 1) as f32);
        xy.push(quantize_xy_mm(xy_local[row * 2]));
        xy.push(quantize_xy_mm(xy_local[row * 2 + 1]));
    }
    xy
}

fn quantize_xy_mm(value: f64) -> f32 {
    let scaled = (value as f32) * 1000.0;
    let abs_scaled = scaled.abs();
    let frac = abs_scaled - abs_scaled.floor();
    let rounded = if frac == 0.5 {
        scaled.signum() * (abs_scaled + 0.5).floor()
    } else {
        scaled.round()
    };
    rounded / 1000.0
}

fn local_xy_from_lonlat(
    lonlat: &[f64],
    heading_deg: f64,
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    let rows = lonlat.len() / 2;
    if rows == 0 {
        return Ok((Vec::new(), vec![0.0, 0.0]));
    }
    let mut min_lon = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for row in 0..rows {
        let lon = lonlat[row * 2];
        let lat = lonlat[row * 2 + 1];
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
    }
    let ll0 = vec![(max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0];
    let origin_lon = ll0[0].to_radians();
    let origin_lat = ll0[1].to_radians();
    let a = 6_378_137.0f64;
    let e = 0.082_094_437_949_70f64;
    let m0 = meridian_arc(a, e, origin_lat);
    let mut xy = vec![0.0; rows * 2];

    for row in 0..rows {
        let lon = lonlat[row * 2].to_radians();
        let lat = lonlat[row * 2 + 1].to_radians();
        let dlambda = lon - origin_lon;
        if lat != 0.0 {
            let m = meridian_arc(a, e, lat);
            let n = a / (1.0 - e.powi(2) * lat.sin().powi(2)).sqrt();
            let east = dlambda * lat.sin();
            let cot_lat = 1.0 / lat.tan();
            xy[row * 2] = n * cot_lat * east.sin();
            xy[row * 2 + 1] = m - m0 + n * cot_lat * (1.0 - east.cos());
        } else {
            xy[row * 2] = a * dlambda;
            xy[row * 2 + 1] = -m0;
        }
    }

    let theta = (180.0 - heading_deg).to_radians();
    let theta = if theta > std::f64::consts::PI {
        theta - 2.0 * std::f64::consts::PI
    } else {
        theta
    };
    let cos = theta.cos();
    let sin = theta.sin();
    let mut rotated = vec![0.0; xy.len()];
    for row in 0..rows {
        let x = xy[row * 2];
        let y = xy[row * 2 + 1];
        rotated[row * 2] = cos * x + sin * y;
        rotated[row * 2 + 1] = -sin * x + cos * y;
    }
    if ptp_xy(&rotated, 0) < ptp_xy(&xy, 0) && ptp_xy(&rotated, 1) < ptp_xy(&xy, 1) {
        Ok((rotated, ll0))
    } else {
        Ok((xy, ll0))
    }
}

fn meridian_arc(a: f64, e: f64, lat: f64) -> f64 {
    a * ((1.0 - e.powi(2) / 4.0 - 3.0 * e.powi(4) / 64.0 - 5.0 * e.powi(6) / 256.0) * lat
        - (3.0 * e.powi(2) / 8.0 + 3.0 * e.powi(4) / 32.0 + 45.0 * e.powi(6) / 1024.0)
            * (2.0 * lat).sin()
        + (15.0 * e.powi(4) / 256.0 + 45.0 * e.powi(6) / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e.powi(6) / 3072.0) * (6.0 * lat).sin())
}

fn ptp_xy(values: &[f64], col: usize) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for row in 0..values.len() / 2 {
        let value = values[row * 2 + col];
        min = min.min(value);
        max = max.max(value);
    }
    max - min
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
            ArtifactComparisonSpec::new(
                "PATCH_1/pm2.mat",
                ["K_ps", "C_ps", "coh_ps", "ph_patch", "ph_res"],
            ),
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
    fn stage5_promotes_hdf5_la1_vector() {
        let root = temp_root("stage5-hdf5-la");
        create_stage5_fixture(&root, "n");
        let patch = root.join("PATCH_1");
        fs::remove_file(patch.join("la1.mat")).unwrap();
        write_matlab_hdf5_f64_vector(
            &patch,
            "la-raw.h5",
            "la1.mat",
            "la",
            &[0.1, 0.2, 0.3, 0.4, 0.5],
        );

        run_stage5_patch_native(&patch).unwrap();

        let la2 = MatData::read(patch.join("la2.mat")).unwrap();
        assert_eq!(la2.get_f64_matrix("la").unwrap().values, vec![0.1, 0.5]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage5_promotes_hdf5_hgt1_vector() {
        let root = temp_root("stage5-hdf5-hgt");
        create_stage5_fixture(&root, "n");
        let patch = root.join("PATCH_1");
        fs::remove_file(patch.join("hgt1.mat")).unwrap();
        write_matlab_hdf5_f64_vector(
            &patch,
            "hgt-raw.h5",
            "hgt1.mat",
            "hgt",
            &[10.0, 20.0, 30.0, 40.0, 50.0],
        );

        run_stage5_patch_native(&patch).unwrap();

        let hgt2 = MatData::read(patch.join("hgt2.mat")).unwrap();
        assert_eq!(hgt2.get_f32_matrix("hgt").unwrap().values, vec![10.0, 50.0]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage5_unreadable_optional_vector_returns_structured_error() {
        let root = temp_root("stage5-bad-optional");
        create_stage5_fixture(&root, "n");
        let patch = root.join("PATCH_1");
        fs::write(patch.join("la1.mat"), b"not a readable mat file").unwrap();

        let err = run_stage5_patch_native(&patch).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 5);
                assert!(message.contains("la1.mat.la"));
                assert!(message.contains("unable to read"));
            }
            other => panic!("expected structured Stage 5 error, got {other:?}"),
        }
        assert!(!patch.join("la2.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn synthetic_stage5_merge_matches_python_and_is_faster() {
        let root = temp_root("stage5-merge");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage5_merge_fixture(&python_root);
        create_stage5_merge_fixture(&rust_root);

        let python_start = Instant::now();
        run_python_stage5_merge(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage5_merge_native(&rust_root).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let specs = vec![
            ArtifactComparisonSpec::new(
                "ps2.mat",
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
            ArtifactComparisonSpec::new("ph2.mat", ["ph"]),
            ArtifactComparisonSpec::new(
                "pm2.mat",
                ["K_ps", "C_ps", "coh_ps", "ph_patch", "ph_res"],
            ),
            ArtifactComparisonSpec::new("bp2.mat", ["bperp_mat"]),
            ArtifactComparisonSpec::new("hgt2.mat", ["hgt"]),
            ArtifactComparisonSpec::new("la2.mat", ["la"]),
            ArtifactComparisonSpec::new("rc2.mat", ["ph_rc"]),
            ArtifactComparisonSpec::new("psver.mat", ["psver"]),
            ArtifactComparisonSpec::new("ifgstd2.mat", ["ifg_std"]),
        ];
        let summary = compare_fixture_artifacts(
            5,
            "merged",
            "synthetic_stage5_merge",
            &python_root,
            &rust_root,
            &specs,
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 5 merge failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 5 merge should beat Python path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage5_merge_rejects_subset_patch_list_when_legacy_manifest_has_more_patches() {
        let root = temp_root("stage5-merge-subset-manifest");
        create_stage5_merge_fixture(&root);
        fs::write(root.join("patch.list"), "PATCH_1\n").unwrap();
        fs::write(
            root.join("patch.list_old"),
            "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n",
        )
        .unwrap();

        let err = run_stage5_merge_native(&root).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 5);
                assert!(message.contains("patch.list"));
                assert!(message.contains("patch.list_old"));
                assert!(message.contains("restore the full patch.list"));
            }
            other => panic!("expected structured merged Stage 5 error, got {other:?}"),
        }
        assert!(!root.join("ifgstd2.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage5_merge_missing_patch_ph2_returns_structured_error() {
        let root = temp_root("stage5-merge-missing-ph");
        create_stage5_merge_fixture(&root);
        fs::remove_file(root.join("PATCH_1/ph2.mat")).unwrap();

        let err = run_stage5_merge_native(&root).unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 5);
                assert!(message.contains("PATCH_1/ph2.mat"));
                assert!(message.contains("Patch missing stage-5 outputs"));
            }
            other => panic!("expected structured merged Stage 5 error, got {other:?}"),
        }
        assert!(!root.join("ifgstd2.mat").exists());
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
        select
            .add_f64_col_vector("coh_ps2", vec![0.8, 0.7])
            .unwrap();
        select
            .add_f32_matrix("ph_res2", 2, 3, vec![0.0; 6])
            .unwrap();
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
        mat.add_f64_row_vector("bperp", vec![-12.0, 0.0, 14.0, 28.0])
            .unwrap();
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
        mat.add_complex_f32_matrix("ph_patch", 5, 3, ph_patch)
            .unwrap();
        mat.add_f32_matrix("ph_res", 5, 3, ph_res).unwrap();
        mat.add_f64_row_vector("K_ps", vec![0.01, 0.02, 0.03, 0.04, 0.05])
            .unwrap();
        mat.add_f64_row_vector("C_ps", vec![0.1, 0.2, 0.3, 0.4, 0.5])
            .unwrap();
        mat.add_f64_row_vector("coh_ps", vec![0.9, 0.8, 0.7, 0.6, 0.5])
            .unwrap();
        mat.write().unwrap();
    }

    fn write_select1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("select1.mat"));
        mat.add_f64_col_vector("ix", vec![1.0, 3.0, 4.0, 5.0])
            .unwrap();
        mat.add_u8_matrix("keep_ix", 4, 1, vec![1, 1, 0, 1])
            .unwrap();
        mat.add_f64_col_vector("K_ps2", vec![0.11, 0.22, 0.33, 0.44])
            .unwrap();
        mat.add_f64_col_vector("C_ps2", vec![0.15, 0.25, 0.35, 0.45])
            .unwrap();
        mat.add_f64_col_vector("coh_ps2", vec![0.91, 0.82, 0.73, 0.64])
            .unwrap();
        mat.add_f32_matrix(
            "ph_res2",
            4,
            3,
            vec![
                0.01, 0.02, 0.03, 0.11, 0.12, 0.13, 0.21, 0.22, 0.23, 0.31, 0.32, 0.33,
            ],
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
        hgt.add_f32_row_vector("hgt", vec![100.0, 110.0, 120.0, 130.0, 140.0])
            .unwrap();
        hgt.write().unwrap();

        let mut la = MatFile::new(patch.join("la1.mat"));
        la.add_f64_row_vector("la", vec![0.1, 0.2, 0.3, 0.4, 0.5])
            .unwrap();
        la.write().unwrap();

        let mut da = MatFile::new(patch.join("da1.mat"));
        da.add_f64_row_vector("D_A", vec![1.0, 1.1, 1.2, 1.3, 1.4])
            .unwrap();
        da.write().unwrap();
    }

    fn write_matlab_hdf5_f64_vector(
        patch: &Path,
        raw_name: &str,
        matlab_name: &str,
        variable: &str,
        values: &[f64],
    ) {
        let raw_hdf5 = patch.join(raw_name);
        let h5 = rust_hdf5::H5File::create(&raw_hdf5).unwrap();
        let ds = h5
            .new_dataset::<f64>()
            .shape(&[1usize, values.len()])
            .create(variable)
            .unwrap();
        ds.write_raw(values).unwrap();
        h5.close().unwrap();
        let mut matlab_hdf5 = fs::File::create(patch.join(matlab_name)).unwrap();
        matlab_hdf5.write_all(&vec![b' '; 512]).unwrap();
        matlab_hdf5
            .write_all(&fs::read(&raw_hdf5).unwrap())
            .unwrap();
        fs::remove_file(raw_hdf5).unwrap();
    }

    fn create_stage5_merge_fixture(root: &Path) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("patch.list"), "PATCH_2\nPATCH_1\n").unwrap();
        write_root_parms(root);
        write_promoted_patch(
            &root.join("PATCH_2"),
            &[
                PromotedRow {
                    key: (100.0, 200.0),
                    lonlat: (-118.03, 34.01),
                    k: 0.31,
                    c: 0.11,
                    coh: 0.62,
                    hgt: 210.0,
                    la: 0.51,
                },
                PromotedRow {
                    key: (102.0, 202.0),
                    lonlat: (-118.01, 34.04),
                    k: 0.33,
                    c: 0.13,
                    coh: 0.74,
                    hgt: 220.0,
                    la: 0.52,
                },
            ],
        );
        write_promoted_patch(
            &root.join("PATCH_1"),
            &[
                PromotedRow {
                    key: (100.0, 200.0),
                    lonlat: (-118.02, 34.02),
                    k: 0.21,
                    c: 0.21,
                    coh: 0.91,
                    hgt: 110.0,
                    la: 0.41,
                },
                PromotedRow {
                    key: (101.0, 201.0),
                    lonlat: (-118.00, 34.03),
                    k: 0.22,
                    c: 0.22,
                    coh: 0.82,
                    hgt: 120.0,
                    la: 0.42,
                },
                PromotedRow {
                    key: (103.0, 203.0),
                    lonlat: (-117.99, 34.05),
                    k: 0.23,
                    c: 0.23,
                    coh: 0.73,
                    hgt: 130.0,
                    la: 0.43,
                },
            ],
        );
    }

    #[derive(Clone, Copy)]
    struct PromotedRow {
        key: (f64, f64),
        lonlat: (f64, f64),
        k: f64,
        c: f64,
        coh: f64,
        hgt: f64,
        la: f64,
    }

    fn write_root_parms(root: &Path) {
        let mut mat = MatFile::new(root.join("parms.mat"));
        mat.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_f64_scalar("heading", 167.0).unwrap();
        mat.write().unwrap();
    }

    fn write_promoted_patch(patch: &Path, rows: &[PromotedRow]) {
        fs::create_dir_all(patch).unwrap();
        let n_ps = rows.len();
        let mut ij = Vec::with_capacity(n_ps * 3);
        let mut lonlat = Vec::with_capacity(n_ps * 2);
        let mut xy = Vec::with_capacity(n_ps * 3);
        let mut ph = Vec::with_capacity(n_ps * 4);
        let mut ph_patch = Vec::with_capacity(n_ps * 3);
        let mut ph_res = Vec::with_capacity(n_ps * 3);
        let mut bp = Vec::with_capacity(n_ps * 3);
        let mut rc = Vec::with_capacity(n_ps * 4);
        for (row_ix, row) in rows.iter().enumerate() {
            ij.extend_from_slice(&[(row_ix + 1) as f64, row.key.0, row.key.1]);
            lonlat.extend_from_slice(&[row.lonlat.0, row.lonlat.1]);
            xy.extend_from_slice(&[
                (row_ix + 1) as f32,
                row.key.0 as f32 * 10.0,
                row.key.1 as f32 * 10.0,
            ]);
            for col in 0..4 {
                let real = 1.0 + row_ix as f32 * 0.2 + col as f32 * 0.03;
                let imag = 0.4 + row_ix as f32 * 0.1 - col as f32 * 0.02;
                ph.push((real, imag));
                rc.push((real * 0.8, imag * 0.6));
            }
            for col in 0..3 {
                ph_patch.push((0.7 + row_ix as f32 * 0.1, -0.2 + col as f32 * 0.05));
                ph_res.push(row_ix as f32 * 0.01 + col as f32 * 0.015);
                bp.push(5.0 + row_ix as f32 * 2.0 + col as f32);
            }
        }

        let mut ps2 = MatFile::new(patch.join("ps2.mat"));
        ps2.add_f32_col_vector("bperp", vec![-12.0, 0.0, 14.0, 28.0])
            .unwrap();
        ps2.add_f64_col_vector("day", vec![738_572.0, 738_584.0, 738_596.0, 738_608.0])
            .unwrap();
        ps2.add_f64_matrix("ij", n_ps, 3, ij).unwrap();
        ps2.add_f64_row_vector("ll0", vec![-118.0, 34.0]).unwrap();
        ps2.add_f64_matrix("lonlat", n_ps, 2, lonlat).unwrap();
        ps2.add_f64_scalar("master_day", 738_584.0).unwrap();
        ps2.add_f64_scalar("master_ix", 2.0).unwrap();
        ps2.add_f64_scalar("n_ifg", 4.0).unwrap();
        ps2.add_f64_scalar("n_image", 4.0).unwrap();
        ps2.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps2.add_f32_matrix("xy", n_ps, 3, xy).unwrap();
        ps2.add_f64_scalar("mean_incidence", 0.42).unwrap();
        ps2.add_f64_scalar("mean_range", 820_000.0).unwrap();
        ps2.write().unwrap();

        let mut ph2 = MatFile::new(patch.join("ph2.mat"));
        ph2.add_complex_f32_matrix("ph", n_ps, 4, ph).unwrap();
        ph2.write().unwrap();

        let mut pm2 = MatFile::new(patch.join("pm2.mat"));
        pm2.add_f64_col_vector("K_ps", rows.iter().map(|row| row.k).collect())
            .unwrap();
        pm2.add_f64_col_vector("C_ps", rows.iter().map(|row| row.c).collect())
            .unwrap();
        pm2.add_f64_col_vector("coh_ps", rows.iter().map(|row| row.coh).collect())
            .unwrap();
        pm2.add_complex_f32_matrix("ph_patch", n_ps, 3, ph_patch)
            .unwrap();
        pm2.add_f32_matrix("ph_res", n_ps, 3, ph_res).unwrap();
        pm2.write().unwrap();

        let mut bp2 = MatFile::new(patch.join("bp2.mat"));
        bp2.add_f32_matrix("bperp_mat", n_ps, 3, bp).unwrap();
        bp2.write().unwrap();

        let mut hgt2 = MatFile::new(patch.join("hgt2.mat"));
        hgt2.add_f32_col_vector("hgt", rows.iter().map(|row| row.hgt as f32).collect())
            .unwrap();
        hgt2.write().unwrap();

        let mut la2 = MatFile::new(patch.join("la2.mat"));
        la2.add_f64_col_vector("la", rows.iter().map(|row| row.la).collect())
            .unwrap();
        la2.write().unwrap();

        let mut rc2 = MatFile::new(patch.join("rc2.mat"));
        rc2.add_complex_f32_matrix("ph_rc", n_ps, 4, rc).unwrap();
        rc2.write().unwrap();

        write_psver(patch).unwrap();
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

    fn run_python_stage5_merge(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage5_merge_and_ifgstd; stage5_merge_and_ifgstd(Path(sys.argv[1]))";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage5 merge failed: {}\nstdout: {}",
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
