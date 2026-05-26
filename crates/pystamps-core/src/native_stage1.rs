use crate::CoreError;
use pystamps_mat::{
    f32_matrix, f64_matrix, write_baseline_artifact, write_da_artifact, write_hgt_artifact,
    write_la_artifact, write_phase_artifact, write_psver_artifact, MatData, Matrix, Ps1Artifact,
    VAR_BPERP, VAR_BPERP_MAT, VAR_DAY, VAR_IJ, VAR_LL0, VAR_LONLAT, VAR_MASTER_DAY, VAR_MASTER_IX,
    VAR_SORT_IX, VAR_XY,
};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

const DEFAULT_MEAN_RANGE: f64 = 830_000.0;
const DEFAULT_MEAN_INCIDENCE: f64 = 23.0_f64.to_radians();

pub fn run_stage1_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let ij_path = patch_dir.join("pscands.1.ij");
    let ph_path = patch_dir.join("pscands.1.ph");
    let ll_path = patch_dir.join("pscands.1.ll");
    for path in [&ij_path, &ph_path, &ll_path] {
        if !path.exists() {
            return stage1_err(format!("missing required input {}", path.display()));
        }
    }

    let ij = read_text_matrix(&ij_path)?;
    if ij.is_empty() {
        return stage1_err("pscands.1.ij is empty");
    }
    let n_ps = ij.len();
    if ij.iter().any(|row| row.len() < 3) {
        return stage1_err("pscands.1.ij must have at least three columns");
    }

    resolve_file(patch_dir, "width.txt")?;
    resolve_file(patch_dir, "len.txt")?;
    let metadata = resolve_stage1_metadata(patch_dir, &ij)?;
    let day_full = metadata.day_full;
    let bperp_full = metadata.bperp_full;
    let master_day = metadata.master_day;
    let master_ix = metadata.master_ix;
    let day_ix = metadata.day_ix;

    let ph_raw = read_complex_columns(&ph_path, n_ps)?;
    if ph_raw.cols != day_ix.len() {
        return stage1_err(format!(
            "phase file has {} columns but metadata has {} entries",
            ph_raw.cols,
            day_ix.len()
        ));
    }
    let mut ph_reordered = vec![(0.0_f32, 0.0_f32); n_ps * day_full.len()];
    for (sorted_col, &original_col) in day_ix.iter().enumerate() {
        let out_col = if sorted_col + 1 >= master_ix {
            sorted_col + 1
        } else {
            sorted_col
        };
        for row in 0..n_ps {
            ph_reordered[row * day_full.len() + out_col] =
                ph_raw.values[row * ph_raw.cols + original_col];
        }
    }
    for row in 0..n_ps {
        ph_reordered[row * day_full.len() + (master_ix - 1)] = (1.0, 0.0);
    }

    let lonlat_raw = read_binary_f32(&ll_path, BinaryKind::LonLat)?;
    if lonlat_raw.len() != n_ps * 2 {
        return stage1_err(format!(
            "pscands.1.ll has {} float32 values for {n_ps} candidates",
            lonlat_raw.len()
        ));
    }
    let lonlat: Vec<[f64; 2]> = lonlat_raw
        .chunks_exact(2)
        .map(|pair| [pair[0] as f64, pair[1] as f64])
        .collect();
    let heading_deg = stage1_heading_deg(patch_dir);
    let (xy_local, ll0) = local_xy_from_lonlat(&lonlat, heading_deg);
    let xy_sort = xy_local
        .iter()
        .map(|pair| [pair[0] as f32, pair[1] as f32])
        .collect::<Vec<_>>();
    let mut sort_ix: Vec<usize> = (0..n_ps).collect();
    sort_ix.sort_by(|&left, &right| {
        xy_sort[left][1]
            .total_cmp(&xy_sort[right][1])
            .then_with(|| xy_sort[left][0].total_cmp(&xy_sort[right][0]))
    });

    let mut ij_sorted = Vec::with_capacity(n_ps * ij[0].len());
    let mut lonlat_sorted = Vec::with_capacity(n_ps * 2);
    let mut xy_out = Vec::with_capacity(n_ps * 3);
    let mut ph_sorted = vec![(0.0_f32, 0.0_f32); n_ps * day_full.len()];
    for (out_row, &src_row) in sort_ix.iter().enumerate() {
        for col in 0..ij[src_row].len() {
            ij_sorted.push(if col == 0 {
                (out_row + 1) as f64
            } else {
                ij[src_row][col]
            });
        }
        lonlat_sorted.extend_from_slice(&lonlat[src_row]);
        xy_out.push((out_row + 1) as f32);
        xy_out.push(quantize_mm(xy_sort[src_row][0]));
        xy_out.push(quantize_mm(xy_sort[src_row][1]));
        for col in 0..day_full.len() {
            ph_sorted[out_row * day_full.len() + col] =
                ph_reordered[src_row * day_full.len() + col];
        }
    }

    let (mean_range, mean_incidence) =
        stage1_geometry(patch_dir, &ij).unwrap_or((DEFAULT_MEAN_RANGE, DEFAULT_MEAN_INCIDENCE));
    write_ps1(
        patch_dir,
        &ij_sorted,
        ij[0].len(),
        &lonlat_sorted,
        &xy_out,
        &bperp_full,
        &day_full,
        master_day,
        master_ix,
        n_ps,
        &sort_ix,
        ll0,
        mean_range,
        mean_incidence,
    )?;
    write_phase_artifact(patch_dir.join("ph1.mat"), n_ps, day_full.len(), ph_sorted)?;
    write_psver_artifact(patch_dir.join("psver.mat"), 1.0)?;

    if patch_dir.join("pscands.1.da").exists() {
        let da = read_text_flat(&patch_dir.join("pscands.1.da"))?;
        if da.len() == n_ps {
            let sorted = sort_ix.iter().map(|&ix| da[ix]).collect::<Vec<_>>();
            write_da_artifact(patch_dir.join("da1.mat"), sorted)?;
        }
    }

    if patch_dir.join("pscands.1.hgt").exists() {
        let hgt = read_binary_f32(&patch_dir.join("pscands.1.hgt"), BinaryKind::Generic)?;
        if hgt.len() == n_ps {
            let sorted = sort_ix.iter().map(|&ix| hgt[ix]).collect::<Vec<_>>();
            write_hgt_artifact(patch_dir.join("hgt1.mat"), sorted)?;
        }
    }

    if patch_dir.join("pscands.1.la").exists() {
        let la = read_text_flat(&patch_dir.join("pscands.1.la"))?;
        if la.len() == n_ps {
            let sorted = sort_ix.iter().map(|&ix| la[ix]).collect::<Vec<_>>();
            write_la_artifact(patch_dir.join("la1.mat"), sorted)?;
        }
    }

    let bperp_cols = day_full.len() - 1;
    let bperp_mat = if let Some(source) = metadata.bperp_mat {
        if source.rows != n_ps || source.cols != day_ix.len() {
            return stage1_err(format!(
                "metadata bperp_mat has shape {}x{} for {n_ps} candidates and {} interferograms",
                source.rows,
                source.cols,
                day_ix.len()
            ));
        }
        let mut sorted = Vec::with_capacity(n_ps * source.cols);
        for &src_row in &sort_ix {
            for &src_col in &day_ix {
                sorted.push(source.values[src_row * source.cols + src_col]);
            }
        }
        sorted
    } else {
        let no_master_bperp: Vec<f32> = bperp_full
            .iter()
            .enumerate()
            .filter_map(|(ix, &value)| (ix != master_ix - 1).then_some(value as f32))
            .collect();
        let mut tiled = Vec::with_capacity(n_ps * no_master_bperp.len());
        for _ in 0..n_ps {
            tiled.extend_from_slice(&no_master_bperp);
        }
        tiled
    };
    write_baseline_artifact(patch_dir.join("bp1.mat"), n_ps, bperp_cols, bperp_mat)?;

    Ok(format!("Stage 1 created ps1/ph1 for {n_ps} candidates"))
}

#[derive(Clone, Debug)]
struct Stage1Metadata {
    day_full: Vec<f64>,
    bperp_full: Vec<f64>,
    master_day: f64,
    master_ix: usize,
    day_ix: Vec<usize>,
    bperp_mat: Option<Matrix<f32>>,
}

fn resolve_stage1_metadata(patch_dir: &Path, ij: &[Vec<f64>]) -> Result<Stage1Metadata, CoreError> {
    let day_path = resolve_file_optional(patch_dir, "day.1.in");
    let master_day_path = resolve_file_optional(patch_dir, "master_day.1.in");
    let bperp_path = resolve_file_optional(patch_dir, "bperp.1.in");
    if let (Some(day_path), Some(master_day_path), Some(bperp_path)) =
        (day_path, master_day_path, bperp_path)
    {
        return metadata_from_text(&day_path, &master_day_path, &bperp_path, None);
    }

    if let Some(existing) = metadata_from_existing_ps1(patch_dir, ij.len())? {
        return Ok(existing);
    }

    metadata_from_snap(patch_dir, ij)
}

fn metadata_from_text(
    day_path: &Path,
    master_day_path: &Path,
    bperp_path: &Path,
    bperp_mat: Option<Matrix<f32>>,
) -> Result<Stage1Metadata, CoreError> {
    let day_raw = read_text_flat(day_path)?;
    let bperp_raw = read_text_flat(bperp_path)?;
    if day_raw.len() != bperp_raw.len() {
        return stage1_err(format!(
            "day.1.in has {} rows but bperp.1.in has {} rows",
            day_raw.len(),
            bperp_raw.len()
        ));
    }
    let master_day_yyyymmdd = read_text_flat(master_day_path)?
        .first()
        .copied()
        .ok_or_else(|| CoreError::NativeStage {
            stage: 1,
            message: "master_day.1.in is empty".to_string(),
        })?;
    let slave_day = day_raw
        .iter()
        .map(|&day| yyyymmdd_to_ordinal(day as i64))
        .collect::<Vec<_>>();
    let master_day = yyyymmdd_to_ordinal(master_day_yyyymmdd as i64);
    build_metadata(slave_day, bperp_raw, master_day, bperp_mat)
}

fn metadata_from_existing_ps1(
    patch_dir: &Path,
    n_ps: usize,
) -> Result<Option<Stage1Metadata>, CoreError> {
    let ps1_file = patch_dir.join("ps1.mat");
    if !ps1_file.exists() {
        return Ok(None);
    }
    let ps1 = MatData::read(&ps1_file)?;
    let day_full = ps1.get_f64_matrix(VAR_DAY)?.values;
    let bperp_full = ps1.get_f64_matrix(VAR_BPERP)?.values;
    let master_day = scalar_from_mat(&ps1, VAR_MASTER_DAY)?;
    let master_ix = scalar_from_mat(&ps1, VAR_MASTER_IX)?.round() as usize;
    if day_full.is_empty()
        || bperp_full.len() != day_full.len()
        || master_ix == 0
        || master_ix > day_full.len()
    {
        return Ok(None);
    }

    let mut slave_day = Vec::with_capacity(day_full.len() - 1);
    let mut slave_bperp = Vec::with_capacity(day_full.len() - 1);
    for (ix, (&day, &bperp)) in day_full.iter().zip(bperp_full.iter()).enumerate() {
        if ix != master_ix - 1 {
            slave_day.push(day);
            slave_bperp.push(bperp);
        }
    }

    let bperp_mat = match MatData::read(patch_dir.join("bp1.mat")) {
        Ok(bp1) => match bp1.get_f32_matrix(VAR_BPERP_MAT) {
            Ok(candidate) if candidate.rows == n_ps && candidate.cols == day_full.len() - 1 => {
                Some(candidate)
            }
            _ => None,
        },
        Err(_) => None,
    };
    Ok(Some(build_metadata(
        slave_day,
        slave_bperp,
        master_day,
        bperp_mat,
    )?))
}

fn build_metadata(
    slave_day: Vec<f64>,
    slave_bperp: Vec<f64>,
    master_day: f64,
    bperp_mat: Option<Matrix<f32>>,
) -> Result<Stage1Metadata, CoreError> {
    if slave_day.len() != slave_bperp.len() {
        return stage1_err(format!(
            "Stage 1 metadata mismatch: day has {} rows but bperp has {}",
            slave_day.len(),
            slave_bperp.len()
        ));
    }
    let mut day_ix: Vec<usize> = (0..slave_day.len()).collect();
    day_ix.sort_by(|&left, &right| slave_day[left].total_cmp(&slave_day[right]));
    let master_ix = day_ix
        .iter()
        .filter(|&&ix| slave_day[ix] < master_day)
        .count()
        + 1;

    let mut day_full = Vec::with_capacity(slave_day.len() + 1);
    let mut bperp_full = Vec::with_capacity(slave_day.len() + 1);
    for (sorted_ix, &source_ix) in day_ix.iter().enumerate() {
        if sorted_ix == master_ix - 1 {
            day_full.push(master_day);
            bperp_full.push(0.0);
        }
        day_full.push(slave_day[source_ix]);
        bperp_full.push(slave_bperp[source_ix]);
    }
    if master_ix == slave_day.len() + 1 {
        day_full.push(master_day);
        bperp_full.push(0.0);
    }

    Ok(Stage1Metadata {
        day_full,
        bperp_full,
        master_day,
        master_ix,
        day_ix,
        bperp_mat,
    })
}

fn scalar_from_mat(mat: &MatData, name: &str) -> Result<f64, CoreError> {
    mat.get_f64_matrix(name)?
        .values
        .first()
        .copied()
        .ok_or_else(|| CoreError::NativeStage {
            stage: 1,
            message: format!("{name} is empty"),
        })
}

#[derive(Clone, Debug)]
struct SnapRecord {
    master: String,
    slave: String,
    base_file: PathBuf,
}

fn metadata_from_snap(patch_dir: &Path, ij: &[Vec<f64>]) -> Result<Stage1Metadata, CoreError> {
    let dataset_root = stage1_dataset_root(patch_dir);
    let records = snap_ifg_records(&dataset_root)?;
    let master_day = single_master_day(&records)?;
    let rslc_par = resolve_rslc_par(&dataset_root, &master_day)?;

    let mut bperp_cols = Vec::with_capacity(records.len());
    for record in &records {
        bperp_cols.push(snap_patch_bperp_vector(&record.base_file, &rslc_par, ij)?);
    }
    if bperp_cols.is_empty() {
        return stage1_err(
            "Stage 1 SNAP metadata synthesis did not produce any perpendicular baselines",
        );
    }

    let n_ps = ij.len();
    let n_cols = bperp_cols.len();
    let mut bperp_values = Vec::with_capacity(n_ps * n_cols);
    for row in 0..n_ps {
        for col in &bperp_cols {
            bperp_values.push(col[row]);
        }
    }
    let bperp_mat = Matrix {
        name: VAR_BPERP_MAT.to_string(),
        rows: n_ps,
        cols: n_cols,
        values: bperp_values,
    };

    let day_path = patch_dir.join("day.1.in");
    let master_day_path = patch_dir.join("master_day.1.in");
    let bperp_path = patch_dir.join("bperp.1.in");
    write_lines_if_missing(
        &day_path,
        records.iter().map(|record| record.slave.clone()).collect(),
    )?;
    write_lines_if_missing(&master_day_path, vec![master_day])?;
    let mean_bperp = (0..n_cols)
        .map(|col| {
            let sum: f64 = bperp_cols[col].iter().map(|&value| value as f64).sum();
            format!("{:.12}", sum / n_ps as f64)
        })
        .collect::<Vec<_>>();
    write_lines_if_missing(&bperp_path, mean_bperp)?;

    metadata_from_text(&day_path, &master_day_path, &bperp_path, Some(bperp_mat))
}

fn stage1_dataset_root(path: &Path) -> PathBuf {
    if path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name.starts_with("PATCH_"))
    {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    }
}

fn snap_ifg_records(dataset_root: &Path) -> Result<Vec<SnapRecord>, CoreError> {
    let diff_dir = dataset_root.join("diff0");
    if !diff_dir.exists() {
        return stage1_err(format!(
            "Stage 1 SNAP metadata synthesis requires {}",
            diff_dir.display()
        ));
    }
    let mut records = Vec::new();
    let entries = fs::read_dir(&diff_dir).map_err(|source| CoreError::FileIo {
        path: diff_dir.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| CoreError::FileIo {
            path: diff_dir.clone(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("base") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if let Some((master, slave)) = parse_date_pair_from_name(name) {
            records.push(SnapRecord {
                master,
                slave,
                base_file: path,
            });
        }
    }
    records.sort_by(|left, right| left.base_file.cmp(&right.base_file));
    if records.is_empty() {
        return stage1_err(format!(
            "Stage 1 SNAP metadata synthesis requires parseable diff0/*.base files in {}",
            diff_dir.display()
        ));
    }
    Ok(records)
}

fn single_master_day(records: &[SnapRecord]) -> Result<String, CoreError> {
    let mut masters = records
        .iter()
        .map(|record| record.master.clone())
        .collect::<Vec<_>>();
    masters.sort();
    masters.dedup();
    if masters.len() != 1 {
        return stage1_err(format!(
            "Stage 1 SNAP metadata synthesis requires a single-master diff0 stack; found masters {}",
            masters.join(", ")
        ));
    }
    Ok(masters.remove(0))
}

fn parse_date_pair_from_name(name: &str) -> Option<(String, String)> {
    let bytes = name.as_bytes();
    if bytes.len() < 17 {
        return None;
    }
    for ix in 0..=bytes.len() - 17 {
        if bytes[ix + 8] != b'_' {
            continue;
        }
        let master = &bytes[ix..ix + 8];
        let slave = &bytes[ix + 9..ix + 17];
        if master.iter().all(u8::is_ascii_digit) && slave.iter().all(u8::is_ascii_digit) {
            return Some((
                String::from_utf8(master.to_vec()).ok()?,
                String::from_utf8(slave.to_vec()).ok()?,
            ));
        }
    }
    None
}

fn resolve_rslc_par(dataset_root: &Path, master_day: &str) -> Result<PathBuf, CoreError> {
    let rslc_dir = dataset_root.join("rslc");
    let preferred = rslc_dir.join(format!("{master_day}.rslc.par"));
    if preferred.exists() {
        return Ok(preferred);
    }
    let entries = match fs::read_dir(&rslc_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == ErrorKind::NotFound => {
            return stage1_err(format!(
                "Stage 1 SNAP metadata synthesis requires rslc/*.rslc.par with a file matching master date {master_day} under {}",
                rslc_dir.display()
            ));
        }
        Err(source) => {
            return Err(CoreError::FileIo {
                path: rslc_dir,
                source,
            });
        }
    };
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| CoreError::FileIo {
            path: rslc_dir.clone(),
            source,
        })?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".rslc.par"))
        {
            candidates.push(path);
        }
    }
    candidates.sort();
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }
    stage1_err(format!(
        "Stage 1 SNAP metadata synthesis requires rslc/*.rslc.par with a file matching master date {master_day} under {}",
        rslc_dir.display()
    ))
}

fn snap_patch_bperp_vector(
    base_file: &Path,
    rslc_par: &Path,
    ij: &[Vec<f64>],
) -> Result<Vec<f32>, CoreError> {
    let b_tcn = read_named_float_vector(base_file, "initial_baseline(TCN)", 3)?;
    let br_tcn = read_named_float_vector(base_file, "initial_baseline_rate", 3)?;
    let range_pixel_spacing = read_named_scalar(rslc_par, "range_pixel_spacing")?;
    let near_range_slc = read_named_scalar(rslc_par, "near_range_slc")?;
    let sar_to_earth_center = read_named_scalar(rslc_par, "sar_to_earth_center")?;
    let earth_radius_below_sensor = read_named_scalar(rslc_par, "earth_radius_below_sensor")?;
    let azimuth_lines = read_named_scalar(rslc_par, "azimuth_lines")?;
    let prf = read_named_scalar(rslc_par, "prf")?;
    if prf == 0.0 {
        return stage1_err(format!("Invalid PRF in {}", rslc_par.display()));
    }

    let mean_az = azimuth_lines / 2.0 - 0.5;
    Ok(ij
        .iter()
        .map(|row| {
            let azimuth = row[1];
            let rg = near_range_slc + row[2] * range_pixel_spacing;
            let look_arg = (sar_to_earth_center.powi(2) + rg.powi(2)
                - earth_radius_below_sensor.powi(2))
                / (2.0 * sar_to_earth_center * rg);
            let look = look_arg.clamp(-1.0, 1.0).acos();
            let bc = b_tcn[1] + br_tcn[1] * (azimuth - mean_az) / prf;
            let bn = b_tcn[2] + br_tcn[2] * (azimuth - mean_az) / prf;
            (bc * look.cos() - bn * look.sin()) as f32
        })
        .collect())
}

fn read_named_scalar(path: &Path, key: &str) -> Result<f64, CoreError> {
    read_named_float_vector(path, key, 1).map(|values| values[0])
}

fn read_named_float_vector(path: &Path, key: &str, count: usize) -> Result<Vec<f64>, CoreError> {
    let text = fs::read_to_string(path).map_err(|source| CoreError::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    for line in text.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if name.trim() != key {
            continue;
        }
        let values = extract_float_tokens(rest);
        if values.len() >= count {
            return Ok(values.into_iter().take(count).collect());
        }
        break;
    }
    stage1_err(format!("Unable to parse '{key}' from {}", path.display()))
}

fn extract_float_tokens(text: &str) -> Vec<f64> {
    text.replace(',', " ")
        .split_whitespace()
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn write_lines_if_missing(path: &Path, values: Vec<String>) -> Result<(), CoreError> {
    if path.exists() {
        return Ok(());
    }
    let mut text = values.join("\n");
    text.push('\n');
    fs::write(path, text).map_err(|source| CoreError::FileIo {
        path: path.to_path_buf(),
        source,
    })
}

fn stage1_heading_deg(patch_dir: &Path) -> Option<f64> {
    let dataset_root = stage1_dataset_root(patch_dir);
    if !dataset_root.join("diff0").exists() || !dataset_root.join("rslc").exists() {
        return None;
    }
    let records = snap_ifg_records(&dataset_root).ok()?;
    let master_day = single_master_day(&records).ok()?;
    let rslc_par = resolve_rslc_par(&dataset_root, &master_day).ok()?;
    read_named_scalar(&rslc_par, "heading").ok()
}

fn stage1_geometry(patch_dir: &Path, ij: &[Vec<f64>]) -> Option<(f64, f64)> {
    let dataset_root = stage1_dataset_root(patch_dir);
    if !dataset_root.join("diff0").exists() || !dataset_root.join("rslc").exists() {
        return None;
    }
    let records = snap_ifg_records(&dataset_root).ok()?;
    let master_day = single_master_day(&records).ok()?;
    let rslc_par = resolve_rslc_par(&dataset_root, &master_day).ok()?;
    let range_pixel_spacing = read_named_scalar(&rslc_par, "range_pixel_spacing").ok()?;
    let near_range_slc = read_named_scalar(&rslc_par, "near_range_slc").ok()?;
    let sar_to_earth_center = read_named_scalar(&rslc_par, "sar_to_earth_center").ok()?;
    let earth_radius_below_sensor =
        read_named_scalar(&rslc_par, "earth_radius_below_sensor").ok()?;
    let center_range_slc = read_named_scalar(&rslc_par, "center_range_slc").ok()?;
    let mut sum = 0.0;
    for row in ij {
        let rg = near_range_slc + row[2] * range_pixel_spacing;
        let incidence_arg =
            (sar_to_earth_center.powi(2) - earth_radius_below_sensor.powi(2) - rg.powi(2))
                / (2.0 * earth_radius_below_sensor * rg);
        sum += incidence_arg.clamp(-1.0, 1.0).acos();
    }
    Some((center_range_slc, sum / ij.len() as f64))
}

fn write_ps1(
    patch_dir: &Path,
    ij_sorted: &[f64],
    ij_cols: usize,
    lonlat_sorted: &[f64],
    xy_out: &[f32],
    bperp_full: &[f64],
    day_full: &[f64],
    master_day: f64,
    master_ix: usize,
    n_ps: usize,
    sort_ix: &[usize],
    ll0: [f64; 2],
    mean_range: f64,
    mean_incidence: f64,
) -> Result<(), CoreError> {
    let ps = Ps1Artifact {
        ij: f64_matrix(VAR_IJ, n_ps, ij_cols, ij_sorted.to_vec())?,
        lonlat: f64_matrix(VAR_LONLAT, n_ps, 2, lonlat_sorted.to_vec())?,
        xy: f32_matrix(VAR_XY, n_ps, 3, xy_out.to_vec())?,
        bperp: f64_matrix(VAR_BPERP, 1, bperp_full.len(), bperp_full.to_vec())?,
        day: f64_matrix(VAR_DAY, 1, day_full.len(), day_full.to_vec())?,
        master_day,
        master_ix: master_ix as f64,
        n_ifg: day_full.len() as f64,
        n_image: day_full.len() as f64,
        n_ps: n_ps as f64,
        sort_ix: f64_matrix(
            VAR_SORT_IX,
            1,
            sort_ix.len(),
            sort_ix.iter().map(|&ix| (ix + 1) as f64).collect(),
        )?,
        ll0: f64_matrix(VAR_LL0, 1, 2, vec![ll0[0], ll0[1]])?,
        mean_range,
        mean_incidence,
    };
    Ok(ps.write(patch_dir.join("ps1.mat"))?)
}

#[derive(Clone, Copy)]
enum BinaryKind {
    LonLat,
    Generic,
}

struct ComplexMatrix {
    cols: usize,
    values: Vec<(f32, f32)>,
}

fn stage1_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(CoreError::NativeStage {
        stage: 1,
        message: message.into(),
    })
}

fn resolve_file(patch_dir: &Path, filename: &str) -> Result<PathBuf, CoreError> {
    resolve_file_optional(patch_dir, filename).ok_or_else(|| CoreError::NativeStage {
        stage: 1,
        message: format!("{filename} not found near {}", patch_dir.display()),
    })
}

fn resolve_file_optional(patch_dir: &Path, filename: &str) -> Option<PathBuf> {
    for candidate in [
        patch_dir.join(filename),
        patch_dir.parent().unwrap_or(patch_dir).join(filename),
        patch_dir
            .parent()
            .and_then(Path::parent)
            .unwrap_or(patch_dir)
            .join(filename),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn read_text_matrix(path: &Path) -> Result<Vec<Vec<f64>>, CoreError> {
    let text = fs::read_to_string(path).map_err(|source| CoreError::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = line
            .split_whitespace()
            .map(|token| {
                token.parse::<f64>().map_err(|err| CoreError::NativeStage {
                    stage: 1,
                    message: format!("unable to parse {} in {}: {err}", token, path.display()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !row.is_empty() {
            rows.push(row);
        }
    }
    if let Some(width) = rows.first().map(Vec::len) {
        if rows.iter().any(|row| row.len() != width) {
            return stage1_err(format!("inconsistent row width in {}", path.display()));
        }
    }
    Ok(rows)
}

fn read_text_flat(path: &Path) -> Result<Vec<f64>, CoreError> {
    Ok(read_text_matrix(path)?.into_iter().flatten().collect())
}

fn read_complex_columns(path: &Path, rows: usize) -> Result<ComplexMatrix, CoreError> {
    let raw = read_binary_f32(path, BinaryKind::Generic)?;
    if raw.len() % (2 * rows) != 0 {
        return stage1_err(format!(
            "unexpected binary size for phase file {}",
            path.display()
        ));
    }
    let cols = raw.len() / (2 * rows);
    let mut values = vec![(0.0_f32, 0.0_f32); rows * cols];
    for col in 0..cols {
        let offset = col * rows * 2;
        for row in 0..rows {
            values[row * cols + col] = (raw[offset + row * 2], raw[offset + row * 2 + 1]);
        }
    }
    Ok(ComplexMatrix { cols, values })
}

fn read_binary_f32(path: &Path, kind: BinaryKind) -> Result<Vec<f32>, CoreError> {
    let bytes = fs::read(path).map_err(|source| CoreError::FileIo {
        path: path.to_path_buf(),
        source,
    })?;
    if bytes.len() % 4 != 0 {
        return stage1_err(format!(
            "{} byte count is not divisible by 4",
            path.display()
        ));
    }
    let little = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let big = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok(if endian_score(&big, kind) > endian_score(&little, kind) {
        big
    } else {
        little
    })
}

fn endian_score(values: &[f32], kind: BinaryKind) -> i64 {
    let finite = values.iter().filter(|value| value.is_finite()).count() as i64;
    let plausible = match kind {
        BinaryKind::LonLat => values
            .chunks_exact(2)
            .filter(|pair| pair[0].abs() <= 180.0 && pair[1].abs() <= 90.0)
            .count() as i64,
        BinaryKind::Generic => values
            .iter()
            .filter(|value| **value == 0.0 || (value.abs() >= 1.0e-12 && value.abs() <= 1.0e12))
            .count() as i64,
    };
    finite + plausible
}

fn yyyymmdd_to_ordinal(value: i64) -> f64 {
    let year = value / 10_000;
    let month = (value % 10_000) / 100;
    let day = value % 100;
    days_from_civil(year, month, day) as f64 + 719_529.0
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - (month <= 2) as i64;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn quantize_mm(value: f32) -> f32 {
    (value * 1000.0).round() / 1000.0
}

fn local_xy_from_lonlat(
    lonlat: &[[f64; 2]],
    heading_deg: Option<f64>,
) -> (Vec<[f64; 2]>, [f64; 2]) {
    let min_lon = lonlat
        .iter()
        .map(|pair| pair[0])
        .fold(f64::INFINITY, f64::min);
    let max_lon = lonlat
        .iter()
        .map(|pair| pair[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lat = lonlat
        .iter()
        .map(|pair| pair[1])
        .fold(f64::INFINITY, f64::min);
    let max_lat = lonlat
        .iter()
        .map(|pair| pair[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let origin = [(min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0];
    let origin_rad = [origin[0].to_radians(), origin[1].to_radians()];
    let a = 6_378_137.0_f64;
    let e = 0.082_094_437_949_70_f64;
    let m0 = meridian_arc(a, e, origin_rad[1]);
    let mut xy: Vec<[f64; 2]> = lonlat
        .iter()
        .map(|pair| {
            let lon = pair[0].to_radians();
            let lat = pair[1].to_radians();
            if lat != 0.0 {
                let dlambda = lon - origin_rad[0];
                let m = meridian_arc(a, e, lat);
                let n = a / (1.0 - e.powi(2) * lat.sin().powi(2)).sqrt();
                let angle = dlambda * lat.sin();
                let cot = 1.0 / lat.tan();
                [
                    n * cot * angle.sin(),
                    m - m0 + n * cot * (1.0 - angle.cos()),
                ]
            } else {
                [a * (lon - origin_rad[0]), -m0]
            }
        })
        .collect();
    if let Some(heading_deg) = heading_deg {
        let mut theta = (180.0 - heading_deg).to_radians();
        if theta > std::f64::consts::PI {
            theta -= 2.0 * std::f64::consts::PI;
        }
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let rotated = xy
            .iter()
            .map(|pair| {
                [
                    cos_t * pair[0] + sin_t * pair[1],
                    -sin_t * pair[0] + cos_t * pair[1],
                ]
            })
            .collect::<Vec<_>>();
        if range(rotated.iter().map(|pair| pair[0])) < range(xy.iter().map(|pair| pair[0]))
            && range(rotated.iter().map(|pair| pair[1])) < range(xy.iter().map(|pair| pair[1]))
        {
            xy = rotated;
        }
    }
    (xy, origin)
}

fn range(values: impl Iterator<Item = f64>) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    max - min
}

fn meridian_arc(a: f64, e: f64, lat: f64) -> f64 {
    a * ((1.0 - e.powi(2) / 4.0 - 3.0 * e.powi(4) / 64.0 - 5.0 * e.powi(6) / 256.0) * lat
        - (3.0 * e.powi(2) / 8.0 + 3.0 * e.powi(4) / 32.0 + 45.0 * e.powi(6) / 1024.0)
            * (2.0 * lat).sin()
        + (15.0 * e.powi(4) / 256.0 + 45.0 * e.powi(6) / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e.powi(6) / 3072.0) * (6.0 * lat).sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::io::Write;
    use std::process::Command;

    #[test]
    fn stage1_native_writes_expected_artifacts() {
        let root = temp_root("native");
        let _ = fs::remove_dir_all(&root);
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::write(root.join("width.txt"), "10\n").unwrap();
        fs::write(root.join("len.txt"), "10\n").unwrap();
        fs::write(root.join("day.1.in"), "20200104\n20200102\n").unwrap();
        fs::write(root.join("master_day.1.in"), "20200103\n").unwrap();
        fs::write(root.join("bperp.1.in"), "20\n10\n").unwrap();
        fs::write(patch.join("pscands.1.ij"), "9 2 2\n8 1 1\n7 3 3\n").unwrap();
        write_f32_file(
            &patch.join("pscands.1.ll"),
            &[-120.0, 35.0, -119.99, 35.01, -120.01, 34.99],
        );
        write_f32_file(
            &patch.join("pscands.1.ph"),
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0],
        );

        let details = run_stage1_native(&patch).unwrap();

        assert_eq!(details, "Stage 1 created ps1/ph1 for 3 candidates");
        for name in ["ps1.mat", "ph1.mat", "bp1.mat", "psver.mat"] {
            assert!(patch.join(name).exists(), "{name} missing");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn synthetic_raw_stage1_matches_python_reference() {
        let root = temp_root("parity");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_raw_fixture(&python_root);
        create_raw_fixture(&rust_root);

        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage1_load_initial; stage1_load_initial(Path(sys.argv[1]))";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(python_root.join("PATCH_1"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage1 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        run_stage1_native(rust_root.join("PATCH_1")).unwrap();

        let specs = [
            ArtifactComparisonSpec::new(
                "PATCH_1/ps1.mat",
                [
                    "ij",
                    "lonlat",
                    "xy",
                    "bperp",
                    "day",
                    "master_day",
                    "master_ix",
                    "n_ifg",
                    "n_image",
                    "n_ps",
                    "sort_ix",
                    "ll0",
                    "mean_range",
                    "mean_incidence",
                ],
            ),
            ArtifactComparisonSpec::new("PATCH_1/ph1.mat", ["ph"]),
            ArtifactComparisonSpec::new("PATCH_1/bp1.mat", ["bperp_mat"]),
            ArtifactComparisonSpec::new("PATCH_1/psver.mat", ["psver"]),
            ArtifactComparisonSpec::new("PATCH_1/da1.mat", ["D_A"]),
            ArtifactComparisonSpec::new("PATCH_1/hgt1.mat", ["hgt"]),
        ];
        let summary = compare_fixture_artifacts(
            1,
            "patch",
            "synthetic_raw",
            &python_root,
            &rust_root,
            &specs,
            &ParityTolerance::default(),
        )
        .unwrap();

        assert!(
            summary.all_ok(),
            "Stage 1 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validation_stage1_ps1_ij_matches_golden_for_representative_patches() {
        let golden_root = workspace_validation_root();
        assert!(
            golden_root.join("PATCH_1/ps1.mat").is_file(),
            "validation golden dataset is missing at {}",
            golden_root.display()
        );
        let root = prepare_validation_stage1_fixture(&golden_root, &["PATCH_1", "PATCH_2"]);

        for patch_name in ["PATCH_1", "PATCH_2"] {
            let patch_dir = root.join(patch_name);
            run_stage1_native(&patch_dir).unwrap();
        }
        assert_stage1_ps1_matches_golden_with_python(&golden_root, &root, &["PATCH_1", "PATCH_2"]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validation_stage1_ij_parity_rejects_zero_based_shifts() {
        let golden_root = workspace_validation_root();
        let script = r#"
import sys
from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat

root = Path(sys.argv[1])
ij = np.asarray(read_mat(root / "PATCH_1" / "ps1.mat")["ij"], dtype=np.float64)

def expect_ij_mismatch(label, shifted):
    try:
        np.testing.assert_array_equal(ij, shifted)
    except AssertionError:
        return
    raise AssertionError(f"{label} unexpectedly passed exact ij parity")

row_shifted = ij.copy()
row_shifted[:, 0] -= 1
expect_ij_mismatch("zero-based row indices", row_shifted)

column_shifted = ij.copy()
column_shifted[:, 1] -= 1
expect_ij_mismatch("zero-based coordinate columns", column_shifted)
"#;
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(&golden_root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "negative ij shift check failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn stage1_reuses_existing_ps1_metadata_without_text_files() {
        let root = temp_root("existing");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::write(patch.join("width.txt"), "10\n").unwrap();
        fs::write(patch.join("len.txt"), "10\n").unwrap();
        fs::write(patch.join("pscands.1.ij"), "1 10 20\n2 30 40\n").unwrap();
        write_f32_file(&patch.join("pscands.1.ll"), &[12.0, 45.0, 13.0, 46.0]);
        write_f32_file(
            &patch.join("pscands.1.ph"),
            &[0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
        );
        write_existing_ps1(&patch);

        run_stage1_native(&patch).unwrap();

        let ps1 = MatData::read(patch.join("ps1.mat")).unwrap();
        assert_eq!(
            ps1.get_f64_matrix(VAR_DAY).unwrap().values,
            vec![738949.0, 738961.0, 738973.0]
        );
        assert_eq!(scalar_from_mat(&ps1, VAR_MASTER_IX).unwrap(), 1.0);
        let bp1 = MatData::read(patch.join("bp1.mat")).unwrap();
        let bperp = bp1.get_f32_matrix(VAR_BPERP_MAT).unwrap();
        assert_eq!(bperp.rows, 2);
        assert_eq!(bperp.cols, 2);
        assert_eq!(bperp.values, vec![-354.25, -148.0, -354.125, -147.875]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage1_synthesizes_snap_metadata() {
        let root = temp_root("snap");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::create_dir_all(root.join("diff0")).unwrap();
        fs::create_dir_all(root.join("rslc")).unwrap();
        fs::write(root.join("width.txt"), "10\n").unwrap();
        fs::write(root.join("len.txt"), "10\n").unwrap();
        fs::write(patch.join("pscands.1.ij"), "1 10 20\n2 30 40\n").unwrap();
        write_f32_file(&patch.join("pscands.1.ll"), &[12.0, 45.0, 13.0, 46.0]);
        write_f32_file(
            &patch.join("pscands.1.ph"),
            &[0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
        );
        fs::write(
            root.join("diff0").join("20200103_20200104.base"),
            "initial_baseline(TCN): 0 20 10\ninitial_baseline_rate: 0 0.2 0.1\n",
        )
        .unwrap();
        fs::write(
            root.join("diff0").join("20200103_20200102.base"),
            "initial_baseline(TCN): 0 10 5\ninitial_baseline_rate: 0 0.1 0.05\n",
        )
        .unwrap();
        fs::write(
            root.join("rslc").join("20200103.rslc.par"),
            "\
range_pixel_spacing: 2.3
near_range_slc: 830000
center_range_slc: 831000
sar_to_earth_center: 7000000
earth_radius_below_sensor: 6371000
azimuth_lines: 100
prf: 1000
heading: 12
",
        )
        .unwrap();

        run_stage1_native(&patch).unwrap();

        assert!(patch.join("day.1.in").exists());
        assert!(patch.join("master_day.1.in").exists());
        assert!(patch.join("bperp.1.in").exists());
        let bp1 = MatData::read(patch.join("bp1.mat")).unwrap();
        let bperp = bp1.get_f32_matrix(VAR_BPERP_MAT).unwrap();
        assert_eq!((bperp.rows, bperp.cols), (2, 2));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_phase_input_fails_without_outputs() {
        let root = temp_root("missing-ph");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::write(root.join("width.txt"), "10\n").unwrap();
        fs::write(root.join("len.txt"), "10\n").unwrap();
        fs::write(root.join("day.1.in"), "20200104\n").unwrap();
        fs::write(root.join("master_day.1.in"), "20200103\n").unwrap();
        fs::write(root.join("bperp.1.in"), "20\n").unwrap();
        fs::write(patch.join("pscands.1.ij"), "1 2 2\n").unwrap();
        write_f32_file(&patch.join("pscands.1.ll"), &[-120.0, 35.0]);

        let err = run_stage1_native(&patch).unwrap_err();

        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 1);
                assert!(message.contains("pscands.1.ph"));
            }
            other => panic!("expected structured Stage 1 error, got {other:?}"),
        }
        for name in ["ps1.mat", "ph1.mat", "bp1.mat", "psver.mat"] {
            assert!(!patch.join(name).exists(), "{name} should not be written");
        }
        fs::remove_dir_all(root).unwrap();
    }

    fn write_f32_file(path: &Path, values: &[f32]) {
        let mut file = fs::File::create(path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn create_raw_fixture(root: &Path) {
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::write(root.join("width.txt"), "10\n").unwrap();
        fs::write(root.join("len.txt"), "10\n").unwrap();
        fs::write(root.join("day.1.in"), "20200104\n20200102\n").unwrap();
        fs::write(root.join("master_day.1.in"), "20200103\n").unwrap();
        fs::write(root.join("bperp.1.in"), "20\n10\n").unwrap();
        fs::write(patch.join("pscands.1.ij"), "9 2 2\n8 1 1\n7 3 3\n").unwrap();
        fs::write(patch.join("pscands.1.da"), "0.4\n0.2\n0.3\n").unwrap();
        write_f32_file(
            &patch.join("pscands.1.ll"),
            &[-120.0, 35.0, -119.99, 35.01, -120.01, 34.99],
        );
        write_f32_file(&patch.join("pscands.1.hgt"), &[100.0, 200.0, 150.0]);
        write_f32_file(
            &patch.join("pscands.1.ph"),
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0],
        );
    }

    fn write_existing_ps1(patch: &Path) {
        let mut ps1 = pystamps_mat::MatFile::new(patch.join("ps1.mat"));
        ps1.add_f64_row_vector(VAR_DAY, vec![738949.0, 738961.0, 738973.0])
            .unwrap();
        ps1.add_f64_scalar(VAR_MASTER_DAY, 738949.0).unwrap();
        ps1.add_f64_scalar(VAR_MASTER_IX, 1.0).unwrap();
        ps1.add_f32_row_vector(VAR_BPERP, vec![0.0, -354.25, -148.0])
            .unwrap();
        ps1.write().unwrap();

        let mut bp1 = pystamps_mat::MatFile::new(patch.join("bp1.mat"));
        bp1.add_f32_matrix(
            VAR_BPERP_MAT,
            2,
            2,
            vec![-354.25, -148.0, -354.125, -147.875],
        )
        .unwrap();
        bp1.write().unwrap();
    }

    fn workspace_validation_root() -> PathBuf {
        workspace_root().join("inputs_and_outputs/InSAR_dataset_test")
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf()
    }

    fn prepare_validation_stage1_fixture(golden_root: &Path, patch_names: &[&str]) -> PathBuf {
        let root = temp_workspace_root("validation-parity");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        for filename in ["width.txt", "len.txt"] {
            link_or_copy(&golden_root.join(filename), &root.join(filename));
        }

        copy_matching_files(&golden_root.join("diff0"), &root.join("diff0"), "base");
        copy_matching_files(&golden_root.join("rslc"), &root.join("rslc"), "par");
        fs::write(
            root.join("patch.list"),
            patch_names
                .iter()
                .map(|name| format!("{name}\n"))
                .collect::<String>(),
        )
        .unwrap();

        for patch_name in patch_names {
            let golden_patch = golden_root.join(patch_name);
            let patch = root.join(patch_name);
            fs::create_dir_all(&patch).unwrap();
            for filename in [
                "pscands.1.ij",
                "pscands.1.ll",
                "pscands.1.ph",
                "pscands.1.da",
                "pscands.1.hgt",
                "pscands.1.la",
            ] {
                let source = golden_patch.join(filename);
                if source.exists() {
                    link_or_copy(&source, &patch.join(filename));
                }
            }
        }
        root
    }

    fn copy_matching_files(source_dir: &Path, target_dir: &Path, extension: &str) {
        fs::create_dir_all(target_dir).unwrap();
        for entry in fs::read_dir(source_dir).unwrap() {
            let source = entry.unwrap().path();
            if source.extension().and_then(|value| value.to_str()) == Some(extension) {
                link_or_copy(&source, &target_dir.join(source.file_name().unwrap()));
            }
        }
    }

    fn link_or_copy(source: &Path, target: &Path) {
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        if fs::hard_link(source, target).is_err() {
            fs::copy(source, target).unwrap();
        }
    }

    fn assert_stage1_ps1_matches_golden_with_python(
        golden_root: &Path,
        observed_root: &Path,
        patch_names: &[&str],
    ) {
        let script = r#"
import sys
from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat

golden_root = Path(sys.argv[1])
observed_root = Path(sys.argv[2])
patches = sys.argv[3:]
keys = [
    "bperp",
    "day",
    "ij",
    "ll0",
    "lonlat",
    "master_day",
    "master_ix",
    "mean_incidence",
    "mean_range",
    "n_ifg",
    "n_image",
    "n_ps",
    "sort_ix",
    "xy",
]
tolerances = {
    "bperp": (1.0e-4, 1.0e-5),
    "xy": (1.0e-4, 1.0e-5),
    "ll0": (1.0e-6, 1.0e-8),
    "lonlat": (1.0e-6, 1.0e-8),
    "mean_incidence": (1.0e-6, 1.0e-8),
    "mean_range": (1.0e-6, 1.0e-8),
}

for patch in patches:
    golden = read_mat(golden_root / patch / "ps1.mat")
    observed = read_mat(observed_root / patch / "ps1.mat")
    if sorted(golden) != sorted(observed):
        raise AssertionError(f"{patch} ps1.mat keys differ: {sorted(observed)} != {sorted(golden)}")
    for key in keys:
        expected = np.asarray(golden[key])
        actual = np.asarray(observed[key])
        if actual.shape != expected.shape:
            raise AssertionError(f"{patch}/{key} shape {actual.shape} != {expected.shape}")
        atol, rtol = tolerances.get(key, (0.0, 0.0))
        if atol == 0.0 and rtol == 0.0:
            ok = np.array_equal(expected, actual)
        else:
            ok = np.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True)
        if not ok:
            max_abs = float(np.nanmax(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
            raise AssertionError(f"{patch}/{key} max_abs={max_abs} exceeds atol={atol} rtol={rtol}")
    np.testing.assert_array_equal(golden["ij"][0, :], observed["ij"][0, :])
    np.testing.assert_array_equal(golden["ij"][-1, :], observed["ij"][-1, :])
"#;
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(golden_root)
            .arg(observed_root)
            .args(patch_names)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Stage 1 validation parity check failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn temp_workspace_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        workspace_root().join("target").join(format!(
            "pystamps-stage1-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "pystamps-stage1-{name}-{}-{nanos}",
            std::process::id()
        ))
    }
}
