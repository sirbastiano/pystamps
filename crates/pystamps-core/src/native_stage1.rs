use crate::CoreError;
use pystamps_mat::MatFile;
use std::fs;
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

    resolve_file(patch_dir, "width.txt")?;
    resolve_file(patch_dir, "len.txt")?;
    let day_path = resolve_file(patch_dir, "day.1.in")?;
    let master_day_path = resolve_file(patch_dir, "master_day.1.in")?;
    let bperp_path = resolve_file(patch_dir, "bperp.1.in")?;

    let ij = read_text_matrix(&ij_path)?;
    if ij.is_empty() {
        return stage1_err("pscands.1.ij is empty");
    }
    let n_ps = ij.len();
    let day_raw = read_text_flat(&day_path)?;
    let bperp_raw = read_text_flat(&bperp_path)?;
    if day_raw.len() != bperp_raw.len() {
        return stage1_err(format!(
            "day.1.in has {} rows but bperp.1.in has {} rows",
            day_raw.len(),
            bperp_raw.len()
        ));
    }
    let master_day_yyyymmdd = read_text_flat(&master_day_path)?
        .first()
        .copied()
        .ok_or_else(|| CoreError::NativeStage {
            stage: 1,
            message: "master_day.1.in is empty".to_string(),
        })?;

    let mut slave: Vec<(f64, f64, usize)> = day_raw
        .iter()
        .zip(bperp_raw.iter())
        .enumerate()
        .map(|(ix, (&day, &bp))| (yyyymmdd_to_ordinal(day as i64), bp, ix))
        .collect();
    slave.sort_by(|left, right| left.0.total_cmp(&right.0));
    let master_day = yyyymmdd_to_ordinal(master_day_yyyymmdd as i64);
    let master_ix = slave.iter().filter(|(day, _, _)| *day < master_day).count() + 1;

    let mut day_full = Vec::with_capacity(slave.len() + 1);
    let mut bperp_full = Vec::with_capacity(slave.len() + 1);
    for (ix, (day, bp, _)) in slave.iter().enumerate() {
        if ix == master_ix - 1 {
            day_full.push(master_day);
            bperp_full.push(0.0_f64);
        }
        day_full.push(*day);
        bperp_full.push(*bp);
    }
    if master_ix == slave.len() + 1 {
        day_full.push(master_day);
        bperp_full.push(0.0_f64);
    }

    let ph_raw = read_complex_columns(&ph_path, n_ps)?;
    if ph_raw.cols != slave.len() {
        return stage1_err(format!(
            "phase file has {} columns but metadata has {} entries",
            ph_raw.cols,
            slave.len()
        ));
    }
    let mut ph_reordered = vec![(0.0_f32, 0.0_f32); n_ps * day_full.len()];
    for (sorted_col, (_, _, original_col)) in slave.iter().enumerate() {
        let out_col = if sorted_col + 1 >= master_ix {
            sorted_col + 1
        } else {
            sorted_col
        };
        for row in 0..n_ps {
            ph_reordered[row * day_full.len() + out_col] = ph_raw.values[row * ph_raw.cols + original_col];
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
    let (xy_local, ll0) = local_xy_from_lonlat(&lonlat);
    let mut sort_ix: Vec<usize> = (0..n_ps).collect();
    sort_ix.sort_by(|&left, &right| {
        xy_local[left][1]
            .total_cmp(&xy_local[right][1])
            .then_with(|| xy_local[left][0].total_cmp(&xy_local[right][0]))
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
        xy_out.push(quantize_mm(xy_local[src_row][0] as f32));
        xy_out.push(quantize_mm(xy_local[src_row][1] as f32));
        for col in 0..day_full.len() {
            ph_sorted[out_row * day_full.len() + col] = ph_reordered[src_row * day_full.len() + col];
        }
    }

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
    )?;
    let mut ph_mat = MatFile::new(patch_dir.join("ph1.mat"));
    ph_mat.add_complex_f32_matrix("ph", n_ps, day_full.len(), ph_sorted)?;
    ph_mat.write()?;
    let mut psver = MatFile::new(patch_dir.join("psver.mat"));
    psver.add_f64_matrix("psver", 1, 1, vec![1.0])?;
    psver.write()?;

    if patch_dir.join("pscands.1.da").exists() {
        let da = read_text_flat(&patch_dir.join("pscands.1.da"))?;
        if da.len() == n_ps {
            let sorted = sort_ix.iter().map(|&ix| da[ix]).collect::<Vec<_>>();
            let mut da_mat = MatFile::new(patch_dir.join("da1.mat"));
            da_mat.add_f64_matrix("D_A", 1, n_ps, sorted)?;
            da_mat.write()?;
        }
    }

    if patch_dir.join("pscands.1.hgt").exists() {
        let hgt = read_binary_f32(&patch_dir.join("pscands.1.hgt"), BinaryKind::Generic)?;
        if hgt.len() == n_ps {
            let sorted = sort_ix.iter().map(|&ix| hgt[ix]).collect::<Vec<_>>();
            let mut hgt_mat = MatFile::new(patch_dir.join("hgt1.mat"));
            hgt_mat.add_f32_matrix("hgt", 1, n_ps, sorted)?;
            hgt_mat.write()?;
        }
    }

    let no_master_bperp: Vec<f32> = bperp_full
        .iter()
        .enumerate()
        .filter_map(|(ix, &value)| (ix != master_ix - 1).then_some(value as f32))
        .collect();
    let mut bperp_mat = Vec::with_capacity(n_ps * no_master_bperp.len());
    for _ in 0..n_ps {
        bperp_mat.extend_from_slice(&no_master_bperp);
    }
    let mut bp_mat = MatFile::new(patch_dir.join("bp1.mat"));
    bp_mat.add_f32_matrix("bperp_mat", n_ps, no_master_bperp.len(), bperp_mat)?;
    bp_mat.write()?;

    Ok(format!("Stage 1 created ps1/ph1 for {n_ps} candidates"))
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
) -> Result<(), CoreError> {
    let mut ps = MatFile::new(patch_dir.join("ps1.mat"));
    ps.add_f64_matrix("ij", n_ps, ij_cols, ij_sorted.to_vec())?;
    ps.add_f64_matrix("lonlat", n_ps, 2, lonlat_sorted.to_vec())?;
    ps.add_f32_matrix("xy", n_ps, 3, xy_out.to_vec())?;
    ps.add_f64_matrix("bperp", 1, bperp_full.len(), bperp_full.to_vec())?;
    ps.add_f64_matrix("day", 1, day_full.len(), day_full.to_vec())?;
    ps.add_f64_matrix("master_day", 1, 1, vec![master_day])?;
    ps.add_f64_matrix("master_ix", 1, 1, vec![master_ix as f64])?;
    ps.add_f64_matrix("n_ifg", 1, 1, vec![day_full.len() as f64])?;
    ps.add_f64_matrix("n_image", 1, 1, vec![day_full.len() as f64])?;
    ps.add_f64_matrix("n_ps", 1, 1, vec![n_ps as f64])?;
    ps.add_f64_matrix(
        "sort_ix",
        1,
        sort_ix.len(),
        sort_ix.iter().map(|&ix| (ix + 1) as f64).collect(),
    )?;
    ps.add_f64_matrix("ll0", 1, 2, vec![ll0[0], ll0[1]])?;
    ps.add_f64_matrix("mean_range", 1, 1, vec![DEFAULT_MEAN_RANGE])?;
    ps.add_f64_matrix("mean_incidence", 1, 1, vec![DEFAULT_MEAN_INCIDENCE])?;
    Ok(ps.write()?)
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
            return Ok(candidate);
        }
    }
    stage1_err(format!("{filename} not found near {}", patch_dir.display()))
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
        return stage1_err(format!("unexpected binary size for phase file {}", path.display()));
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
        return stage1_err(format!("{} byte count is not divisible by 4", path.display()));
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

fn local_xy_from_lonlat(lonlat: &[[f64; 2]]) -> (Vec<[f64; 2]>, [f64; 2]) {
    let min_lon = lonlat.iter().map(|pair| pair[0]).fold(f64::INFINITY, f64::min);
    let max_lon = lonlat
        .iter()
        .map(|pair| pair[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lat = lonlat.iter().map(|pair| pair[1]).fold(f64::INFINITY, f64::min);
    let max_lat = lonlat
        .iter()
        .map(|pair| pair[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let origin = [(min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0];
    let origin_rad = [origin[0].to_radians(), origin[1].to_radians()];
    let a = 6_378_137.0_f64;
    let e = 0.082_094_437_949_70_f64;
    let m0 = meridian_arc(a, e, origin_rad[1]);
    let xy = lonlat
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
                [n * cot * angle.sin(), m - m0 + n * cot * (1.0 - angle.cos())]
            } else {
                [a * (lon - origin_rad[0]), -m0]
            }
        })
        .collect();
    (xy, origin)
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
    use std::io::Write;

    #[test]
    fn stage1_native_writes_expected_artifacts() {
        let root = std::env::temp_dir().join(format!("pystamps-stage1-native-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        fs::write(root.join("width.txt"), "10\n").unwrap();
        fs::write(root.join("len.txt"), "10\n").unwrap();
        fs::write(root.join("day.1.in"), "20200104\n20200102\n").unwrap();
        fs::write(root.join("master_day.1.in"), "20200103\n").unwrap();
        fs::write(root.join("bperp.1.in"), "20\n10\n").unwrap();
        fs::write(
            patch.join("pscands.1.ij"),
            "9 2 2\n8 1 1\n7 3 3\n",
        )
        .unwrap();
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

    fn write_f32_file(path: &Path, values: &[f32]) {
        let mut file = fs::File::create(path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}
