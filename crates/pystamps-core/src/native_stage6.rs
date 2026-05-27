use crate::CoreError;
use num_complex::Complex64;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

const DIST_INF: usize = usize::MAX / 4;
const MAT_V5_HEADER_BYTES: u64 = 128;
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT32: u32 = 5;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;

#[derive(Clone, Debug)]
struct Stage6Parms {
    small_baseline_flag: String,
    unwrap_patch_phase: String,
    unwrap_grid_size: f64,
    drop_ifg_index: Vec<i64>,
}

impl Default for Stage6Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            unwrap_patch_phase: "n".to_string(),
            unwrap_grid_size: 20.0,
            drop_ifg_index: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
struct WrappedPhase {
    values: Vec<(f32, f32)>,
    phase_restore: Vec<f32>,
    cols: usize,
}

#[derive(Clone, Debug)]
struct UwGrid {
    ph: Matrix<(f32, f32)>,
    ph_in: Matrix<(f32, f32)>,
    nzix: Matrix<u8>,
    grid_ij: Matrix<f64>,
    n_i: usize,
    n_j: usize,
    n_ps: usize,
    xy: Matrix<f64>,
    ij: Matrix<f64>,
    grid_x_min: f32,
    grid_y_min: f32,
    pix_size: f64,
}

#[derive(Clone, Debug)]
struct UwInterp {
    edgs: Matrix<f64>,
    rowix: Matrix<f64>,
    colix: Matrix<f64>,
    z: Matrix<f64>,
    n_edge: usize,
    edge_counts: Vec<usize>,
}

pub fn run_stage6_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let mut timer = Stage6Timer::new();
    let ps2 = read_mat_stage6_selected(
        dataset_root,
        "ps2.mat",
        &["n_ps", "n_ifg", "master_ix", "xy", "bperp"],
    )?;
    let parms = load_stage6_parms(dataset_root);
    timer.mark("read ps/parms");

    let n_ps = scalar_from_mat(&ps2, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage6_err("ps2.mat missing valid n_ps");
    }
    let rc2_shape = if parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
        None
    } else {
        rc2_phase_shape_if_compatible(dataset_root, n_ps)?
    };
    let mut n_ifg = scalar_from_mat(&ps2, "n_ifg", 0.0).round() as usize;
    if n_ifg == 0 {
        if let Some((_, cols)) = rc2_shape {
            n_ifg = cols;
        }
    }
    let need_ph2 = !parms.unwrap_patch_phase.eq_ignore_ascii_case("y") && rc2_shape.is_none();
    let ph2 = if need_ph2 || n_ifg == 0 {
        let ph2 = read_mat_stage6_selected(dataset_root, "ph2.mat", &["ph"])?;
        let ph2 = complex_ps_matrix(&ph2, "ph", n_ps, "ph2.ph")?;
        if n_ifg == 0 {
            n_ifg = ph2.cols;
        }
        Some(ph2)
    } else {
        None
    };
    if n_ifg == 0 {
        return stage6_err("ph2.ph must contain at least one interferogram");
    }
    if let Some(ph2) = &ph2 {
        if ph2.cols != n_ifg {
            return stage6_err(format!(
                "ph2.ph has {} interferograms but ps2.n_ifg is {n_ifg}",
                ph2.cols
            ));
        }
    }
    let pm_vars: &[&str] = if parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
        &["ph_patch"]
    } else if need_ph2 {
        &["K_ps", "C_ps", "ph_patch"]
    } else {
        &["K_ps"]
    };
    let pm2 = read_mat_stage6_selected(dataset_root, "pm2.mat", pm_vars)?;
    let bp2 = read_mat_stage6_selected(dataset_root, "bp2.mat", &["bperp_mat"])?;
    ensure_mat_stage6(dataset_root, "ifgstd2.mat")?;
    timer.mark("read phase inputs");
    let master_ix = scalar_from_mat(&ps2, "master_ix", 1.0).round() as usize;
    if master_ix == 0 || master_ix > n_ifg {
        return stage6_err(format!(
            "ps2.master_ix must be 1-based within ph2.ph columns; got {master_ix}"
        ));
    }
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    if small_baseline {
        return stage6_err(
            "Stage 6 native unwrap currently supports single-master merged artifacts",
        );
    }

    let drop_set: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    let unwrap_cols: Vec<usize> = (0..n_ifg)
        .filter(|col| !drop_set.contains(&((*col + 1) as i64)) && *col != master_ix - 1)
        .collect();
    if unwrap_cols.is_empty() {
        return stage6_err("No interferograms available for stage-6 unwrapping");
    }

    let bperp_full = expand_bperp_matrix(&bp2, &ps2, n_ps, n_ifg, master_ix)?;
    timer.mark("expand bperp");
    let wrapped = build_wrapped_phase(
        dataset_root,
        ph2.as_ref(),
        &pm2,
        &bperp_full,
        n_ps,
        n_ifg,
        master_ix,
        &parms,
    )?;
    timer.mark("build wrapped phase");
    let uw_grid = if dataset_root.join("uw_grid.mat").exists() {
        read_uw_grid(dataset_root, n_ps)?
    } else {
        let grid = build_uw_grid(&ps2, &wrapped, &unwrap_cols, n_ps, &parms)?;
        write_uw_grid(dataset_root, &grid)?;
        grid
    };
    timer.mark("uw_grid");
    let uw_interp = if dataset_root.join("uw_interp.mat").exists() {
        read_uw_interp(dataset_root, uw_grid.n_i, uw_grid.n_j)?
    } else {
        let interp = build_uw_interp(&uw_grid)?;
        write_uw_interp(dataset_root, &interp)?;
        interp
    };
    timer.mark("uw_interp");
    validate_connected_graph(&uw_interp, uw_grid.n_ps)?;

    let ph_uw_some = unwrap_grid_phase(&uw_grid, &uw_interp)?;
    timer.mark("unwrap grid phase");
    let msd_some = grid_msd(&ph_uw_some, uw_grid.n_ps, unwrap_cols.len(), &uw_interp);
    write_uw_phaseuw(
        dataset_root,
        &ph_uw_some,
        uw_grid.n_ps,
        unwrap_cols.len(),
        &msd_some,
    )?;
    timer.mark("write uw_phaseuw");
    write_phuw2(
        dataset_root,
        &uw_grid,
        &wrapped,
        &ph_uw_some,
        &msd_some,
        &unwrap_cols,
        n_ps,
        n_ifg,
    )?;
    timer.mark("write phuw2");

    Ok(format!(
        "Stage 6 natively unwrapped {n_ps} PS across {n_ifg} interferograms using Rust graph unwrap"
    ))
}

struct Stage6Timer {
    enabled: bool,
    last: Instant,
}

impl Stage6Timer {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("PYSTAMPS_STAGE6_TIMINGS").is_some(),
            last: Instant::now(),
        }
    }

    fn mark(&mut self, label: &str) {
        if self.enabled {
            let elapsed = self.last.elapsed();
            eprintln!("stage6 timing {label}: {:.3}s", elapsed.as_secs_f64());
            self.last = Instant::now();
        }
    }
}

fn build_wrapped_phase(
    dataset_root: &Path,
    ph2: Option<&ComplexMatrixF32>,
    pm2: &MatData,
    bperp_full: &Matrix<f32>,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
    parms: &Stage6Parms,
) -> Result<WrappedPhase, CoreError> {
    let (mut ph_w, used_rc2): (Vec<(f32, f32)>, bool) =
        if parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
            let ph_patch = complex_ps_matrix(pm2, "ph_patch", n_ps, "pm2.ph_patch")?;
            if ph_patch.cols + 1 != n_ifg {
                return stage6_err(format!(
                    "pm2.ph_patch has {} columns but single-master ph2.ph has {n_ifg}",
                    ph_patch.cols
                ));
            }
            let mut values = vec![(1.0f32, 0.0f32); n_ps * n_ifg];
            for row in 0..n_ps {
                for col in 0..n_ifg {
                    if col == master_ix - 1 {
                        continue;
                    }
                    let src_col = if col < master_ix - 1 { col } else { col - 1 };
                    values[row * n_ifg + col] = ph_patch.values[row * ph_patch.cols + src_col];
                }
            }
            (values, false)
        } else if dataset_root.join("rc2.mat").exists() {
            match read_rc2_phase_if_compatible(dataset_root, n_ps) {
                Ok(Some(ph_rc)) => (ph_rc.values, true),
                Ok(None) => (
                    ph2.ok_or_else(|| {
                        stage6_err_owned(
                            "ph2.ph is required when rc2.ph_rc is missing or incompatible"
                                .to_string(),
                        )
                    })?
                    .values
                    .iter()
                    .copied()
                    .collect(),
                    false,
                ),
                Err(err) => return Err(err),
            }
        } else {
            (
                ph2.ok_or_else(|| {
                    stage6_err_owned("ph2.ph is required when rc2.mat is not available".to_string())
                })?
                .values
                .iter()
                .copied()
                .collect(),
                false,
            )
        };

    if !parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
        let k_ps = optional_vector_f64(pm2, "K_ps").filter(|values| values.len() == n_ps);
        if used_rc2 {
            if let Some(k_ps) = k_ps {
                ph_w.par_chunks_mut(n_ifg)
                    .enumerate()
                    .for_each(|(row, row_values)| {
                        let bperp_row = &bperp_full.values[row * n_ifg..(row + 1) * n_ifg];
                        for col in 0..n_ifg {
                            let theta = k_ps[row] as f32 * bperp_row[col];
                            rotate_tuple(&mut row_values[col], theta);
                        }
                    });
            }
        } else if !parms.small_baseline_flag.eq_ignore_ascii_case("y") {
            if let Ok(ph_patch) = complex_ps_matrix(pm2, "ph_patch", n_ps, "pm2.ph_patch") {
                if ph_patch.cols + 1 == n_ifg {
                    ph_w.par_chunks_mut(n_ifg)
                        .enumerate()
                        .for_each(|(row, row_values)| {
                            let patch_row =
                                &ph_patch.values[row * ph_patch.cols..(row + 1) * ph_patch.cols];
                            for col in 0..n_ifg {
                                if col == master_ix - 1 {
                                    continue;
                                }
                                let src_col = if col < master_ix - 1 { col } else { col - 1 };
                                multiply_tuple_conj_patch(&mut row_values[col], patch_row[src_col]);
                            }
                        });
                }
            }
            if let Some(k_ps) = k_ps {
                let c_ps = optional_vector_f64(pm2, "C_ps")
                    .filter(|values| values.len() == n_ps)
                    .unwrap_or_else(|| vec![0.0; n_ps]);
                ph_w.par_chunks_mut(n_ifg)
                    .enumerate()
                    .for_each(|(row, row_values)| {
                        let bperp_row = &bperp_full.values[row * n_ifg..(row + 1) * n_ifg];
                        for col in 0..n_ifg {
                            let theta = k_ps[row] as f32 * bperp_row[col] + c_ps[row] as f32;
                            rotate_tuple(&mut row_values[col], -theta);
                        }
                    });
            }
        } else if let Some(k_ps) = k_ps {
            ph_w.par_chunks_mut(n_ifg)
                .enumerate()
                .for_each(|(row, row_values)| {
                    let bperp_row = &bperp_full.values[row * n_ifg..(row + 1) * n_ifg];
                    for col in 0..n_ifg {
                        let theta = k_ps[row] as f32 * bperp_row[col];
                        rotate_tuple(&mut row_values[col], theta);
                    }
                });
        }
    }

    let mut phase_restore = vec![0.0f32; n_ps * n_ifg];
    let scla_path = dataset_root.join("scla_smooth2.mat");
    if scla_path.exists() {
        if let Ok(scla) = MatData::read(&scla_path) {
            if let Some(k_ps_uw) = optional_vector_f64(&scla, "K_ps_uw") {
                if k_ps_uw.len() == n_ps {
                    for row in 0..n_ps {
                        for col in 0..n_ifg {
                            let theta = k_ps_uw[row] as f32 * bperp_full.values[row * n_ifg + col];
                            rotate_tuple(&mut ph_w[row * n_ifg + col], -theta);
                            phase_restore[row * n_ifg + col] += theta;
                        }
                    }
                }
            }
            if let Some(c_ps_uw) = optional_vector_f64(&scla, "C_ps_uw") {
                if c_ps_uw.len() == n_ps {
                    for row in 0..n_ps {
                        for col in 0..n_ifg {
                            rotate_tuple(&mut ph_w[row * n_ifg + col], -(c_ps_uw[row] as f32));
                            phase_restore[row * n_ifg + col] += c_ps_uw[row] as f32;
                        }
                    }
                }
            }
            if let Ok(ph_ramp) = ps_matrix_f32(&scla, "ph_ramp", n_ps, "scla_smooth2.ph_ramp") {
                if ph_ramp.cols == n_ifg {
                    for row in 0..n_ps {
                        for col in 0..n_ifg {
                            let theta = ph_ramp.values[row * n_ifg + col];
                            rotate_tuple(&mut ph_w[row * n_ifg + col], -theta);
                            phase_restore[row * n_ifg + col] += theta;
                        }
                    }
                }
            }
        }
    }

    if ph_w.par_iter().any(|&(re, im)| {
        let mag2 = re.mul_add(re, im * im);
        mag2 > 0.0 && (mag2 - 1.0).abs() > 1.0e-4
    }) {
        ph_w.par_iter_mut().for_each(normalize_tuple);
    }

    Ok(WrappedPhase {
        values: ph_w,
        phase_restore,
        cols: n_ifg,
    })
}

fn build_uw_grid(
    ps2: &MatData,
    wrapped: &WrappedPhase,
    unwrap_cols: &[usize],
    n_ps: usize,
    parms: &Stage6Parms,
) -> Result<UwGrid, CoreError> {
    let xy = ps_dim_f32(ps2, "xy", n_ps, 3, "ps2.xy")?;
    let pix_size = if parms.unwrap_grid_size > 0.0 {
        parms.unwrap_grid_size as f32
    } else {
        20.0f32
    };
    let grid_x_min = (0..n_ps)
        .map(|row| xy.values[row * 3 + 1])
        .fold(f32::INFINITY, f32::min);
    let grid_y_min = (0..n_ps)
        .map(|row| xy.values[row * 3 + 2])
        .fold(f32::INFINITY, f32::min);
    let mut grid_i = vec![1usize; n_ps];
    let mut grid_j = vec![1usize; n_ps];
    for row in 0..n_ps {
        let x = xy.values[row * 3 + 1];
        let y = xy.values[row * 3 + 2];
        grid_i[row] = ((y - grid_y_min + 1.0e-3f32) / pix_size).ceil().max(1.0) as usize;
        grid_j[row] = ((x - grid_x_min + 1.0e-3f32) / pix_size).ceil().max(1.0) as usize;
    }
    if let Some(max_i) = grid_i.iter().copied().max() {
        if max_i > 1 {
            for value in &mut grid_i {
                if *value == max_i {
                    *value = max_i - 1;
                }
            }
        }
    }
    if let Some(max_j) = grid_j.iter().copied().max() {
        if max_j > 1 {
            for value in &mut grid_j {
                if *value == max_j {
                    *value = max_j - 1;
                }
            }
        }
    }
    let n_i = grid_i.iter().copied().max().unwrap_or(1).max(1);
    let n_j = grid_j.iter().copied().max().unwrap_or(1).max(1);
    let n_unwrap = unwrap_cols.len();

    let mut grouped: BTreeMap<usize, Vec<Complex64>> = BTreeMap::new();
    let mut ph_in = vec![(0.0f32, 0.0f32); n_ps * n_unwrap];
    for row in 0..n_ps {
        let lin = (grid_j[row] - 1) * n_i + (grid_i[row] - 1);
        let entry = grouped
            .entry(lin)
            .or_insert_with(|| vec![Complex64::new(0.0, 0.0); n_unwrap]);
        for (out_col, &src_col) in unwrap_cols.iter().enumerate() {
            let value = wrapped.values[row * wrapped.cols + src_col];
            entry[out_col] += tuple_to_complex(value);
            ph_in[row * n_unwrap + out_col] = value;
        }
    }

    let mut nz_flat = vec![false; n_i * n_j];
    let mut ph_values = Vec::new();
    let mut nz_lins = Vec::new();
    for (lin, values) in grouped {
        if values
            .first()
            .map(|value| value.norm() > 0.0)
            .unwrap_or(false)
        {
            nz_flat[lin] = true;
            nz_lins.push(lin);
            for value in values {
                ph_values.push((value.re as f32, value.im as f32));
            }
        }
    }
    let n_grid = nz_lins.len();
    if n_grid == 0 {
        return stage6_err("uw_grid has no non-zero points in first interferogram");
    }
    let mut nzix = vec![0u8; n_i * n_j];
    for (lin, &keep) in nz_flat.iter().enumerate() {
        if keep {
            let row = lin % n_i;
            let col = lin / n_i;
            nzix[row * n_j + col] = 1;
        }
    }
    let mut grid_ij = Vec::with_capacity(n_ps * 2);
    for row in 0..n_ps {
        grid_ij.push(grid_i[row] as f64);
        grid_ij.push(grid_j[row] as f64);
    }
    let mut xy_grid = Vec::with_capacity(n_grid * 3);
    let mut ij_grid = Vec::with_capacity(n_grid * 2);
    let pix_size_f64 = pix_size as f64;
    for (pos, &lin) in nz_lins.iter().enumerate() {
        let i = (lin % n_i) + 1;
        let j = (lin / n_i) + 1;
        xy_grid.push((pos + 1) as f64);
        xy_grid.push((j as f64 - 0.5) * pix_size_f64);
        xy_grid.push((i as f64 - 0.5) * pix_size_f64);
        ij_grid.push(i as f64);
        ij_grid.push(j as f64);
    }

    Ok(UwGrid {
        ph: Matrix {
            name: "ph".to_string(),
            rows: n_grid,
            cols: n_unwrap,
            values: ph_values,
        },
        ph_in: Matrix {
            name: "ph_in".to_string(),
            rows: n_ps,
            cols: n_unwrap,
            values: ph_in,
        },
        nzix: Matrix {
            name: "nzix".to_string(),
            rows: n_i,
            cols: n_j,
            values: nzix,
        },
        grid_ij: Matrix {
            name: "grid_ij".to_string(),
            rows: n_ps,
            cols: 2,
            values: grid_ij,
        },
        n_i,
        n_j,
        n_ps: n_grid,
        xy: Matrix {
            name: "xy".to_string(),
            rows: n_grid,
            cols: 3,
            values: xy_grid,
        },
        ij: Matrix {
            name: "ij".to_string(),
            rows: n_grid,
            cols: 2,
            values: ij_grid,
        },
        grid_x_min,
        grid_y_min,
        pix_size: pix_size_f64,
    })
}

fn build_uw_interp(uw_grid: &UwGrid) -> Result<UwInterp, CoreError> {
    let z = nearest_grid_labels(uw_grid)?;

    let mut edge_keys = Vec::with_capacity(
        uw_grid.n_i.saturating_sub(1) * uw_grid.n_j + uw_grid.n_i * uw_grid.n_j.saturating_sub(1),
    );
    for row in 0..uw_grid.n_i.saturating_sub(1) {
        for col in 0..uw_grid.n_j {
            if let Some(key) =
                label_edge_key(z[row * uw_grid.n_j + col], z[(row + 1) * uw_grid.n_j + col])
            {
                edge_keys.push(key);
            }
        }
    }
    for row in 0..uw_grid.n_i {
        for col in 0..uw_grid.n_j.saturating_sub(1) {
            if let Some(key) =
                label_edge_key(z[row * uw_grid.n_j + col], z[row * uw_grid.n_j + col + 1])
            {
                edge_keys.push(key);
            }
        }
    }
    edge_keys.sort_unstable();
    edge_keys.dedup();

    let edge_ids: HashMap<u64, usize> = edge_keys
        .iter()
        .enumerate()
        .map(|(ix, &edge)| (edge, ix + 1))
        .collect();

    let mut rowix = vec![0.0; uw_grid.n_i.saturating_sub(1) * uw_grid.n_j];
    let mut colix = vec![0.0; uw_grid.n_i * uw_grid.n_j.saturating_sub(1)];
    let mut edge_counts = vec![0usize; edge_ids.len()];
    for row in 0..uw_grid.n_i.saturating_sub(1) {
        for col in 0..uw_grid.n_j {
            let value = signed_label_edge_id(
                &edge_ids,
                z[row * uw_grid.n_j + col],
                z[(row + 1) * uw_grid.n_j + col],
            );
            if value != 0 {
                edge_counts[value.unsigned_abs() - 1] += 1;
            }
            rowix[row * uw_grid.n_j + col] = value as f64;
        }
    }
    for row in 0..uw_grid.n_i {
        for col in 0..uw_grid.n_j.saturating_sub(1) {
            let value = signed_label_edge_id(
                &edge_ids,
                z[row * uw_grid.n_j + col],
                z[row * uw_grid.n_j + col + 1],
            );
            if value != 0 {
                edge_counts[value.unsigned_abs() - 1] += 1;
            }
            colix[row * uw_grid.n_j.saturating_sub(1) + col] = value as f64;
        }
    }

    let mut edgs = vec![0.0; edge_keys.len() * 3];
    for (row, &key) in edge_keys.iter().enumerate() {
        let (a, b) = decode_label_edge_key(key);
        edgs[row * 3] = (row + 1) as f64;
        edgs[row * 3 + 1] = a as f64;
        edgs[row * 3 + 2] = b as f64;
    }
    Ok(UwInterp {
        edgs: Matrix {
            name: "edgs".to_string(),
            rows: edge_keys.len(),
            cols: 3,
            values: edgs,
        },
        rowix: Matrix {
            name: "rowix".to_string(),
            rows: uw_grid.n_i.saturating_sub(1),
            cols: uw_grid.n_j,
            values: rowix,
        },
        colix: Matrix {
            name: "colix".to_string(),
            rows: uw_grid.n_i,
            cols: uw_grid.n_j.saturating_sub(1),
            values: colix,
        },
        z: Matrix {
            name: "Z".to_string(),
            rows: uw_grid.n_i,
            cols: uw_grid.n_j,
            values: z.iter().map(|&value| value as f64).collect(),
        },
        n_edge: edge_keys.len(),
        edge_counts,
    })
}

fn nearest_grid_labels(uw_grid: &UwGrid) -> Result<Vec<usize>, CoreError> {
    let rows = uw_grid.n_i;
    let cols = uw_grid.n_j;
    let len = rows * cols;
    let mut vertical_dist = vec![DIST_INF; len];
    let mut vertical_label = vec![0usize; len];
    let mut f = vec![DIST_INF; rows.max(cols)];
    let mut labels = vec![0usize; rows.max(cols)];
    let mut dist = vec![DIST_INF; rows.max(cols)];
    let mut out_labels = vec![0usize; rows.max(cols)];
    let mut next_label = 1usize;

    for col in 0..cols {
        f[..rows].fill(DIST_INF);
        labels[..rows].fill(0);
        for row in 0..rows {
            let ix = row * cols + col;
            if uw_grid.nzix.values[ix] != 0 {
                f[row] = 0;
                labels[row] = next_label;
                next_label += 1;
            }
        }
        distance_transform_1d(
            &f[..rows],
            &labels[..rows],
            &mut dist[..rows],
            &mut out_labels[..rows],
        );
        for row in 0..rows {
            let ix = row * cols + col;
            vertical_dist[ix] = dist[row];
            vertical_label[ix] = out_labels[row];
        }
    }

    let mut z = vec![0usize; len];
    for row in 0..rows {
        for col in 0..cols {
            let ix = row * cols + col;
            f[col] = vertical_dist[ix];
            labels[col] = vertical_label[ix];
        }
        distance_transform_1d(
            &f[..cols],
            &labels[..cols],
            &mut dist[..cols],
            &mut out_labels[..cols],
        );
        for col in 0..cols {
            z[row * cols + col] = out_labels[col];
        }
    }
    if next_label - 1 != uw_grid.n_ps {
        return Err(CoreError::NativeStage {
            stage: 6,
            message: format!(
                "uw_grid.nzix labels {} occupied points but uw_grid.n_ps is {}",
                next_label - 1,
                uw_grid.n_ps
            ),
        });
    }
    Ok(z)
}

fn distance_transform_1d(
    f: &[usize],
    labels: &[usize],
    dist_out: &mut [usize],
    label_out: &mut [usize],
) {
    let n = f.len();
    let mut sites = Vec::with_capacity(n);
    for (idx, (&value, &label)) in f.iter().zip(labels.iter()).enumerate() {
        if value < DIST_INF && label != 0 {
            sites.push(idx);
        }
    }
    if sites.is_empty() {
        dist_out.fill(DIST_INF);
        label_out.fill(0);
        return;
    }

    let mut v = vec![0usize; sites.len()];
    let mut z = vec![0.0f64; sites.len() + 1];
    let mut k = 0usize;
    v[0] = sites[0];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for &q in sites.iter().skip(1) {
        let mut s = parabola_intersection(f, q, v[k]);
        while s <= z[k] {
            if k == 0 {
                break;
            }
            k -= 1;
            s = parabola_intersection(f, q, v[k]);
        }
        if s <= z[k] && k == 0 {
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
        } else {
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        }
    }

    k = 0;
    for q in 0..n {
        // MATLAB dsearchn picks the later node on exact grid ties for these artifacts.
        while z[k + 1] <= q as f64 {
            k += 1;
        }
        let site = v[k];
        let delta = q.abs_diff(site);
        dist_out[q] = f[site].saturating_add(delta * delta);
        label_out[q] = labels[site];
    }
}

fn parabola_intersection(f: &[usize], q: usize, p: usize) -> f64 {
    let qf = q as f64;
    let pf = p as f64;
    ((f[q] as f64 + qf * qf) - (f[p] as f64 + pf * pf)) / (2.0 * (qf - pf))
}

fn unwrap_grid_phase(uw_grid: &UwGrid, uw_interp: &UwInterp) -> Result<Vec<f32>, CoreError> {
    let adjacency = graph_adjacency(&uw_interp.edgs, uw_grid.n_ps)?;
    let traversal = graph_unwrap_traversal(&adjacency)?;
    let mut output = vec![0.0f32; uw_grid.n_ps * uw_grid.ph.cols];
    let mut wrapped = vec![0.0f64; uw_grid.n_ps * uw_grid.ph.cols];
    for row in 0..uw_grid.n_ps {
        for col in 0..uw_grid.ph.cols {
            wrapped[row * uw_grid.ph.cols + col] =
                tuple_to_complex(uw_grid.ph.values[row * uw_grid.ph.cols + col]).arg();
        }
    }
    for col in 0..uw_grid.ph.cols {
        output[col] = wrapped[0] as f32;
    }
    for &(parent, child) in &traversal {
        for col in 0..uw_grid.ph.cols {
            let parent_ix = parent * uw_grid.ph.cols + col;
            let child_ix = child * uw_grid.ph.cols + col;
            let delta = wrap_phase(wrapped[child_ix] - wrapped[parent_ix]);
            output[child_ix] = output[parent_ix] + delta as f32;
        }
    }
    Ok(output)
}

fn graph_unwrap_traversal(adjacency: &[Vec<usize>]) -> Result<Vec<(usize, usize)>, CoreError> {
    if adjacency.is_empty() {
        return Ok(Vec::new());
    }
    let mut traversal = Vec::with_capacity(adjacency.len().saturating_sub(1));
    let mut visited = vec![false; adjacency.len()];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &next in &adjacency[node] {
            if !visited[next] {
                visited[next] = true;
                traversal.push((node, next));
                queue.push_back(next);
            }
        }
    }
    if visited.iter().any(|&seen| !seen) {
        return stage6_err("disconnected unwrap graph: not all grid points are reachable");
    }
    Ok(traversal)
}

fn grid_msd(ph_uw: &[f32], n_ps_grid: usize, n_unwrap: usize, uw_interp: &UwInterp) -> Vec<f64> {
    let mut msd = vec![0.0; n_unwrap];
    if uw_interp.edgs.rows == 0 {
        return msd;
    }
    msd.par_iter_mut().enumerate().for_each(|(col, value)| {
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in 0..uw_interp.edgs.rows {
            let edge_count = uw_interp.edge_counts.get(row).copied().unwrap_or(1);
            if edge_count == 0 {
                continue;
            }
            let a = uw_interp.edgs.values[row * 3 + 1].round() as isize - 1;
            let b = uw_interp.edgs.values[row * 3 + 2].round() as isize - 1;
            if a < 0 || b < 0 || a as usize >= n_ps_grid || b as usize >= n_ps_grid {
                continue;
            }
            let diff = ph_uw[a as usize * n_unwrap + col] as f64
                - ph_uw[b as usize * n_unwrap + col] as f64;
            if diff != 0.0 {
                sum += diff * diff * edge_count as f64;
                count += edge_count;
            }
        }
        if count > 0 {
            *value = sum / count as f64;
        }
    });
    msd
}

fn write_phuw2(
    dataset_root: &Path,
    uw_grid: &UwGrid,
    wrapped: &WrappedPhase,
    ph_uw_some: &[f32],
    msd_some: &[f64],
    unwrap_cols: &[usize],
    n_ps: usize,
    n_ifg: usize,
) -> Result<(), CoreError> {
    let mut gridix = vec![0usize; uw_grid.n_i * uw_grid.n_j];
    let mut node = 1usize;
    for row in 0..uw_grid.n_i {
        for col in 0..uw_grid.n_j {
            if uw_grid.nzix.values[row * uw_grid.n_j + col] != 0 {
                gridix[row * uw_grid.n_j + col] = node;
                node += 1;
            }
        }
    }

    let mut ph_uw = vec![0.0f32; n_ps * n_ifg];
    ph_uw
        .par_chunks_mut(n_ifg)
        .enumerate()
        .for_each(|(row, out_row)| {
            let grid_i = uw_grid.grid_ij.values[row * 2].round() as isize;
            let grid_j = uw_grid.grid_ij.values[row * 2 + 1].round() as isize;
            if grid_i <= 0
                || grid_j <= 0
                || grid_i as usize > uw_grid.n_i
                || grid_j as usize > uw_grid.n_j
            {
                return;
            }
            let ps_grid_idx = gridix[(grid_i as usize - 1) * uw_grid.n_j + (grid_j as usize - 1)];
            if ps_grid_idx == 0 {
                return;
            }
            for (out_col, &src_col) in unwrap_cols.iter().enumerate() {
                let ph_pix = ph_uw_some[(ps_grid_idx - 1) * unwrap_cols.len() + out_col];
                let ph_in = uw_grid.ph_in.values[row * unwrap_cols.len() + out_col];
                let residual = residual_phase(ph_in, ph_pix);
                out_row[src_col] = ph_pix + residual + wrapped.phase_restore[row * n_ifg + src_col];
            }
        });
    let mut msd = vec![0.0f32; n_ifg];
    for (out_col, &src_col) in unwrap_cols.iter().enumerate() {
        msd[src_col] = msd_some[out_col] as f32;
    }
    let mut mat = MatFile::new(dataset_root.join("phuw2.mat"));
    mat.add_f32_matrix("ph_uw", n_ps, n_ifg, ph_uw)?;
    mat.add_f32_col_vector("msd", msd)?;
    mat.write()?;
    Ok(())
}

fn write_uw_phaseuw(
    dataset_root: &Path,
    ph_uw: &[f32],
    rows: usize,
    cols: usize,
    msd: &[f64],
) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("uw_phaseuw.mat"));
    mat.add_f32_matrix("ph_uw", rows, cols, ph_uw.to_vec())?;
    mat.add_f64_col_vector("msd", msd.to_vec())?;
    mat.write()?;
    Ok(())
}

fn write_uw_grid(dataset_root: &Path, grid: &UwGrid) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("uw_grid.mat"));
    mat.add_complex_f32_matrix("ph", grid.ph.rows, grid.ph.cols, grid.ph.values.clone())?;
    mat.add_complex_f32_matrix(
        "ph_in",
        grid.ph_in.rows,
        grid.ph_in.cols,
        grid.ph_in.values.clone(),
    )?;
    mat.add_complex_f32_matrix("ph_lowpass", 0, 0, Vec::new())?;
    mat.add_complex_f32_matrix("ph_uw_predef", 0, 0, Vec::new())?;
    mat.add_complex_f32_matrix("ph_in_predef", 0, 0, Vec::new())?;
    mat.add_f64_matrix("xy", grid.xy.rows, grid.xy.cols, grid.xy.values.clone())?;
    mat.add_f64_matrix("ij", grid.ij.rows, grid.ij.cols, grid.ij.values.clone())?;
    mat.add_u8_matrix(
        "nzix",
        grid.nzix.rows,
        grid.nzix.cols,
        grid.nzix.values.clone(),
    )?;
    mat.add_f32_scalar("grid_x_min", grid.grid_x_min)?;
    mat.add_f32_scalar("grid_y_min", grid.grid_y_min)?;
    mat.add_f32_scalar("n_i", grid.n_i as f32)?;
    mat.add_f32_scalar("n_j", grid.n_j as f32)?;
    mat.add_f64_scalar("n_ifg", grid.ph.cols as f64)?;
    mat.add_f64_scalar("n_ps", grid.n_ps as f64)?;
    mat.add_f64_matrix(
        "grid_ij",
        grid.grid_ij.rows,
        grid.grid_ij.cols,
        grid.grid_ij.values.clone(),
    )?;
    mat.add_f64_scalar("pix_size", grid.pix_size)?;
    mat.write()?;
    Ok(())
}

fn write_uw_interp(dataset_root: &Path, interp: &UwInterp) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("uw_interp.mat"));
    mat.add_f64_matrix(
        "edgs",
        interp.edgs.rows,
        interp.edgs.cols,
        interp.edgs.values.clone(),
    )?;
    mat.add_f64_scalar("n_edge", interp.n_edge as f64)?;
    mat.add_f64_matrix(
        "rowix",
        interp.rowix.rows,
        interp.rowix.cols,
        interp.rowix.values.clone(),
    )?;
    mat.add_f64_matrix(
        "colix",
        interp.colix.rows,
        interp.colix.cols,
        interp.colix.values.clone(),
    )?;
    mat.add_f64_matrix("Z", interp.z.rows, interp.z.cols, interp.z.values.clone())?;
    mat.write()?;
    Ok(())
}

fn read_uw_grid(dataset_root: &Path, n_ps: usize) -> Result<UwGrid, CoreError> {
    let mat = read_mat_stage6(dataset_root, "uw_grid.mat")?;
    let n_grid = scalar_from_mat(&mat, "n_ps", 0.0).round() as usize;
    if n_grid == 0 {
        return stage6_err("uw_grid.mat missing valid n_ps");
    }
    let ph = complex_ps_matrix(&mat, "ph", n_grid, "uw_grid.ph")?;
    let ph_in =
        complex_ps_matrix(&mat, "ph_in", n_ps, "uw_grid.ph_in").unwrap_or(ComplexMatrixF32 {
            name: "ph_in".to_string(),
            rows: n_ps,
            cols: ph.cols,
            values: vec![(0.0, 0.0); n_ps * ph.cols],
        });
    let nzix_source = mat
        .get_f32_matrix("nzix")
        .or_else(|_| {
            mat.get_f64_matrix("nzix").map(|m| Matrix {
                name: m.name,
                rows: m.rows,
                cols: m.cols,
                values: m.values.iter().map(|&value| value as f32).collect(),
            })
        })
        .map_err(|err| CoreError::NativeStage {
            stage: 6,
            message: format!("uw_grid.nzix is invalid: {err}"),
        })?;
    let nzix = Matrix {
        name: "nzix".to_string(),
        rows: nzix_source.rows,
        cols: nzix_source.cols,
        values: nzix_source
            .values
            .iter()
            .map(|&value| u8::from(value != 0.0))
            .collect(),
    };
    let grid_ij = ps_dim_f64(&mat, "grid_ij", n_ps, 2, "uw_grid.grid_ij")?;
    let xy = mat.get_f64_matrix("xy").unwrap_or(Matrix {
        name: "xy".to_string(),
        rows: 0,
        cols: 0,
        values: Vec::new(),
    });
    let ij = mat.get_f64_matrix("ij").unwrap_or(Matrix {
        name: "ij".to_string(),
        rows: 0,
        cols: 0,
        values: Vec::new(),
    });
    let n_i = nzix.rows;
    let n_j = nzix.cols;
    Ok(UwGrid {
        ph: Matrix {
            name: ph.name,
            rows: ph.rows,
            cols: ph.cols,
            values: ph.values,
        },
        ph_in: Matrix {
            name: ph_in.name,
            rows: ph_in.rows,
            cols: ph_in.cols,
            values: ph_in.values,
        },
        nzix,
        grid_ij,
        n_i,
        n_j,
        n_ps: n_grid,
        xy,
        ij,
        grid_x_min: scalar_from_mat(&mat, "grid_x_min", 0.0) as f32,
        grid_y_min: scalar_from_mat(&mat, "grid_y_min", 0.0) as f32,
        pix_size: scalar_from_mat(&mat, "pix_size", 20.0),
    })
}

fn read_uw_interp(dataset_root: &Path, n_i: usize, n_j: usize) -> Result<UwInterp, CoreError> {
    let mat = read_mat_stage6(dataset_root, "uw_interp.mat")?;
    let edgs = mat
        .get_f64_matrix("edgs")
        .map_err(|err| CoreError::NativeStage {
            stage: 6,
            message: format!("uw_interp.edgs is invalid: {err}"),
        })?;
    let rowix = mat.get_f64_matrix("rowix").unwrap_or(Matrix {
        name: "rowix".to_string(),
        rows: n_i.saturating_sub(1),
        cols: n_j,
        values: vec![0.0; n_i.saturating_sub(1) * n_j],
    });
    let colix = mat.get_f64_matrix("colix").unwrap_or(Matrix {
        name: "colix".to_string(),
        rows: n_i,
        cols: n_j.saturating_sub(1),
        values: vec![0.0; n_i * n_j.saturating_sub(1)],
    });
    let z = mat.get_f64_matrix("Z").unwrap_or(Matrix {
        name: "Z".to_string(),
        rows: n_i,
        cols: n_j,
        values: vec![1.0; n_i * n_j],
    });
    let edge_counts = edge_counts_from_indices(&rowix, &colix, edgs.rows);
    Ok(UwInterp {
        n_edge: scalar_from_mat(&mat, "n_edge", edgs.rows as f64).round() as usize,
        edgs,
        rowix,
        colix,
        z,
        edge_counts,
    })
}

fn expand_bperp_matrix(
    bp2: &MatData,
    ps2: &MatData,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Matrix<f32>, CoreError> {
    if let Ok(bp_nm) = ps_matrix_f32(bp2, "bperp_mat", n_ps, "bp2.bperp_mat") {
        if bp_nm.cols == n_ifg {
            return Ok(bp_nm);
        }
        if bp_nm.cols + 1 == n_ifg {
            let mut values = vec![0.0; n_ps * n_ifg];
            for row in 0..n_ps {
                for col in 0..n_ifg {
                    if col == master_ix - 1 {
                        continue;
                    }
                    let src_col = if col < master_ix - 1 { col } else { col - 1 };
                    values[row * n_ifg + col] = bp_nm.values[row * bp_nm.cols + src_col];
                }
            }
            return Ok(Matrix {
                name: "bperp_mat".to_string(),
                rows: n_ps,
                cols: n_ifg,
                values,
            });
        }
    }
    let bperp = ps_vector_f64(ps2, "bperp", n_ifg, "ps2.bperp")?;
    let mut values = Vec::with_capacity(n_ps * n_ifg);
    for _ in 0..n_ps {
        values.extend(bperp.iter().map(|&value| value as f32));
    }
    Ok(Matrix {
        name: "bperp_mat".to_string(),
        rows: n_ps,
        cols: n_ifg,
        values,
    })
}

#[cfg(test)]
fn grid_points(uw_grid: &UwGrid) -> Result<Vec<(f64, f64)>, CoreError> {
    let mut points = Vec::with_capacity(uw_grid.n_ps);
    for col in 0..uw_grid.n_j {
        for row in 0..uw_grid.n_i {
            if uw_grid.nzix.values[row * uw_grid.n_j + col] != 0 {
                points.push(((col + 1) as f64, (row + 1) as f64));
            }
        }
    }
    if points.len() != uw_grid.n_ps {
        return stage6_err("uw_grid.nzix and uw_grid.n_ps are inconsistent");
    }
    Ok(points)
}

#[cfg(test)]
fn nearest_point(points: &[(f64, f64)], target: (f64, f64)) -> usize {
    points
        .iter()
        .enumerate()
        .map(|(ix, &point)| {
            (
                ix,
                (point.0 - target.0).powi(2) + (point.1 - target.1).powi(2),
            )
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(ix, _)| ix)
        .unwrap_or(0)
}

fn label_edge_key(a: usize, b: usize) -> Option<u64> {
    (a != b).then(|| {
        let lo = a.min(b) as u64;
        let hi = a.max(b) as u64;
        (lo << 32) | hi
    })
}

fn decode_label_edge_key(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xffff_ffff) as usize)
}

fn signed_label_edge_id(edge_ids: &HashMap<u64, usize>, a: usize, b: usize) -> isize {
    if a == b {
        return 0;
    }
    let Some(key) = label_edge_key(a, b) else {
        return 0;
    };
    let id = *edge_ids.get(&key).unwrap_or(&0);
    if a <= b {
        id as isize
    } else {
        -(id as isize)
    }
}

fn edge_counts_from_indices(rowix: &Matrix<f64>, colix: &Matrix<f64>, n_edge: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_edge];
    for &value in rowix.values.iter().chain(colix.values.iter()) {
        if !value.is_finite() {
            continue;
        }
        let edge_id = value.round().abs() as usize;
        if (1..=n_edge).contains(&edge_id) {
            counts[edge_id - 1] += 1;
        }
    }
    counts
}

fn validate_connected_graph(uw_interp: &UwInterp, n_nodes: usize) -> Result<(), CoreError> {
    if n_nodes <= 1 {
        return Ok(());
    }
    let adjacency = graph_adjacency(&uw_interp.edgs, n_nodes)?;
    let mut visited = vec![false; n_nodes];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &next in &adjacency[node] {
            if !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }
    if visited.iter().any(|&seen| !seen) {
        return stage6_err("disconnected unwrap graph: not all grid points are reachable");
    }
    Ok(())
}

fn graph_adjacency(edgs: &Matrix<f64>, n_nodes: usize) -> Result<Vec<Vec<usize>>, CoreError> {
    if edgs.cols < 3 && edgs.rows > 0 {
        return stage6_err(format!(
            "uw_interp.edgs must have at least 3 columns, got {}",
            edgs.cols
        ));
    }
    let mut adjacency = vec![Vec::new(); n_nodes];
    for row in 0..edgs.rows {
        let a = edgs.values[row * edgs.cols + 1].round() as isize - 1;
        let b = edgs.values[row * edgs.cols + 2].round() as isize - 1;
        if a < 0 || b < 0 || a as usize >= n_nodes || b as usize >= n_nodes || a == b {
            return stage6_err(format!(
                "invalid Stage 6 unwrap graph edge {}: ({}, {}) for n_nodes={n_nodes}",
                row + 1,
                a + 1,
                b + 1
            ));
        }
        adjacency[a as usize].push(b as usize);
        adjacency[b as usize].push(a as usize);
    }
    Ok(adjacency)
}

fn load_stage6_parms(dataset_root: &Path) -> Stage6Parms {
    let path = dataset_root.join("parms.mat");
    if !path.exists() {
        return Stage6Parms::default();
    }
    let Ok(mat) = MatData::read(path) else {
        return Stage6Parms::default();
    };
    Stage6Parms {
        small_baseline_flag: text_from_mat(&mat, "small_baseline_flag", "n"),
        unwrap_patch_phase: text_from_mat(&mat, "unwrap_patch_phase", "n"),
        unwrap_grid_size: scalar_from_mat(&mat, "unwrap_grid_size", 20.0),
        drop_ifg_index: optional_vector_f64(&mat, "drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| (value > 0.0).then_some(value.round() as i64))
            .collect(),
    }
}

fn read_mat_stage6(dataset_root: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(dataset_root.join(filename))
        .map_err(|err| stage6_err_owned(format!("unable to read {filename}: {err}")))
}

fn read_mat_stage6_selected(
    dataset_root: &Path,
    filename: &str,
    variables: &[&str],
) -> Result<MatData, CoreError> {
    MatData::read_selected(dataset_root.join(filename), variables)
        .map_err(|err| stage6_err_owned(format!("unable to read {filename}: {err}")))
}

fn ensure_mat_stage6(dataset_root: &Path, filename: &str) -> Result<(), CoreError> {
    if dataset_root.join(filename).is_file() {
        Ok(())
    } else {
        stage6_err(format!(
            "Missing required artifact: {filename} before stage 6"
        ))
    }
}

fn read_rc2_phase_if_compatible(
    dataset_root: &Path,
    n_ps: usize,
) -> Result<Option<ComplexMatrixF32>, CoreError> {
    let path = dataset_root.join("rc2.mat");
    if let Some((rows, cols)) = mat_v5_variable_shape(&path, "ph_rc")? {
        if rows != n_ps && cols != n_ps {
            return Ok(None);
        }
    }
    let rc2 = read_mat_stage6_selected(dataset_root, "rc2.mat", &["ph_rc"])?;
    let Ok(source) = rc2.get_complex_f32_matrix("ph_rc") else {
        return Ok(None);
    };
    if source.rows == n_ps {
        Ok(Some(source))
    } else if source.cols == n_ps {
        Ok(Some(transpose_complex_f32(source)))
    } else {
        Ok(None)
    }
}

fn rc2_phase_shape_if_compatible(
    dataset_root: &Path,
    n_ps: usize,
) -> Result<Option<(usize, usize)>, CoreError> {
    let path = dataset_root.join("rc2.mat");
    if !path.exists() {
        return Ok(None);
    }
    let Some((rows, cols)) = mat_v5_variable_shape(&path, "ph_rc")? else {
        return Ok(None);
    };
    if rows == n_ps {
        Ok(Some((rows, cols)))
    } else if cols == n_ps {
        Ok(Some((cols, rows)))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy)]
enum MatEndian {
    Little,
    Big,
}

impl MatEndian {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            MatEndian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
            MatEndian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
        }
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        match self {
            MatEndian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            MatEndian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        match self {
            MatEndian::Little => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            MatEndian::Big => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

struct MatElementHeader {
    data_type: u32,
    data_size: usize,
    data_start: usize,
    padded_end: usize,
}

fn mat_v5_variable_shape(path: &Path, variable: &str) -> Result<Option<(usize, usize)>, CoreError> {
    let mut file = File::open(path)
        .map_err(|err| stage6_err_owned(format!("unable to inspect rc2.mat: {err}")))?;
    let mut header = vec![0u8; 4096];
    let bytes_read = file
        .read(&mut header)
        .map_err(|err| stage6_err_owned(format!("unable to inspect rc2.mat: {err}")))?;
    header.truncate(bytes_read);
    if header.len() < MAT_V5_HEADER_BYTES as usize {
        return Ok(None);
    }
    let endian = match &header[126..128] {
        b"IM" => MatEndian::Little,
        b"MI" => MatEndian::Big,
        _ => return Ok(None),
    };

    let mut offset = MAT_V5_HEADER_BYTES as usize;
    while offset + 8 <= header.len() {
        let top = mat_element_header(&header, offset, endian)?;
        if top.data_type == MI_COMPRESSED {
            return Ok(None);
        }
        if top.data_type != MI_MATRIX {
            offset = top.padded_end;
            continue;
        }
        let Some(shape) = mat_matrix_shape_from_prefix(&header, top.data_start, endian, variable)?
        else {
            return Ok(None);
        };
        return Ok(Some(shape));
    }
    Ok(None)
}

fn mat_matrix_shape_from_prefix(
    bytes: &[u8],
    matrix_start: usize,
    endian: MatEndian,
    variable: &str,
) -> Result<Option<(usize, usize)>, CoreError> {
    let flags = mat_element_header(bytes, matrix_start, endian)?;
    let dims = mat_element_header(bytes, flags.padded_end, endian)?;
    if dims.data_type != MI_INT32
        || dims.data_size < 8
        || dims.data_start + dims.data_size > bytes.len()
    {
        return Ok(None);
    }
    let dim_values = bytes[dims.data_start..dims.data_start + dims.data_size]
        .chunks_exact(4)
        .map(|chunk| endian.read_i32(chunk))
        .collect::<Vec<_>>();
    if dim_values.len() < 2 || dim_values.iter().any(|&dim| dim < 0) {
        return Ok(None);
    }
    let name = mat_element_header(bytes, dims.padded_end, endian)?;
    if name.data_start + name.data_size > bytes.len()
        || (name.data_type != MI_INT8 && name.data_type != MI_UINT8)
    {
        return Ok(None);
    }
    let name_text =
        String::from_utf8_lossy(&bytes[name.data_start..name.data_start + name.data_size]);
    if name_text != variable {
        return Ok(None);
    }
    let rows = dim_values[0] as usize;
    let mut cols = 1usize;
    for dim in &dim_values[1..] {
        cols = cols
            .checked_mul(*dim as usize)
            .ok_or_else(|| stage6_err_owned("MAT v5 dimensions overflow usize".to_string()))?;
    }
    Ok(Some((rows, cols)))
}

fn mat_element_header(
    bytes: &[u8],
    offset: usize,
    endian: MatEndian,
) -> Result<MatElementHeader, CoreError> {
    if offset + 8 > bytes.len() {
        return stage6_err("MAT v5 element header is truncated");
    }
    let small_type = endian.read_u16(&bytes[offset..offset + 2]) as u32;
    let small_size = endian.read_u16(&bytes[offset + 2..offset + 4]) as usize;
    if small_size > 0 {
        if small_size > 4 {
            return stage6_err("MAT v5 small data element is malformed");
        }
        return Ok(MatElementHeader {
            data_type: small_type,
            data_size: small_size,
            data_start: offset + 4,
            padded_end: offset + 8,
        });
    }
    let data_type = endian.read_u32(&bytes[offset..offset + 4]);
    let data_size = endian.read_u32(&bytes[offset + 4..offset + 8]) as usize;
    let data_start = offset + 8;
    let padded = (8 - (data_size % 8)) % 8;
    let padded_end = data_start
        .checked_add(data_size)
        .and_then(|end| end.checked_add(padded))
        .ok_or_else(|| stage6_err_owned("MAT v5 data element size overflows usize".to_string()))?;
    Ok(MatElementHeader {
        data_type,
        data_size,
        data_start,
        padded_end,
    })
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
        stage: 6,
        message: format!("{label} is missing"),
    })?;
    if values.len() != len {
        return stage6_err(format!(
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
            stage: 6,
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
            stage: 6,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f64(source));
    }
    stage6_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}, expected {n_ps}x{n_dim}",
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
            stage: 6,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f32(source));
    }
    stage6_err(format!(
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
            stage: 6,
            message: format!("{label} is invalid: {err}"),
        })?;
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        return Ok(transpose_complex_f32(source));
    }
    stage6_err(format!(
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
    stage6_err(format!(
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

fn tuple_to_complex(value: (f32, f32)) -> Complex64 {
    Complex64::new(value.0 as f64, value.1 as f64)
}

fn rotate_tuple(value: &mut (f32, f32), theta: f32) {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let re = value.0 * cos_theta - value.1 * sin_theta;
    let im = value.0 * sin_theta + value.1 * cos_theta;
    *value = (re, im);
}

fn multiply_tuple_conj_patch(value: &mut (f32, f32), patch: (f32, f32)) {
    let re = value.0 * patch.0 + value.1 * patch.1;
    let im = value.1 * patch.0 - value.0 * patch.1;
    *value = (re, im);
}

fn normalize_tuple(value: &mut (f32, f32)) {
    let mag = value.0.hypot(value.1);
    if mag > 0.0 {
        value.0 /= mag;
        value.1 /= mag;
    }
}

fn residual_phase(value: (f32, f32), ph_pix: f32) -> f32 {
    let (sin_theta, cos_theta) = (-ph_pix).sin_cos();
    let re = value.0 * cos_theta - value.1 * sin_theta;
    let im = value.0 * sin_theta + value.1 * cos_theta;
    im.atan2(re)
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

fn stage6_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage6_err_owned(message.into()))
}

fn stage6_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 6, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Instant;

    #[test]
    fn stage6_synthetic_fixture_writes_unwrap_artifacts() {
        let root = temp_dataset("pystamps-stage6-synthetic");
        write_stage6_inputs(&root, 4);

        let native_start = Instant::now();
        let details = run_stage6_native(&root).unwrap();
        let native_elapsed = native_start.elapsed();

        assert!(details.contains("natively unwrapped 4 PS"));
        let phuw2 = MatData::read(root.join("phuw2.mat")).unwrap();
        let ph_uw = phuw2.get_f32_matrix("ph_uw").unwrap();
        assert_eq!((ph_uw.rows, ph_uw.cols), (4, 3));
        let expected_col0 = [0.0f32, 0.4, 0.8, 1.2];
        let expected_col2 = [1.0f32, 1.4, 1.8, 2.2];
        for row in 0..4 {
            assert!((ph_uw.values[row * 3] - expected_col0[row]).abs() < 1.0e-5);
            assert_eq!(ph_uw.values[row * 3 + 1], 0.0);
            assert!((ph_uw.values[row * 3 + 2] - expected_col2[row]).abs() < 1.0e-5);
        }

        let uw_phaseuw = MatData::read(root.join("uw_phaseuw.mat")).unwrap();
        assert_eq!(uw_phaseuw.get_f32_matrix("ph_uw").unwrap().cols, 2);
        let uw_grid = MatData::read(root.join("uw_grid.mat")).unwrap();
        assert_eq!(scalar_from_mat(&uw_grid, "n_ps", 0.0), 4.0);
        let uw_interp = MatData::read(root.join("uw_interp.mat")).unwrap();
        assert!(scalar_from_mat(&uw_interp, "n_edge", 0.0) >= 3.0);
        assert!(!root.join("snaphu.in").exists());
        assert!(!root.join("unwrap.1.node").exists());
        assert!(native_elapsed.as_millis() < 500);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nearest_grid_labels_match_bruteforce_nearest_point() {
        let uw_grid = UwGrid {
            ph: Matrix {
                name: "ph".to_string(),
                rows: 4,
                cols: 1,
                values: vec![(1.0, 0.0); 4],
            },
            ph_in: Matrix {
                name: "ph_in".to_string(),
                rows: 4,
                cols: 1,
                values: vec![(1.0, 0.0); 4],
            },
            nzix: Matrix {
                name: "nzix".to_string(),
                rows: 4,
                cols: 5,
                values: vec![1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            },
            grid_ij: Matrix {
                name: "grid_ij".to_string(),
                rows: 4,
                cols: 2,
                values: Vec::new(),
            },
            n_i: 4,
            n_j: 5,
            n_ps: 4,
            xy: Matrix {
                name: "xy".to_string(),
                rows: 4,
                cols: 3,
                values: Vec::new(),
            },
            ij: Matrix {
                name: "ij".to_string(),
                rows: 4,
                cols: 2,
                values: Vec::new(),
            },
            grid_x_min: 0.0,
            grid_y_min: 0.0,
            pix_size: 20.0,
        };

        let points = grid_points(&uw_grid).unwrap();
        let observed = nearest_grid_labels(&uw_grid).unwrap();
        let expected = (0..uw_grid.n_i)
            .flat_map(|row| {
                let points = &points;
                (0..uw_grid.n_j)
                    .map(move |col| nearest_point(points, ((col + 1) as f64, (row + 1) as f64)) + 1)
            })
            .collect::<Vec<_>>();

        assert_eq!(observed, expected);
    }

    #[test]
    fn stage6_disconnected_unwrap_graph_returns_structured_error() {
        let root = temp_dataset("pystamps-stage6-disconnected");
        write_stage6_inputs(&root, 2);
        let mut uw_grid = MatFile::new(root.join("uw_grid.mat"));
        uw_grid
            .add_complex_f32_matrix("ph", 2, 2, vec![(1.0, 0.0); 4])
            .unwrap();
        uw_grid
            .add_complex_f32_matrix("ph_in", 2, 2, vec![(1.0, 0.0); 4])
            .unwrap();
        uw_grid.add_u8_matrix("nzix", 1, 2, vec![1, 1]).unwrap();
        uw_grid
            .add_f64_matrix("grid_ij", 2, 2, vec![1.0, 1.0, 1.0, 2.0])
            .unwrap();
        uw_grid.add_f64_scalar("n_ps", 2.0).unwrap();
        uw_grid.write().unwrap();

        let mut uw_interp = MatFile::new(root.join("uw_interp.mat"));
        uw_interp.add_f64_matrix("edgs", 0, 3, Vec::new()).unwrap();
        uw_interp.add_f64_scalar("n_edge", 0.0).unwrap();
        uw_interp.add_f64_matrix("rowix", 0, 2, Vec::new()).unwrap();
        uw_interp.add_f64_matrix("colix", 1, 1, vec![0.0]).unwrap();
        uw_interp.add_f64_matrix("Z", 1, 2, vec![1.0, 2.0]).unwrap();
        uw_interp.write().unwrap();

        let err = run_stage6_native(&root).unwrap_err().to_string();
        assert!(err.contains("stage 6 native implementation error"));
        assert!(err.contains("disconnected unwrap graph"));
        assert!(!root.join("phuw2.mat").exists());

        fs::remove_dir_all(root).unwrap();
    }

    fn write_stage6_inputs(root: &Path, n_ps: usize) {
        if root.exists() {
            fs::remove_dir_all(root).unwrap();
        }
        fs::create_dir_all(root).unwrap();
        let n_ifg = 3usize;
        let master_ix = 2usize;
        let xy = if n_ps == 4 {
            vec![
                1.0, 0.0, 0.0, 2.0, 41.0, 0.0, 3.0, 0.0, 41.0, 4.0, 41.0, 41.0,
            ]
        } else {
            vec![1.0, 0.0, 0.0, 2.0, 41.0, 0.0]
        };
        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps2.add_f64_scalar("n_ifg", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("n_image", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("master_ix", master_ix as f64).unwrap();
        ps2.add_f64_col_vector("day", vec![10.0, 20.0, 30.0])
            .unwrap();
        ps2.add_f32_col_vector("bperp", vec![10.0, 0.0, 20.0])
            .unwrap();
        ps2.add_f64_matrix("xy", n_ps, 3, xy).unwrap();
        ps2.add_f64_scalar("mean_range", 830000.0).unwrap();
        ps2.add_f64_scalar("mean_incidence", 23.0_f64.to_radians())
            .unwrap();
        ps2.write().unwrap();

        let mut phases = Vec::with_capacity(n_ps * n_ifg);
        for row in 0..n_ps {
            let base = row as f32 * 0.4;
            for col in 0..n_ifg {
                let phase = if col == 1 {
                    0.0
                } else {
                    base + col as f32 * 0.5
                };
                phases.push((phase.cos(), phase.sin()));
            }
        }
        let mut ph2 = MatFile::new(root.join("ph2.mat"));
        ph2.add_complex_f32_matrix("ph", n_ps, n_ifg, phases.clone())
            .unwrap();
        ph2.write().unwrap();

        let mut pm2 = MatFile::new(root.join("pm2.mat"));
        pm2.add_f64_col_vector("K_ps", vec![0.0; n_ps]).unwrap();
        pm2.add_f64_col_vector("C_ps", vec![0.0; n_ps]).unwrap();
        pm2.add_f64_col_vector("coh_ps", vec![1.0; n_ps]).unwrap();
        pm2.add_complex_f32_matrix(
            "ph_patch",
            n_ps,
            n_ifg - 1,
            vec![(1.0, 0.0); n_ps * (n_ifg - 1)],
        )
        .unwrap();
        pm2.add_f32_matrix("ph_res", n_ps, n_ifg - 1, vec![0.0; n_ps * (n_ifg - 1)])
            .unwrap();
        pm2.write().unwrap();

        let mut bp2 = MatFile::new(root.join("bp2.mat"));
        bp2.add_f32_matrix("bperp_mat", n_ps, n_ifg - 1, vec![0.0; n_ps * (n_ifg - 1)])
            .unwrap();
        bp2.write().unwrap();

        let mut ifgstd2 = MatFile::new(root.join("ifgstd2.mat"));
        ifgstd2
            .add_f64_col_vector("ifg_std", vec![1.0; n_ifg])
            .unwrap();
        ifgstd2.write().unwrap();

        let mut parms = MatFile::new(root.join("parms.mat"));
        parms
            .add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        parms
            .add_u32_matrix("unwrap_patch_phase", 1, 1, vec!['n' as u32])
            .unwrap();
        parms.add_f64_scalar("unwrap_grid_size", 20.0).unwrap();
        parms.write().unwrap();
    }

    fn temp_dataset(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{name}-{}-{}", std::process::id(), unique_nanos()))
    }

    fn unique_nanos() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
