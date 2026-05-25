use crate::CoreError;
use delaunator::{triangulate, Point};
use num_complex::Complex64;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

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
    values: Vec<Complex64>,
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
}

pub fn run_stage6_native(dataset_root: impl AsRef<Path>) -> Result<String, CoreError> {
    let dataset_root = dataset_root.as_ref();
    let ps2 = read_mat_stage6(dataset_root, "ps2.mat")?;
    let ph2 = read_mat_stage6(dataset_root, "ph2.mat")?;
    let pm2 = read_mat_stage6(dataset_root, "pm2.mat")?;
    let bp2 = read_mat_stage6(dataset_root, "bp2.mat")?;
    let _ifgstd2 = read_mat_stage6(dataset_root, "ifgstd2.mat")?;
    let parms = load_stage6_parms(dataset_root);

    let n_ps = scalar_from_mat(&ps2, "n_ps", 0.0).round() as usize;
    if n_ps == 0 {
        return stage6_err("ps2.mat missing valid n_ps");
    }
    let ph2 = complex_ps_matrix(&ph2, "ph", n_ps, "ph2.ph")?;
    let n_ifg = ph2.cols;
    if n_ifg == 0 {
        return stage6_err("ph2.ph must contain at least one interferogram");
    }
    let master_ix = scalar_from_mat(&ps2, "master_ix", 1.0).round() as usize;
    if master_ix == 0 || master_ix > n_ifg {
        return stage6_err(format!("ps2.master_ix must be 1-based within ph2.ph columns; got {master_ix}"));
    }
    let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
    if small_baseline {
        return stage6_err("Stage 6 native unwrap currently supports single-master merged artifacts");
    }

    let drop_set: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    let unwrap_cols: Vec<usize> = (0..n_ifg)
        .filter(|col| !drop_set.contains(&((*col + 1) as i64)) && *col != master_ix - 1)
        .collect();
    if unwrap_cols.is_empty() {
        return stage6_err("No interferograms available for stage-6 unwrapping");
    }

    let bperp_full = expand_bperp_matrix(&bp2, &ps2, n_ps, n_ifg, master_ix)?;
    let wrapped = build_wrapped_phase(dataset_root, &ph2, &pm2, &bperp_full, n_ps, n_ifg, master_ix, &parms)?;
    let uw_grid = if dataset_root.join("uw_grid.mat").exists() {
        read_uw_grid(dataset_root, n_ps)?
    } else {
        let grid = build_uw_grid(&ps2, &wrapped, &unwrap_cols, n_ps, &parms)?;
        write_uw_grid(dataset_root, &grid)?;
        grid
    };
    let uw_interp = if dataset_root.join("uw_interp.mat").exists() {
        read_uw_interp(dataset_root, uw_grid.n_i, uw_grid.n_j)?
    } else {
        let interp = build_uw_interp(&uw_grid)?;
        write_uw_interp(dataset_root, &interp)?;
        interp
    };
    validate_connected_graph(&uw_interp, uw_grid.n_ps)?;

    let ph_uw_some = unwrap_grid_phase(&uw_grid, &uw_interp)?;
    let msd_some = grid_msd(&ph_uw_some, uw_grid.n_ps, unwrap_cols.len(), &uw_interp);
    write_uw_phaseuw(dataset_root, &ph_uw_some, uw_grid.n_ps, unwrap_cols.len(), &msd_some)?;
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

    Ok(format!(
        "Stage 6 natively unwrapped {n_ps} PS across {n_ifg} interferograms using Rust graph unwrap"
    ))
}

fn build_wrapped_phase(
    dataset_root: &Path,
    ph2: &ComplexMatrixF32,
    pm2: &MatData,
    bperp_full: &Matrix<f64>,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
    parms: &Stage6Parms,
) -> Result<WrappedPhase, CoreError> {
    let mut ph_w: Vec<Complex64> = if parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
        let ph_patch = complex_ps_matrix(pm2, "ph_patch", n_ps, "pm2.ph_patch")?;
        if ph_patch.cols + 1 != n_ifg {
            return stage6_err(format!(
                "pm2.ph_patch has {} columns but single-master ph2.ph has {n_ifg}",
                ph_patch.cols
            ));
        }
        let mut values = vec![Complex64::new(1.0, 0.0); n_ps * n_ifg];
        for row in 0..n_ps {
            for col in 0..n_ifg {
                if col == master_ix - 1 {
                    continue;
                }
                let src_col = if col < master_ix - 1 { col } else { col - 1 };
                values[row * n_ifg + col] = tuple_to_complex(ph_patch.values[row * ph_patch.cols + src_col]);
            }
        }
        values
    } else if dataset_root.join("rc2.mat").exists() {
        let rc2 = read_mat_stage6(dataset_root, "rc2.mat")?;
        let ph_rc = complex_ps_matrix(&rc2, "ph_rc", n_ps, "rc2.ph_rc")?;
        ph_rc.values.iter().map(|&value| tuple_to_complex(value)).collect()
    } else {
        ph2.values.iter().map(|&value| tuple_to_complex(value)).collect()
    };

    if !parms.unwrap_patch_phase.eq_ignore_ascii_case("y") {
        if let Some(k_ps) = optional_vector_f64(pm2, "K_ps") {
            if k_ps.len() == n_ps {
                for row in 0..n_ps {
                    for col in 0..n_ifg {
                        let theta = k_ps[row] * bperp_full.values[row * n_ifg + col];
                        ph_w[row * n_ifg + col] *= Complex64::from_polar(1.0, theta);
                    }
                }
            }
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
                            let theta = k_ps_uw[row] * bperp_full.values[row * n_ifg + col];
                            ph_w[row * n_ifg + col] *= Complex64::from_polar(1.0, -theta);
                            phase_restore[row * n_ifg + col] += theta as f32;
                        }
                    }
                }
            }
            if let Some(c_ps_uw) = optional_vector_f64(&scla, "C_ps_uw") {
                if c_ps_uw.len() == n_ps {
                    for row in 0..n_ps {
                        for col in 0..n_ifg {
                            ph_w[row * n_ifg + col] *= Complex64::from_polar(1.0, -c_ps_uw[row]);
                            phase_restore[row * n_ifg + col] += c_ps_uw[row] as f32;
                        }
                    }
                }
            }
            if let Ok(ph_ramp) = ps_matrix_f32(&scla, "ph_ramp", n_ps, "scla_smooth2.ph_ramp") {
                if ph_ramp.cols == n_ifg {
                    for row in 0..n_ps {
                        for col in 0..n_ifg {
                            let theta = ph_ramp.values[row * n_ifg + col] as f64;
                            ph_w[row * n_ifg + col] *= Complex64::from_polar(1.0, -theta);
                            phase_restore[row * n_ifg + col] += theta as f32;
                        }
                    }
                }
            }
        }
    }

    for value in &mut ph_w {
        let norm = value.norm();
        if norm > 0.0 {
            *value /= norm;
        }
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
    let xy = ps_dim_f64(ps2, "xy", n_ps, 3, "ps2.xy")?;
    let pix_size = if parms.unwrap_grid_size > 0.0 {
        parms.unwrap_grid_size
    } else {
        20.0
    };
    let grid_x_min = (0..n_ps)
        .map(|row| xy.values[row * 3 + 1])
        .fold(f64::INFINITY, f64::min);
    let grid_y_min = (0..n_ps)
        .map(|row| xy.values[row * 3 + 2])
        .fold(f64::INFINITY, f64::min);
    let mut grid_i = vec![1usize; n_ps];
    let mut grid_j = vec![1usize; n_ps];
    for row in 0..n_ps {
        let x = xy.values[row * 3 + 1];
        let y = xy.values[row * 3 + 2];
        grid_i[row] = ((y - grid_y_min + 1.0e-3) / pix_size).ceil().max(1.0) as usize;
        grid_j[row] = ((x - grid_x_min + 1.0e-3) / pix_size).ceil().max(1.0) as usize;
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
        let entry = grouped.entry(lin).or_insert_with(|| vec![Complex64::new(0.0, 0.0); n_unwrap]);
        for (out_col, &src_col) in unwrap_cols.iter().enumerate() {
            let value = wrapped.values[row * wrapped.cols + src_col];
            entry[out_col] += value;
            ph_in[row * n_unwrap + out_col] = (value.re as f32, value.im as f32);
        }
    }

    let mut nz_flat = vec![false; n_i * n_j];
    let mut ph_values = Vec::new();
    let mut nz_lins = Vec::new();
    for (lin, values) in grouped {
        if values.first().map(|value| value.norm() > 0.0).unwrap_or(false) {
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
    for (pos, &lin) in nz_lins.iter().enumerate() {
        let i = (lin % n_i) + 1;
        let j = (lin / n_i) + 1;
        xy_grid.push((pos + 1) as f64);
        xy_grid.push((j as f64 - 0.5) * pix_size);
        xy_grid.push((i as f64 - 0.5) * pix_size);
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
        grid_x_min: grid_x_min as f32,
        grid_y_min: grid_y_min as f32,
        pix_size,
    })
}

fn build_uw_interp(uw_grid: &UwGrid) -> Result<UwInterp, CoreError> {
    let points = grid_points(uw_grid)?;
    let mut z = vec![0usize; uw_grid.n_i * uw_grid.n_j];
    for row in 0..uw_grid.n_i {
        for col in 0..uw_grid.n_j {
            let target = ((col + 1) as f64, (row + 1) as f64);
            let nearest = nearest_point(&points, target) + 1;
            z[row * uw_grid.n_j + col] = nearest;
        }
    }

    let mut edge_ids = BTreeMap::<(usize, usize), usize>::new();
    for (a, b) in native_graph_edges(&points) {
        let _ = edge_id_signed(&mut edge_ids, a + 1, b + 1);
    }
    let mut rowix = vec![0.0; uw_grid.n_i.saturating_sub(1) * uw_grid.n_j];
    for row in 0..uw_grid.n_i.saturating_sub(1) {
        for col in 0..uw_grid.n_j {
            let value = edge_id_signed(&mut edge_ids, z[row * uw_grid.n_j + col], z[(row + 1) * uw_grid.n_j + col]);
            rowix[row * uw_grid.n_j + col] = value as f64;
        }
    }
    let mut colix = vec![0.0; uw_grid.n_i * uw_grid.n_j.saturating_sub(1)];
    for row in 0..uw_grid.n_i {
        for col in 0..uw_grid.n_j.saturating_sub(1) {
            let value = edge_id_signed(&mut edge_ids, z[row * uw_grid.n_j + col], z[row * uw_grid.n_j + col + 1]);
            colix[row * uw_grid.n_j.saturating_sub(1) + col] = value as f64;
        }
    }
    let mut edgs = vec![0.0; edge_ids.len() * 3];
    for ((a, b), id) in &edge_ids {
        let row = *id - 1;
        edgs[row * 3] = *id as f64;
        edgs[row * 3 + 1] = *a as f64;
        edgs[row * 3 + 2] = *b as f64;
    }
    Ok(UwInterp {
        edgs: Matrix {
            name: "edgs".to_string(),
            rows: edge_ids.len(),
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
        n_edge: edge_ids.len(),
    })
}

fn unwrap_grid_phase(uw_grid: &UwGrid, uw_interp: &UwInterp) -> Result<Vec<f32>, CoreError> {
    validate_connected_graph(uw_interp, uw_grid.n_ps)?;
    let adjacency = graph_adjacency(&uw_interp.edgs, uw_grid.n_ps)?;
    let mut output = vec![0.0f32; uw_grid.n_ps * uw_grid.ph.cols];
    for col in 0..uw_grid.ph.cols {
        let wrapped: Vec<f64> = (0..uw_grid.n_ps)
            .map(|row| tuple_to_complex(uw_grid.ph.values[row * uw_grid.ph.cols + col]).arg())
            .collect();
        let mut visited = vec![false; uw_grid.n_ps];
        let mut queue = VecDeque::new();
        output[col] = wrapped[0] as f32;
        visited[0] = true;
        queue.push_back(0usize);
        while let Some(node) = queue.pop_front() {
            for &next in &adjacency[node] {
                if visited[next] {
                    continue;
                }
                let delta = wrap_phase(wrapped[next] - wrapped[node]);
                output[next * uw_grid.ph.cols + col] = output[node * uw_grid.ph.cols + col] + delta as f32;
                visited[next] = true;
                queue.push_back(next);
            }
        }
        if visited.iter().any(|&seen| !seen) {
            return stage6_err("disconnected unwrap graph: not all grid points are reachable");
        }
    }
    Ok(output)
}

fn grid_msd(ph_uw: &[f32], n_ps_grid: usize, n_unwrap: usize, uw_interp: &UwInterp) -> Vec<f64> {
    let mut msd = vec![0.0; n_unwrap];
    if uw_interp.edgs.rows == 0 {
        return msd;
    }
    for col in 0..n_unwrap {
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in 0..uw_interp.edgs.rows {
            let a = uw_interp.edgs.values[row * 3 + 1].round() as isize - 1;
            let b = uw_interp.edgs.values[row * 3 + 2].round() as isize - 1;
            if a < 0 || b < 0 || a as usize >= n_ps_grid || b as usize >= n_ps_grid {
                continue;
            }
            let diff = ph_uw[a as usize * n_unwrap + col] as f64 - ph_uw[b as usize * n_unwrap + col] as f64;
            sum += diff * diff;
            count += 1;
        }
        if count > 0 {
            msd[col] = sum / count as f64;
        }
    }
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
    for row in 0..n_ps {
        let grid_i = uw_grid.grid_ij.values[row * 2].round() as isize;
        let grid_j = uw_grid.grid_ij.values[row * 2 + 1].round() as isize;
        if grid_i <= 0 || grid_j <= 0 || grid_i as usize > uw_grid.n_i || grid_j as usize > uw_grid.n_j {
            continue;
        }
        let ps_grid_idx = gridix[(grid_i as usize - 1) * uw_grid.n_j + (grid_j as usize - 1)];
        if ps_grid_idx == 0 {
            continue;
        }
        for (out_col, &src_col) in unwrap_cols.iter().enumerate() {
            let ph_pix = ph_uw_some[(ps_grid_idx - 1) * unwrap_cols.len() + out_col];
            let ph_in = tuple_to_complex(uw_grid.ph_in.values[row * unwrap_cols.len() + out_col]);
            let residual = (ph_in * Complex64::from_polar(1.0, -(ph_pix as f64))).arg() as f32;
            ph_uw[row * n_ifg + src_col] = ph_pix + residual + wrapped.phase_restore[row * n_ifg + src_col];
        }
    }
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
    mat.add_complex_f32_matrix("ph_in", grid.ph_in.rows, grid.ph_in.cols, grid.ph_in.values.clone())?;
    mat.add_complex_f32_matrix("ph_lowpass", 0, 0, Vec::new())?;
    mat.add_complex_f32_matrix("ph_uw_predef", 0, 0, Vec::new())?;
    mat.add_complex_f32_matrix("ph_in_predef", 0, 0, Vec::new())?;
    mat.add_f64_matrix("xy", grid.xy.rows, grid.xy.cols, grid.xy.values.clone())?;
    mat.add_f64_matrix("ij", grid.ij.rows, grid.ij.cols, grid.ij.values.clone())?;
    mat.add_u8_matrix("nzix", grid.nzix.rows, grid.nzix.cols, grid.nzix.values.clone())?;
    mat.add_f32_scalar("grid_x_min", grid.grid_x_min)?;
    mat.add_f32_scalar("grid_y_min", grid.grid_y_min)?;
    mat.add_f32_scalar("n_i", grid.n_i as f32)?;
    mat.add_f32_scalar("n_j", grid.n_j as f32)?;
    mat.add_f64_scalar("n_ifg", grid.ph.cols as f64)?;
    mat.add_f64_scalar("n_ps", grid.n_ps as f64)?;
    mat.add_f64_matrix("grid_ij", grid.grid_ij.rows, grid.grid_ij.cols, grid.grid_ij.values.clone())?;
    mat.add_f64_scalar("pix_size", grid.pix_size)?;
    mat.write()?;
    Ok(())
}

fn write_uw_interp(dataset_root: &Path, interp: &UwInterp) -> Result<(), CoreError> {
    let mut mat = MatFile::new(dataset_root.join("uw_interp.mat"));
    mat.add_f64_matrix("edgs", interp.edgs.rows, interp.edgs.cols, interp.edgs.values.clone())?;
    mat.add_f64_scalar("n_edge", interp.n_edge as f64)?;
    mat.add_f64_matrix("rowix", interp.rowix.rows, interp.rowix.cols, interp.rowix.values.clone())?;
    mat.add_f64_matrix("colix", interp.colix.rows, interp.colix.cols, interp.colix.values.clone())?;
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
    let ph_in = complex_ps_matrix(&mat, "ph_in", n_ps, "uw_grid.ph_in").unwrap_or(ComplexMatrixF32 {
        name: "ph_in".to_string(),
        rows: n_ps,
        cols: ph.cols,
        values: vec![(0.0, 0.0); n_ps * ph.cols],
    });
    let nzix_source = mat.get_f32_matrix("nzix").or_else(|_| mat.get_f64_matrix("nzix").map(|m| Matrix {
        name: m.name,
        rows: m.rows,
        cols: m.cols,
        values: m.values.iter().map(|&value| value as f32).collect(),
    })).map_err(|err| CoreError::NativeStage {
        stage: 6,
        message: format!("uw_grid.nzix is invalid: {err}"),
    })?;
    let nzix = Matrix {
        name: "nzix".to_string(),
        rows: nzix_source.rows,
        cols: nzix_source.cols,
        values: nzix_source.values.iter().map(|&value| u8::from(value != 0.0)).collect(),
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
    let edgs = mat.get_f64_matrix("edgs").map_err(|err| CoreError::NativeStage {
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
    Ok(UwInterp {
        n_edge: scalar_from_mat(&mat, "n_edge", edgs.rows as f64).round() as usize,
        edgs,
        rowix,
        colix,
        z,
    })
}

fn expand_bperp_matrix(
    bp2: &MatData,
    ps2: &MatData,
    n_ps: usize,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Matrix<f64>, CoreError> {
    if let Ok(bp_nm) = ps_matrix_f32(bp2, "bperp_mat", n_ps, "bp2.bperp_mat") {
        if bp_nm.cols == n_ifg {
            return Ok(Matrix {
                name: "bperp_mat".to_string(),
                rows: n_ps,
                cols: n_ifg,
                values: bp_nm.values.iter().map(|&value| value as f64).collect(),
            });
        }
        if bp_nm.cols + 1 == n_ifg {
            let mut values = vec![0.0; n_ps * n_ifg];
            for row in 0..n_ps {
                for col in 0..n_ifg {
                    if col == master_ix - 1 {
                        continue;
                    }
                    let src_col = if col < master_ix - 1 { col } else { col - 1 };
                    values[row * n_ifg + col] = bp_nm.values[row * bp_nm.cols + src_col] as f64;
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
        values.extend_from_slice(&bperp);
    }
    Ok(Matrix {
        name: "bperp_mat".to_string(),
        rows: n_ps,
        cols: n_ifg,
        values,
    })
}

fn native_graph_edges(points: &[(f64, f64)]) -> Vec<(usize, usize)> {
    if points.len() < 2 {
        return Vec::new();
    }
    if points.len() == 2 {
        return vec![(0, 1)];
    }
    let delaunay_points: Vec<Point> = points.iter().map(|&(x, y)| Point { x, y }).collect();
    let triangulation = triangulate(&delaunay_points);
    let mut edges = BTreeSet::new();
    for tri in triangulation.triangles.chunks_exact(3) {
        insert_edge(&mut edges, tri[0], tri[1]);
        insert_edge(&mut edges, tri[1], tri[2]);
        insert_edge(&mut edges, tri[0], tri[2]);
    }
    if edges.is_empty() {
        nearest_neighbor_edges(points)
    } else {
        edges.into_iter().collect()
    }
}

fn nearest_neighbor_edges(points: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for (i, &point) in points.iter().enumerate() {
        if let Some((j, _)) = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, &other)| (j, (point.0 - other.0).powi(2) + (point.1 - other.1).powi(2)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        {
            insert_edge(&mut edges, i, j);
        }
    }
    edges.into_iter().collect()
}

fn insert_edge(edges: &mut BTreeSet<(usize, usize)>, a: usize, b: usize) {
    if a != b {
        edges.insert((a.min(b), a.max(b)));
    }
}

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

fn nearest_point(points: &[(f64, f64)], target: (f64, f64)) -> usize {
    points
        .iter()
        .enumerate()
        .map(|(ix, &point)| (ix, (point.0 - target.0).powi(2) + (point.1 - target.1).powi(2)))
        .min_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .map(|(ix, _)| ix)
        .unwrap_or(0)
}

fn edge_id_signed(edge_ids: &mut BTreeMap<(usize, usize), usize>, a: usize, b: usize) -> isize {
    if a == b {
        return 0;
    }
    let edge = (a.min(b), a.max(b));
    let next_id = edge_ids.len() + 1;
    let id = *edge_ids.entry(edge).or_insert(next_id);
    if a <= b {
        id as isize
    } else {
        -(id as isize)
    }
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
        return stage6_err(format!("uw_interp.edgs must have at least 3 columns, got {}", edgs.cols));
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
    MatData::read(dataset_root.join(filename)).map_err(|err| stage6_err_owned(format!("unable to read {filename}: {err}")))
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
        stage: 6,
        message: format!("{label} is missing"),
    })?;
    if values.len() != len {
        return stage6_err(format!("{label} has incompatible length {} for expected length {len}", values.len()));
    }
    Ok(values)
}

fn ps_matrix_f32(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
    let source = mat.get_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 6,
        message: format!("{label} is invalid: {err}"),
    })?;
    orient_matrix_f32(source, n_ps, label)
}

fn ps_dim_f64(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| CoreError::NativeStage {
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

fn complex_ps_matrix(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat.get_complex_f32_matrix(name).map_err(|err| CoreError::NativeStage {
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

fn orient_matrix_f32(source: Matrix<f32>, n_ps: usize, label: &str) -> Result<Matrix<f32>, CoreError> {
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
    fn stage6_disconnected_unwrap_graph_returns_structured_error() {
        let root = temp_dataset("pystamps-stage6-disconnected");
        write_stage6_inputs(&root, 2);
        let mut uw_grid = MatFile::new(root.join("uw_grid.mat"));
        uw_grid.add_complex_f32_matrix("ph", 2, 2, vec![(1.0, 0.0); 4]).unwrap();
        uw_grid.add_complex_f32_matrix("ph_in", 2, 2, vec![(1.0, 0.0); 4]).unwrap();
        uw_grid.add_u8_matrix("nzix", 1, 2, vec![1, 1]).unwrap();
        uw_grid.add_f64_matrix("grid_ij", 2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();
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
                1.0, 0.0, 0.0,
                2.0, 41.0, 0.0,
                3.0, 0.0, 41.0,
                4.0, 41.0, 41.0,
            ]
        } else {
            vec![1.0, 0.0, 0.0, 2.0, 41.0, 0.0]
        };
        let mut ps2 = MatFile::new(root.join("ps2.mat"));
        ps2.add_f64_scalar("n_ps", n_ps as f64).unwrap();
        ps2.add_f64_scalar("n_ifg", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("n_image", n_ifg as f64).unwrap();
        ps2.add_f64_scalar("master_ix", master_ix as f64).unwrap();
        ps2.add_f64_col_vector("day", vec![10.0, 20.0, 30.0]).unwrap();
        ps2.add_f32_col_vector("bperp", vec![10.0, 0.0, 20.0]).unwrap();
        ps2.add_f64_matrix("xy", n_ps, 3, xy).unwrap();
        ps2.add_f64_scalar("mean_range", 830000.0).unwrap();
        ps2.add_f64_scalar("mean_incidence", 23.0_f64.to_radians()).unwrap();
        ps2.write().unwrap();

        let mut phases = Vec::with_capacity(n_ps * n_ifg);
        for row in 0..n_ps {
            let base = row as f32 * 0.4;
            for col in 0..n_ifg {
                let phase = if col == 1 { 0.0 } else { base + col as f32 * 0.5 };
                phases.push((phase.cos(), phase.sin()));
            }
        }
        let mut ph2 = MatFile::new(root.join("ph2.mat"));
        ph2.add_complex_f32_matrix("ph", n_ps, n_ifg, phases.clone()).unwrap();
        ph2.write().unwrap();

        let mut pm2 = MatFile::new(root.join("pm2.mat"));
        pm2.add_f64_col_vector("K_ps", vec![0.0; n_ps]).unwrap();
        pm2.add_f64_col_vector("C_ps", vec![0.0; n_ps]).unwrap();
        pm2.add_f64_col_vector("coh_ps", vec![1.0; n_ps]).unwrap();
        pm2.add_complex_f32_matrix("ph_patch", n_ps, n_ifg - 1, vec![(1.0, 0.0); n_ps * (n_ifg - 1)]).unwrap();
        pm2.add_f32_matrix("ph_res", n_ps, n_ifg - 1, vec![0.0; n_ps * (n_ifg - 1)]).unwrap();
        pm2.write().unwrap();

        let mut bp2 = MatFile::new(root.join("bp2.mat"));
        bp2.add_f32_matrix("bperp_mat", n_ps, n_ifg - 1, vec![0.0; n_ps * (n_ifg - 1)]).unwrap();
        bp2.write().unwrap();

        let mut ifgstd2 = MatFile::new(root.join("ifgstd2.mat"));
        ifgstd2.add_f64_col_vector("ifg_std", vec![1.0; n_ifg]).unwrap();
        ifgstd2.write().unwrap();

        let mut parms = MatFile::new(root.join("parms.mat"));
        parms.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32]).unwrap();
        parms.add_u32_matrix("unwrap_patch_phase", 1, 1, vec!['n' as u32]).unwrap();
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
