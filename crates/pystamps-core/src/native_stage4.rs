use crate::CoreError;
use delaunator::{triangulate, Point};
use num_complex::Complex64;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Stage4Parms {
    small_baseline_flag: String,
    drop_ifg_index: Vec<i64>,
    weed_standard_dev: f64,
    weed_max_noise: f64,
    weed_zero_elevation: String,
    weed_neighbours: String,
    weed_time_win: f64,
}

impl Default for Stage4Parms {
    fn default() -> Self {
        Self {
            small_baseline_flag: "n".to_string(),
            drop_ifg_index: Vec::new(),
            weed_standard_dev: std::f64::consts::PI,
            weed_max_noise: std::f64::consts::PI,
            weed_zero_elevation: "n".to_string(),
            weed_neighbours: "y".to_string(),
            weed_time_win: 360.0,
        }
    }
}

#[derive(Clone, Debug)]
struct EdgeStats {
    ps_std: Vec<f64>,
    ps_max: Vec<f64>,
}

pub fn run_stage4_native(patch_dir: impl AsRef<Path>) -> Result<String, CoreError> {
    let patch_dir = patch_dir.as_ref();
    let select1 = read_mat_stage4(patch_dir, "select1.mat")?;
    let ps1 = read_mat_stage4(patch_dir, "ps1.mat")?;
    let ph1 = read_mat_stage4(patch_dir, "ph1.mat")?;
    let _pm1 = read_mat_stage4(patch_dir, "pm1.mat")?;
    let parms = load_stage4_parms(patch_dir);

    let n_ps_total = scalar_from_mat(&ps1, "n_ps", 0.0).round() as usize;
    if n_ps_total == 0 {
        return stage4_err("ps1.mat missing valid n_ps");
    }

    let ix = vector_i64(&select1, "ix", "select1.ix")?;
    if ix.is_empty() {
        return stage4_err("select1.mat has empty ix");
    }
    let keep_ix = bool_vector_or_default(&select1, "keep_ix", ix.len(), true);
    let ix2: Vec<i64> = ix
        .iter()
        .zip(keep_ix.iter())
        .filter_map(|(&value, &keep)| keep.then_some(value))
        .collect();
    validate_one_based_indices(&ix2, n_ps_total, "select1.ix after keep_ix")?;

    if ix2.is_empty() {
        write_weed1(
            patch_dir,
            &ifg_index_for_weed(&ps1, &parms),
            &[],
            &[],
            &[],
            &[],
        )?;
        return Ok("Stage 4 retained 0/0 selected PS".to_string());
    }

    let coh_ps2_all = ps_vector_f64(&select1, "coh_ps2", ix.len(), "select1.coh_ps2")?;
    let k_ps2_all = ps_vector_f64(&select1, "K_ps2", ix.len(), "select1.K_ps2")?;
    let c_ps2_all = ps_vector_f64(&select1, "C_ps2", ix.len(), "select1.C_ps2")?;
    let coh_ps2 = select_values_by_mask(&coh_ps2_all, &keep_ix);
    let k_ps2 = select_values_by_mask(&k_ps2_all, &keep_ix);
    let c_ps2 = select_values_by_mask(&c_ps2_all, &keep_ix);
    let ix2_rows: Vec<usize> = ix2.iter().map(|&value| (value - 1) as usize).collect();

    let ij_all = ps_dim_f64(&ps1, "ij", n_ps_total, 3, "ps1.ij")?;
    let xy_all = ps_dim_f64(&ps1, "xy", n_ps_total, 3, "ps1.xy")?;
    let ij2 = select_rows_matrix_f64(&ij_all, &ix2_rows);
    let xy2 = select_rows_matrix_f64(&xy_all, &ix2_rows);
    let n_ps = ix2.len();
    let mut ix_weed = vec![true; n_ps];

    if parms.weed_neighbours.eq_ignore_ascii_case("y") {
        let ij_cols23: Vec<(i64, i64)> = (0..n_ps)
            .map(|row| {
                (
                    ij2[row * 3 + 1].round() as i64,
                    ij2[row * 3 + 2].round() as i64,
                )
            })
            .collect();
        let keep_adj = adjacent_component_keep_mask(&ij_cols23, &coh_ps2);
        for (keep, &adj_keep) in ix_weed.iter_mut().zip(keep_adj.iter()) {
            *keep &= adj_keep;
        }
    }

    if parms.weed_zero_elevation.eq_ignore_ascii_case("y") {
        if let Some(hgt) = load_hgt1(patch_dir, n_ps_total)? {
            for (pos, &source_row) in ix2_rows.iter().enumerate() {
                if hgt[source_row] < 1.0e-6 {
                    ix_weed[pos] = false;
                }
            }
        }
    }

    remove_duplicate_xy(&xy2, &coh_ps2, &mut ix_weed);

    let n_pre_noise = ix_weed.iter().filter(|&&keep| keep).count();
    let mut ix_weed2 = vec![true; n_pre_noise];
    let mut ps_std = vec![0.0_f64; n_pre_noise];
    let mut ps_max = vec![0.0_f64; n_pre_noise];
    let no_weed_noisy = parms.weed_standard_dev >= std::f64::consts::PI
        && parms.weed_max_noise >= std::f64::consts::PI;

    if !no_weed_noisy && n_pre_noise > 0 {
        let ph1 = ps_complex_matrix(&ph1, "ph", n_ps_total, "ph1.ph")?;
        let bperp = vector_f64(&ps1, "bperp", "ps1.bperp")?;
        if bperp.len() != ph1.cols {
            return stage4_err(format!(
                "ps1.bperp has length {} but ph1.ph has {} interferograms",
                bperp.len(),
                ph1.cols
            ));
        }
        let ifg_index = ifg_index_for_weed(&ps1, &parms);
        let ifg_cols: Vec<usize> = ifg_index
            .iter()
            .filter_map(|&value| {
                let ix = value.round() as i64 - 1;
                (ix >= 0 && (ix as usize) < ph1.cols).then_some(ix as usize)
            })
            .collect();
        let kept_positions: Vec<usize> = ix_weed
            .iter()
            .enumerate()
            .filter_map(|(pos, &keep)| keep.then_some(pos))
            .collect();
        let points: Vec<(f64, f64)> = kept_positions
            .iter()
            .map(|&pos| (xy2[pos * 3 + 1], xy2[pos * 3 + 2]))
            .collect();
        let edges = stage4_graph_edges(&points)?;
        validate_stage4_edge_topology(&edges, n_pre_noise)?;
        ps_std = vec![f64::INFINITY; n_pre_noise];
        ps_max = vec![f64::INFINITY; n_pre_noise];

        if !edges.is_empty() && !ifg_cols.is_empty() {
            let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
            let master_ix = scalar_from_mat(&ps1, "master_ix", 1.0).round() as usize;
            if !small_baseline && (master_ix == 0 || master_ix > ph1.cols) {
                return stage4_err(format!("ps1.master_ix must be 1-based within ph1 width {}; got {master_ix}", ph1.cols));
            }

            let mut ph_weed = Vec::with_capacity(n_pre_noise * ifg_cols.len());
            for &selected_pos in &kept_positions {
                let source_row = ix2_rows[selected_pos];
                let mut row = Vec::with_capacity(ph1.cols);
                for col in 0..ph1.cols {
                    let source = ph1.values[source_row * ph1.cols + col];
                    let phase = mul_exp_neg_i(
                        Complex64::new(source.0 as f64, source.1 as f64),
                        k_ps2[selected_pos] * bperp[col],
                    );
                    row.push(normalize_complex(phase));
                }
                if !small_baseline {
                    row[master_ix - 1] = Complex64::from_polar(1.0, c_ps2[selected_pos]);
                }
                for &col in &ifg_cols {
                    ph_weed.push(row[col]);
                }
            }
            let b_use: Vec<f64> = ifg_cols.iter().map(|&col| bperp[col]).collect();
            let day_use = if small_baseline {
                Vec::new()
            } else {
                let day = vector_f64(&ps1, "day", "ps1.day")?;
                if day.len() != ph1.cols {
                    return stage4_err(format!(
                        "ps1.day has length {} but ph1.ph has {} interferograms",
                        day.len(),
                        ph1.cols
                    ));
                }
                ifg_cols.iter().map(|&col| day[col]).collect()
            };
            let stats = stage4_edge_stats_kernel(
                &ph_weed,
                n_pre_noise,
                ifg_cols.len(),
                &edges,
                &b_use,
                &day_use,
                parms.weed_time_win,
                small_baseline,
            )?;
            ps_std = stats.ps_std;
            ps_max = stats.ps_max;
        }

        ix_weed2 = ps_std
            .iter()
            .zip(ps_max.iter())
            .map(|(&std, &max)| std < parms.weed_standard_dev && max < parms.weed_max_noise)
            .collect();
        let mut pre_noise_pos = 0usize;
        for keep in &mut ix_weed {
            if *keep {
                *keep = ix_weed2[pre_noise_pos];
                pre_noise_pos += 1;
            }
        }
    }

    write_weed1(
        patch_dir,
        &ifg_index_for_weed(&ps1, &parms),
        &ix_weed,
        &ix_weed2,
        &ps_max,
        &ps_std,
    )?;
    Ok(format!(
        "Stage 4 retained {}/{} selected PS",
        ix_weed.iter().filter(|&&keep| keep).count(),
        ix_weed.len()
    ))
}

fn read_mat_stage4(patch_dir: &Path, filename: &str) -> Result<MatData, CoreError> {
    MatData::read(patch_dir.join(filename)).map_err(|err| stage4_err_owned(format!("unable to read {filename}: {err}")))
}

fn load_stage4_parms(patch_dir: &Path) -> Stage4Parms {
    let Some(path) = resolve_file_optional(patch_dir, "parms.mat") else {
        return Stage4Parms::default();
    };
    let Ok(mat) = MatData::read(path) else {
        return Stage4Parms::default();
    };
    Stage4Parms {
        small_baseline_flag: text_from_mat(&mat, "small_baseline_flag", "n"),
        drop_ifg_index: optional_vector_f64(&mat, "drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter(|value| value.is_finite())
            .map(|value| value.round() as i64)
            .collect(),
        weed_standard_dev: scalar_from_mat(&mat, "weed_standard_dev", std::f64::consts::PI),
        weed_max_noise: scalar_from_mat(&mat, "weed_max_noise", std::f64::consts::PI),
        weed_zero_elevation: text_from_mat(&mat, "weed_zero_elevation", "n"),
        weed_neighbours: text_from_mat(&mat, "weed_neighbours", "y"),
        weed_time_win: scalar_from_mat(&mat, "weed_time_win", 360.0),
    }
}

fn write_weed1(
    patch_dir: &Path,
    ifg_index: &[f64],
    ix_weed: &[bool],
    ix_weed2: &[bool],
    ps_max: &[f64],
    ps_std: &[f64],
) -> Result<(), CoreError> {
    let mut mat = MatFile::new(patch_dir.join("weed1.mat"));
    mat.add_f64_row_vector("ifg_index", ifg_index.to_vec())?;
    mat.add_u8_matrix("ix_weed", ix_weed.len(), 1, ix_weed.iter().map(|&keep| u8::from(keep)).collect())?;
    mat.add_u8_matrix("ix_weed2", ix_weed2.len(), 1, ix_weed2.iter().map(|&keep| u8::from(keep)).collect())?;
    mat.add_f32_col_vector("ps_max", ps_max.iter().map(|&value| value as f32).collect())?;
    mat.add_f32_col_vector("ps_std", ps_std.iter().map(|&value| value as f32).collect())?;
    mat.write()?;
    Ok(())
}

fn adjacent_component_keep_mask(ij_cols23: &[(i64, i64)], coh: &[f64]) -> Vec<bool> {
    let n_ps = ij_cols23.len();
    if n_ps == 0 {
        return Vec::new();
    }
    let min_r = ij_cols23.iter().map(|&(r, _)| r).min().unwrap_or(0);
    let min_c = ij_cols23.iter().map(|&(_, c)| c).min().unwrap_or(0);
    let shifted: Vec<(usize, usize)> = ij_cols23
        .iter()
        .map(|&(r, c)| ((r + 2 - min_r) as usize, (c + 2 - min_c) as usize))
        .collect();
    let n_r = shifted.iter().map(|&(r, _)| r).max().unwrap_or(0) + 2;
    let n_c = shifted.iter().map(|&(_, c)| c).max().unwrap_or(0) + 2;
    let mut neigh_ix = vec![0usize; n_r * n_c];
    for (i, &(r, c)) in shifted.iter().enumerate() {
        for rr in r - 1..=r + 1 {
            for cc in c - 1..=c + 1 {
                if rr == r && cc == c {
                    continue;
                }
                let idx = rr * n_c + cc;
                if neigh_ix[idx] == 0 {
                    neigh_ix[idx] = i + 1;
                }
            }
        }
    }

    let mut neigh_ps = vec![Vec::<usize>::new(); n_ps + 1];
    for (i, &(r, c)) in shifted.iter().enumerate() {
        let my_neigh_ix = neigh_ix[r * n_c + c];
        if my_neigh_ix != 0 {
            neigh_ps[my_neigh_ix].push(i + 1);
        }
    }

    let mut ix_weed = vec![true; n_ps];
    for i in 1..=n_ps {
        if neigh_ps[i].is_empty() {
            continue;
        }
        let mut same_ps = vec![i];
        let mut i2 = 0usize;
        while i2 < same_ps.len() {
            let ps_i = same_ps[i2];
            if !neigh_ps[ps_i].is_empty() {
                let neighbors = std::mem::take(&mut neigh_ps[ps_i]);
                same_ps.extend(neighbors);
            }
            i2 += 1;
        }
        same_ps.sort_unstable();
        same_ps.dedup();
        let best = same_ps
            .iter()
            .copied()
            .max_by(|&left, &right| coh[left - 1].total_cmp(&coh[right - 1]))
            .unwrap_or(i);
        for same in same_ps {
            if same != best {
                ix_weed[same - 1] = false;
            }
        }
    }
    ix_weed
}

fn remove_duplicate_xy(xy2: &[f64], coh: &[f64], ix_weed: &mut [bool]) {
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (row, &keep) in ix_weed.iter().enumerate() {
        if keep {
            groups
                .entry((xy2[row * 3 + 1].to_bits(), xy2[row * 3 + 2].to_bits()))
                .or_default()
                .push(row);
        }
    }
    for rows in groups.values() {
        if rows.len() <= 1 {
            continue;
        }
        let best = rows
            .iter()
            .copied()
            .max_by(|&left, &right| coh[left].total_cmp(&coh[right]))
            .unwrap_or(rows[0]);
        for &row in rows {
            if row != best {
                ix_weed[row] = false;
            }
        }
    }
}

fn stage4_graph_edges(points: &[(f64, f64)]) -> Result<Vec<(usize, usize)>, CoreError> {
    let n = points.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    if n == 2 {
        return Ok(vec![(0, 1)]);
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
        return Ok(nearest_neighbor_edges(points));
    }
    Ok(edges.into_iter().collect())
}

fn nearest_neighbor_edges(points: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for (i, &point) in points.iter().enumerate() {
        let Some((j, _)) = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, &other)| (j, (point.0 - other.0).powi(2) + (point.1 - other.1).powi(2)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        else {
            continue;
        };
        insert_edge(&mut edges, i, j);
    }
    edges.into_iter().collect()
}

fn insert_edge(edges: &mut BTreeSet<(usize, usize)>, a: usize, b: usize) {
    if a != b {
        edges.insert((a.min(b), a.max(b)));
    }
}

fn stage4_edge_stats_kernel(
    ph: &[Complex64],
    n_node: usize,
    n_ifg: usize,
    edges: &[(usize, usize)],
    bperp: &[f64],
    day: &[f64],
    time_win: f64,
    small_baseline: bool,
) -> Result<EdgeStats, CoreError> {
    validate_stage4_edge_topology(edges, n_node)?;
    if ph.len() != n_node * n_ifg {
        return stage4_err(format!("stage4_edge_stats phase matrix has {} values for {n_node}x{n_ifg}", ph.len()));
    }
    if bperp.len() != n_ifg {
        return stage4_err("stage4_edge_stats bperp vector must match phase width");
    }
    if !small_baseline && day.len() != n_ifg {
        return stage4_err("stage4_edge_stats day vector must match phase width for non-small-baseline mode");
    }
    let mut ps_std = vec![f64::INFINITY; n_node];
    let mut ps_max = vec![f64::INFINITY; n_node];
    let n_edge = edges.len();
    if n_edge == 0 || n_ifg == 0 {
        return Ok(EdgeStats { ps_std, ps_max });
    }

    let mut dph_space = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
    for (edge_ix, &(a, b)) in edges.iter().enumerate() {
        for ifg_ix in 0..n_ifg {
            dph_space[edge_ix * n_ifg + ifg_ix] = ph[b * n_ifg + ifg_ix] * ph[a * n_ifg + ifg_ix].conj();
        }
    }

    let (edge_std, edge_max) = if !small_baseline {
        stage4_single_master_edge_stats(&dph_space, n_edge, n_ifg, bperp, day, time_win)
    } else {
        stage4_small_baseline_edge_stats(&dph_space, n_edge, n_ifg, bperp)
    };

    for (edge_ix, &(a, b)) in edges.iter().enumerate() {
        ps_std[a] = ps_std[a].min(edge_std[edge_ix]);
        ps_std[b] = ps_std[b].min(edge_std[edge_ix]);
        ps_max[a] = ps_max[a].min(edge_max[edge_ix]);
        ps_max[b] = ps_max[b].min(edge_max[edge_ix]);
    }
    Ok(EdgeStats { ps_std, ps_max })
}

fn stage4_single_master_edge_stats(
    dph_space: &[Complex64],
    n_edge: usize,
    n_ifg: usize,
    bperp: &[f64],
    day: &[f64],
    time_win: f64,
) -> (Vec<f64>, Vec<f64>) {
    let time_win = time_win.max(1.0e-6);
    let mut time_diff_all = vec![0.0; n_ifg * n_ifg];
    let mut weight_all = vec![0.0; n_ifg * n_ifg];
    for row in 0..n_ifg {
        let mut weight_sum = 0.0;
        for col in 0..n_ifg {
            let diff = day[row] - day[col];
            time_diff_all[row * n_ifg + col] = diff;
            let weight = (-(diff * diff) / (2.0 * time_win * time_win)).exp();
            weight_all[row * n_ifg + col] = weight;
            weight_sum += weight;
        }
        if weight_sum <= 0.0 {
            let fill = 1.0 / n_ifg as f64;
            for col in 0..n_ifg {
                weight_all[row * n_ifg + col] = fill;
            }
        } else {
            for col in 0..n_ifg {
                weight_all[row * n_ifg + col] /= weight_sum;
            }
        }
    }

    let mut dph_smooth0 = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
    for edge in 0..n_edge {
        for out_ix in 0..n_ifg {
            let mut accum = Complex64::new(0.0, 0.0);
            for src_ix in 0..n_ifg {
                accum += dph_space[edge * n_ifg + src_ix] * weight_all[out_ix * n_ifg + src_ix];
            }
            dph_smooth0[edge * n_ifg + out_ix] = accum;
        }
    }

    let mut dph_smooth2 = dph_smooth0.clone();
    for edge in 0..n_edge {
        for ifg in 0..n_ifg {
            dph_smooth2[edge * n_ifg + ifg] -= dph_space[edge * n_ifg + ifg] * weight_all[ifg * n_ifg + ifg];
        }
    }

    let mut dph_smooth = dph_smooth0.clone();
    for ifg in 0..n_ifg {
        let time_diff = &time_diff_all[ifg * n_ifg..(ifg + 1) * n_ifg];
        let weight = &weight_all[ifg * n_ifg..(ifg + 1) * n_ifg];
        let mut dph_mean_adj = vec![0.0; n_edge * n_ifg];
        let mut dph_mean = vec![Complex64::new(0.0, 0.0); n_edge];
        for edge in 0..n_edge {
            let mean = dph_smooth0[edge * n_ifg + ifg];
            dph_mean[edge] = mean;
            let mean_conj = mean.conj();
            for col in 0..n_ifg {
                dph_mean_adj[edge * n_ifg + col] = (dph_space[edge * n_ifg + col] * mean_conj).arg();
            }
        }
        let (m0, m1) = weighted_affine_fit_rows(time_diff, &dph_mean_adj, n_edge, n_ifg, weight);
        let mut dph_mean_adj2 = vec![0.0; n_edge * n_ifg];
        for edge in 0..n_edge {
            for col in 0..n_ifg {
                let detrended = dph_mean_adj[edge * n_ifg + col] - (m0[edge] + m1[edge] * time_diff[col]);
                dph_mean_adj2[edge * n_ifg + col] = wrap_phase(detrended);
            }
        }
        let (m20, _) = weighted_affine_fit_rows(time_diff, &dph_mean_adj2, n_edge, n_ifg, weight);
        for edge in 0..n_edge {
            dph_smooth[edge * n_ifg + ifg] = dph_mean[edge] * Complex64::from_polar(1.0, m0[edge] + m20[edge]);
        }
    }

    let mut dph_noise = vec![0.0; n_edge * n_ifg];
    let mut dph_noise2 = vec![0.0; n_edge * n_ifg];
    for edge in 0..n_edge {
        for ifg in 0..n_ifg {
            dph_noise[edge * n_ifg + ifg] = (dph_space[edge * n_ifg + ifg] * dph_smooth[edge * n_ifg + ifg].conj()).arg();
            dph_noise2[edge * n_ifg + ifg] = (dph_space[edge * n_ifg + ifg] * dph_smooth2[edge * n_ifg + ifg].conj()).arg();
        }
    }

    let ifg_var = variance_cols_real(&dph_noise2, n_edge, n_ifg, usize::from(n_edge > 1));
    let w_ifg: Vec<f64> = ifg_var
        .iter()
        .map(|&value| if value == 0.0 { f64::INFINITY } else { 1.0 / value })
        .collect();
    let k_edge = weighted_slope_fit_rows_real(bperp, &dph_noise, n_edge, n_ifg, &w_ifg);
    for edge in 0..n_edge {
        for ifg in 0..n_ifg {
            dph_noise[edge * n_ifg + ifg] -= k_edge[edge] * bperp[ifg];
        }
    }
    std_max_rows_real(&dph_noise, n_edge, n_ifg, usize::from(n_ifg > 1))
}

fn stage4_small_baseline_edge_stats(
    dph_space: &[Complex64],
    n_edge: usize,
    n_ifg: usize,
    bperp: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ifg_var = variance_cols_complex(dph_space, n_edge, n_ifg, usize::from(n_edge > 1));
    let w_ifg: Vec<f64> = ifg_var
        .iter()
        .map(|&value| if value == 0.0 { f64::INFINITY } else { 1.0 / value })
        .collect();
    let k_edge = weighted_slope_fit_rows_complex(bperp, dph_space, n_edge, n_ifg, &w_ifg);
    let mut ang = vec![0.0; n_edge * n_ifg];
    for edge in 0..n_edge {
        for ifg in 0..n_ifg {
            ang[edge * n_ifg + ifg] = (dph_space[edge * n_ifg + ifg] - k_edge[edge] * bperp[ifg]).arg();
        }
    }
    std_max_rows_real(&ang, n_edge, n_ifg, usize::from(n_ifg > 1))
}

fn validate_stage4_edge_topology(edges: &[(usize, usize)], n_nodes: usize) -> Result<(), CoreError> {
    for (pos, &(a, b)) in edges.iter().enumerate() {
        if a >= n_nodes || b >= n_nodes || a == b {
            return stage4_err(format!(
                "invalid Stage 4 edge topology at edge {}: ({a}, {b}) for n_nodes={n_nodes}",
                pos + 1
            ));
        }
    }
    Ok(())
}

fn weighted_affine_fit_rows(time_diff: &[f64], y: &[f64], n_row: usize, n_col: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut intercept = vec![0.0; n_row];
    let mut slope = vec![0.0; n_row];
    if n_row == 0 || n_col == 0 {
        return (intercept, slope);
    }
    let s0: f64 = w.iter().sum();
    let s1: f64 = w.iter().zip(time_diff.iter()).map(|(&wi, &ti)| wi * ti).sum();
    let s2: f64 = w.iter().zip(time_diff.iter()).map(|(&wi, &ti)| wi * ti * ti).sum();
    let det = s0 * s2 - s1 * s1;
    if det == 0.0 {
        if s0 != 0.0 {
            for row in 0..n_row {
                intercept[row] = (0..n_col).map(|col| y[row * n_col + col] * w[col]).sum::<f64>() / s0;
            }
        }
        return (intercept, slope);
    }
    for row in 0..n_row {
        let mut wy0 = 0.0;
        let mut wy1 = 0.0;
        for col in 0..n_col {
            let value = y[row * n_col + col];
            wy0 += value * w[col];
            wy1 += value * w[col] * time_diff[col];
        }
        intercept[row] = (wy0 * s2 - wy1 * s1) / det;
        slope[row] = (wy1 * s0 - wy0 * s1) / det;
    }
    (intercept, slope)
}

fn weighted_slope_fit_rows_real(x: &[f64], y: &[f64], n_row: usize, n_col: usize, w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let inf_idx: Vec<usize> = w.iter().enumerate().filter_map(|(idx, &value)| value.is_infinite().then_some(idx)).collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row in 0..n_row {
            out[row] = inf_idx.iter().map(|&col| y[row * n_col + col] * x[col]).sum::<f64>() / den;
        }
        return out;
    }
    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value.is_finite() && value > 0.0).then_some(idx))
        .collect();
    if pos_idx.is_empty() {
        return out;
    }
    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row in 0..n_row {
        out[row] = pos_idx
            .iter()
            .map(|&col| y[row * n_col + col] * w[col] * x[col])
            .sum::<f64>()
            / den;
    }
    out
}

fn weighted_slope_fit_rows_complex(x: &[f64], y: &[Complex64], n_row: usize, n_col: usize, w: &[f64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n_row];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let inf_idx: Vec<usize> = w.iter().enumerate().filter_map(|(idx, &value)| value.is_infinite().then_some(idx)).collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return out;
        }
        for row in 0..n_row {
            out[row] = inf_idx
                .iter()
                .map(|&col| y[row * n_col + col] * x[col])
                .sum::<Complex64>()
                / den;
        }
        return out;
    }
    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value.is_finite() && value > 0.0).then_some(idx))
        .collect();
    if pos_idx.is_empty() {
        return out;
    }
    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return out;
    }
    for row in 0..n_row {
        out[row] = pos_idx
            .iter()
            .map(|&col| y[row * n_col + col] * (w[col] * x[col]))
            .sum::<Complex64>()
            / den;
    }
    out
}

fn variance_cols_real(data: &[f64], n_row: usize, n_col: usize, ddof: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col in 0..n_col {
        let mean = (0..n_row).map(|row| data[row * n_col + col]).sum::<f64>() / n_row as f64;
        out[col] = (0..n_row)
            .map(|row| {
                let delta = data[row * n_col + col] - mean;
                delta * delta
            })
            .sum::<f64>()
            / denom as f64;
    }
    out
}

fn variance_cols_complex(data: &[Complex64], n_row: usize, n_col: usize, ddof: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_col];
    if n_row == 0 || n_col == 0 {
        return out;
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return out;
    }
    for col in 0..n_col {
        let mean = (0..n_row).map(|row| data[row * n_col + col]).sum::<Complex64>() / n_row as f64;
        out[col] = (0..n_row)
            .map(|row| (data[row * n_col + col] - mean).norm_sqr())
            .sum::<f64>()
            / denom as f64;
    }
    out
}

fn std_max_rows_real(data: &[f64], n_row: usize, n_col: usize, ddof: usize) -> (Vec<f64>, Vec<f64>) {
    let mut std = vec![0.0; n_row];
    let mut max_abs = vec![0.0; n_row];
    if n_row == 0 || n_col == 0 {
        return (std, max_abs);
    }
    let denom = n_col.saturating_sub(ddof);
    for row in 0..n_row {
        let values = &data[row * n_col..(row + 1) * n_col];
        let mean = values.iter().sum::<f64>() / n_col as f64;
        let mut accum = 0.0;
        let mut max_value = 0.0_f64;
        for &value in values {
            accum += (value - mean) * (value - mean);
            max_value = max_value.max(value.abs());
        }
        std[row] = if denom == 0 { 0.0 } else { (accum / denom as f64).sqrt() };
        max_abs[row] = max_value;
    }
    (std, max_abs)
}

fn wrap_phase(value: f64) -> f64 {
    value.sin().atan2(value.cos())
}

fn mul_exp_neg_i(value: Complex64, theta: f64) -> Complex64 {
    let (sin, cos) = theta.sin_cos();
    Complex64::new(value.re * cos + value.im * sin, value.im * cos - value.re * sin)
}

fn normalize_complex(value: Complex64) -> Complex64 {
    let norm = value.norm();
    if norm == 0.0 {
        Complex64::new(0.0, 0.0)
    } else {
        value / norm
    }
}

fn load_hgt1(patch_dir: &Path, n_ps: usize) -> Result<Option<Vec<f64>>, CoreError> {
    let path = patch_dir.join("hgt1.mat");
    if !path.exists() {
        return Ok(None);
    }
    let hgt = read_mat_stage4(patch_dir, "hgt1.mat")?;
    let values = optional_vector_f64(&hgt, "hgt")
        .or_else(|| optional_vector_f32(&hgt, "hgt").map(|values| values.into_iter().map(|value| value as f64).collect()))
        .ok_or_else(|| CoreError::NativeStage {
            stage: 4,
            message: "hgt1.mat missing hgt".to_string(),
        })?;
    if values.len() != n_ps {
        return stage4_err(format!("hgt1.hgt has incompatible length {} for n_ps={n_ps}", values.len()));
    }
    Ok(Some(values))
}

fn ifg_index_for_weed(ps: &MatData, parms: &Stage4Parms) -> Vec<f64> {
    let n_ifg = scalar_from_mat(ps, "n_ifg", 0.0).round() as i64;
    let drop: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    (1..=n_ifg)
        .filter(|value| !drop.contains(value))
        .map(|value| value as f64)
        .collect()
}

fn scalar_from_mat(mat: &MatData, name: &str, default: f64) -> f64 {
    optional_vector_f64(mat, name)
        .and_then(|values| values.into_iter().next())
        .unwrap_or(default)
}

fn vector_f64(mat: &MatData, name: &str, label: &str) -> Result<Vec<f64>, CoreError> {
    optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 4,
        message: format!("{label} is missing"),
    })
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
}

fn optional_vector_f32(mat: &MatData, name: &str) -> Option<Vec<f32>> {
    mat.get_f32_matrix(name).ok().map(|matrix| matrix.values)
}

fn vector_i64(mat: &MatData, name: &str, label: &str) -> Result<Vec<i64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 4,
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

fn ps_vector_f64(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<Vec<f64>, CoreError> {
    let values = optional_vector_f64(mat, name).ok_or_else(|| CoreError::NativeStage {
        stage: 4,
        message: format!("{label} is missing"),
    })?;
    if values.len() != n_ps {
        return stage4_err(format!("{label} has incompatible length {} for n_ps={n_ps}", values.len()));
    }
    Ok(values)
}

fn ps_dim_f64(mat: &MatData, name: &str, n_ps: usize, n_dim: usize, label: &str) -> Result<Matrix<f64>, CoreError> {
    let source = mat.get_f64_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 4,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps && source.cols == n_dim {
        return Ok(source);
    }
    if source.rows == n_dim && source.cols == n_ps {
        return Ok(transpose_f64(source));
    }
    stage4_err(format!(
        "{label} has incompatible shape {}x{}; expected {n_ps}x{n_dim}",
        source.rows, source.cols
    ))
}

fn ps_complex_matrix(mat: &MatData, name: &str, n_ps: usize, label: &str) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat.get_complex_f32_matrix(name).map_err(|err| CoreError::NativeStage {
        stage: 4,
        message: format!("{label} is missing or invalid: {err}"),
    })?;
    if source.rows == n_ps {
        return Ok(source);
    }
    if source.cols == n_ps {
        return Ok(transpose_complex(source));
    }
    stage4_err(format!(
        "{label} has incompatible shape {}x{} for n_ps={n_ps}",
        source.rows, source.cols
    ))
}

fn validate_one_based_indices(values: &[i64], n_ps: usize, label: &str) -> Result<(), CoreError> {
    for (pos, &value) in values.iter().enumerate() {
        if value < 1 || value as usize > n_ps {
            return stage4_err(format!(
                "{label} contains out-of-bounds 1-based index {value} at position {} for n_ps={n_ps}",
                pos + 1
            ));
        }
    }
    Ok(())
}

fn select_values_by_mask(values: &[f64], mask: &[bool]) -> Vec<f64> {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(&value, &keep)| keep.then_some(value))
        .collect()
}

fn select_rows_matrix_f64(matrix: &Matrix<f64>, rows: &[usize]) -> Vec<f64> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(&matrix.values[row * matrix.cols..(row + 1) * matrix.cols]);
    }
    values
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
    let text = values
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

fn stage4_err<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(stage4_err_owned(message.into()))
}

fn stage4_err_owned(message: String) -> CoreError {
    CoreError::NativeStage { stage: 4, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_parity::{compare_fixture_artifacts, ArtifactComparisonSpec, ParityTolerance};
    use std::fs;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn synthetic_neighboring_stage4_matches_python_reference_and_is_faster() {
        let root = temp_root("stage4-neighboring");
        let python_root = root.join("python");
        let rust_root = root.join("rust");
        create_stage4_fixture(&python_root);
        create_stage4_fixture(&rust_root);

        let python_start = Instant::now();
        run_python_stage4(&python_root);
        let python_elapsed = python_start.elapsed();
        let rust_start = Instant::now();
        run_stage4_native(rust_root.join("PATCH_1")).unwrap();
        let rust_elapsed = rust_start.elapsed();

        let summary = compare_fixture_artifacts(
            4,
            "patch",
            "synthetic_stage4_neighboring_ps",
            &python_root,
            &rust_root,
            &[ArtifactComparisonSpec::new(
                "PATCH_1/weed1.mat",
                ["ifg_index", "ix_weed", "ix_weed2", "ps_max", "ps_std"],
            )],
            &ParityTolerance::default(),
        )
        .unwrap();
        assert!(
            summary.all_ok(),
            "Stage 4 parity failures: {:?}",
            summary.failures().collect::<Vec<_>>()
        );
        assert!(
            rust_elapsed < python_elapsed,
            "Rust Stage 4 should beat Python/native-kernel path: rust={rust_elapsed:?} python={python_elapsed:?}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_edge_topology_returns_structured_stage4_error() {
        let ph = vec![Complex64::new(1.0, 0.0); 2 * 3];
        let err = stage4_edge_stats_kernel(
            &ph,
            2,
            3,
            &[(0, 2)],
            &[0.0, 10.0, 20.0],
            &[1.0, 2.0, 3.0],
            360.0,
            false,
        )
        .unwrap_err();
        match err {
            CoreError::NativeStage { stage, message } => {
                assert_eq!(stage, 4);
                assert!(message.contains("invalid Stage 4 edge topology"));
            }
            other => panic!("expected structured Stage 4 error, got {other:?}"),
        }
    }

    fn create_stage4_fixture(root: &Path) {
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        write_parms(&patch);
        write_ps1(&patch);
        write_ph1(&patch);
        write_pm1(&patch);
        write_select1(&patch);
        write_hgt1(&patch);
    }

    fn write_parms(patch: &Path) {
        let mut mat = MatFile::new(patch.join("parms.mat"));
        mat.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32]).unwrap();
        mat.add_u32_matrix("weed_neighbours", 1, 1, vec!['n' as u32]).unwrap();
        mat.add_u32_matrix("weed_zero_elevation", 1, 1, vec!['y' as u32]).unwrap();
        mat.add_f64_scalar("weed_standard_dev", 1.0).unwrap();
        mat.add_f64_scalar("weed_max_noise", 1.0).unwrap();
        mat.add_f64_scalar("weed_time_win", 360.0).unwrap();
        mat.write().unwrap();
    }

    fn write_ps1(patch: &Path) {
        let ij = vec![
            1.0, 10.0, 10.0,
            2.0, 10.0, 20.0,
            3.0, 20.0, 10.0,
        ];
        let xy = vec![
            1.0, 0.0, 0.0,
            2.0, 1.0, 0.0,
            3.0, 0.0, 1.0,
        ];
        let mut mat = MatFile::new(patch.join("ps1.mat"));
        mat.add_f64_scalar("n_ps", 3.0).unwrap();
        mat.add_f64_scalar("n_ifg", 4.0).unwrap();
        mat.add_f64_scalar("master_ix", 1.0).unwrap();
        mat.add_f64_row_vector("bperp", vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        mat.add_f64_row_vector("day", vec![20200101.0, 20200113.0, 20200125.0, 20200206.0]).unwrap();
        mat.add_f64_matrix("ij", 3, 3, ij).unwrap();
        mat.add_f64_matrix("xy", 3, 3, xy).unwrap();
        mat.write().unwrap();
    }

    fn write_ph1(patch: &Path) {
        let mut values = Vec::new();
        for _row in 0..3 {
            for _col in 0..4 {
                values.push((1.0_f32, 0.0_f32));
            }
        }
        let mut mat = MatFile::new(patch.join("ph1.mat"));
        mat.add_complex_f32_matrix("ph", 3, 4, values).unwrap();
        mat.write().unwrap();
    }

    fn write_pm1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("pm1.mat"));
        mat.add_f64_row_vector("coh_ps", vec![0.8, 0.7, 0.6]).unwrap();
        mat.write().unwrap();
    }

    fn write_select1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("select1.mat"));
        mat.add_f64_col_vector("ix", vec![1.0, 2.0, 3.0]).unwrap();
        mat.add_u8_matrix("keep_ix", 3, 1, vec![1, 1, 1]).unwrap();
        mat.add_f64_col_vector("K_ps2", vec![0.0, 0.0, 0.0]).unwrap();
        mat.add_f64_col_vector("C_ps2", vec![0.0, 0.0, 0.0]).unwrap();
        mat.add_f64_col_vector("coh_ps2", vec![0.8, 0.7, 0.6]).unwrap();
        mat.write().unwrap();
    }

    fn write_hgt1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("hgt1.mat"));
        mat.add_f32_col_vector("hgt", vec![10.0, 11.0, 12.0]).unwrap();
        mat.write().unwrap();
    }

    fn run_python_stage4(root: &Path) {
        let script = "import sys; from pathlib import Path; from pystamps.pipeline.ported import stage4_weed_ps; stage4_weed_ps(Path(sys.argv[1]) / 'PATCH_1', backend='native')";
        let output = Command::new("uv")
            .args(["run", "python", "-c", script])
            .arg(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python stage4 failed: {}",
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
