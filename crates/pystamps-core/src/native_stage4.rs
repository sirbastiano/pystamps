use crate::CoreError;
use delaunator::{triangulate, Point};
use num_complex::Complex64;
use pystamps_mat::{ComplexMatrixF32, MatData, MatFile, Matrix};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const HDF5_SIGNATURE_SCAN_BYTES: usize = 1024 * 1024;

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
            weed_standard_dev: 1.0,
            weed_max_noise: f64::INFINITY,
            weed_zero_elevation: "n".to_string(),
            weed_neighbours: "n".to_string(),
            weed_time_win: 730.0,
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
    let select1 = Stage4MatSource::read(patch_dir.join("select1.mat"));
    let ps1 = Stage4MatSource::read(patch_dir.join("ps1.mat"));
    let ph1 = Stage4MatSource::read(patch_dir.join("ph1.mat"));
    let parms = load_stage4_parms(patch_dir);

    let n_ps_total = ps1.scalar("n_ps", 0.0).round() as usize;
    if n_ps_total == 0 {
        return stage4_err("ps1.mat missing valid n_ps");
    }

    let ix = select1.vector_i64("ix", "select1.ix")?;
    if ix.is_empty() {
        return stage4_err("select1.mat has empty ix");
    }
    let keep_ix = select1.bool_vector_or_default("keep_ix", ix.len(), true);
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

    let coh_ps2_all = select1.ps_vector_f64("coh_ps2", ix.len(), "select1.coh_ps2")?;
    let k_ps2_all = select1.ps_vector_f64("K_ps2", ix.len(), "select1.K_ps2")?;
    let c_ps2_all = select1.ps_vector_f64("C_ps2", ix.len(), "select1.C_ps2")?;
    let coh_ps2 = select_values_by_mask(&coh_ps2_all, &keep_ix);
    let k_ps2 = select_values_by_mask(&k_ps2_all, &keep_ix);
    let c_ps2 = select_values_by_mask(&c_ps2_all, &keep_ix);
    let ix2_rows: Vec<usize> = ix2.iter().map(|&value| (value - 1) as usize).collect();

    let ij_all = ps1.ps_dim_f64("ij", n_ps_total, 3, "ps1.ij")?;
    let xy_all = ps1.ps_dim_f64("xy", n_ps_total, 3, "ps1.xy")?;
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
        let ph1 = ph1.ps_complex_matrix("ph", n_ps_total, "ph1.ph")?;
        let bperp = ps1.vector_f64_required("bperp", "ps1.bperp")?;
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
        let edges = stage4_graph_edges(patch_dir, &points)?;
        validate_stage4_edge_topology(&edges, n_pre_noise)?;
        ps_std = vec![f64::INFINITY; n_pre_noise];
        ps_max = vec![f64::INFINITY; n_pre_noise];

        if !edges.is_empty() && !ifg_cols.is_empty() {
            let small_baseline = parms.small_baseline_flag.eq_ignore_ascii_case("y");
            let master_ix = ps1.scalar("master_ix", 1.0).round() as usize;
            if !small_baseline && (master_ix == 0 || master_ix > ph1.cols) {
                return stage4_err(format!(
                    "ps1.master_ix must be 1-based within ph1 width {}; got {master_ix}",
                    ph1.cols
                ));
            }

            let mut ph_weed = Vec::with_capacity(n_pre_noise * ifg_cols.len());
            for &selected_pos in &kept_positions {
                let source_row = ix2_rows[selected_pos];
                let row_offset = source_row * ph1.cols;
                let topo_phase = k_ps2[selected_pos];
                let master_phase = c_ps2[selected_pos];
                for &col in &ifg_cols {
                    if !small_baseline && col == master_ix - 1 {
                        ph_weed.push(Complex64::from_polar(1.0, master_phase));
                        continue;
                    }
                    let source = ph1.values[row_offset + col];
                    let phase = mul_exp_neg_i(
                        Complex64::new(source.0 as f64, source.1 as f64),
                        topo_phase * bperp[col],
                    );
                    ph_weed.push(normalize_complex(phase));
                }
            }
            let b_use: Vec<f64> = ifg_cols.iter().map(|&col| bperp[col]).collect();
            let day_use = if small_baseline {
                Vec::new()
            } else {
                let day = ps1.vector_f64_required("day", "ps1.day")?;
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

#[derive(Debug)]
struct Stage4MatSource {
    path: PathBuf,
    mat: Option<MatData>,
}

impl Stage4MatSource {
    fn read(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let mat = if find_hdf5_signature_offset(&path).is_ok() {
            None
        } else {
            MatData::read(&path).ok()
        };
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

    fn vector_f64_required(&self, name: &str, label: &str) -> Result<Vec<f64>, CoreError> {
        self.vector_f64(name).ok_or_else(|| CoreError::NativeStage {
            stage: 4,
            message: format!("{label} is missing"),
        })
    }

    fn vector_i64(&self, name: &str, label: &str) -> Result<Vec<i64>, CoreError> {
        Ok(self
            .vector_f64_required(name, label)?
            .into_iter()
            .filter(|value| value.is_finite())
            .map(|value| value.round() as i64)
            .collect())
    }

    fn bool_vector_or_default(
        &self,
        name: &str,
        expected_len: usize,
        default_value: bool,
    ) -> Vec<bool> {
        let Some(values) = self.vector_f64(name) else {
            return vec![default_value; expected_len];
        };
        if values.len() != expected_len {
            return vec![default_value; expected_len];
        }
        values.into_iter().map(|value| value != 0.0).collect()
    }

    fn ps_vector_f64(&self, name: &str, n_ps: usize, label: &str) -> Result<Vec<f64>, CoreError> {
        let values = self.vector_f64_required(name, label)?;
        if values.len() != n_ps {
            return stage4_err(format!(
                "{label} has incompatible length {} for n_ps={n_ps}",
                values.len()
            ));
        }
        Ok(values)
    }

    fn ps_dim_f64(
        &self,
        name: &str,
        n_ps: usize,
        n_dim: usize,
        label: &str,
    ) -> Result<Matrix<f64>, CoreError> {
        if let Some(mat) = &self.mat {
            if let Ok(matrix) = ps_dim_f64(mat, name, n_ps, n_dim, label) {
                return Ok(matrix);
            }
        }
        if let Ok(matrix) = read_hdf5_matrix_f64(&self.path, name) {
            return orient_ps_dim_f64(matrix, n_ps, n_dim, label);
        }
        stage4_err(format!(
            "{label} is missing or has incompatible shape; expected {n_ps}x{n_dim}"
        ))
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
        stage4_err(format!("{label} is missing or invalid"))
    }
}

fn load_stage4_parms(patch_dir: &Path) -> Stage4Parms {
    let Some(path) = resolve_file_optional(patch_dir, "parms.mat") else {
        return Stage4Parms::default();
    };
    let source = Stage4MatSource::read(path);
    let small_baseline_flag = source.text("small_baseline_flag", "n");
    let default_standard_dev = if small_baseline_flag.eq_ignore_ascii_case("y") {
        f64::INFINITY
    } else {
        1.0
    };
    Stage4Parms {
        small_baseline_flag,
        drop_ifg_index: source
            .vector_f64("drop_ifg_index")
            .unwrap_or_default()
            .into_iter()
            .filter(|value| value.is_finite())
            .map(|value| value.round() as i64)
            .collect(),
        weed_standard_dev: source.scalar("weed_standard_dev", default_standard_dev),
        weed_max_noise: source.scalar("weed_max_noise", f64::INFINITY),
        weed_zero_elevation: source.text("weed_zero_elevation", "n"),
        weed_neighbours: source.text("weed_neighbours", "n"),
        weed_time_win: source.scalar("weed_time_win", 730.0),
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
    mat.add_u8_matrix(
        "ix_weed",
        ix_weed.len(),
        1,
        ix_weed.iter().map(|&keep| u8::from(keep)).collect(),
    )?;
    mat.add_u8_matrix(
        "ix_weed2",
        ix_weed2.len(),
        1,
        ix_weed2.iter().map(|&keep| u8::from(keep)).collect(),
    )?;
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

fn stage4_graph_edges(
    patch_dir: &Path,
    points: &[(f64, f64)],
) -> Result<Vec<(usize, usize)>, CoreError> {
    let n = points.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    if let Some(edges) = load_triangle_edges(patch_dir, n)? {
        return Ok(edges);
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

fn load_triangle_edges(
    patch_dir: &Path,
    n_nodes: usize,
) -> Result<Option<Vec<(usize, usize)>>, CoreError> {
    let edge_path = patch_dir.join("psweed.2.edge");
    if n_nodes < 2 || !edge_path.exists() {
        return Ok(None);
    }
    if let Some(node_count) = triangle_node_count(&patch_dir.join("psweed.1.node"))? {
        if node_count != n_nodes {
            return Ok(None);
        }
    }

    let text = fs::read_to_string(&edge_path).map_err(|err| {
        stage4_err_owned(format!(
            "unable to read Stage 4 triangle edge file {}: {err}",
            edge_path.display()
        ))
    })?;
    let mut edges = Vec::new();
    let mut seen = BTreeSet::new();
    for line in text.lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let Ok(a_1b) = cols[1].parse::<i64>() else {
            continue;
        };
        let Ok(b_1b) = cols[2].parse::<i64>() else {
            continue;
        };
        if a_1b < 1 || b_1b < 1 {
            continue;
        }
        let a = (a_1b - 1) as usize;
        let b = (b_1b - 1) as usize;
        if a >= n_nodes || b >= n_nodes || a == b {
            continue;
        }
        let edge = (a.min(b), a.max(b));
        if seen.insert(edge) {
            edges.push(edge);
        }
    }
    if edges.is_empty() {
        Ok(None)
    } else {
        Ok(Some(edges))
    }
}

fn triangle_node_count(path: &Path) -> Result<Option<usize>, CoreError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|err| {
        stage4_err_owned(format!(
            "unable to read Stage 4 triangle node file {}: {err}",
            path.display()
        ))
    })?;
    let Some(first_line) = text.lines().find(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    let Some(first_col) = first_line.split_whitespace().next() else {
        return Ok(None);
    };
    match first_col.parse::<usize>() {
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
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
        return stage4_err(format!(
            "stage4_edge_stats phase matrix has {} values for {n_node}x{n_ifg}",
            ph.len()
        ));
    }
    if bperp.len() != n_ifg {
        return stage4_err("stage4_edge_stats bperp vector must match phase width");
    }
    if !small_baseline && day.len() != n_ifg {
        return stage4_err(
            "stage4_edge_stats day vector must match phase width for non-small-baseline mode",
        );
    }
    let mut ps_std = vec![f64::INFINITY; n_node];
    let mut ps_max = vec![f64::INFINITY; n_node];
    let n_edge = edges.len();
    if n_edge == 0 || n_ifg == 0 {
        return Ok(EdgeStats { ps_std, ps_max });
    }

    let edge_stats = if !small_baseline {
        stage4_single_master_edge_stats(ph, n_ifg, edges, bperp, day, time_win)
    } else {
        stage4_small_baseline_edge_stats(ph, n_ifg, edges, bperp)
    };

    for (edge_ix, &(a, b)) in edges.iter().enumerate() {
        let (edge_std, edge_max) = edge_stats[edge_ix];
        ps_std[a] = ps_std[a].min(edge_std);
        ps_std[b] = ps_std[b].min(edge_std);
        ps_max[a] = ps_max[a].min(edge_max);
        ps_max[b] = ps_max[b].min(edge_max);
    }
    Ok(EdgeStats { ps_std, ps_max })
}

fn stage4_single_master_edge_stats(
    ph: &[Complex64],
    n_ifg: usize,
    edges: &[(usize, usize)],
    bperp: &[f64],
    day: &[f64],
    time_win: f64,
) -> Vec<(f64, f64)> {
    let n_edge = edges.len();
    let time_win = time_win.max(1.0e-6);
    let mut time_diff_all = vec![0.0; n_ifg * n_ifg];
    let mut weight_all = vec![0.0; n_ifg * n_ifg];
    let mut affine = Vec::with_capacity(n_ifg);
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
        let time_diff = &time_diff_all[row * n_ifg..(row + 1) * n_ifg];
        let weight = &weight_all[row * n_ifg..(row + 1) * n_ifg];
        affine.push(weighted_affine_coefficients(time_diff, weight));
    }

    let mut dph_noise = vec![0.0; n_edge * n_ifg];
    let chunk_edges = 512usize;
    let (noise2_sum, noise2_sumsq) = dph_noise
        .par_chunks_mut(n_ifg * chunk_edges)
        .zip(edges.par_chunks(chunk_edges))
        .map(|(noise_chunk, edge_chunk)| {
            let mut dph_re = vec![0.0; n_ifg];
            let mut dph_im = vec![0.0; n_ifg];
            let mut dph_phase = vec![0.0; n_ifg];
            let mut dph_mean_adj = vec![0.0; n_ifg];
            let mut noise2_row = vec![0.0; n_ifg];
            let mut noise2_sum = vec![0.0; n_ifg];
            let mut noise2_sumsq = vec![0.0; n_ifg];
            for (local_ix, &edge) in edge_chunk.iter().enumerate() {
                fill_edge_phase_components(
                    ph,
                    n_ifg,
                    edge,
                    &mut dph_re,
                    &mut dph_im,
                    &mut dph_phase,
                );
                let noise_row = &mut noise_chunk[local_ix * n_ifg..(local_ix + 1) * n_ifg];
                single_master_edge_noise(
                    &dph_re,
                    &dph_im,
                    &dph_phase,
                    n_ifg,
                    &time_diff_all,
                    &weight_all,
                    &affine,
                    &mut dph_mean_adj,
                    noise_row,
                    &mut noise2_row,
                );
                for ifg in 0..n_ifg {
                    noise2_sum[ifg] += noise2_row[ifg];
                    noise2_sumsq[ifg] += noise2_row[ifg] * noise2_row[ifg];
                }
            }
            (noise2_sum, noise2_sumsq)
        })
        .reduce(
            || (vec![0.0; n_ifg], vec![0.0; n_ifg]),
            |mut left, right| {
                for ifg in 0..n_ifg {
                    left.0[ifg] += right.0[ifg];
                    left.1[ifg] += right.1[ifg];
                }
                left
            },
        );

    let ifg_var =
        variance_from_sum_sumsq(&noise2_sum, &noise2_sumsq, n_edge, usize::from(n_edge > 1));
    let w_ifg: Vec<f64> = ifg_var
        .iter()
        .map(|&value| {
            if value == 0.0 {
                f64::INFINITY
            } else {
                1.0 / value
            }
        })
        .collect();
    corrected_edge_stats_rows_real(bperp, &dph_noise, n_ifg, &w_ifg, usize::from(n_ifg > 1))
}

fn stage4_small_baseline_edge_stats(
    ph: &[Complex64],
    n_ifg: usize,
    edges: &[(usize, usize)],
    bperp: &[f64],
) -> Vec<(f64, f64)> {
    let n_edge = edges.len();
    let ifg_var = variance_cols_complex_edges(ph, n_ifg, edges, usize::from(n_edge > 1));
    let w_ifg: Vec<f64> = ifg_var
        .iter()
        .map(|&value| {
            if value == 0.0 {
                f64::INFINITY
            } else {
                1.0 / value
            }
        })
        .collect();
    edges
        .par_iter()
        .map(|&edge| {
            let dph = edge_phase_row(ph, n_ifg, edge);
            let k_edge = weighted_slope_fit_complex(bperp, &dph, &w_ifg);
            let mut ang = vec![0.0; n_ifg];
            for ifg in 0..n_ifg {
                ang[ifg] = (dph[ifg] - k_edge * bperp[ifg]).arg();
            }
            std_max_row_real(&ang, usize::from(n_ifg > 1))
        })
        .collect()
}

fn edge_phase_row(ph: &[Complex64], n_ifg: usize, edge: (usize, usize)) -> Vec<Complex64> {
    let (a, b) = edge;
    let mut dph = Vec::with_capacity(n_ifg);
    for ifg_ix in 0..n_ifg {
        dph.push(ph[b * n_ifg + ifg_ix] * ph[a * n_ifg + ifg_ix].conj());
    }
    dph
}

fn fill_edge_phase_components(
    ph: &[Complex64],
    n_ifg: usize,
    edge: (usize, usize),
    re_out: &mut [f64],
    im_out: &mut [f64],
    phase_out: &mut [f64],
) {
    let (a, b) = edge;
    for ifg_ix in 0..n_ifg {
        let value = ph[b * n_ifg + ifg_ix] * ph[a * n_ifg + ifg_ix].conj();
        re_out[ifg_ix] = value.re;
        im_out[ifg_ix] = value.im;
        phase_out[ifg_ix] = value.arg();
    }
}

#[derive(Clone, Copy, Debug)]
struct AffineCoefficients {
    s0: f64,
    s1: f64,
    s2: f64,
    det: f64,
}

fn weighted_affine_coefficients(time_diff: &[f64], weight: &[f64]) -> AffineCoefficients {
    let s0: f64 = weight.iter().sum();
    let s1: f64 = weight
        .iter()
        .zip(time_diff.iter())
        .map(|(&wi, &ti)| wi * ti)
        .sum();
    let s2: f64 = weight
        .iter()
        .zip(time_diff.iter())
        .map(|(&wi, &ti)| wi * ti * ti)
        .sum();
    AffineCoefficients {
        s0,
        s1,
        s2,
        det: s0 * s2 - s1 * s1,
    }
}

fn weighted_affine_fit_from_sums(wy0: f64, wy1: f64, coeffs: AffineCoefficients) -> (f64, f64) {
    if coeffs.s0 == 0.0 {
        return (0.0, 0.0);
    }
    if coeffs.det == 0.0 {
        return (wy0 / coeffs.s0, 0.0);
    }
    (
        (wy0 * coeffs.s2 - wy1 * coeffs.s1) / coeffs.det,
        (wy1 * coeffs.s0 - wy0 * coeffs.s1) / coeffs.det,
    )
}

fn single_master_edge_noise(
    dph_re: &[f64],
    dph_im: &[f64],
    dph_phase: &[f64],
    n_ifg: usize,
    time_diff_all: &[f64],
    weight_all: &[f64],
    affine: &[AffineCoefficients],
    dph_mean_adj: &mut [f64],
    noise_row: &mut [f64],
    noise2_row: &mut [f64],
) {
    for ifg in 0..n_ifg {
        let weight = &weight_all[ifg * n_ifg..(ifg + 1) * n_ifg];
        let time_diff = &time_diff_all[ifg * n_ifg..(ifg + 1) * n_ifg];
        let mut mean_re = 0.0;
        let mut mean_im = 0.0;
        for col in 0..n_ifg {
            mean_re += dph_re[col] * weight[col];
            mean_im += dph_im[col] * weight[col];
        }
        let mean_angle = if mean_re * mean_re + mean_im * mean_im == 0.0 {
            None
        } else {
            Some(mean_im.atan2(mean_re))
        };
        let mut fit_wy0 = 0.0;
        let mut fit_wy1 = 0.0;
        if let Some(angle) = mean_angle {
            for col in 0..n_ifg {
                let adjusted = wrap_phase_bounded(dph_phase[col] - angle);
                dph_mean_adj[col] = adjusted;
                fit_wy0 += adjusted * weight[col];
                fit_wy1 += adjusted * weight[col] * time_diff[col];
            }
        } else {
            dph_mean_adj.fill(0.0);
        }
        let (m0, m1) = weighted_affine_fit_from_sums(fit_wy0, fit_wy1, affine[ifg]);
        let mut wy0 = 0.0;
        let mut wy1 = 0.0;
        for col in 0..n_ifg {
            let detrended = wrap_phase_if_needed(dph_mean_adj[col] - (m0 + m1 * time_diff[col]));
            wy0 += detrended * weight[col];
            wy1 += detrended * weight[col] * time_diff[col];
        }
        let m20 = if affine[ifg].det == 0.0 {
            if affine[ifg].s0 == 0.0 {
                0.0
            } else {
                wy0 / affine[ifg].s0
            }
        } else {
            (wy0 * affine[ifg].s2 - wy1 * affine[ifg].s1) / affine[ifg].det
        };
        let smooth2_re = mean_re - dph_re[ifg] * weight[ifg];
        let smooth2_im = mean_im - dph_im[ifg] * weight[ifg];
        noise_row[ifg] = mean_angle
            .map(|angle| wrap_phase_if_needed(dph_phase[ifg] - (angle + m0 + m20)))
            .unwrap_or(0.0);
        noise2_row[ifg] = if smooth2_re * smooth2_re + smooth2_im * smooth2_im == 0.0 {
            0.0
        } else {
            wrap_phase_if_needed(dph_phase[ifg] - smooth2_im.atan2(smooth2_re))
        };
    }
}

fn variance_cols_complex_edges(
    ph: &[Complex64],
    n_ifg: usize,
    edges: &[(usize, usize)],
    ddof: usize,
) -> Vec<f64> {
    let n_edge = edges.len();
    if n_edge == 0 || n_ifg == 0 {
        return vec![0.0; n_ifg];
    }
    let denom = n_edge.saturating_sub(ddof);
    if denom == 0 {
        return vec![0.0; n_ifg];
    }
    let (sum, sum_norm) = edges
        .par_iter()
        .map(|&edge| {
            let dph = edge_phase_row(ph, n_ifg, edge);
            let mut sum = vec![Complex64::new(0.0, 0.0); n_ifg];
            let mut sum_norm = vec![0.0; n_ifg];
            for ifg in 0..n_ifg {
                sum[ifg] = dph[ifg];
                sum_norm[ifg] = dph[ifg].norm_sqr();
            }
            (sum, sum_norm)
        })
        .reduce(
            || (vec![Complex64::new(0.0, 0.0); n_ifg], vec![0.0; n_ifg]),
            |mut left, right| {
                for ifg in 0..n_ifg {
                    left.0[ifg] += right.0[ifg];
                    left.1[ifg] += right.1[ifg];
                }
                left
            },
        );
    let mut out = vec![0.0; n_ifg];
    for ifg in 0..n_ifg {
        let mean = sum[ifg] / n_edge as f64;
        out[ifg] = (sum_norm[ifg] - n_edge as f64 * mean.norm_sqr()) / denom as f64;
        if out[ifg] < 0.0 && out[ifg] > -1.0e-12 {
            out[ifg] = 0.0;
        }
    }
    out
}

fn weighted_slope_fit_complex(x: &[f64], y: &[Complex64], w: &[f64]) -> Complex64 {
    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| value.is_infinite().then_some(idx))
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        return inf_idx
            .iter()
            .map(|&col| y[col] * x[col])
            .sum::<Complex64>()
            / den;
    }
    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value.is_finite() && value > 0.0).then_some(idx))
        .collect();
    if pos_idx.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    pos_idx
        .iter()
        .map(|&col| y[col] * (w[col] * x[col]))
        .sum::<Complex64>()
        / den
}

fn std_max_row_real(values: &[f64], ddof: usize) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n_col = values.len();
    let mean = values.iter().sum::<f64>() / n_col as f64;
    let mut accum = 0.0;
    let mut max_value = 0.0_f64;
    for &value in values {
        accum += (value - mean) * (value - mean);
        max_value = max_value.max(value.abs());
    }
    let denom = n_col.saturating_sub(ddof);
    let std = if denom == 0 {
        0.0
    } else {
        (accum / denom as f64).sqrt()
    };
    (std, max_value)
}

fn corrected_edge_stats_rows_real(
    x: &[f64],
    data: &[f64],
    n_col: usize,
    w: &[f64],
    ddof: usize,
) -> Vec<(f64, f64)> {
    if n_col == 0 {
        return vec![(0.0, 0.0); data.len()];
    }
    let inf_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| value.is_infinite().then_some(idx))
        .collect();
    if !inf_idx.is_empty() {
        let den: f64 = inf_idx.iter().map(|&idx| x[idx] * x[idx]).sum();
        if den == 0.0 {
            return data
                .par_chunks(n_col)
                .map(|row| std_max_row_real(row, ddof))
                .collect();
        }
        return data
            .par_chunks(n_col)
            .map(|row| {
                let mut numerator = 0.0;
                for &col in &inf_idx {
                    numerator += row[col] * x[col];
                }
                let k = numerator / den;
                std_max_corrected_row_real(row, x, k, ddof)
            })
            .collect();
    }
    let pos_idx: Vec<usize> = w
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value.is_finite() && value > 0.0).then_some(idx))
        .collect();
    if pos_idx.is_empty() {
        return data
            .par_chunks(n_col)
            .map(|row| std_max_row_real(row, ddof))
            .collect();
    }
    let den: f64 = pos_idx.iter().map(|&idx| w[idx] * x[idx] * x[idx]).sum();
    if den == 0.0 {
        return data
            .par_chunks(n_col)
            .map(|row| std_max_row_real(row, ddof))
            .collect();
    }
    data.par_chunks(n_col)
        .map(|row| {
            let mut numerator = 0.0;
            for &col in &pos_idx {
                numerator += row[col] * w[col] * x[col];
            }
            let k = numerator / den;
            std_max_corrected_row_real(row, x, k, ddof)
        })
        .collect()
}

fn std_max_corrected_row_real(values: &[f64], x: &[f64], k: f64, ddof: usize) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n_col = values.len();
    let mut sum = 0.0;
    for col in 0..n_col {
        sum += values[col] - k * x[col];
    }
    let mean = sum / n_col as f64;
    let mut accum = 0.0;
    let mut max_value = 0.0_f64;
    for col in 0..n_col {
        let corrected = values[col] - k * x[col];
        accum += (corrected - mean) * (corrected - mean);
        max_value = max_value.max(corrected.abs());
    }
    let denom = n_col.saturating_sub(ddof);
    let std = if denom == 0 {
        0.0
    } else {
        (accum / denom as f64).sqrt()
    };
    (std, max_value)
}

fn validate_stage4_edge_topology(
    edges: &[(usize, usize)],
    n_nodes: usize,
) -> Result<(), CoreError> {
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

fn variance_from_sum_sumsq(sum: &[f64], sumsq: &[f64], n_row: usize, ddof: usize) -> Vec<f64> {
    if n_row == 0 {
        return vec![0.0; sum.len()];
    }
    let denom = n_row.saturating_sub(ddof);
    if denom == 0 {
        return vec![0.0; sum.len()];
    }
    sum.iter()
        .zip(sumsq.iter())
        .map(|(&sum_value, &sumsq_value)| {
            let mean = sum_value / n_row as f64;
            let mut variance = (sumsq_value - n_row as f64 * mean * mean) / denom as f64;
            if variance < 0.0 && variance > -1.0e-12 {
                variance = 0.0;
            }
            variance
        })
        .collect()
}

fn wrap_phase(value: f64) -> f64 {
    let wrapped = (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    if wrapped == -std::f64::consts::PI && value > 0.0 {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

fn wrap_phase_if_needed(value: f64) -> f64 {
    if (-std::f64::consts::PI..=std::f64::consts::PI).contains(&value) {
        value
    } else {
        wrap_phase(value)
    }
}

fn wrap_phase_bounded(value: f64) -> f64 {
    let mut wrapped = value;
    if wrapped < -std::f64::consts::PI {
        wrapped += 2.0 * std::f64::consts::PI;
    } else if wrapped > std::f64::consts::PI {
        wrapped -= 2.0 * std::f64::consts::PI;
    }
    if wrapped == -std::f64::consts::PI && value > 0.0 {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

fn mul_exp_neg_i(value: Complex64, theta: f64) -> Complex64 {
    let (sin, cos) = theta.sin_cos();
    Complex64::new(
        value.re * cos + value.im * sin,
        value.im * cos - value.re * sin,
    )
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
    let hgt = Stage4MatSource::read(path);
    let values = hgt
        .vector_f64("hgt")
        .ok_or_else(|| CoreError::NativeStage {
            stage: 4,
            message: "hgt1.mat missing hgt".to_string(),
        })?;
    if values.len() != n_ps {
        return stage4_err(format!(
            "hgt1.hgt has incompatible length {} for n_ps={n_ps}",
            values.len()
        ));
    }
    Ok(Some(values))
}

fn ifg_index_for_weed(ps: &Stage4MatSource, parms: &Stage4Parms) -> Vec<f64> {
    let n_ifg = ps.scalar("n_ifg", 0.0).round() as i64;
    let drop: BTreeSet<i64> = parms.drop_ifg_index.iter().copied().collect();
    (1..=n_ifg)
        .filter(|value| !drop.contains(value))
        .map(|value| value as f64)
        .collect()
}

fn optional_vector_f64(mat: &MatData, name: &str) -> Option<Vec<f64>> {
    mat.get_f64_matrix(name).ok().map(|matrix| matrix.values)
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
            stage: 4,
            message: format!("{label} is missing or invalid: {err}"),
        })?;
    orient_ps_dim_f64(source, n_ps, n_dim, label)
}

fn orient_ps_dim_f64(
    source: Matrix<f64>,
    n_ps: usize,
    n_dim: usize,
    label: &str,
) -> Result<Matrix<f64>, CoreError> {
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

fn ps_complex_matrix(
    mat: &MatData,
    name: &str,
    n_ps: usize,
    label: &str,
) -> Result<ComplexMatrixF32, CoreError> {
    let source = mat
        .get_complex_f32_matrix(name)
        .map_err(|err| CoreError::NativeStage {
            stage: 4,
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

fn text_from_mat_opt(mat: &MatData, name: &str) -> Option<String> {
    let Some(values) = optional_vector_f64(mat, name) else {
        return None;
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
    let values = read_hdf5_numeric_f64(&dataset)?;
    let (rows, cols) = hdf5_matrix_shape(&dataset);
    Ok(Matrix {
        name: variable.to_string(),
        rows,
        cols,
        values,
    })
}

fn read_hdf5_numeric_f64(dataset: &rust_hdf5::H5Dataset) -> Result<Vec<f64>, String> {
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values);
    }
    if let Ok(values) = dataset.read_raw::<f32>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<u8>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<i8>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<u16>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<i16>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<u32>() {
        return Ok(values.into_iter().map(|value| value as f64).collect());
    }
    if let Ok(values) = dataset.read_raw::<i32>() {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    if let Ok(values) = dataset.read_raw::<u64>() {
        return Ok(values.into_iter().map(|value| value as f64).collect());
    }
    if let Ok(values) = dataset.read_raw::<i64>() {
        return Ok(values.into_iter().map(|value| value as f64).collect());
    }
    Err("unsupported HDF5 numeric dataset type".to_string())
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
        "pystamps-stage4-hdf5-{}-{}.h5",
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

    #[test]
    fn duplicate_coordinates_keep_highest_coherence_and_preserve_shapes() {
        let root = temp_root("stage4-duplicates");
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        write_parms_no_noise(&patch, "n");
        write_ps1_custom(
            &patch,
            &[
                (1.0, 10.0, 10.0, 0.0, 0.0),
                (2.0, 10.0, 11.0, 0.0, 0.0),
                (3.0, 10.0, 12.0, 1.0, 0.0),
                (4.0, 10.0, 13.0, 2.0, 0.0),
                (5.0, 10.0, 14.0, 3.0, 0.0),
            ],
        );
        write_ph1_custom(&patch, 5);
        write_select1_custom(
            &patch,
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.2, 0.9, 0.4, 0.5, 0.6],
        );

        run_stage4_native(&patch).unwrap();

        let weed = MatData::read(patch.join("weed1.mat")).unwrap();
        let ix_weed = weed.get_f32_matrix("ix_weed").unwrap().values;
        let ix_weed2 = weed.get_f32_matrix("ix_weed2").unwrap().values;
        assert_eq!(ix_weed, vec![0.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(ix_weed2, vec![1.0, 1.0, 1.0, 1.0]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn boundary_neighbor_weeding_keeps_valid_boundary_pixel() {
        let keep = adjacent_component_keep_mask(&[(1, 1), (2, 2), (5, 5)], &[0.9, 0.2, 0.8]);
        assert_eq!(keep, vec![true, false, true]);
    }

    #[test]
    fn one_based_indices_preserve_first_and_last_rows() {
        assert!(validate_one_based_indices(&[1, 5], 5, "test").is_ok());
        assert!(validate_one_based_indices(&[0], 5, "test").is_err());
        assert!(validate_one_based_indices(&[6], 5, "test").is_err());
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
        mat.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_u32_matrix("weed_neighbours", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_u32_matrix("weed_zero_elevation", 1, 1, vec!['y' as u32])
            .unwrap();
        mat.add_f64_scalar("weed_standard_dev", 1.0).unwrap();
        mat.add_f64_scalar("weed_max_noise", 1.0).unwrap();
        mat.add_f64_scalar("weed_time_win", 360.0).unwrap();
        mat.write().unwrap();
    }

    fn write_ps1(patch: &Path) {
        let ij = vec![1.0, 10.0, 10.0, 2.0, 10.0, 20.0, 3.0, 20.0, 10.0];
        let xy = vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0];
        let mut mat = MatFile::new(patch.join("ps1.mat"));
        mat.add_f64_scalar("n_ps", 3.0).unwrap();
        mat.add_f64_scalar("n_ifg", 4.0).unwrap();
        mat.add_f64_scalar("master_ix", 1.0).unwrap();
        mat.add_f64_row_vector("bperp", vec![0.0, 10.0, 20.0, 30.0])
            .unwrap();
        mat.add_f64_row_vector("day", vec![20200101.0, 20200113.0, 20200125.0, 20200206.0])
            .unwrap();
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
        mat.add_f64_row_vector("coh_ps", vec![0.8, 0.7, 0.6])
            .unwrap();
        mat.write().unwrap();
    }

    fn write_select1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("select1.mat"));
        mat.add_f64_col_vector("ix", vec![1.0, 2.0, 3.0]).unwrap();
        mat.add_u8_matrix("keep_ix", 3, 1, vec![1, 1, 1]).unwrap();
        mat.add_f64_col_vector("K_ps2", vec![0.0, 0.0, 0.0])
            .unwrap();
        mat.add_f64_col_vector("C_ps2", vec![0.0, 0.0, 0.0])
            .unwrap();
        mat.add_f64_col_vector("coh_ps2", vec![0.8, 0.7, 0.6])
            .unwrap();
        mat.write().unwrap();
    }

    fn write_parms_no_noise(patch: &Path, weed_neighbours: &str) {
        let mut mat = MatFile::new(patch.join("parms.mat"));
        mat.add_u32_matrix("small_baseline_flag", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_u32_matrix(
            "weed_neighbours",
            1,
            weed_neighbours.len(),
            weed_neighbours.chars().map(|ch| ch as u32).collect(),
        )
        .unwrap();
        mat.add_u32_matrix("weed_zero_elevation", 1, 1, vec!['n' as u32])
            .unwrap();
        mat.add_f64_scalar("weed_standard_dev", std::f64::consts::PI)
            .unwrap();
        mat.add_f64_scalar("weed_max_noise", std::f64::consts::PI)
            .unwrap();
        mat.add_f64_scalar("weed_time_win", 360.0).unwrap();
        mat.write().unwrap();
    }

    fn write_ps1_custom(patch: &Path, rows: &[(f64, f64, f64, f64, f64)]) {
        let mut ij = Vec::with_capacity(rows.len() * 3);
        let mut xy = Vec::with_capacity(rows.len() * 3);
        for &(id, ij_r, ij_c, x, y) in rows {
            ij.extend_from_slice(&[id, ij_r, ij_c]);
            xy.extend_from_slice(&[id, x, y]);
        }
        let mut mat = MatFile::new(patch.join("ps1.mat"));
        mat.add_f64_scalar("n_ps", rows.len() as f64).unwrap();
        mat.add_f64_scalar("n_ifg", 4.0).unwrap();
        mat.add_f64_scalar("master_ix", 1.0).unwrap();
        mat.add_f64_row_vector("bperp", vec![0.0, 10.0, 20.0, 30.0])
            .unwrap();
        mat.add_f64_row_vector("day", vec![20200101.0, 20200113.0, 20200125.0, 20200206.0])
            .unwrap();
        mat.add_f64_matrix("ij", rows.len(), 3, ij).unwrap();
        mat.add_f64_matrix("xy", rows.len(), 3, xy).unwrap();
        mat.write().unwrap();
    }

    fn write_ph1_custom(patch: &Path, n_ps: usize) {
        let mut values = Vec::new();
        for _row in 0..n_ps {
            for _col in 0..4 {
                values.push((1.0_f32, 0.0_f32));
            }
        }
        let mut mat = MatFile::new(patch.join("ph1.mat"));
        mat.add_complex_f32_matrix("ph", n_ps, 4, values).unwrap();
        mat.write().unwrap();
    }

    fn write_select1_custom(patch: &Path, ix: &[f64], coh: &[f64]) {
        let mut mat = MatFile::new(patch.join("select1.mat"));
        mat.add_f64_col_vector("ix", ix.to_vec()).unwrap();
        mat.add_u8_matrix("keep_ix", ix.len(), 1, vec![1; ix.len()])
            .unwrap();
        mat.add_f64_col_vector("K_ps2", vec![0.0; ix.len()])
            .unwrap();
        mat.add_f64_col_vector("C_ps2", vec![0.0; ix.len()])
            .unwrap();
        mat.add_f64_col_vector("coh_ps2", coh.to_vec()).unwrap();
        mat.write().unwrap();
    }

    fn write_hgt1(patch: &Path) {
        let mut mat = MatFile::new(patch.join("hgt1.mat"));
        mat.add_f32_col_vector("hgt", vec![10.0, 11.0, 12.0])
            .unwrap();
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
