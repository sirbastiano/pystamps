import json
from pathlib import Path

import numpy as np
import pytest
import pystamps.pipeline.ported as ported

from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline.ported import (
    _compute_active_single_master_uw_space_time,
    _extract_grid_values_for_ps,
    _maybe_resolve_external_tool,
    _single_master_insert_master_ix,
    _write_binary_matrix,
    _write_complex_raster,
)


RUN_FULL_GATE = Path("inputs_and_outputs/RUN_FULL_GATE_1e10")


pytestmark = [
    pytest.mark.skipif(
        not RUN_FULL_GATE.exists(),
        reason="requires local parity run copy under inputs_and_outputs/RUN_FULL_GATE_1e10",
    ),
]


def test_write_complex_raster_matches_matlab_fwrite_layout(tmp_path: Path) -> None:
    values = np.asarray(
        [
            [1 + 10j, 2 + 20j],
            [3 + 30j, 4 + 40j],
        ],
        dtype=np.complex64,
    )

    out = tmp_path / "snaphu.in"
    _write_complex_raster(out, values)

    raw = np.fromfile(out, dtype=np.float32)
    expected = np.asarray([1, 10, 2, 20, 3, 30, 4, 40], dtype=np.float32)
    assert np.array_equal(raw, expected)


def test_write_binary_matrix_matches_matlab_fwrite_transpose_layout(tmp_path: Path) -> None:
    values = np.asarray(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int16,
    )

    out = tmp_path / "snaphu.costinfile"
    _write_binary_matrix(out, values)

    raw = np.fromfile(out, dtype=np.int16)
    expected = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int16)
    assert np.array_equal(raw, expected)


def test_maybe_resolve_external_tool_prefers_local_build_deps_bin(tmp_path: Path, monkeypatch) -> None:
    tool = tmp_path / ".build-deps" / "bin" / "triangle"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    resolved = _maybe_resolve_external_tool("triangle")

    assert resolved == str(tool.resolve())


def test_compute_active_single_master_uses_smoother_noise(monkeypatch) -> None:
    expected_smooth = np.asarray([[1.0, 1.5]], dtype=np.float32)
    expected_noise = np.asarray([[0.25, -0.25]], dtype=np.float32)

    def fake_estimate(*args, **kwargs):
        return np.zeros((1,), dtype=np.float32)

    def fake_smooth(*args, **kwargs):
        return expected_smooth, expected_noise

    monkeypatch.setattr(ported, "_estimate_la_error_single_master", fake_estimate)
    monkeypatch.setattr(ported, "_smooth_3d_full_single_master", fake_smooth)

    uw_ph = np.asarray([[1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j]], dtype=np.complex64)
    edgs = np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64)

    _G, _dph_space, dph_smooth_ifg, dph_noise, dph_space_uw = _compute_active_single_master_uw_space_time(
        uw_ph,
        edgs,
        day=np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        master_ix=1,
        bperp=np.asarray([100.0, 200.0], dtype=np.float64),
        unwrap_ifg=np.asarray([2, 3], dtype=np.int64),
        time_win=36.0,
        n_trial_wraps=1.0,
    )

    assert np.array_equal(dph_smooth_ifg, expected_smooth)
    assert np.array_equal(dph_noise, expected_noise)
    assert np.allclose(dph_space_uw, expected_smooth + expected_noise)


def test_compute_active_single_master_normalizes_arc_phasors(monkeypatch) -> None:
    captured = {}

    def fake_estimate(dph_space, **kwargs):
        captured["abs"] = np.abs(dph_space)
        return np.zeros((1,), dtype=np.float32)

    def fake_smooth(dph_space, **kwargs):
        captured["smooth_abs"] = np.abs(dph_space)
        return np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)

    monkeypatch.setattr(ported, "_estimate_la_error_single_master", fake_estimate)
    monkeypatch.setattr(ported, "_smooth_3d_full_single_master", fake_smooth)

    uw_ph = np.asarray(
        [
            [2.0 + 0.0j, 3.0 + 0.0j],
            [0.0 + 4.0j, 0.0 + 5.0j],
        ],
        dtype=np.complex64,
    )
    edgs = np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64)

    ported._compute_active_single_master_uw_space_time(
        uw_ph,
        edgs,
        day=np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
        master_ix=1,
        bperp=np.asarray([100.0, 200.0], dtype=np.float64),
        unwrap_ifg=np.asarray([2, 3], dtype=np.int64),
        time_win=36.0,
        n_trial_wraps=1.0,
    )

    assert np.allclose(captured["abs"], 1.0)
    assert np.allclose(captured["smooth_abs"], 1.0)


def test_compute_active_single_master_chunked_matches_default() -> None:
    phase = np.asarray(
        [
            [0.10, -0.20, 0.30, -0.40],
            [0.25, -0.15, 0.05, 0.45],
            [-0.35, 0.40, -0.10, 0.15],
            [0.50, -0.30, 0.20, -0.05],
            [-0.45, 0.10, 0.35, -0.25],
        ],
        dtype=np.float32,
    )
    uw_ph = np.exp(1j * phase).astype(np.complex64)
    edgs = np.asarray(
        [
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 3.0],
            [3.0, 3.0, 4.0],
            [4.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )
    kwargs = dict(
        day=np.asarray([-10.0, 0.0, 15.0, 30.0, 45.0], dtype=np.float64),
        master_ix=2,
        bperp=np.asarray([80.0, 120.0, 160.0, 200.0], dtype=np.float64),
        unwrap_ifg=np.asarray([1, 3, 4, 5], dtype=np.int64),
        time_win=36.0,
        n_trial_wraps=2.0,
    )

    out_default = _compute_active_single_master_uw_space_time(uw_ph, edgs, **kwargs)
    out_chunked = _compute_active_single_master_uw_space_time(uw_ph, edgs, chunk_edges=2, **kwargs)

    assert np.array_equal(np.asarray(out_default[0]), np.asarray(out_chunked[0]))
    for lhs, rhs in zip(out_default[1:], out_chunked[1:], strict=True):
        assert np.allclose(lhs, rhs, equal_nan=True)


def test_extract_grid_values_for_ps_uses_matlab_column_major_order() -> None:
    ifguw = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    nzix = np.asarray([[True, False, True], [False, True, False]], dtype=bool)

    out = _extract_grid_values_for_ps(ifguw, nzix)

    np.testing.assert_array_equal(out, np.asarray([1.0, 5.0, 3.0], dtype=np.float32))


def test_compute_active_single_master_masks_noise_above_legacy_cutoff(monkeypatch) -> None:
    def fake_estimate(*args, **kwargs):
        return np.zeros((1,), dtype=np.float32)

    def fake_smooth(*args, **kwargs):
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([[0.0, 0.0, 2.2]], dtype=np.float32),
        )

    monkeypatch.setattr(ported, "_estimate_la_error_single_master", fake_estimate)
    monkeypatch.setattr(ported, "_smooth_3d_full_single_master", fake_smooth)

    uw_ph = np.asarray([[1 + 0j, 1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j, 1 + 0j]], dtype=np.complex64)
    edgs = np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64)

    _G, _dph_space, _dph_smooth_ifg, dph_noise, dph_space_uw = _compute_active_single_master_uw_space_time(
        uw_ph,
        edgs,
        day=np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
        master_ix=1,
        bperp=np.asarray([100.0, 200.0, 300.0], dtype=np.float64),
        unwrap_ifg=np.asarray([2, 3, 4], dtype=np.int64),
        time_win=36.0,
        n_trial_wraps=1.0,
    )

    assert np.isnan(dph_noise).all()
    assert np.isnan(dph_space_uw).all()


def test_stage6_unwrap_restores_scla_terms_in_phuw2(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    unwrap_cols = np.asarray([0, 2], dtype=np.int64)
    bp_nm = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    bperp_full = np.concatenate(
        [bp_nm[:, : master_ix - 1], np.zeros((n_ps, 1), dtype=np.float32), bp_nm[:, master_ix - 1 :]],
        axis=1,
    )
    k_ps_uw = np.asarray([0.1, 0.2], dtype=np.float32)
    c_ps_uw = np.asarray([0.5, -0.25], dtype=np.float32)
    ph_ramp = np.asarray([[0.05, 0.0, 0.15], [0.2, 0.0, -0.1]], dtype=np.float32)
    restore = (k_ps_uw[:, None] * bperp_full + c_ps_uw[:, None] + ph_ramp).astype(np.float32)
    unwrapped_without_restore = np.asarray([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]], dtype=np.float32)
    ph_rc = np.exp(1j * (unwrapped_without_restore + restore)).astype(np.complex64)

    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 41.0, 0.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(dataset_root / "ph2.mat", {"ph": np.ones((n_ps, n_ifg), dtype=np.complex64)})
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": np.ones((n_ps, n_ifg - 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(dataset_root / "rc2.mat", {"ph_rc": ph_rc})
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": bp_nm})
    write_mat(
        dataset_root / "scla_smooth2.mat",
        {
            "K_ps_uw": np.asarray(k_ps_uw[:, None], dtype=np.float32),
            "C_ps_uw": np.asarray(c_ps_uw[:, None], dtype=np.float32),
            "ph_ramp": ph_ramp,
        },
    )
    write_mat(
        dataset_root / "uw_grid.mat",
        {
            "ph": np.ones((2, 2), dtype=np.complex64),
            "nzix": np.asarray([[True, True], [False, False]], dtype=bool),
            "grid_ij": np.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            "n_ps": np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "uw_interp.mat",
        {
            "edgs": np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "Z": np.asarray([[1, 2], [1, 2]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )

    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(ported, "_run_external_command", lambda *args, **kwargs: None)

    grids = [
        np.asarray([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [11.0, 12.0]], dtype=np.float32),
    ]

    def fake_load_float_grid(path: Path, ncol: int) -> np.ndarray:
        return grids.pop(0)

    monkeypatch.setattr(ported, "_load_float_grid", fake_load_float_grid)
    debug_path = dataset_root / "stage6_debug.json"
    monkeypatch.setenv("PYSTAMPS_STAGE6_DEBUG_JSON", str(debug_path))

    ported.stage6_unwrap(dataset_root, backend="python", enable_mat_cache=False, snaphu_path="/bin/true")

    uw_phaseuw = read_mat(dataset_root / "uw_phaseuw.mat")
    np.testing.assert_allclose(
        np.asarray(uw_phaseuw["ph_uw"], dtype=np.float32),
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
    )

    phuw2 = read_mat(dataset_root / "phuw2.mat")
    expected = np.zeros((n_ps, n_ifg), dtype=np.float32)
    expected[:, unwrap_cols] = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32) + restore[:, unwrap_cols]
    np.testing.assert_allclose(np.asarray(phuw2["ph_uw"], dtype=np.float32), expected)
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug_payload["status"] == "completed"
    assert debug_payload["phase"] == "completed"
    assert debug_payload["unwrap_ifg_total"] == 2
    assert debug_payload["ifg_completed"] == 2
    assert debug_payload["timings_sec"]["snaphu_loop"] >= 0.0
    assert debug_payload["timings_sec"]["uw_space_time"] >= 0.0


def test_stage6_unwrap_ignores_incompatible_scla_smooth_seed(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    unwrap_cols = np.asarray([0, 2], dtype=np.int64)
    unwrapped_without_restore = np.asarray([[1.0, 0.0, 3.0], [2.0, 0.0, 4.0]], dtype=np.float32)
    ph_rc = np.exp(1j * unwrapped_without_restore).astype(np.complex64)

    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 41.0, 0.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(dataset_root / "ph2.mat", {"ph": np.ones((n_ps, n_ifg), dtype=np.complex64)})
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": np.ones((n_ps, n_ifg - 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(dataset_root / "rc2.mat", {"ph_rc": ph_rc})
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)})
    write_mat(
        dataset_root / "scla_smooth2.mat",
        {
            "K_ps_uw": np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
            "C_ps_uw": np.asarray([[0.5], [-0.25], [0.75]], dtype=np.float32),
            "ph_ramp": np.zeros((3, 3), dtype=np.float32),
        },
    )
    write_mat(
        dataset_root / "uw_grid.mat",
        {
            "ph": np.ones((2, 2), dtype=np.complex64),
            "nzix": np.asarray([[True, True], [False, False]], dtype=bool),
            "grid_ij": np.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            "n_ps": np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "uw_interp.mat",
        {
            "edgs": np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "Z": np.asarray([[1, 2], [1, 2]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )

    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(ported, "_run_external_command", lambda *args, **kwargs: None)

    grids = [
        np.asarray([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [11.0, 12.0]], dtype=np.float32),
    ]

    monkeypatch.setattr(ported, "_load_float_grid", lambda path, ncol: grids.pop(0))

    ported.stage6_unwrap(dataset_root, backend="python", enable_mat_cache=False, snaphu_path="/bin/true")

    phuw2 = read_mat(dataset_root / "phuw2.mat")
    expected = np.zeros((n_ps, n_ifg), dtype=np.float32)
    expected[:, unwrap_cols] = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(phuw2["ph_uw"], dtype=np.float32), expected)


def test_stage6_unwrap_uses_stored_uw_grid_ph_in_for_reconstruction(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    unwrap_cols = np.asarray([0, 2], dtype=np.int64)
    stored_phase = np.asarray([[0.7, -0.8], [-0.2, 0.3]], dtype=np.float32)
    rc_phase = np.asarray([[-0.4, 0.0, 0.9], [0.1, 0.0, -0.6]], dtype=np.float32)

    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 41.0, 41.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "ph2.mat",
        {"ph": np.exp(1j * rc_phase).astype(np.complex64)},
    )
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": np.ones((n_ps, n_ifg - 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(
        dataset_root / "rc2.mat",
        {"ph_rc": np.exp(1j * rc_phase).astype(np.complex64)},
    )
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": np.zeros((n_ps, n_ifg - 1), dtype=np.float32)})
    write_mat(
        dataset_root / "uw_grid.mat",
        {
            "ph": np.ones((2, 2), dtype=np.complex64),
            "ph_in": np.exp(1j * stored_phase).astype(np.complex64),
            "nzix": np.asarray([[True, True], [False, False]], dtype=bool),
            "grid_ij": np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
            "n_ps": np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "uw_interp.mat",
        {
            "edgs": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.zeros((2, 1), dtype=np.float64),
            "Z": np.asarray([[1, 1], [1, 1]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )

    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((1, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(ported, "_run_external_command", lambda *args, **kwargs: None)

    grids = [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]

    def fake_load_float_grid(path: Path, ncol: int) -> np.ndarray:
        return grids.pop(0)

    monkeypatch.setattr(ported, "_load_float_grid", fake_load_float_grid)

    ported.stage6_unwrap(dataset_root, backend="python", enable_mat_cache=False, snaphu_path="/bin/true")

    phuw2 = read_mat(dataset_root / "phuw2.mat")
    expected = np.zeros((n_ps, n_ifg), dtype=np.float32)
    expected[:, unwrap_cols] = stored_phase
    np.testing.assert_allclose(np.asarray(phuw2["ph_uw"], dtype=np.float32), expected)


def test_stage6_generates_uw_grid_ph_in_from_rc2_without_patch_residual(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    unwrap_cols = np.asarray([0, 2], dtype=np.int64)
    rc_phase = np.asarray([[0.4, 0.0, -0.6], [0.2, 0.0, 0.7]], dtype=np.float32)
    patch_phase_nm = np.asarray([[0.1, -0.2], [-0.3, 0.25]], dtype=np.float32)
    ph_patch_nm = np.exp(1j * patch_phase_nm).astype(np.complex64)
    k_ps = np.asarray([0.3, -0.25], dtype=np.float32)
    bp_nm = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    bperp_full = np.concatenate(
        [bp_nm[:, : master_ix - 1], np.zeros((n_ps, 1), dtype=np.float32), bp_nm[:, master_ix - 1 :]],
        axis=1,
    )
    expected_ph_in = np.exp(1j * (rc_phase[:, unwrap_cols] + k_ps[:, None] * bperp_full[:, unwrap_cols])).astype(
        np.complex64
    )

    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 41.0, 41.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(dataset_root / "ph2.mat", {"ph": np.exp(1j * rc_phase).astype(np.complex64)})
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray(k_ps[:, None], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": ph_patch_nm,
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(dataset_root / "rc2.mat", {"ph_rc": np.exp(1j * rc_phase).astype(np.complex64)})
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": bp_nm})
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_prefilter_flag": np.asarray("n"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )

    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((1, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(ported, "_run_external_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ported,
        "_build_uw_interp_payload",
        lambda *args, **kwargs: {
            "edgs": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.zeros((2, 1), dtype=np.float64),
            "Z": np.asarray([[1, 1], [1, 1]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )

    grids = [np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
    monkeypatch.setattr(ported, "_load_float_grid", lambda *args, **kwargs: grids.pop(0))

    ported.stage6_unwrap(dataset_root, backend="python", enable_mat_cache=False, snaphu_path="/bin/true")

    uw_grid = read_mat(dataset_root / "uw_grid.mat")
    np.testing.assert_allclose(np.asarray(uw_grid["ph_in"], dtype=np.complex64), expected_ph_in, atol=1e-6, rtol=0.0)


def test_stage6_unwrap_applies_grid_backprojection_residual(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path
    n_ps = 2
    n_ifg = 3
    master_ix = 2
    unwrap_cols = np.asarray([0, 2], dtype=np.int64)
    ph2 = np.exp(1j * np.asarray([[0.6, 0.0, -0.4], [0.2, 0.0, 0.8]], dtype=np.float32)).astype(np.complex64)

    write_mat(
        dataset_root / "ps2.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "n_ifg": np.asarray(float(n_ifg), dtype=np.float64),
            "n_image": np.asarray(float(n_ifg), dtype=np.float64),
            "master_ix": np.asarray(float(master_ix), dtype=np.float64),
            "day": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64),
            "bperp": np.asarray([[10.0], [0.0], [20.0]], dtype=np.float32),
            "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
            "ij": np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
            "lonlat": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
            "mean_range": np.asarray(830000.0, dtype=np.float64),
            "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
        },
    )
    write_mat(dataset_root / "ph2.mat", {"ph": ph2})
    write_mat(
        dataset_root / "pm2.mat",
        {
            "K_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "C_ps": np.asarray([[0.0], [0.0]], dtype=np.float64),
            "coh_ps": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "ph_patch": np.ones((n_ps, n_ifg - 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, n_ifg - 1), dtype=np.float32),
        },
    )
    write_mat(dataset_root / "rc2.mat", {"ph_rc": ph2})
    write_mat(dataset_root / "bp2.mat", {"bperp_mat": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)})
    write_mat(
        dataset_root / "uw_grid.mat",
        {
            "ph": np.ones((2, 2), dtype=np.complex64),
            "nzix": np.asarray([[True, True], [False, False]], dtype=bool),
            "grid_ij": np.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            "n_ps": np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "uw_interp.mat",
        {
            "edgs": np.asarray([[1.0, 1.0, 2.0]], dtype=np.float64),
            "rowix": np.zeros((1, 2), dtype=np.float64),
            "colix": np.asarray([[1.0], [1.0]], dtype=np.float64),
            "Z": np.asarray([[1, 2], [1, 2]], dtype=np.int64),
            "n_edge": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        dataset_root / "parms.mat",
        {
            "small_baseline_flag": np.asarray("n"),
            "unwrap_patch_phase": np.asarray("n"),
            "unwrap_method": np.asarray("3D"),
            "unwrap_la_error_flag": np.asarray("y"),
            "unwrap_spatial_cost_func_flag": np.asarray("n"),
            "unwrap_time_win": np.asarray(36.0, dtype=np.float64),
            "lambda": np.asarray(0.0555, dtype=np.float64),
            "max_topo_err": np.asarray(15.0, dtype=np.float64),
        },
    )

    monkeypatch.setattr(
        ported,
        "_compute_active_single_master_uw_space_time",
        lambda *args, **kwargs: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((1, 2), dtype=np.complex64),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(ported, "_run_external_command", lambda *args, **kwargs: None)

    grids = [
        np.asarray([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [11.0, 12.0]], dtype=np.float32),
    ]

    monkeypatch.setattr(ported, "_load_float_grid", lambda *args, **kwargs: grids.pop(0))

    ported.stage6_unwrap(dataset_root, backend="python", enable_mat_cache=False, snaphu_path="/bin/true")

    phuw2 = read_mat(dataset_root / "phuw2.mat")
    expected = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_uw_pix = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    expected[:, unwrap_cols] = ph_uw_pix + np.angle(ph2[:, unwrap_cols] * np.exp(-1j * ph_uw_pix)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(phuw2["ph_uw"], dtype=np.float32), expected)


def test_single_master_insert_master_ix_uses_first_positive_slot() -> None:
    day = np.asarray([-20.0, -5.0, 4.0, 18.0], dtype=np.float64)

    assert _single_master_insert_master_ix(day) == 2


@pytest.mark.dataset_parity
def test_active_single_master_matches_full_gate_probe_slice() -> None:
    ps2 = read_mat(RUN_FULL_GATE / "ps2.mat")
    uw_grid = read_mat(RUN_FULL_GATE / "uw_grid.mat")
    uw_interp = read_mat(RUN_FULL_GATE / "uw_interp.mat")
    gold = read_mat(RUN_FULL_GATE / "uw_space_time.mat")

    uw_ph = np.asarray(uw_grid["ph"], dtype=np.complex64)
    edgs = np.asarray(uw_interp["edgs"], dtype=np.float64)[:256]
    gold_noise = np.asarray(gold["dph_noise"], dtype=np.float32)[:256]
    gold_uw = np.asarray(gold["dph_space_uw"], dtype=np.float32)[:256]

    day_full = np.asarray(ps2["day"], dtype=np.float64).reshape(-1)
    master_ix = int(round(float(np.asarray(ps2["master_ix"]).reshape(-1)[0])))
    unwrap_ifg = np.asarray([i for i in range(1, day_full.size + 1) if i != master_ix], dtype=np.int64)
    bperp_full = ported._as_ps_vector(ps2.get("bperp"), day_full.size, "ps2.bperp").astype(np.float64)
    bperp_use = bperp_full[unwrap_ifg - 1]
    parms = read_mat(RUN_FULL_GATE / "parms.mat")
    max_topo_err = float(ported._mat_scalar(parms.get("max_topo_err", 15.0), 15.0))
    lambda_m = float(ported._mat_scalar(parms.get("lambda", 0.0555), 0.0555))
    mean_range = float(ported._mat_scalar(ps2.get("mean_range", 830000.0), 830000.0))
    mean_incidence = float(ported._mat_scalar(ps2.get("mean_incidence", np.deg2rad(23.0)), np.deg2rad(23.0)))
    max_K = max_topo_err / (lambda_m * mean_range * np.sin(mean_incidence) / (4.0 * np.pi))
    n_trial_wraps = float(np.max(bperp_full) - np.min(bperp_full)) * max_K / (2.0 * np.pi)
    time_win = float(ported._mat_scalar(parms.get("unwrap_time_win", 36.0), 36.0))

    _G, _dph_space, _dph_smooth_ifg, dph_noise, dph_space_uw = _compute_active_single_master_uw_space_time(
        uw_ph,
        edgs,
        day=day_full - day_full[master_ix - 1],
        master_ix=master_ix,
        bperp=bperp_use,
        unwrap_ifg=unwrap_ifg,
        time_win=time_win,
        n_trial_wraps=n_trial_wraps,
    )

    assert float(np.nanmax(np.abs(dph_noise - gold_noise))) <= 5e-6
    assert float(np.nanmax(np.abs(dph_space_uw - gold_uw))) <= 5e-6


@pytest.mark.dataset_parity
def test_active_single_master_matches_full_gate_critical_rows() -> None:
    ps2 = read_mat(RUN_FULL_GATE / "ps2.mat")
    uw_grid = read_mat(RUN_FULL_GATE / "uw_grid.mat")
    uw_interp = read_mat(RUN_FULL_GATE / "uw_interp.mat")
    gold = read_mat(RUN_FULL_GATE / "uw_space_time.mat")

    uw_ph = np.asarray(uw_grid["ph"], dtype=np.complex64)
    day_full = np.asarray(ps2["day"], dtype=np.float64).reshape(-1)
    master_ix = int(round(float(np.asarray(ps2["master_ix"]).reshape(-1)[0])))
    unwrap_ifg = np.asarray([i for i in range(1, day_full.size + 1) if i != master_ix], dtype=np.int64)
    bperp_full = ported._as_ps_vector(ps2.get("bperp"), day_full.size, "ps2.bperp").astype(np.float64)
    bperp_use = bperp_full[unwrap_ifg - 1]
    parms = read_mat(RUN_FULL_GATE / "parms.mat")
    max_topo_err = float(ported._mat_scalar(parms.get("max_topo_err", 15.0), 15.0))
    lambda_m = float(ported._mat_scalar(parms.get("lambda", 0.0555), 0.0555))
    mean_range = float(ported._mat_scalar(ps2.get("mean_range", 830000.0), 830000.0))
    mean_incidence = float(ported._mat_scalar(ps2.get("mean_incidence", np.deg2rad(23.0)), np.deg2rad(23.0)))
    max_K = max_topo_err / (lambda_m * mean_range * np.sin(mean_incidence) / (4.0 * np.pi))
    n_trial_wraps = float(np.max(bperp_full) - np.min(bperp_full)) * max_K / (2.0 * np.pi)
    time_win = float(ported._mat_scalar(parms.get("unwrap_time_win", 36.0), 36.0))

    for row in (670353, 224153):
        edgs = np.asarray(uw_interp["edgs"], dtype=np.float64)[row : row + 1]
        gold_noise = np.asarray(gold["dph_noise"], dtype=np.float32)[row : row + 1]
        gold_uw = np.asarray(gold["dph_space_uw"], dtype=np.float32)[row : row + 1]

        _G, _dph_space, _dph_smooth_ifg, dph_noise, dph_space_uw = _compute_active_single_master_uw_space_time(
            uw_ph,
            edgs,
            day=day_full - day_full[master_ix - 1],
            master_ix=master_ix,
            bperp=bperp_use,
            unwrap_ifg=unwrap_ifg,
            time_win=time_win,
            n_trial_wraps=n_trial_wraps,
        )

        assert float(np.nanmax(np.abs(dph_noise - gold_noise))) <= 5e-6
        assert float(np.nanmax(np.abs(dph_space_uw - gold_uw))) <= 5e-6
