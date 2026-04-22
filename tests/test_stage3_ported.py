from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline import ported


def test_stage3_reestimate_writes_reestimated_threshold_coeffs(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls = {"count": 0}

    def fake_clap(ph_bit: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
        vals = np.asarray([0.5 + 0.0j, 0.25 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_bit, dtype=np.complex128)
        out[0, 0] = vals[clap_calls["count"]]
        clap_calls["count"] += 1
        return out

    monkeypatch.setattr(ported, "_clap_filt_patch", fake_clap)

    coeff_calls = {"count": 0}
    initial_coeffs = np.asarray([9.0, 8.0], dtype=np.float64)
    reestimated_coeffs = np.asarray([1.5, -0.2], dtype=np.float64)

    def fake_threshold(*args, **kwargs):
        coeff_calls["count"] += 1
        coeffs = initial_coeffs if coeff_calls["count"] == 1 else reestimated_coeffs
        return np.zeros(1, dtype=np.float64), coeffs

    monkeypatch.setattr(ported, "_coh_threshold_from_dist", fake_threshold)

    def fake_topofit(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        return 0.1, 0.2, 0.9, np.ones(2, dtype=np.complex64)

    monkeypatch.setattr(ported, "_ps_topofit_single", fake_topofit)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 1 PS"
    payload = read_mat(patch_dir / "select1.mat")
    np.testing.assert_allclose(
        np.asarray(payload["coh_thresh_coeffs"], dtype=np.float64).reshape(-1),
        reestimated_coeffs,
        atol=0.0,
        rtol=0.0,
    )


def test_stage3_reestimate_skips_topofit_when_any_ifg_is_zero(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls = {"count": 0}

    def fake_clap(ph_bit: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
        vals = np.asarray([0.5 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_bit, dtype=np.complex128)
        out[0, 0] = vals[clap_calls["count"]]
        clap_calls["count"] += 1
        return out

    monkeypatch.setattr(ported, "_clap_filt_patch", fake_clap)
    monkeypatch.setattr(
        ported,
        "_coh_threshold_from_dist",
        lambda *args, **kwargs: (np.zeros(1, dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)),
    )

    called = {"count": 0}

    def fake_topofit(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        called["count"] += 1
        return 0.15, 0.25, 0.85, np.ones(2, dtype=np.complex64)

    monkeypatch.setattr(ported, "_ps_topofit_single", fake_topofit)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 1 PS"
    assert called["count"] == 0
    payload = read_mat(patch_dir / "select1.mat")
    assert np.isnan(np.asarray(payload["K_ps2"], dtype=np.float64).reshape(-1)[0])
    assert np.isnan(np.asarray(payload["coh_ps2"], dtype=np.float64).reshape(-1)[0])


def test_stage3_reestimate_keep_ix_uses_strict_source_threshold(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls = {"count": 0}

    def fake_clap(ph_bit: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
        vals = np.asarray([0.5 + 0.0j, 0.25 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_bit, dtype=np.complex128)
        out[0, 0] = vals[clap_calls["count"]]
        clap_calls["count"] += 1
        return out

    monkeypatch.setattr(ported, "_clap_filt_patch", fake_clap)
    monkeypatch.setattr(
        ported,
        "_coh_threshold_from_dist",
        lambda *args, **kwargs: (np.asarray([0.5], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)),
    )

    def fake_topofit(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        return 0.1, 0.2, 0.5000005, np.ones(2, dtype=np.complex64)

    monkeypatch.setattr(ported, "_ps_topofit_single", fake_topofit)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 1 PS"
    payload = read_mat(patch_dir / "select1.mat")
    assert np.asarray(payload["keep_ix"]).reshape(-1)[0]


def test_stage3_oversample_zeroing_matches_matlab_first_plane_semantics() -> None:
    ph_bit = np.stack(
        (
            np.full((4, 4), 10.0 + 0.0j, dtype=np.complex64),
            np.full((4, 4), 20.0 + 0.0j, dtype=np.complex64),
        ),
        axis=2,
    )
    ps_bit_i = 2
    ps_bit_j = 2
    slc_osf = 2

    ph_bit[ps_bit_i - 1, ps_bit_j - 1, :] = 0
    rad = slc_osf - 1
    ii = np.arange(ps_bit_i - rad, ps_bit_i + rad + 1, dtype=np.int64)
    jj = np.arange(ps_bit_j - rad, ps_bit_j + rad + 1, dtype=np.int64)
    ii = ii[(ii > 0) & (ii <= ph_bit.shape[0])] - 1
    jj = jj[(jj > 0) & (jj <= ph_bit.shape[1])] - 1
    ph_bit[np.ix_(ii, jj, np.asarray([0], dtype=np.int64))] = 0

    np.testing.assert_allclose(ph_bit[:3, :3, 0], 0.0, atol=0.0, rtol=0.0)
    assert ph_bit[1, 1, 1] == 0.0
    assert ph_bit[0, 0, 1] == 20.0 + 0.0j


def test_clap_filt_patch_stack_returns_complex128_workspace() -> None:
    ph = np.ones((4, 4, 2), dtype=np.complex64)
    low_pass = np.ones((4, 4), dtype=np.float64)

    out = ported._clap_filt_patch_stack(ph, alpha=1.0, beta=0.3, low_pass=low_pass)

    assert out.dtype == np.complex128


def test_stage3_density_threshold_uses_matlab_da_bin_edges(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    n_ps = 50000
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0], dtype=np.float64),
            "xy": np.zeros((n_ps, 3), dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.arange(1, n_ps + 1, dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.zeros(n_ps, dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.zeros((n_ps, 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, 1), dtype=np.float32),
            "K_ps": np.zeros(n_ps, dtype=np.float64),
            "C_ps": np.zeros(n_ps, dtype=np.float64),
        },
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms: np.asarray([1.0], dtype=np.float64))
    monkeypatch.setattr(
        ported,
        "_as_ps_ifg_complex",
        lambda values, n_ps_arg, name: np.zeros((n_ps_arg, 1), dtype=np.complex64),
    )
    monkeypatch.setattr(
        ported,
        "_as_ps_matrix",
        lambda values, n_ps_arg, name: np.zeros((n_ps_arg, 1), dtype=np.float32),
    )

    captured: dict[str, np.ndarray] = {}

    def fake_threshold(*, D_A_max: np.ndarray, **kwargs):
        captured["D_A_max"] = np.asarray(D_A_max, dtype=np.float64).copy()
        return np.ones(n_ps, dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(ported, "_coh_threshold_from_dist", fake_threshold)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 0 PS"
    np.testing.assert_array_equal(
        captured["D_A_max"],
        np.asarray([0.0, 10001.0, 20001.0, 30001.0, 50000.0], dtype=np.float64),
    )


def test_stage3_saved_patch1_row_matches_oracle_residual_angles() -> None:
    patch_dir = Path("inputs_and_outputs/InSAR_dataset_test/PATCH_1")

    sel = read_mat(patch_dir / "select1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    ph1 = read_mat(patch_dir / "ph1.mat")
    bp1 = read_mat(patch_dir / "bp1.mat")
    pm1 = read_mat(patch_dir / "pm1.mat")

    row = 56350
    ix = np.asarray(sel["ix"], dtype=np.float64).reshape(-1).astype(np.int64)
    ps_idx = int(ix[row] - 1)

    ph_all = np.asarray(ph1["ph"], dtype=np.complex128)
    master_ix = int(round(float(np.asarray(ps["master_ix"], dtype=np.float64).reshape(-1)[0])))
    ph_work = ph_all[:, np.arange(ph_all.shape[1]) != (master_ix - 1)]
    ph_patch2 = np.asarray(sel["ph_patch2"], dtype=np.complex128)
    bperp_mat = np.asarray(bp1["bperp_mat"], dtype=np.float64)
    ifg_index_ix = np.asarray(sel["ifg_index"], dtype=np.float64).reshape(-1).astype(np.int64) - 1

    psdph = ph_work[ps_idx, :] * np.conj(ph_patch2[row, :])
    psdph = np.divide(psdph, np.abs(psdph), out=np.zeros_like(psdph), where=np.abs(psdph) != 0)
    _, _, _, phase_residual = ported._ps_topofit_single(
        psdph[ifg_index_ix].astype(np.complex64, copy=False),
        bperp_mat[ps_idx, :][ifg_index_ix],
        float(np.asarray(pm1["n_trial_wraps"], dtype=np.float64).reshape(-1)[0]),
    )

    observed = np.angle(phase_residual).astype(np.float32, copy=False)
    expected = np.asarray(sel["ph_res2"], dtype=np.float32)[row, ifg_index_ix]

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=np.finfo(np.float32).eps)
