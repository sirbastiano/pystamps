from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import signal

from pystamps.pipeline import ported


def test_matlab_v5_uniform_rng_uses_column_major_matrix_fill() -> None:
    flat_rng = ported._MatlabV5UniformRNG(2005)
    matrix_rng = ported._MatlabV5UniformRNG(2005)

    expected = flat_rng.uniform(6).reshape((2, 3), order="F")
    observed = matrix_rng.uniform((2, 3))

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_stage2_random_phase_chunks_match_full_matrix_layout() -> None:
    rng_expected = ported._MatlabV5UniformRNG(2005)
    rng_observed = ported._MatlabV5UniformRNG(2005)
    rng_naive = ported._MatlabV5UniformRNG(2005)

    full = rng_expected.uniform((6, 3)) * (2 * np.pi)
    expected = [
        np.exp(1j * full[0:2, :]),
        np.exp(1j * full[2:4, :]),
        np.exp(1j * full[4:6, :]),
    ]
    observed = list(
        ported._stage2_random_phase_chunks(
            rng_observed,
            6,
            2,
            3,
            small_baseline=False,
        )
    )
    naive = [
        np.exp(1j * (rng_naive.uniform((2, 3)) * (2 * np.pi))),
        np.exp(1j * (rng_naive.uniform((2, 3)) * (2 * np.pi))),
        np.exp(1j * (rng_naive.uniform((2, 3)) * (2 * np.pi))),
    ]

    assert len(observed) == len(expected) == len(naive) == 3
    for observed_chunk, expected_chunk in zip(observed, expected, strict=True):
        assert observed_chunk.dtype == np.complex128
        np.testing.assert_allclose(observed_chunk, expected_chunk, rtol=0.0, atol=0.0)
    assert any(not np.array_equal(observed_chunk, naive_chunk) for observed_chunk, naive_chunk in zip(observed, naive, strict=True))


def test_matlab_interp_matches_stage2_firwin_path() -> None:
    x = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    observed = ported._matlab_interp(x, 10)

    q = 10
    n = 4
    coeff = signal.firwin(
        2 * q * n + 2,
        0.5 / q,
        window="hamming",
        scale=True,
        fs=2.0,
    ).astype(np.float64)
    y = np.zeros(x.size * q + q * n + 1, dtype=np.float64)
    y[: x.size * q : q] = x
    y = q * signal.lfilter(coeff, [1.0], y)
    expected = y[q * n + 1 :]

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_stage2_psquare_weighting_uses_matlab_rounding_for_bin_lookup() -> None:
    Nr = np.ones(5, dtype=np.float64)
    Na = np.ones(5, dtype=np.float64)
    coh_ps = np.asarray([0.0015], dtype=np.float64)

    _, prand_hi, prand_ps, weighting = ported._stage2_psquare_weighting(
        Nr,
        Na,
        low_coh_thresh=0,
        nr_max_nz_ix=5,
        coh_ps=coh_ps,
    )

    expected_ix = int(ported._round_half_away_from_zero(coh_ps * 1000.0)[0])
    np.testing.assert_allclose(prand_ps, prand_hi[[expected_ix]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weighting, (1.0 - prand_hi[[expected_ix]]) ** 2, rtol=0.0, atol=0.0)


def test_stage2_psquare_weighting_trims_interp_tail_like_matlab() -> None:
    Nr = np.ones(100, dtype=np.float64)
    Na = np.ones(100, dtype=np.float64)
    coh_ps = np.asarray([0.995], dtype=np.float64)

    _, prand_hi, prand_ps, _ = ported._stage2_psquare_weighting(
        Nr,
        Na,
        low_coh_thresh=0,
        nr_max_nz_ix=100,
        coh_ps=coh_ps,
    )

    assert prand_hi.shape == (1001,)
    np.testing.assert_allclose(prand_ps, prand_hi[[995]], rtol=0.0, atol=0.0)


def test_stage2_grid_indices_keep_single_precision_boundary_bin() -> None:
    xy = np.asarray(
        [
            [1.0, -23497.43359375, 0.0],
            [2.0, -21497.43359375, 0.0],
            [3.0, -20497.43359375, 0.0],
        ],
        dtype=np.float32,
    )

    observed = ported._stage2_grid_indices(xy, 50.0)

    x64 = xy.astype(np.float64)[:, 1]
    naive = np.ceil((x64 - np.min(x64) + 1e-6) / 50.0).astype(np.int64)

    assert int(observed[1, 1]) == 40
    assert int(naive[1]) == 41


def test_stage2_row_invariant_bperp_vector_prefers_invariant_bp1_rows() -> None:
    ps_bperp = np.asarray([15.0, 30.0], dtype=np.float64)
    bp1 = np.tile(np.asarray([10.0, 20.0], dtype=np.float64), (3, 1))

    observed = ported._stage2_row_invariant_bperp_vector(ps_bperp, bp1)

    np.testing.assert_allclose(observed, np.asarray([10.0, 20.0], dtype=np.float64), rtol=0.0, atol=0.0)


def test_prepare_clap_stack_retains_complex128_scratch_buffer() -> None:
    prepared = ported._prepare_clap_filt_grid_stack((24, 24, 3), n_win=24, n_pad=8, low_pass=np.zeros((32, 32)))

    assert prepared.ph_bit.dtype == np.complex128


def test_clap_stack_matches_scalar_per_ifg_legacy_path() -> None:
    rng = np.random.default_rng(3)
    ph_stack = (
        rng.normal(size=(36, 36, 3)) + 1j * rng.normal(size=(36, 36, 3))
    ).astype(np.complex64)
    low_pass = np.zeros((32, 32), dtype=np.float64)
    prepared = ported._prepare_clap_filt_grid_stack(ph_stack.shape, n_win=24, n_pad=8, low_pass=low_pass)

    observed = ported._clap_filt_grid_stack_prepared(ph_stack, alpha=1.0, beta=0.3, prepared=prepared)
    expected = np.empty_like(ph_stack)
    for i_ifg in range(ph_stack.shape[2]):
        expected[:, :, i_ifg] = ported._clap_filt_grid(
            ph_stack[:, :, i_ifg],
            alpha=1.0,
            beta=0.3,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_clap_stack_matches_scalar_per_ifg_with_ifg_parallelism() -> None:
    rng = np.random.default_rng(3)
    ph_stack = (
        rng.normal(size=(36, 36, 3)) + 1j * rng.normal(size=(36, 36, 3))
    ).astype(np.complex64)
    low_pass = np.zeros((32, 32), dtype=np.float64)
    prepared = ported._prepare_clap_filt_grid_stack(ph_stack.shape, n_win=24, n_pad=8, low_pass=low_pass)

    observed = ported._clap_filt_grid_stack_prepared(
        ph_stack,
        alpha=1.0,
        beta=0.3,
        prepared=prepared,
        workers=3,
    )
    expected = np.empty_like(ph_stack)
    for i_ifg in range(ph_stack.shape[2]):
        expected[:, :, i_ifg] = ported._clap_filt_grid(
            ph_stack[:, :, i_ifg],
            alpha=1.0,
            beta=0.3,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_stage2_estimate_gamma_uses_legacy_trial_wrap_inputs(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    (patch_dir / "la1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(3.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 50.0, 50.0],
                [3.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(900000.0, dtype=np.float64),
        "mean_incidence": np.asarray(0.4, dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
                [1.0 + 0.0j, 0.6 + 0.4j, 0.4 + 0.6j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([15.0, 30.0], dtype=np.float64), (3, 1))}
    la_payload = {"la": np.asarray([0.55, 0.55, 0.55], dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "la1.mat":
            return la_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(ported, "_build_stage_options", lambda patch: ported.StageOptions())
    monkeypatch.setattr(ported, "_load_parms", lambda patch: ported.Parms())
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )

    def fake_clap(
        ph_stack: np.ndarray,
        alpha: float,
        beta: float,
        prepared: object,
        out: np.ndarray | None = None,
        workers: int = 1,
        preserve_precision: bool = False,
    ):
        if out is None:
            return np.asarray(ph_stack, dtype=np.complex64).copy()
        out[...] = np.asarray(ph_stack, dtype=np.complex64)
        return out

    monkeypatch.setattr(ported, "_clap_filt_grid_stack_prepared", fake_clap)
    monkeypatch.setattr(ported._MatlabV5UniformRNG, "uniform", lambda self, size: np.zeros(size, dtype=np.float64))

    seen_trial_wraps: list[float] = []

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        seen_trial_wraps.append(float(n_trial_wraps))
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ):
        seen_trial_wraps.append(float(n_trial_wraps))
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_topofit)

    result = ported.stage2_estimate_gamma(patch_dir, debug=False)

    opts = ported.StageOptions()
    expected_mean_inc = 0.55 + 0.052
    expected_max_k = opts.max_topo_err / (opts.lambda_m * 830000.0 * np.sin(expected_mean_inc) / (4 * np.pi))
    expected_trial_wraps = 15.0 * expected_max_k / (2 * np.pi)
    wrong_max_k = opts.max_topo_err / (opts.lambda_m * 900000.0 * np.sin(0.4) / (4 * np.pi))
    wrong_trial_wraps = 15.0 * wrong_max_k / (2 * np.pi)

    assert result == "Stage 2 computed coherence for 3 candidates in 3 iterations"
    assert seen_trial_wraps
    assert abs(expected_trial_wraps - wrong_trial_wraps) > 1e-3
    np.testing.assert_allclose(seen_trial_wraps[0], expected_trial_wraps, rtol=0.0, atol=1e-12)


def test_stage2_estimate_gamma_splits_trial_wrap_inputs_from_row_invariant_phase_ramp_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    (patch_dir / "la1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(3.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 50.0, 50.0],
                [3.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(900000.0, dtype=np.float64),
        "mean_incidence": np.asarray(0.4, dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
                [1.0 + 0.0j, 0.6 + 0.4j, 0.4 + 0.6j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([10.0, 20.0], dtype=np.float64), (3, 1))}
    la_payload = {"la": np.asarray([0.55, 0.55, 0.55], dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "la1.mat":
            return la_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(ported, "_build_stage_options", lambda patch: ported.StageOptions())
    monkeypatch.setattr(ported, "_load_parms", lambda patch: ported.Parms())
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ported,
        "_stage2_random_phase_chunks",
        lambda *args, **kwargs: [np.ones((3, 2), dtype=np.complex64)],
    )
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported._MatlabV5UniformRNG, "uniform", lambda self, size: np.zeros(size, dtype=np.float64))

    seen_hist_bperp: list[np.ndarray] = []
    seen_hist_wraps: list[float] = []
    seen_topofit_bperp: list[np.ndarray] = []
    seen_topofit_wraps: list[float] = []

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        seen_hist_bperp.append(np.asarray(bperp, dtype=np.float64).copy())
        seen_hist_wraps.append(float(n_trial_wraps))
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_row_invariant_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ):
        seen_topofit_bperp.append(np.asarray(bperp, dtype=np.float64).copy())
        seen_topofit_wraps.append(float(n_trial_wraps))
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_row_invariant_topofit)

    ported.stage2_estimate_gamma(patch_dir, debug=False)

    opts = ported.StageOptions()
    expected_mean_inc = 0.55 + 0.052
    expected_max_k = opts.max_topo_err / (opts.lambda_m * 830000.0 * np.sin(expected_mean_inc) / (4 * np.pi))
    expected_trial_wraps = 15.0 * expected_max_k / (2 * np.pi)
    wrong_trial_wraps = 10.0 * expected_max_k / (2 * np.pi)

    assert seen_hist_bperp
    assert seen_topofit_bperp
    assert seen_hist_wraps
    assert seen_topofit_wraps
    np.testing.assert_allclose(seen_hist_bperp[0], np.asarray([15.0, 30.0], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        seen_topofit_bperp[0],
        np.asarray([10.0, 20.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert abs(expected_trial_wraps - wrong_trial_wraps) > 1e-3
    np.testing.assert_allclose(seen_hist_wraps[0], expected_trial_wraps, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(seen_topofit_wraps[0], expected_trial_wraps, rtol=0.0, atol=1e-12)
