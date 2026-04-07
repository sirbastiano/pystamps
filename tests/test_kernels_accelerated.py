import numpy as np
import pytest
import importlib.util

import pystamps.kernels.accelerated as accel
from pystamps.kernels import (
    BackendUnavailableError,
    describe_backend_matrix,
    run_stage4_edge_stats_kernel,
    run_stage2_grid_accumulate_kernel,
    run_stage2_histogram_kernel,
    run_stage2_topofit_coh_row_invariant_kernel,
    run_stage2_topofit_kernel,
    run_stage2_topofit_row_invariant_kernel,
    run_stage7_scla_kernel,
    run_stage8_edge_noise_kernel,
)
from pystamps.pipeline import ported


def _install_fake_stage78_native_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[object, ...]], object]:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(accel.os, "cpu_count", lambda: 6)

    class _FakeNative:
        def stage4_edge_stats(
            self,
            ph_weed: np.ndarray,
            node_a: np.ndarray,
            node_b: np.ndarray,
            bperp: np.ndarray,
            day: np.ndarray,
            time_win: float,
            small_baseline: bool,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage4", int(threads), bool(small_baseline), tuple(np.asarray(ph_weed).shape)))
            n_node, n_ifg = np.asarray(ph_weed).shape
            assert np.asarray(node_a).shape == np.asarray(node_b).shape
            assert np.asarray(bperp).shape == (n_ifg,)
            if not bool(small_baseline):
                assert np.asarray(day).shape == (n_ifg,)
            return {
                "ps_std": np.full(n_node, 3.0, dtype=np.float64),
                "ps_max": np.full(n_node, 4.0, dtype=np.float64),
            }

        def stage7_scla_parity(
            self,
            ph_proc: np.ndarray,
            ph_mean_v: np.ndarray,
            bperp_mat: np.ndarray,
            unwrap_ix: np.ndarray,
            solve_ix: np.ndarray,
            day: np.ndarray,
            master_ix: int,
            ifg_std: np.ndarray,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage7", int(threads), tuple(np.asarray(ph_proc).shape)))
            n_ps, n_ifg = np.asarray(ph_proc).shape
            assert np.asarray(ph_mean_v).shape == (n_ps, n_ifg)
            assert np.asarray(bperp_mat).shape[0] == n_ps
            assert np.asarray(unwrap_ix).ndim == 1
            assert np.asarray(solve_ix).ndim == 1
            assert np.asarray(day).shape == (n_ifg,)
            assert np.asarray(ifg_std).shape == (n_ifg,)
            assert int(master_ix) == 1
            return {
                "K_ps_uw": np.full(n_ps, 11.0, dtype=np.float64),
                "C_ps_uw": np.full(n_ps, 12.0, dtype=np.float32),
                "ph_scla": np.full((n_ps, n_ifg), 13.0, dtype=np.float32),
                "ph_ramp": np.full((n_ps, n_ifg), 14.0, dtype=np.float64),
                "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
                "mean_v": np.full(n_ps, 15.0, dtype=np.float32),
                "m": np.full((2, n_ps), 16.0, dtype=np.float32),
            }

        def stage8_edge_noise(
            self,
            uw_ph: np.ndarray,
            node_a: np.ndarray,
            node_b: np.ndarray,
            chunk_edges: int = 0,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage8", int(chunk_edges), int(threads), tuple(np.asarray(uw_ph).shape)))
            n_edge = np.asarray(node_a).size
            n_ifg = np.asarray(uw_ph).shape[1]
            assert np.asarray(node_b).shape == np.asarray(node_a).shape
            return {
                "dph_noise": np.full((n_edge, n_ifg), -1.0, dtype=np.float32),
                "dph_space_uw": np.full((n_edge, n_ifg), 2.0, dtype=np.float32),
            }

    native_mod = _FakeNative()
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: native_mod)
    return calls, native_mod


def test_stage4_kernel_cpu_small_baseline_zero_noise() -> None:
    ph_weed = np.ones((3, 3), dtype=np.complex128)
    out = run_stage4_edge_stats_kernel(
        ph_weed,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([], dtype=np.float64),
        time_win=30.0,
        small_baseline=True,
        backend="python",
    )

    np.testing.assert_allclose(out["ps_std"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out["ps_max"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)


def test_stage4_kernel_cpu_non_small_baseline_zero_noise() -> None:
    ph_weed = np.ones((3, 3), dtype=np.complex128)
    out = run_stage4_edge_stats_kernel(
        ph_weed,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        time_win=30.0,
        small_baseline=False,
        backend="python",
    )

    np.testing.assert_allclose(out["ps_std"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out["ps_max"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)


def test_stage7_kernel_cpu_shapes() -> None:
    ph_proc = np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64)
    ph_mean_v = ph_proc.copy()
    b = np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=np.float64)
    unwrap_ix = np.asarray([0, 1, 2], dtype=np.int64)
    solve_ix = np.asarray([1, 2], dtype=np.int64)
    day = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    out = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        b,
        unwrap_ix,
        solve_ix,
        day,
        master_ix=1,
        ifg_std=ifg_std,
        backend="cpu",
        chunk_ps=1,
    )

    assert out["K_ps_uw"].shape == (2,)
    assert out["C_ps_uw"].shape == (2,)
    assert out["ph_scla"].shape == (2, 3)
    assert out["ph_ramp"].shape == (2, 3)
    assert out["ifg_vcm"].shape == (3, 3)
    assert out["mean_v"].shape == (2,)
    assert out["m"].shape == (2, 2)


def test_stage8_kernel_cpu_shapes() -> None:
    uw_ph = np.asarray([[1 + 0j, 1 + 0j], [1j, -1j], [1 + 1j, 1 - 1j]], dtype=np.complex64)
    node_a = np.asarray([0, 1], dtype=np.int64)
    node_b = np.asarray([1, 2], dtype=np.int64)
    out = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="cpu")

    assert out["dph_noise"].shape == (2, 2)
    assert out["dph_space_uw"].shape == (2, 2)


def test_stage8_kernel_uses_forward_edge_orientation() -> None:
    uw_ph = np.asarray(
        [
            [1 + 0j, 1j],
            [1j, 1 + 0j],
        ],
        dtype=np.complex64,
    )
    out = run_stage8_edge_noise_kernel(
        uw_ph,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        backend="cpu",
    )

    expected = np.angle(uw_ph[[1], :] * np.conj(uw_ph[[0], :])).astype(np.float32)

    np.testing.assert_allclose(out["dph_space_uw"], expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        out["dph_noise"],
        (expected - np.mean(expected, axis=1, keepdims=True)) * np.float32(0.5),
        atol=0.0,
        rtol=0.0,
    )


def test_gpu_backend_requires_cupy() -> None:
    if importlib.util.find_spec("cupy") is not None:
        out = run_stage8_edge_noise_kernel(
            np.asarray([[1 + 0j]], dtype=np.complex64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            backend="gpu",
        )
        assert out["dph_noise"].shape == (1, 1)
        return

    with pytest.raises(BackendUnavailableError, match="CuPy"):
        run_stage8_edge_noise_kernel(
            np.asarray([[1 + 0j]], dtype=np.complex64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            backend="gpu",
        )


def test_stage2_grid_accumulate_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)

    ph_weight = np.asarray(
        [
            [1 + 1j, 2 + 0j],
            [3 + 0j, 4 + 1j],
            [5 - 1j, 6 + 2j],
        ],
        dtype=np.complex64,
    )
    grid_lin = np.asarray([0, 2, 0], dtype=np.int64)

    observed = run_stage2_grid_accumulate_kernel(ph_weight, grid_lin, 3, 1, backend="auto")
    expected = accel._stage2_grid_accumulate_cpu(ph_weight, grid_lin, 3, 1)

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


def test_stage2_topofit_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.ones(n_row, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    out = run_stage2_topofit_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    assert out[0].shape == (2,)


def test_stage2_row_invariant_topofit_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        assert bperp.shape == (2, 3)
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.ones(n_row, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    out = run_stage2_topofit_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    assert out[0].shape == (2,)


def test_stage2_row_invariant_coh_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        assert bperp.shape == (2, 3)
        return np.full(cpxphase.shape[0], 0.75, dtype=np.float64)

    out = run_stage2_topofit_coh_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    np.testing.assert_allclose(out, np.full(2, 0.75, dtype=np.float64))


def test_stage2_histogram_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)

    values = np.asarray([0.1, 0.4, 0.49, 0.8, np.nan], dtype=np.float64)
    centers = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    observed = run_stage2_histogram_kernel(values, centers, backend="auto")
    expected = accel._stage2_histogram_with_centers_cpu(values, centers)

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed, np.asarray([1.0, 2.0, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)


def test_stage2_histogram_kernel_equal_spacing_matches_octave_rule() -> None:
    centers = np.asarray([0.005, 0.015, 0.025, 0.035], dtype=np.float64)
    values = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
    observed = accel._stage2_histogram_with_centers_cpu(values, centers)
    np.testing.assert_allclose(observed, np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)


def test_describe_backend_matrix_reports_registered_coverage() -> None:
    matrix = describe_backend_matrix()

    assert "providers" in matrix
    assert "kernels" in matrix
    assert matrix["providers"]["python"]["available"] is True
    assert matrix["providers"]["cuda"]["aliases"] == ["gpu"]
    assert matrix["kernels"]["stage2_topofit"]["baseline_backend"] == "python"
    assert matrix["kernels"]["stage4_edge_stats"]["baseline_backend"] == "python"
    assert "native" in matrix["kernels"]["stage2_topofit"]["supported_backends"]
    assert matrix["kernels"]["stage7_scla"]["baseline_backend"] == "python"
    assert "cuda" in matrix["kernels"]["stage8_edge_noise"]["supported_backends"]


def test_describe_backend_matrix_reports_stage7_stage8_native_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_stage78_native_backends(monkeypatch)

    matrix = describe_backend_matrix()

    assert "native" in matrix["kernels"]["stage4_edge_stats"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage7_scla"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage8_edge_noise"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage4_edge_stats"]["available_backends"]
    assert "native" in matrix["kernels"]["stage7_scla"]["available_backends"]
    assert "native" in matrix["kernels"]["stage8_edge_noise"]["available_backends"]


def test_stage4_stage7_stage8_native_wrappers_dispatch_to_fake_native_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _native_mod = _install_fake_stage78_native_backends(monkeypatch)

    stage4 = run_stage4_edge_stats_kernel(
        np.ones((3, 3), dtype=np.complex128),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        time_win=30.0,
        small_baseline=False,
        backend="native",
    )
    stage7 = run_stage7_scla_kernel(
        np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64),
        np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64),
        np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=np.float64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        master_ix=1,
        ifg_std=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        backend="native",
        chunk_ps=4,
    )
    stage8 = run_stage8_edge_noise_kernel(
        np.asarray([[1 + 0j, 1 + 0j], [1j, -1j], [1 + 1j, 1 - 1j]], dtype=np.complex64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        backend="native",
        chunk_edges=3,
    )

    assert calls == [("stage4", 6, False, (3, 3)), ("stage7", 6, (2, 3)), ("stage8", 3, 6, (3, 2))]
    np.testing.assert_allclose(stage4["ps_std"], np.full(3, 3.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage4["ps_max"], np.full(3, 4.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage7["K_ps_uw"], np.full(2, 11.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage7["ph_ramp"], np.full((2, 3), 14.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage8["dph_noise"], np.full((2, 2), -1.0, dtype=np.float32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage8["dph_space_uw"], np.full((2, 2), 2.0, dtype=np.float32), atol=0.0, rtol=0.0)


def test_native_loader_retries_after_initial_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    class _FakeNative:
        def stage7_scla_parity(
            self,
            ph_proc: np.ndarray,
            ph_mean_v: np.ndarray,
            bperp_mat: np.ndarray,
            unwrap_ix: np.ndarray,
            solve_ix: np.ndarray,
            day: np.ndarray,
            master_ix: int,
            ifg_std: np.ndarray,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            n_ps, n_ifg = np.asarray(ph_proc).shape
            return {
                "K_ps_uw": np.zeros(n_ps, dtype=np.float64),
                "C_ps_uw": np.zeros(n_ps, dtype=np.float32),
                "ph_scla": np.zeros((n_ps, n_ifg), dtype=np.float32),
                "ph_ramp": np.zeros((n_ps, n_ifg), dtype=np.float64),
                "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
                "mean_v": np.zeros(n_ps, dtype=np.float32),
                "m": np.zeros((2, n_ps), dtype=np.float32),
            }

    def fake_import_module(name: str) -> object:
        assert name == "pystamps.kernels._stage2_native"
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ImportError("native extension not built yet")
        return _FakeNative()

    monkeypatch.setattr(accel.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(accel.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(accel, "_STAGE2_NATIVE_MODULE", None)
    monkeypatch.setattr(accel, "_STAGE2_NATIVE_IMPORT_ATTEMPTED", False)

    assert accel.stage7_native_available() is False
    assert accel.stage7_native_available() is True
    assert attempts["count"] == 2


def test_stage2_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeNative:
        def accumulate_weighted_grid(
            self,
            ph_weight: np.ndarray,
            grid_lin: np.ndarray,
            n_i: int,
            n_j: int,
            threads: int,
        ) -> np.ndarray:
            calls.append(f"grid:{threads}")
            return np.full((n_i, n_j, ph_weight.shape[1]), 7 + 0j, dtype=np.complex64)

        def ps_topofit_batch_generic(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            calls.append(f"topofit:{threads}")
            n_row, n_col = cpxphase.shape
            return (
                np.full(n_row, 1.0, dtype=np.float64),
                np.full(n_row, 2.0, dtype=np.float64),
                np.full(n_row, 3.0, dtype=np.float64),
                np.full((n_row, n_col), 4 + 0j, dtype=np.complex64),
            )

        def ps_topofit_batch_row_invariant(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            calls.append(f"row:{threads}")
            n_row, n_col = cpxphase.shape
            return (
                np.full(n_row, 5.0, dtype=np.float64),
                np.full(n_row, 6.0, dtype=np.float64),
                np.full(n_row, 7.0, dtype=np.float64),
                np.full((n_row, n_col), 8 + 0j, dtype=np.complex64),
            )

        def ps_topofit_coh_row_invariant(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> np.ndarray:
            calls.append(f"rowcoh:{threads}")
            return np.full(cpxphase.shape[0], 9.0, dtype=np.float64)

        def histogram_with_centers(
            self,
            values: np.ndarray,
            centers: np.ndarray,
        ) -> np.ndarray:
            calls.append("hist")
            return np.asarray([2.0, 1.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    grid = run_stage2_grid_accumulate_kernel(
        np.ones((2, 3), dtype=np.complex64),
        np.asarray([0, 1], dtype=np.int64),
        2,
        1,
        backend="native",
        threads=3,
    )
    topofit = run_stage2_topofit_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        1.0,
        backend="native",
        threads=4,
    )
    topofit_row = run_stage2_topofit_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="native",
        threads=2,
    )
    coh_row = run_stage2_topofit_coh_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="native",
        threads=6,
    )
    hist = run_stage2_histogram_kernel(
        np.asarray([0.1, 0.4, 0.7], dtype=np.float64),
        np.asarray([0.0, 0.6, 1.0], dtype=np.float64),
        backend="native",
    )

    expected_topofit = ported._ps_topofit_batch_generic(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        1.0,
    )
    expected_row = ported._ps_topofit_batch_row_invariant(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        1.0,
    )
    expected_coh = ported._ps_topofit_batch_row_invariant_coh(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        1.0,
    )

    assert calls == ["grid:3", "hist"]
    np.testing.assert_allclose(grid, np.full((2, 1, 3), 7 + 0j, dtype=np.complex64))
    np.testing.assert_allclose(topofit[0], expected_topofit[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(topofit_row[0], expected_row[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(coh_row, expected_coh, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(hist, np.asarray([2.0, 1.0, 0.0], dtype=np.float64))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_kernels_match_python_reference() -> None:
    rng = np.random.default_rng(11)
    cpxphase = np.exp(1j * rng.normal(size=(6, 5))).astype(np.complex128)
    bperp = rng.normal(size=(6, 5)).astype(np.float64)
    bperp_row = np.tile(np.asarray([-120.0, -40.0, 0.0, 55.0, 90.0], dtype=np.float64), (6, 1))

    expected_grid = accel._stage2_grid_accumulate_cpu(cpxphase.astype(np.complex64), np.asarray([0, 1, 0, 1, 2, 2]), 3, 1)
    observed_grid = run_stage2_grid_accumulate_kernel(
        cpxphase.astype(np.complex64),
        np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64),
        3,
        1,
        backend="native",
    )
    np.testing.assert_allclose(observed_grid, expected_grid, atol=1e-6, rtol=0.0)

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=1.5)
    observed = run_stage2_topofit_kernel(cpxphase, bperp, 1.5, backend="native")
    np.testing.assert_allclose(observed[0], expected[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-5, rtol=0.0)

    expected_row = ported._ps_topofit_batch_row_invariant(cpxphase, bperp_row, n_trial_wraps=1.5)
    observed_row = run_stage2_topofit_row_invariant_kernel(cpxphase, bperp_row, 1.5, backend="native")
    np.testing.assert_allclose(observed_row[0], expected_row[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[1], expected_row[1], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[2], expected_row[2], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[3], expected_row[3], atol=1e-5, rtol=0.0)

    coh_row = run_stage2_topofit_coh_row_invariant_kernel(cpxphase, bperp_row, 1.5, backend="native")
    np.testing.assert_allclose(coh_row, expected_row[2], atol=1e-10, rtol=0.0)

    hist_expected = accel._stage2_histogram_with_centers_cpu(
        np.asarray([0.1, 0.49, 0.7, 0.9, np.nan], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    hist_observed = run_stage2_histogram_kernel(
        np.asarray([0.1, 0.49, 0.7, 0.9, np.nan], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        backend="native",
    )
    np.testing.assert_allclose(hist_observed, hist_expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(hist_observed, np.asarray([1.0, 2.0, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage4_stage7_stage8_native_kernels_match_python_reference() -> None:
    rng = np.random.default_rng(123)

    ph_weed = np.exp(1j * rng.normal(size=(8, 6))).astype(np.complex128)
    node_a = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_b = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    bperp = np.linspace(-50.0, 80.0, 6, dtype=np.float64)
    day = np.asarray([1.0, 5.0, 9.0, 15.0, 22.0, 30.0], dtype=np.float64)

    for small_baseline, day_vec in (
        (False, day),
        (True, np.asarray([], dtype=np.float64)),
    ):
        expected_stage4 = run_stage4_edge_stats_kernel(
            ph_weed,
            node_a,
            node_b,
            bperp,
            day_vec,
            time_win=30.0,
            small_baseline=small_baseline,
            backend="python",
        )
        observed_stage4 = run_stage4_edge_stats_kernel(
            ph_weed,
            node_a,
            node_b,
            bperp,
            day_vec,
            time_win=30.0,
            small_baseline=small_baseline,
            backend="native",
        )
        np.testing.assert_allclose(observed_stage4["ps_std"], expected_stage4["ps_std"], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(observed_stage4["ps_max"], expected_stage4["ps_max"], atol=1e-12, rtol=0.0)

    ph_proc = rng.normal(size=(5, 6))
    ph_mean_v = rng.normal(size=(5, 6))
    bperp_mat = rng.normal(size=(5, 6))
    unwrap_ix = np.arange(6, dtype=np.int64)
    solve_ix = np.asarray([0, 1, 3, 4, 5], dtype=np.int64)
    day_stage7 = np.cumsum(rng.uniform(1.0, 4.0, size=6))
    ifg_std = rng.uniform(0.5, 2.0, size=6)

    expected_stage7 = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        bperp_mat,
        unwrap_ix,
        solve_ix,
        day_stage7,
        master_ix=3,
        ifg_std=ifg_std,
        backend="python",
    )
    observed_stage7 = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        bperp_mat,
        unwrap_ix,
        solve_ix,
        day_stage7,
        master_ix=3,
        ifg_std=ifg_std,
        backend="native",
    )
    for key in ("K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp", "ifg_vcm", "mean_v", "m"):
        np.testing.assert_allclose(observed_stage7[key], expected_stage7[key], atol=1e-12, rtol=0.0)

    uw_ph = np.exp(1j * rng.normal(size=(8, 6))).astype(np.complex64)
    expected_stage8 = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="python")
    observed_stage8 = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="native")
    np.testing.assert_allclose(observed_stage8["dph_noise"], expected_stage8["dph_noise"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(observed_stage8["dph_space_uw"], expected_stage8["dph_space_uw"], atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_generic_matches_python_single_precision() -> None:
    rng = np.random.default_rng(21)
    cpxphase = np.exp(1j * rng.normal(size=(5, 7))).astype(np.complex64)
    bperp = rng.normal(size=(5, 7)).astype(np.float32)

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=1.5)
    observed = run_stage2_topofit_kernel(cpxphase, bperp, 1.5, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-6, rtol=0.0)
