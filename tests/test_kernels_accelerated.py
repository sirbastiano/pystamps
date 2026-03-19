import numpy as np
import pytest
import importlib.util

from pystamps.kernels import BackendUnavailableError, run_stage7_scla_kernel, run_stage8_edge_noise_kernel


def test_stage7_kernel_cpu_shapes() -> None:
    ph_uw = np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float32)
    b = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    no_master = np.asarray([False, True, True], dtype=bool)
    day = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    out = run_stage7_scla_kernel(ph_uw, b, no_master, day, master_ix=1, backend="cpu")

    assert out["K_ps_uw"].shape == (2,)
    assert out["C_ps_uw"].shape == (2,)
    assert out["ph_scla"].shape == (2, 3)
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
