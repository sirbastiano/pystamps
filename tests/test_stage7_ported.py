from __future__ import annotations

import numpy as np

from pystamps.pipeline.ported import (
    _deramp_unwrapped_phase,
    _select_reference_ps,
    _stage7_mean_velocity_fit,
    _stage7_unwrap_ifg_sets,
    _weighted_lstsq_shared_design,
)


def test_weighted_lstsq_shared_design_solves_multi_rhs() -> None:
    G = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 2.0],
            [1.0, 5.0],
            [1.0, 9.0],
        ],
        dtype=np.float64,
    )
    coeffs_true = np.asarray(
        [
            [3.0, -2.0],
            [0.5, 1.25],
        ],
        dtype=np.float64,
    )
    Y = G @ coeffs_true
    cov = np.diag(np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64))

    coeffs = _weighted_lstsq_shared_design(G, Y, cov=cov)

    np.testing.assert_allclose(coeffs, coeffs_true, atol=1e-10, rtol=0.0)


def test_deramp_unwrapped_phase_removes_linear_plane() -> None:
    ps = {
        "n_ps": np.asarray(4.0),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1000.0, 0.0],
                [3.0, 0.0, 1000.0],
                [4.0, 1000.0, 1000.0],
            ],
            dtype=np.float64,
        ),
    }
    x_km = ps["xy"][:, 1] / 1000.0
    y_km = ps["xy"][:, 2] / 1000.0
    ramp = np.column_stack(
        (
            1.5 * x_km + 0.75 * y_km + 2.0,
            -0.5 * x_km + 1.25 * y_km - 1.0,
        )
    )
    ph = ramp.copy()

    ph_out, ph_ramp = _deramp_unwrapped_phase(ps, ph)

    np.testing.assert_allclose(ph_ramp, ramp, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(ph_out, np.zeros_like(ph), atol=1e-10, rtol=0.0)


def test_select_reference_ps_uses_local_coordinate_units_for_radius() -> None:
    ps = {
        "n_ps": np.asarray(2.0),
        "lonlat": np.asarray(
            [
                [0.0, 0.0],
                [0.0009, 0.0],
            ],
            dtype=np.float64,
        ),
        "ll0": np.asarray([0.0, 0.0], dtype=np.float64),
    }
    parms_raw = {
        "ref_lon": np.asarray([-1.0, 1.0], dtype=np.float64),
        "ref_lat": np.asarray([-1.0, 1.0], dtype=np.float64),
        "ref_centre_lonlat": np.asarray([0.0, 0.0], dtype=np.float64),
        "ref_radius": np.asarray(120.0, dtype=np.float64),
    }

    ref_ix = _select_reference_ps(ps, parms_raw)

    np.testing.assert_array_equal(ref_ix, np.asarray([0, 1], dtype=np.int64))


def test_stage7_unwrap_ifg_sets_keeps_master_for_sequential_diffs() -> None:
    unwrap_ifg, solve_ifg = _stage7_unwrap_ifg_sets(n_ifg=5, master_ix=3, drop_set={5})

    np.testing.assert_array_equal(unwrap_ifg, np.asarray([1, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(solve_ifg, np.asarray([1, 2, 4], dtype=np.int64))


def test_stage7_mean_velocity_fit_uses_full_stack_weights() -> None:
    ph_mean_v = np.asarray(
        [
            [3.0, 0.0, -1.0, 1.0],
            [-2.0, 0.0, 4.0, 7.0],
        ],
        dtype=np.float64,
    )
    day = np.asarray([8.0, 10.0, 13.0, 17.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

    m = _stage7_mean_velocity_fit(ph_mean_v, day, master_ix=2, ifg_std=ifg_std)

    time_diff = day - day[1]
    weights = 1.0 / ((ifg_std * np.pi / 180.0) ** 2)
    s0 = float(np.sum(weights))
    s1 = float(np.sum(weights * time_diff))
    s2 = float(np.sum(weights * time_diff * time_diff))
    det = s0 * s2 - s1 * s1
    wy0 = np.sum(ph_mean_v * weights[None, :], axis=1)
    wy1 = np.sum(ph_mean_v * (weights * time_diff)[None, :], axis=1)
    expected = np.vstack(
        (
            ((wy0 * s2 - wy1 * s1) / det).astype(np.float32),
            ((wy1 * s0 - wy0 * s1) / det).astype(np.float32),
        )
    )

    np.testing.assert_allclose(m, expected, atol=1e-10, rtol=0.0)
