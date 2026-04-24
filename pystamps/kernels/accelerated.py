from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from pystamps.config import ConfigError, normalize_stage2_kernel_backend
from pystamps.kernels.registry import DEFAULT_REGISTRY, KernelResolutionError

_STAGE8_NOISE_SCALE = np.float32(0.5)
_STAGE2_NATIVE_MODULE: Any | None = None
_STAGE2_NATIVE_IMPORT_ATTEMPTED = False


class BackendUnavailableError(RuntimeError):
    """Raised when a requested compute backend is not available."""


def _cupy() -> Any | None:
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None


def _cuda_available() -> bool:
    return _cupy() is not None


def _resolve_backend(backend: str) -> str:
    name = (backend or "auto").strip().lower()
    if name in {"auto"}:
        return "auto"
    if name in {"threads", "thread", "io", "processes", "process", "cpu", "python"}:
        return "python"
    if name in {"native"}:
        return "native"
    if name in {"gpu", "cuda"}:
        return "cuda"
    raise BackendUnavailableError(
        f"Unsupported kernel backend '{backend}'. Use: auto, python, native, or cuda"
    )


def _resolve_stage2_kernel_backend(backend: str) -> str:
    try:
        return normalize_stage2_kernel_backend(backend)
    except ConfigError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _to_numpy(arr: Any) -> np.ndarray:
    cp = _cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _cov_from_accumulators(sum_res: np.ndarray, sum_outer: np.ndarray, count: int) -> np.ndarray:
    k = int(sum_res.size)
    if k == 0:
        return np.empty((0, 0), dtype=np.float64)
    if count <= 1:
        return np.zeros((k, k), dtype=np.float64)
    mean = sum_res / float(count)
    cov = (sum_outer - float(count) * np.outer(mean, mean)) / float(count - 1)
    return cov.astype(np.float64)


def _auto_chunk_size(total_rows: int, width_hint: int, itemsize: int, target_bytes: int = 128 * 1024 * 1024) -> int:
    if total_rows <= 0:
        return 1
    width = max(1, int(width_hint))
    bytes_per_row = max(1, width * max(1, int(itemsize)))
    chunk = max(1, target_bytes // bytes_per_row)
    return max(1, min(int(total_rows), int(chunk)))


def _load_stage2_native_module() -> Any | None:
    global _STAGE2_NATIVE_IMPORT_ATTEMPTED, _STAGE2_NATIVE_MODULE
    if _STAGE2_NATIVE_MODULE is not None:
        return _STAGE2_NATIVE_MODULE
    if _STAGE2_NATIVE_IMPORT_ATTEMPTED:
        # Re-check after failed loads so same-process rebuilds become visible.
        importlib.invalidate_caches()
    _STAGE2_NATIVE_IMPORT_ATTEMPTED = True
    try:
        _STAGE2_NATIVE_MODULE = importlib.import_module("pystamps.kernels._stage2_native")
    except Exception:
        _STAGE2_NATIVE_MODULE = None
    return _STAGE2_NATIVE_MODULE


def _native_export(name: str) -> Any | None:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        return None
    fn = getattr(native_mod, str(name), None)
    return fn if callable(fn) else None


def _native_threads(threads: int = 0) -> int:
    requested = int(threads)
    if requested > 0:
        return requested
    return max(1, os.cpu_count() or 1)


def stage2_native_available() -> bool:
    return _load_stage2_native_module() is not None


def stage4_native_available() -> bool:
    return _native_export("stage4_edge_stats") is not None


def stage7_native_available() -> bool:
    return _native_export("stage7_scla_parity") is not None


def stage8_native_available() -> bool:
    return _native_export("stage8_edge_noise") is not None


def _resolve_stage2_kernel(
    name: str,
    backend: str,
    *,
    implementations: dict[str, Any] | None = None,
) -> Any:
    requested = _resolve_stage2_kernel_backend(backend)
    try:
        return DEFAULT_REGISTRY.resolve(
            name,
            requested="auto" if requested == "auto" else requested,
            fallback_order=("native", "python") if requested == "auto" else (),
            strict_requested=requested != "auto",
            implementations=implementations,
        )
    except KernelResolutionError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _resolve_generic_kernel(
    name: str,
    backend: str,
    *,
    auto_order: tuple[str, ...] = ("python",),
    explicit_fallbacks: dict[str, tuple[str, ...]] | None = None,
    implementations: dict[str, Any] | None = None,
) -> Any:
    requested = _resolve_backend(backend)
    fallback_map = explicit_fallbacks or {}
    try:
        if requested == "auto":
            return DEFAULT_REGISTRY.resolve(
                name,
                requested="auto",
                fallback_order=auto_order,
                implementations=implementations,
            )
        return DEFAULT_REGISTRY.resolve(
            name,
            requested=requested,
            fallback_order=fallback_map.get(requested, ()),
            strict_requested=requested not in fallback_map,
            implementations=implementations,
        )
    except KernelResolutionError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _ported_stage2_module() -> Any:
    from pystamps.pipeline import ported

    return ported


def _stage2_grid_accumulate_cpu(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    ph = np.asarray(ph_weight, dtype=np.complex64)
    grid = np.asarray(grid_lin, dtype=np.int64).reshape(-1)
    if out is None:
        grid_out = np.zeros((int(n_i), int(n_j), ph.shape[1]), dtype=np.complex64)
    else:
        grid_out = out
        grid_out.fill(0)
    for i_ifg in range(ph.shape[1]):
        flat = grid_out[:, :, i_ifg].reshape(-1)
        np.add.at(flat, grid, ph[:, i_ifg])
    return grid_out


def _stage2_grid_accumulate_python(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    threads: int = 0,
) -> np.ndarray:
    return _stage2_grid_accumulate_cpu(ph_weight, grid_lin, n_i, n_j)


def _stage2_grid_accumulate_native(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    threads: int = 0,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    return np.asarray(
        native_mod.accumulate_weighted_grid(
            np.ascontiguousarray(ph_weight, dtype=np.complex64),
            np.ascontiguousarray(np.asarray(grid_lin, dtype=np.int64).reshape(-1)),
            int(n_i),
            int(n_j),
            max(1, int(threads)) if int(threads) > 0 else 1,
        ),
        dtype=np.complex64,
    )


def _stage2_histogram_with_centers_cpu(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    bins = np.asarray(centers, dtype=np.float64).reshape(-1)
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if bins.size == 0:
        return np.asarray([], dtype=np.float64)
    if bins.size == 1:
        return np.asarray([float(samples.size)], dtype=np.float64)
    diffs = np.diff(bins)
    equal_spacing = bool(
        np.all(np.abs(diffs - diffs[0]) <= np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(bins)))))
    )
    if equal_spacing:
        d = 1.0 if bins.size < 3 else float((bins[-1] - bins[0]) / (bins.size - 1))
        cutoff0 = float((bins[0] + bins[1]) / 2.0)
        assignments = 1 + np.maximum(
            0.0,
            np.minimum(
                float(bins.size - 1),
                np.ceil((samples - cutoff0) / d),
            ),
        )
        return np.bincount(assignments.astype(np.int64) - 1, minlength=bins.size).astype(np.float64)
    mids = (bins[:-1] + bins[1:]) / 2.0
    assignments = np.searchsorted(mids, samples, side="left")
    assignments = np.clip(assignments, 0, bins.size - 1)
    return np.bincount(assignments, minlength=bins.size).astype(np.float64)


def _stage2_histogram_with_centers_python(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    return _stage2_histogram_with_centers_cpu(values, centers)


def _stage2_histogram_with_centers_native(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    return np.asarray(
        native_mod.histogram_with_centers(
            np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(centers, dtype=np.float64).reshape(-1)),
        ),
        dtype=np.float64,
    )


def _stage2_row_invariant_bperp_matrix(
    bperp: np.ndarray,
    n_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    bp = np.asarray(bperp)
    real_dtype = np.float32 if bp.dtype == np.float32 else np.float64
    if bp.ndim == 1:
        bp_vec = np.ascontiguousarray(bp.reshape(-1), dtype=real_dtype)
        bp_mat = np.broadcast_to(bp_vec, (int(n_row), bp_vec.size)).astype(real_dtype, copy=True)
        return bp_vec, bp_mat
    if bp.ndim == 2:
        if bp.shape[0] not in {1, int(n_row)}:
            raise BackendUnavailableError(
                f"Row-invariant stage-2 bperp has incompatible shape {bp.shape} for n_row={n_row}"
            )
        bp_vec = np.ascontiguousarray(bp.reshape(-1) if bp.shape[0] == 1 else bp[0, :], dtype=real_dtype)
        if bp.shape[0] == int(n_row):
            return bp_vec, np.ascontiguousarray(bp, dtype=real_dtype)
        bp_mat = np.broadcast_to(bp_vec, (int(n_row), bp_vec.size)).astype(real_dtype, copy=True)
        return bp_vec, bp_mat
    raise BackendUnavailableError("Row-invariant stage-2 bperp must be a 1-D vector or 2-D matrix")


def _stage2_topofit_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _ported_stage2_module()._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps)


def _stage2_topofit_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    bperp_arr = np.asarray(bperp)
    if bperp_arr.ndim == 1:
        return _stage2_topofit_row_invariant_native(cpxphase, bperp_arr, n_trial_wraps, threads)
    if bperp_arr.ndim == 2 and bperp_arr.shape[0] > 0:
        row0 = np.ascontiguousarray(bperp_arr[0], dtype=bperp_arr.dtype)
        if np.array_equal(bperp_arr, np.broadcast_to(row0, bperp_arr.shape)):
            return _stage2_topofit_row_invariant_native(cpxphase, row0, n_trial_wraps, threads)
    cpx_arr = np.asarray(cpxphase)
    # Keep the generic path exact until the compiled solver reaches Python parity.
    return _stage2_topofit_python(cpx_arr, bperp_arr, n_trial_wraps, threads)


def _stage2_topofit_row_invariant_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    return _ported_stage2_module()._ps_topofit_batch_row_invariant(cpxphase, bperp_mat, n_trial_wraps)


def _stage2_topofit_row_invariant_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    # Keep explicit native requests exact until the compiled row-invariant solver
    # reaches Python parity again.
    return _stage2_topofit_row_invariant_python(cpxphase, bperp, n_trial_wraps, threads)


def _stage2_topofit_coh_row_invariant_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    return np.asarray(
        _ported_stage2_module()._ps_topofit_batch_row_invariant_coh(cpxphase, bperp_mat, n_trial_wraps),
        dtype=np.float64,
    )


def _stage2_topofit_coh_row_invariant_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    return _stage2_topofit_coh_row_invariant_python(cpxphase, bperp, n_trial_wraps, threads)


DEFAULT_REGISTRY.register_provider(
    "python",
    description="NumPy/Python baseline backend",
    aliases=("cpu",),
)
DEFAULT_REGISTRY.register_provider(
    "native",
    description="Compiled native backend",
    availability_probe=stage2_native_available,
    unavailable_reason="Native stage-2 extension is unavailable",
)
DEFAULT_REGISTRY.register_provider(
    "cuda",
    description="CuPy CUDA backend",
    aliases=("gpu",),
    availability_probe=_cuda_available,
    unavailable_reason="GPU backend requested but CuPy is not available",
)
DEFAULT_REGISTRY.register("stage2_grid_accumulate", python=_stage2_grid_accumulate_python, native=_stage2_grid_accumulate_native)
DEFAULT_REGISTRY.register("stage2_topofit", python=_stage2_topofit_python, native=_stage2_topofit_native)
DEFAULT_REGISTRY.register(
    "stage2_topofit_row_invariant",
    python=_stage2_topofit_row_invariant_python,
    native=_stage2_topofit_row_invariant_native,
)
DEFAULT_REGISTRY.register(
    "stage2_topofit_coh_row_invariant",
    python=_stage2_topofit_coh_row_invariant_python,
    native=_stage2_topofit_coh_row_invariant_native,
)
DEFAULT_REGISTRY.register(
    "stage2_histogram",
    python=_stage2_histogram_with_centers_python,
    native=_stage2_histogram_with_centers_native,
)


def run_stage2_grid_accumulate_kernel(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    *,
    backend: str = "auto",
    threads: int = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    resolved = _resolve_stage2_kernel("stage2_grid_accumulate", backend)
    result = resolved.fn(ph_weight, grid_lin, n_i, n_j, threads)
    if out is not None:
        out[...] = result
        return out
    return np.asarray(result, dtype=np.complex64)


def run_stage2_topofit_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit")
    if cpu_fallback is not None:
        implementations["python"] = lambda arr, bp, wraps, _threads=0: cpu_fallback(arr, bp, wraps)
    resolved = _resolve_stage2_kernel("stage2_topofit", backend, implementations=implementations)
    return resolved.fn(cpxphase, bperp, n_trial_wraps, threads)


def run_stage2_topofit_row_invariant_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit_row_invariant")
    if cpu_fallback is not None:
        _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
        implementations["python"] = lambda arr, _bp, wraps, _threads=0: cpu_fallback(arr, bperp_mat, wraps)
    resolved = _resolve_stage2_kernel("stage2_topofit_row_invariant", backend, implementations=implementations)
    return resolved.fn(cpxphase, bperp, n_trial_wraps, threads)


def run_stage2_topofit_coh_row_invariant_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit_coh_row_invariant")
    if cpu_fallback is not None:
        _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
        implementations["python"] = lambda arr, _bp, wraps, _threads=0: np.asarray(
            cpu_fallback(arr, bperp_mat, wraps), dtype=np.float64
        )
    resolved = _resolve_stage2_kernel("stage2_topofit_coh_row_invariant", backend, implementations=implementations)
    return np.asarray(resolved.fn(cpxphase, bperp, n_trial_wraps, threads), dtype=np.float64)


def run_stage2_histogram_kernel(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    finite_values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    centers_arr = np.asarray(centers, dtype=np.float64).reshape(-1)
    resolved = _resolve_stage2_kernel("stage2_histogram", backend)
    return np.asarray(resolved.fn(finite_values, centers_arr), dtype=np.float64)


def _stage4_edge_stats_core(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
) -> dict[str, np.ndarray]:
    ported = _ported_stage2_module()

    ph_arr = np.asarray(ph_weed, dtype=np.complex128)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    bperp_arr = np.asarray(bperp, dtype=np.float64).reshape(-1)
    day_arr = np.asarray(day, dtype=np.float64).reshape(-1)

    if ph_arr.ndim != 2:
        raise BackendUnavailableError("stage4_edge_stats expects a 2-D phase matrix")
    if a.shape != b.shape:
        raise BackendUnavailableError("stage4_edge_stats node arrays must have matching shapes")
    if ph_arr.shape[1] != bperp_arr.size:
        raise BackendUnavailableError("stage4_edge_stats bperp vector must match phase width")
    if not small_baseline and day_arr.size != ph_arr.shape[1]:
        raise BackendUnavailableError("stage4_edge_stats day vector must match phase width for non-small-baseline mode")

    n_node, n_use = ph_arr.shape
    ps_std = np.full(n_node, np.inf, dtype=np.float64)
    ps_max = np.full(n_node, np.inf, dtype=np.float64)
    if a.size == 0 or n_use == 0:
        return {"ps_std": ps_std, "ps_max": ps_max}

    dph_space = ph_arr[b, :] * np.conj(ph_arr[a, :])
    if not small_baseline:
        time_win_f = max(float(time_win), 1e-6)
        time_diff_all = day_arr[:, None] - day_arr[None, :]
        weight_all = np.exp(-(time_diff_all**2) / (2.0 * time_win_f**2))
        weight_sums = np.sum(weight_all, axis=1, keepdims=True)
        zero_rows = weight_sums[:, 0] <= 0
        if np.any(zero_rows):
            weight_all[zero_rows, :] = 1.0 / float(max(1, n_use))
            weight_sums = np.sum(weight_all, axis=1, keepdims=True)
        weight_all = weight_all / weight_sums
        diag_weights = np.diag(weight_all).copy()
        dph_smooth = dph_space @ weight_all.T
        dph_smooth2 = dph_smooth - (dph_space * diag_weights[None, :])

        for i1 in range(n_use):
            time_diff = time_diff_all[i1]
            weight = weight_all[i1]
            dph_mean = dph_smooth[:, i1].copy()
            dph_mean_adj = np.angle(dph_space * np.conj(dph_mean)[:, None])
            m0, m1 = ported._weighted_affine_fit(time_diff, dph_mean_adj, weight)
            detrended = dph_mean_adj - (m0[:, None] + m1[:, None] * time_diff[None, :])
            dph_mean_adj2 = np.angle(np.exp(1j * detrended))
            m20, _m21 = ported._weighted_affine_fit(time_diff, dph_mean_adj2, weight)
            dph_smooth[:, i1] = dph_mean * np.exp(1j * (m0 + m20))

        dph_noise = np.angle(dph_space * np.conj(dph_smooth)).astype(np.float64)
        dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2)).astype(np.float64)
        ddof_var = 1 if dph_noise2.shape[0] > 1 else 0
        ifg_var = np.var(dph_noise2, axis=0, ddof=ddof_var)
        w_ifg = np.divide(
            1.0,
            ifg_var,
            out=np.full_like(ifg_var, np.inf, dtype=np.float64),
            where=ifg_var != 0,
        )
        k_edge = ported._weighted_slope_fit(bperp_arr, dph_noise, w_ifg.astype(np.float64))
        dph_noise = dph_noise - k_edge[:, None] * bperp_arr[None, :]
        ddof = 1 if n_use > 1 else 0
        edge_std = np.std(dph_noise, axis=1, ddof=ddof)
        edge_max = np.max(np.abs(dph_noise), axis=1)
    else:
        ddof_var = 1 if dph_space.shape[0] > 1 else 0
        ifg_var = np.var(dph_space, axis=0, ddof=ddof_var)
        w_ifg = np.divide(
            1.0,
            ifg_var,
            out=np.full_like(ifg_var, np.inf, dtype=np.float64),
            where=ifg_var != 0,
        )
        k_edge = ported._weighted_slope_fit(bperp_arr, dph_space, w_ifg.astype(np.float64))
        dph_adj = dph_space - k_edge[:, None] * bperp_arr[None, :]
        ang = np.angle(dph_adj)
        ddof = 1 if n_use > 1 else 0
        edge_std = np.std(ang, axis=1, ddof=ddof)
        edge_max = np.max(np.abs(ang), axis=1)

    np.minimum.at(ps_std, a, edge_std)
    np.minimum.at(ps_std, b, edge_std)
    np.minimum.at(ps_max, a, edge_max)
    np.minimum.at(ps_max, b, edge_max)
    return {"ps_std": ps_std, "ps_max": ps_max}


def _stage4_edge_stats_python(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    return _stage4_edge_stats_core(ph_weed, node_a, node_b, bperp, day, time_win, small_baseline)


def _stage4_edge_stats_native(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage4_edge_stats")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_edge_stats but the compiled extension does not export it"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph_weed, dtype=np.complex128)),
        np.ascontiguousarray(np.asarray(node_a, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(node_b, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(bperp, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
        float(time_win),
        bool(small_baseline),
        _native_threads(threads),
    )
    return {
        "ps_std": np.asarray(payload["ps_std"], dtype=np.float64),
        "ps_max": np.asarray(payload["ps_max"], dtype=np.float64),
    }


def _stage7_scla_core(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
) -> dict[str, np.ndarray]:
    ported = _ported_stage2_module()

    ph_proc_arr = np.asarray(ph_proc, dtype=np.float64)
    ph_mean_v_arr = np.asarray(ph_mean_v, dtype=np.float64)
    bperp_arr = np.asarray(bperp_mat, dtype=np.float64)
    unwrap_ix_arr = np.asarray(unwrap_ix, dtype=np.int64).reshape(-1)
    solve_ix_arr = np.asarray(solve_ix, dtype=np.int64).reshape(-1)
    day_arr = np.asarray(day, dtype=np.float64).reshape(-1)
    ifg_std_arr = np.asarray(ifg_std, dtype=np.float64).reshape(-1)

    ph_seq = np.diff(ph_proc_arr[:, unwrap_ix_arr], axis=1)
    bperp_seq = np.diff(bperp_arr[:, unwrap_ix_arr], axis=1)
    day_seq = np.diff(day_arr[unwrap_ix_arr])
    coest_mean_vel = unwrap_ix_arr.size >= 4

    mean_bperp = np.mean(bperp_seq, axis=0)
    if coest_mean_vel:
        g_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp, day_seq))
    else:
        g_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp))
    coeffs_seq = ported._weighted_lstsq_shared_design(g_seq, ph_seq.T, cov=None)
    k_ps_uw = coeffs_seq[1, :].astype(np.float64)
    ph_scla = (k_ps_uw[:, None] * bperp_arr).astype(np.float32)

    ifg_vcm = np.diag((ifg_std_arr * np.pi / 180.0) ** 2).astype(np.float64)
    resid_full = ph_proc_arr[:, solve_ix_arr] - ph_scla[:, solve_ix_arr].astype(np.float64)
    if coest_mean_vel:
        g_c = np.column_stack(
            (
                np.ones(solve_ix_arr.size, dtype=np.float64),
                day_arr[solve_ix_arr] - day_arr[int(master_ix) - 1],
            )
        )
        coeffs_c = ported._weighted_lstsq_shared_design(g_c, resid_full.T, cov=ifg_vcm[np.ix_(solve_ix_arr, solve_ix_arr)])
        c_ps_uw = coeffs_c[0, :].astype(np.float32)
    else:
        c_ps_uw = np.mean(resid_full, axis=1).astype(np.float32)

    m = np.asarray(ported._stage7_mean_velocity_fit(ph_mean_v_arr, day_arr, master_ix, ifg_std_arr), dtype=np.float32)
    return {
        "K_ps_uw": k_ps_uw,
        "C_ps_uw": c_ps_uw,
        "ph_scla": ph_scla,
        "ph_ramp": np.zeros_like(ph_proc_arr, dtype=np.float64),
        "ifg_vcm": ifg_vcm,
        "mean_v": m[1, :].astype(np.float32),
        "m": m,
    }


def _stage7_scla_cpu(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    return _stage7_scla_core(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std)


def _stage7_scla_gpu(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    cp = _cupy()
    if cp is None:
        raise BackendUnavailableError("GPU backend requested but CuPy is not available")
    return _stage7_scla_core(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std)


def _stage7_scla_native(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage7_scla_parity")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_scla but the compiled extension does not export stage7_scla_parity"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph_proc, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(ph_mean_v, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(bperp_mat, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(unwrap_ix, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(solve_ix, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
        int(master_ix),
        np.ascontiguousarray(np.asarray(ifg_std, dtype=np.float64).reshape(-1)),
        _native_threads(),
    )
    return {
        "K_ps_uw": np.asarray(payload["K_ps_uw"], dtype=np.float64),
        "C_ps_uw": np.asarray(payload["C_ps_uw"], dtype=np.float32),
        "ph_scla": np.asarray(payload["ph_scla"], dtype=np.float32),
        "ph_ramp": np.asarray(payload["ph_ramp"], dtype=np.float64),
        "ifg_vcm": np.asarray(payload["ifg_vcm"], dtype=np.float64),
        "mean_v": np.asarray(payload["mean_v"], dtype=np.float32),
        "m": np.asarray(payload["m"], dtype=np.float32),
    }


def _stage8_edge_noise_cpu(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    ph = np.asarray(uw_ph, dtype=np.complex64)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    n_edge = int(a.size)
    n_ifg = ph.shape[1]

    if chunk_edges <= 0:
        chunk_edges = _auto_chunk_size(n_edge, max(1, n_ifg * 3), np.dtype(np.float32).itemsize)
    chunk_edges = max(1, int(chunk_edges))

    dph_space_uw = np.empty((n_edge, n_ifg), dtype=np.float32)
    dph_noise = np.empty((n_edge, n_ifg), dtype=np.float32)
    for start in range(0, n_edge, chunk_edges):
        end = min(start + chunk_edges, n_edge)
        dph_space = np.angle(ph[b[start:end], :] * np.conj(ph[a[start:end], :])).astype(np.float32)
        dph_space_uw[start:end, :] = dph_space
        dph_noise[start:end, :] = (
            (dph_space - np.mean(dph_space, axis=1, keepdims=True)) * _STAGE8_NOISE_SCALE
        ).astype(np.float32)
    return {"dph_noise": dph_noise, "dph_space_uw": dph_space_uw}


def _stage8_edge_noise_gpu(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    cp = _cupy()
    if cp is None:
        raise BackendUnavailableError("GPU backend requested but CuPy is not available")

    ph = cp.asarray(uw_ph, dtype=cp.complex64)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    n_edge = int(a.size)
    n_ifg = ph.shape[1]

    if chunk_edges <= 0:
        chunk_edges = _auto_chunk_size(n_edge, max(1, n_ifg * 6), np.dtype(np.float32).itemsize)
    chunk_edges = max(1, int(chunk_edges))

    dph_space_uw = np.empty((n_edge, n_ifg), dtype=np.float32)
    dph_noise = np.empty((n_edge, n_ifg), dtype=np.float32)
    for start in range(0, n_edge, chunk_edges):
        end = min(start + chunk_edges, n_edge)
        a_c = cp.asarray(a[start:end], dtype=cp.int64)
        b_c = cp.asarray(b[start:end], dtype=cp.int64)
        dph_space = cp.angle(ph[b_c, :] * cp.conj(ph[a_c, :])).astype(cp.float32)
        dph_space_np = _to_numpy(dph_space).astype(np.float32)
        dph_space_uw[start:end, :] = dph_space_np
        dph_noise[start:end, :] = (
            (dph_space_np - np.mean(dph_space_np, axis=1, keepdims=True)) * _STAGE8_NOISE_SCALE
        ).astype(np.float32)
    return {"dph_noise": dph_noise, "dph_space_uw": dph_space_uw}


def _stage8_edge_noise_native(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage8_edge_noise")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage8_edge_noise but the compiled extension does not export it"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(uw_ph, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(node_a, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(node_b, dtype=np.int64).reshape(-1)),
        int(chunk_edges),
        _native_threads(),
    )
    return {
        "dph_noise": np.asarray(payload["dph_noise"], dtype=np.float32),
        "dph_space_uw": np.asarray(payload["dph_space_uw"], dtype=np.float32),
    }


DEFAULT_REGISTRY.register("stage7_scla", python=_stage7_scla_cpu, native=_stage7_scla_native, cuda=_stage7_scla_gpu)
DEFAULT_REGISTRY.register("stage4_edge_stats", python=_stage4_edge_stats_python, native=_stage4_edge_stats_native)
DEFAULT_REGISTRY.register(
    "stage8_edge_noise",
    python=_stage8_edge_noise_cpu,
    native=_stage8_edge_noise_native,
    cuda=_stage8_edge_noise_gpu,
)


def run_stage7_scla_kernel(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    backend: str = "auto",
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage7_scla")
    if not stage7_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_scla",
        backend,
        auto_order=("native", "python"),
        explicit_fallbacks={"native": ("python",), "cuda": ("python",)},
        implementations=implementations,
    )
    return resolved.fn(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std, chunk_ps)


def run_stage4_edge_stats_kernel(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage4_edge_stats")
    if not stage4_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_edge_stats",
        backend,
        auto_order=("native", "python"),
        explicit_fallbacks={"native": ("python",), "cuda": ("python",)},
        implementations=implementations,
    )
    return resolved.fn(ph_weed, node_a, node_b, bperp, day, time_win, small_baseline, threads)


def run_stage8_edge_noise_kernel(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    backend: str = "auto",
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage8_edge_noise")
    if not stage8_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage8_edge_noise",
        backend,
        auto_order=("native", "python"),
        explicit_fallbacks={"native": ("python",)},
        implementations=implementations,
    )
    return resolved.fn(uw_ph, node_a, node_b, chunk_edges)


def describe_backend_matrix() -> dict[str, Any]:
    manifest = DEFAULT_REGISTRY.coverage_manifest()
    kernels = manifest.get("kernels", {})
    stage4_kernel = kernels.get("stage4_edge_stats")
    if stage4_kernel is not None and not stage4_native_available():
        stage4_kernel["available_backends"] = [backend for backend in stage4_kernel["available_backends"] if backend != "native"]
    stage7_kernel = kernels.get("stage7_scla")
    if stage7_kernel is not None and not stage7_native_available():
        stage7_kernel["available_backends"] = [backend for backend in stage7_kernel["available_backends"] if backend != "native"]
    stage8_kernel = kernels.get("stage8_edge_noise")
    if stage8_kernel is not None and not stage8_native_available():
        stage8_kernel["available_backends"] = [backend for backend in stage8_kernel["available_backends"] if backend != "native"]
    return manifest
