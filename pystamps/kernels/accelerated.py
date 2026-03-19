from __future__ import annotations

from typing import Any

import numpy as np

from pystamps.kernels.registry import DEFAULT_REGISTRY

_STAGE8_NOISE_SCALE = np.float32(0.5)


class BackendUnavailableError(RuntimeError):
    """Raised when a requested compute backend is not available."""


def _cupy() -> Any | None:
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None


def _resolve_backend(backend: str) -> str:
    name = (backend or "auto").strip().lower()
    if name in {"threads", "thread", "io", "processes", "process", "cpu", "auto"}:
        return "cpu"
    if name in {"native"}:
        return "native"
    if name in {"gpu"}:
        return "gpu"
    return "cpu"


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


def _stage7_scla_cpu(
    ph_uw: np.ndarray,
    bperp_mat: np.ndarray,
    no_master: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    ph = np.asarray(ph_uw, dtype=np.float32)
    b = np.asarray(bperp_mat, dtype=np.float32)
    no_master_b = np.asarray(no_master, dtype=bool)
    n_ps, n_ifg = ph.shape
    k = int(np.sum(no_master_b))

    if chunk_ps <= 0:
        chunk_ps = _auto_chunk_size(n_ps, n_ifg + (2 * k) + b.shape[1], np.dtype(np.float64).itemsize)
    chunk_ps = max(1, int(chunk_ps))

    K_ps_uw = np.zeros((n_ps,), dtype=np.float64)
    C_ps_uw = np.zeros((n_ps,), dtype=np.float32)
    ph_scla = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_ramp = np.zeros((n_ps, n_ifg), dtype=np.float64)
    mean_v = np.zeros((n_ps,), dtype=np.float32)

    sum_res = np.zeros((k,), dtype=np.float64)
    sum_outer = np.zeros((k, k), dtype=np.float64)
    count = 0

    day_f = np.asarray(day, dtype=np.float64).reshape(-1)
    t = day_f - day_f[int(master_ix) - 1]
    denom_t = float(np.sum(t * t))
    if denom_t == 0.0:
        denom_t = 1.0

    for start in range(0, n_ps, chunk_ps):
        end = min(start + chunk_ps, n_ps)
        ph_c = ph[start:end]
        b_c = b[start:end].astype(np.float64)
        y = ph_c[:, no_master_b].astype(np.float64)

        denom = np.sum(b_c * b_c, axis=1)
        denom[denom == 0] = 1.0
        K_c = np.sum(b_c * y, axis=1) / denom
        C_c = np.mean(y - K_c[:, None] * b_c, axis=1).astype(np.float32)
        ph_scla_nm = (K_c[:, None] * b_c).astype(np.float32)
        resid = y - ph_scla_nm

        K_ps_uw[start:end] = K_c
        C_ps_uw[start:end] = C_c
        ph_scla[start:end, no_master_b] = ph_scla_nm
        mean_v[start:end] = (ph_c @ t / denom_t).astype(np.float32)

        if k > 0:
            sum_res += np.sum(resid, axis=0)
            sum_outer += resid.T @ resid
            count += (end - start)

    cov_nm = _cov_from_accumulators(sum_res, sum_outer, count)
    ifg_vcm = np.zeros((n_ifg, n_ifg), dtype=np.float64)
    ix = np.where(no_master_b)[0]
    if k > 0:
        ifg_vcm[np.ix_(ix, ix)] = cov_nm

    m = np.vstack((mean_v, np.zeros_like(mean_v, dtype=np.float32)))
    return {
        "K_ps_uw": K_ps_uw,
        "C_ps_uw": C_ps_uw,
        "ph_scla": ph_scla,
        "ph_ramp": ph_ramp,
        "ifg_vcm": ifg_vcm,
        "mean_v": mean_v,
        "m": m,
    }


def _stage7_scla_gpu(
    ph_uw: np.ndarray,
    bperp_mat: np.ndarray,
    no_master: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    cp = _cupy()
    if cp is None:
        raise BackendUnavailableError("GPU backend requested but CuPy is not available")

    ph = np.asarray(ph_uw, dtype=np.float32)
    b = np.asarray(bperp_mat, dtype=np.float32)
    no_master_b = np.asarray(no_master, dtype=bool)
    n_ps, n_ifg = ph.shape
    k = int(np.sum(no_master_b))

    if chunk_ps <= 0:
        chunk_ps = _auto_chunk_size(n_ps, n_ifg + (2 * k) + b.shape[1], np.dtype(np.float64).itemsize)
    chunk_ps = max(1, int(chunk_ps))

    K_ps_uw = np.zeros((n_ps,), dtype=np.float64)
    C_ps_uw = np.zeros((n_ps,), dtype=np.float32)
    ph_scla = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_ramp = np.zeros((n_ps, n_ifg), dtype=np.float64)
    mean_v = np.zeros((n_ps,), dtype=np.float32)

    no_master_gpu = cp.asarray(no_master_b, dtype=cp.bool_)
    sum_res = cp.zeros((k,), dtype=cp.float64)
    sum_outer = cp.zeros((k, k), dtype=cp.float64)
    count = 0

    day_gpu = cp.asarray(np.asarray(day, dtype=np.float64).reshape(-1), dtype=cp.float64)
    t = day_gpu - day_gpu[int(master_ix) - 1]
    denom_t = cp.sum(t * t)
    if float(denom_t) == 0.0:
        denom_t = cp.asarray(1.0, dtype=cp.float64)

    for start in range(0, n_ps, chunk_ps):
        end = min(start + chunk_ps, n_ps)
        ph_c = cp.asarray(ph[start:end], dtype=cp.float32)
        b_c = cp.asarray(b[start:end], dtype=cp.float32).astype(cp.float64)
        y = ph_c[:, no_master_gpu].astype(cp.float64)

        denom = cp.sum(b_c * b_c, axis=1)
        denom = cp.where(denom == 0, 1.0, denom)
        K_c = cp.sum(b_c * y, axis=1) / denom
        C_c = cp.mean(y - K_c[:, None] * b_c, axis=1).astype(cp.float32)
        ph_scla_nm = (K_c[:, None] * b_c).astype(cp.float32)
        resid = y - ph_scla_nm

        K_ps_uw[start:end] = _to_numpy(K_c).astype(np.float64)
        C_ps_uw[start:end] = _to_numpy(C_c).astype(np.float32)
        ph_scla[start:end, no_master_b] = _to_numpy(ph_scla_nm).astype(np.float32)
        mean_v[start:end] = _to_numpy((ph_c @ t / denom_t).astype(cp.float32)).astype(np.float32)

        if k > 0:
            sum_res += cp.sum(resid, axis=0)
            sum_outer += resid.T @ resid
            count += (end - start)

    cov_nm = _cov_from_accumulators(_to_numpy(sum_res), _to_numpy(sum_outer), count)
    ifg_vcm = np.zeros((n_ifg, n_ifg), dtype=np.float64)
    ix = np.where(no_master_b)[0]
    if k > 0:
        ifg_vcm[np.ix_(ix, ix)] = cov_nm

    m = np.vstack((mean_v, np.zeros_like(mean_v, dtype=np.float32)))
    return {
        "K_ps_uw": K_ps_uw,
        "C_ps_uw": C_ps_uw,
        "ph_scla": ph_scla,
        "ph_ramp": ph_ramp,
        "ifg_vcm": ifg_vcm,
        "mean_v": mean_v,
        "m": m,
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


DEFAULT_REGISTRY.register("stage7_scla", cpu=_stage7_scla_cpu, gpu=_stage7_scla_gpu, native=_stage7_scla_cpu)
DEFAULT_REGISTRY.register(
    "stage8_edge_noise", cpu=_stage8_edge_noise_cpu, gpu=_stage8_edge_noise_gpu, native=_stage8_edge_noise_cpu
)


def run_stage7_scla_kernel(
    ph_uw: np.ndarray,
    bperp_mat: np.ndarray,
    no_master: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    backend: str = "auto",
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    selected = _resolve_backend(backend)
    fn = DEFAULT_REGISTRY.get("stage7_scla", backend=selected)
    return fn(ph_uw, bperp_mat, no_master, day, master_ix, chunk_ps)


def run_stage8_edge_noise_kernel(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    backend: str = "auto",
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    selected = _resolve_backend(backend)
    fn = DEFAULT_REGISTRY.get("stage8_edge_noise", backend=selected)
    return fn(uw_ph, node_a, node_b, chunk_edges)
