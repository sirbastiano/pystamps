from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, spatial
from scipy import ndimage
from scipy import signal

from pystamps.io.mat import read_mat, write_mat
from pystamps.kernels import BackendUnavailableError, run_stage8_edge_noise_kernel


class PortedStageError(RuntimeError):
    """Raised when a ported stage cannot run due to missing inputs."""


_CANONICAL_STAGE2_WEIGHTING_SNAPSHOT = Path("inputs_and_outputs/validation_runs/stage2_weighting_snapshot.json")


@dataclass(slots=True)
class StageOptions:
    grid_size: float = 50.0
    clap_win: float = 32.0
    clap_low_pass_wavelength: float = 800.0
    clap_alpha: float = 1.0
    clap_beta: float = 0.3
    max_topo_err: float = 15.0
    lambda_m: float = 0.0555
    mean_range: float = 830000.0
    mean_incidence: float = np.deg2rad(23.0)


@dataclass(slots=True)
class Parms:
    select_method: str = "PERCENT"
    percent_rand: float = 1.0
    density_rand: float = 1.0
    small_baseline_flag: str = "n"
    drop_ifg_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int64))
    weed_standard_dev: float = np.pi
    weed_max_noise: float = np.pi
    weed_zero_elevation: str = "n"
    weed_neighbours: str = "y"
    gamma_stdev_reject: float = 0.0
    slc_osf: float = 1.0
    weed_time_win: float = 360.0


@dataclass(slots=True)
class Stage5PatchBundle:
    patch: Path
    ps: dict[str, Any]
    n_ps_patch: int
    ij_patch: np.ndarray
    lonlat_patch: np.ndarray
    ph_patch2: np.ndarray
    k_patch: np.ndarray
    c_patch: np.ndarray
    coh_patch: np.ndarray
    ph_patch_patch: np.ndarray
    ph_res_patch: np.ndarray
    ij_cols: np.ndarray
    ij_keys: list[bytes]
    patch_bounds: tuple[int, int, int, int] | None
    bp_patch: np.ndarray | None = None
    hgt_patch: np.ndarray | None = None
    la_patch: np.ndarray | None = None
    rc_patch: np.ndarray | None = None


@dataclass(slots=True)
class Stage1MetadataResolution:
    day_file: Path
    master_day_file: Path
    bperp_file: Path
    synthesized: bool = False
    bperp_mat: np.ndarray | None = None


_DATE_PAIR_RE = re.compile(r"(?P<master>\d{8})_(?P<slave>\d{8})")


def _resolve_file(patch_dir: Path, filename: str) -> Path | None:
    candidates = [
        patch_dir / filename,
        patch_dir.parent / filename,
        patch_dir.parent.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _stage1_dataset_root(path: Path) -> Path:
    if path.name.startswith("PATCH_"):
        return path.parent
    return path


def _parse_date_pair_from_name(name: str) -> tuple[str, str] | None:
    match = _DATE_PAIR_RE.search(name)
    if match is None:
        return None
    return match.group("master"), match.group("slave")


def _extract_float_tokens(text: str) -> list[float]:
    values: list[float] = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _read_named_float_vector(path: Path, key: str, count: int) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.split(":", 1)[0].strip() != key:
                continue
            values = _extract_float_tokens(line.split(":", 1)[1])
            if len(values) < count:
                break
            return np.asarray(values[:count], dtype=np.float64)
    raise PortedStageError(f"Unable to parse '{key}' from {path}")


def _read_named_scalar(path: Path, key: str) -> float:
    return float(_read_named_float_vector(path, key, 1)[0])


def _write_lines_if_missing(path: Path, values: list[str]) -> None:
    if path.exists():
        return
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(values))
            handle.write("\n")
    except FileExistsError:
        return


def _snap_ifg_records(dataset_root: Path) -> list[tuple[str, str, Path]]:
    diff_dir = dataset_root / "diff0"
    if not diff_dir.exists():
        raise PortedStageError(f"Stage 1 SNAP metadata synthesis requires {diff_dir}")

    records: list[tuple[str, str, Path]] = []
    for base_file in sorted(diff_dir.glob("*.base")):
        pair = _parse_date_pair_from_name(base_file.name)
        if pair is None:
            continue
        records.append((pair[0], pair[1], base_file))
    if not records:
        raise PortedStageError(f"Stage 1 SNAP metadata synthesis requires parseable diff0/*.base files in {diff_dir}")
    return records


def _resolve_rslc_par(dataset_root: Path, master_day: str) -> Path:
    rslc_dir = dataset_root / "rslc"
    preferred = rslc_dir / f"{master_day}.rslc.par"
    if preferred.exists():
        return preferred

    candidates = sorted(rslc_dir.glob("*.rslc.par"))
    if len(candidates) == 1:
        return candidates[0]
    raise PortedStageError(
        "Stage 1 SNAP metadata synthesis requires rslc/*.rslc.par with a file matching master date "
        f"{master_day} under {rslc_dir}"
    )


def _snap_patch_bperp_vector(base_file: Path, rslc_par: Path, ij: np.ndarray) -> np.ndarray:
    b_tcn = _read_named_float_vector(base_file, "initial_baseline(TCN)", 3)
    br_tcn = _read_named_float_vector(base_file, "initial_baseline_rate", 3)
    range_pixel_spacing = _read_named_scalar(rslc_par, "range_pixel_spacing")
    near_range_slc = _read_named_scalar(rslc_par, "near_range_slc")
    sar_to_earth_center = _read_named_scalar(rslc_par, "sar_to_earth_center")
    earth_radius_below_sensor = _read_named_scalar(rslc_par, "earth_radius_below_sensor")
    azimuth_lines = _read_named_scalar(rslc_par, "azimuth_lines")
    prf = _read_named_scalar(rslc_par, "prf")
    if prf == 0.0:
        raise PortedStageError(f"Invalid PRF in {rslc_par}")

    mean_az = azimuth_lines / 2.0 - 0.5
    azimuth = np.asarray(ij[:, 1], dtype=np.float64)
    rg = near_range_slc + np.asarray(ij[:, 2], dtype=np.float64) * range_pixel_spacing
    look_arg = (sar_to_earth_center**2 + rg**2 - earth_radius_below_sensor**2) / (2.0 * sar_to_earth_center * rg)
    look = np.arccos(np.clip(look_arg, -1.0, 1.0))

    bc = b_tcn[1] + br_tcn[1] * (azimuth - mean_az) / prf
    bn = b_tcn[2] + br_tcn[2] * (azimuth - mean_az) / prf
    return (bc * np.cos(look) - bn * np.sin(look)).astype(np.float32)


def resolve_stage1_metadata(patch_dir: Path, ij: np.ndarray) -> Stage1MetadataResolution:
    day_file = _resolve_file(patch_dir, "day.1.in")
    master_day_file = _resolve_file(patch_dir, "master_day.1.in")
    bperp_file = _resolve_file(patch_dir, "bperp.1.in")
    if day_file is not None and master_day_file is not None and bperp_file is not None:
        return Stage1MetadataResolution(day_file=day_file, master_day_file=master_day_file, bperp_file=bperp_file)

    dataset_root = _stage1_dataset_root(patch_dir)
    records = _snap_ifg_records(dataset_root)
    master_days = sorted({master for master, _, _ in records})
    if len(master_days) != 1:
        raise PortedStageError(
            "Stage 1 SNAP metadata synthesis requires a single-master diff0 stack; "
            f"found masters {', '.join(master_days)}"
        )

    master_day = master_days[0]
    rslc_par = _resolve_rslc_par(dataset_root, master_day)
    day_values = [slave for _, slave, _ in records]
    bperp_cols = [_snap_patch_bperp_vector(base_file, rslc_par, ij) for _, _, base_file in records]
    if not bperp_cols:
        raise PortedStageError("Stage 1 SNAP metadata synthesis did not produce any perpendicular baselines")
    bperp_mat = np.column_stack(bperp_cols).astype(np.float32)
    bperp_mean = np.mean(bperp_mat.astype(np.float64), axis=0)

    day_file = patch_dir / "day.1.in"
    master_day_file = patch_dir / "master_day.1.in"
    bperp_file = patch_dir / "bperp.1.in"
    _write_lines_if_missing(day_file, day_values)
    _write_lines_if_missing(master_day_file, [master_day])
    _write_lines_if_missing(bperp_file, [f"{value:.12f}" for value in bperp_mean.tolist()])
    return Stage1MetadataResolution(
        day_file=day_file,
        master_day_file=master_day_file,
        bperp_file=bperp_file,
        synthesized=True,
        bperp_mat=bperp_mat,
    )


def _read_mat_cached(path: Path, cache: dict[Path, dict[str, Any]], enabled: bool = True) -> dict[str, Any]:
    key = path.resolve()
    if enabled and key in cache:
        return cache[key]
    payload = read_mat(key)
    if enabled:
        cache[key] = payload
    return payload


def _cache_mat_payload(path: Path, payload: dict[str, Any], cache: dict[Path, dict[str, Any]], enabled: bool = True) -> None:
    if enabled:
        cache[path.resolve()] = payload


def _resolve_io_workers(io_workers: int, item_count: int) -> int:
    requested = int(io_workers) if io_workers and io_workers > 0 else min(8, max(1, os.cpu_count() or 4))
    return max(1, min(int(item_count), requested))


def _row_keys(rows: np.ndarray) -> list[bytes]:
    arr = np.ascontiguousarray(rows)
    if arr.ndim != 2:
        raise PortedStageError("row key generation expects a 2-D array")
    view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).reshape(-1)
    return [bytes(v) for v in view.tolist()]


def _group_reduce_by_index(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.asarray(indices, dtype=np.int64).reshape(-1)
    arr = np.asarray(values)
    if ix.size == 0:
        empty_cols = arr.shape[1:] if arr.ndim > 1 else ()
        return np.empty((0,), dtype=np.int64), np.empty((0, *empty_cols), dtype=arr.dtype)

    order = np.argsort(ix, kind="mergesort")
    ix_sorted = ix[order]
    arr_sorted = arr[order]
    group_start = np.concatenate(([0], np.flatnonzero(ix_sorted[1:] != ix_sorted[:-1]) + 1))
    reduced = np.add.reduceat(arr_sorted, group_start, axis=0)
    return ix_sorted[group_start].astype(np.int64), np.asarray(reduced, dtype=arr.dtype)


def _accumulate_grid_column(group_ix: np.ndarray, grouped_values: np.ndarray, n_cells: int) -> np.ndarray:
    flat = np.zeros(int(n_cells), dtype=np.complex64)
    if group_ix.size > 0:
        flat[np.asarray(group_ix, dtype=np.int64)] = np.asarray(grouped_values, dtype=np.complex64)
    return flat


def _apply_selector_all(selector: np.ndarray, *arrays: np.ndarray | None) -> tuple[np.ndarray | None, ...]:
    out: list[np.ndarray | None] = []
    sel = np.asarray(selector)
    for arr in arrays:
        if arr is None:
            out.append(None)
            continue
        out.append(np.asarray(arr)[sel, ...])
    return tuple(out)


def _load_text_matrix(path: Path, dtype=float) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype)
    if isinstance(values, np.ndarray):
        return values
    return np.asarray([values], dtype=dtype)


def _binary_float32_endian(path: Path, kind: str) -> str:
    sample_count = min(max(32, path.stat().st_size // 4), 512)
    sample_le = np.fromfile(path, dtype="<f4", count=sample_count)
    sample_be = np.fromfile(path, dtype=">f4", count=sample_count)

    def _score(arr: np.ndarray) -> tuple[float, float]:
        finite = np.isfinite(arr)
        finite_ratio = float(np.mean(finite)) if arr.size else 0.0
        if not finite.any():
            return (-1.0, -np.inf)
        arr_f = np.asarray(arr[finite], dtype=np.float64)
        if kind == "lonlat":
            usable = arr_f[: (arr_f.size // 2) * 2]
            if usable.size == 0:
                return (finite_ratio, -np.inf)
            pairs = usable.reshape(-1, 2)
            plausible = np.logical_and(np.abs(pairs[:, 0]) <= 180.0, np.abs(pairs[:, 1]) <= 90.0)
            return (finite_ratio + float(np.mean(plausible)), -float(np.nanmedian(np.abs(pairs))))
        abs_arr = np.abs(arr_f)
        plausible = np.logical_or(abs_arr == 0.0, np.logical_and(abs_arr >= 1e-12, abs_arr <= 1e12))
        return (finite_ratio + float(np.mean(plausible)), -float(np.nanmedian(abs_arr)))

    return ">f4" if _score(sample_be) > _score(sample_le) else "<f4"


def _load_binary_float32(path: Path, kind: str) -> np.ndarray:
    dtype = _binary_float32_endian(path, kind)
    return np.fromfile(path, dtype=dtype).astype(np.float32, copy=False)


def _coerce_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1)


def _coerce_complex(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.names and {"real", "imag"}.issubset(set(arr.dtype.names)):
        return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    return np.asarray(arr, dtype=np.complex64)


def _mat_scalar(values: Any, default: float) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _mat_text(values: Any, default: str) -> str:
    if values is None:
        return default
    if isinstance(values, str):
        text = values
    else:
        arr = np.asarray(values)
        if arr.size == 0:
            return default
        if arr.dtype.kind in {"u", "i"}:
            chars = [chr(int(v)) for v in arr.reshape(-1) if int(v) != 0]
            text = "".join(chars)
        elif arr.dtype.kind in {"U", "S"}:
            text = "".join(str(v) for v in arr.reshape(-1))
        else:
            text = str(arr.reshape(-1)[0])
    text = text.strip()
    return text if text else default


def _matlab_col(values: Any, dtype: np.dtype[Any] | type[np.generic] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return arr.reshape(-1, 1)


def _matlab_row(values: Any, dtype: np.dtype[Any] | type[np.generic] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return arr.reshape(1, -1)


def _matlab_char_row(text: str) -> np.ndarray:
    if not text:
        return np.empty((1, 0), dtype=np.uint16)
    return np.fromiter((ord(ch) for ch in text), dtype=np.uint16).reshape(1, -1)


def _build_stage_options(patch_dir: Path) -> StageOptions:
    options = StageOptions()
    parms_file = _resolve_file(patch_dir, "parms.mat")
    if parms_file is None:
        return options

    try:
        parms = read_mat(parms_file)
    except Exception:
        return options

    options.grid_size = _mat_scalar(parms.get("filter_grid_size", options.grid_size), options.grid_size)
    options.clap_win = _mat_scalar(parms.get("clap_win", options.clap_win), options.clap_win)
    options.clap_low_pass_wavelength = _mat_scalar(
        parms.get("clap_low_pass_wavelength", options.clap_low_pass_wavelength), options.clap_low_pass_wavelength
    )
    options.clap_alpha = _mat_scalar(parms.get("clap_alpha", options.clap_alpha), options.clap_alpha)
    options.clap_beta = _mat_scalar(parms.get("clap_beta", options.clap_beta), options.clap_beta)
    options.max_topo_err = _mat_scalar(parms.get("max_topo_err", options.max_topo_err), options.max_topo_err)
    options.lambda_m = _mat_scalar(parms.get("lambda", options.lambda_m), options.lambda_m)
    return options


def _normalize_drop_index(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.int64)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=np.int64)
    arr = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
    return arr.astype(np.int64)


def _load_parms(patch_dir: Path) -> Parms:
    parms_file = _resolve_file(patch_dir, "parms.mat")
    if parms_file is None:
        return Parms()

    try:
        raw = read_mat(parms_file)
    except Exception:
        return Parms()

    return Parms(
        select_method=_mat_text(raw.get("select_method", "PERCENT"), "PERCENT"),
        percent_rand=_mat_scalar(raw.get("percent_rand", 1.0), 1.0),
        density_rand=_mat_scalar(raw.get("density_rand", 1.0), 1.0),
        small_baseline_flag=_mat_text(raw.get("small_baseline_flag", "n"), "n"),
        drop_ifg_index=_normalize_drop_index(raw.get("drop_ifg_index", None)),
        weed_standard_dev=_mat_scalar(raw.get("weed_standard_dev", np.pi), np.pi),
        weed_max_noise=_mat_scalar(raw.get("weed_max_noise", np.pi), np.pi),
        weed_zero_elevation=_mat_text(raw.get("weed_zero_elevation", "n"), "n"),
        weed_neighbours=_mat_text(raw.get("weed_neighbours", "y"), "y"),
        gamma_stdev_reject=_mat_scalar(raw.get("gamma_stdev_reject", 0.0), 0.0),
        slc_osf=_mat_scalar(raw.get("slc_osf", 1.0), 1.0),
        weed_time_win=_mat_scalar(raw.get("weed_time_win", 360.0), 360.0),
    )


def _hist_with_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if centers.size == 0:
        return np.asarray([], dtype=np.float64)
    if centers.size == 1:
        return np.asarray([float(values.size)], dtype=np.float64)
    mids = (centers[:-1] + centers[1:]) / 2.0
    assignments = np.searchsorted(mids, values, side="left")
    assignments = np.clip(assignments, 0, centers.size - 1)
    return np.bincount(assignments, minlength=centers.size).astype(np.float64)


class _MatlabV5UniformRNG:
    """MATLAB rand('state', seed) / rng(seed, 'v5uniform') generator."""

    _ULP = 2.0**-53
    _MASK32 = (1 << 32) - 1
    _MASK52 = (1 << 52) - 1

    def __init__(self, seed: int) -> None:
        self._index = 0
        self._borrow = 0.0
        self._j = int(seed) if int(seed) != 0 else 2**31
        self._state = self._randsetup(32, self._j)

    @classmethod
    def _randint32(cls, value: int) -> int:
        value &= cls._MASK32
        value ^= (value << 13) & cls._MASK32
        value ^= value >> 17
        value ^= (value << 5) & cls._MASK32
        return value & cls._MASK32

    def _randsetup(self, n: int, seed: int) -> np.ndarray:
        state = np.empty(n, dtype=np.float64)
        j = seed
        for idx in range(n):
            x = 0
            for _ in range(53):
                j = self._randint32(j)
                x = (x << 1) | ((j >> 19) & 1)
            state[idx] = math.ldexp(x, -53)
        return state

    def _randbits(self, value: float) -> float:
        jlo = self._j & self._MASK32
        jhi = self._randint32(jlo)
        self._j = jhi
        mask = ((jhi << 32) & self._MASK52) ^ jlo
        frac, exp = math.frexp(value)
        mantissa = int(math.ldexp(frac, 53))
        return math.ldexp(mantissa ^ mask, exp - 53)

    def uniform(self, size: int | tuple[int, ...]) -> np.ndarray:
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(int(dim) for dim in size)
        out = np.empty(int(np.prod(shape, dtype=np.int64)), dtype=np.float64)
        for idx in range(out.size):
            value = (
                self._state[(self._index + 20) & 31]
                - self._state[(self._index + 5) & 31]
                - self._borrow
            )
            if value < 0.0:
                value += 1.0
                self._borrow = self._ULP
            else:
                self._borrow = 0.0
            self._state[self._index] = value
            self._index = (self._index + 1) & 31
            out[idx] = self._randbits(value)
        return out.reshape(shape)


def _polyfit_eval_centered(x: np.ndarray, y: np.ndarray, deg: int, x_eval: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.nan
    mu0 = float(np.mean(x))
    mu1 = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    if not np.isfinite(mu1) or mu1 == 0:
        mu1 = 1.0
    x_scaled = (x - mu0) / mu1
    coeff = np.polyfit(x_scaled, y, deg)
    x0_scaled = (float(x_eval) - mu0) / mu1
    return float(np.polyval(coeff, x0_scaled))


def _clap_filter_kernel() -> np.ndarray:
    # MATLAB gausswin(7) default alpha=2.5.
    alpha = 2.5
    std = (7 - 1) / (2.0 * alpha)
    g = signal.windows.gaussian(7, std=std, sym=True)
    return np.outer(g, g).astype(np.float64)


def _clap_filt_patch(ph: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
    ph = np.asarray(ph, dtype=np.complex64).copy()
    ph[np.isnan(ph)] = 0
    ph_fft = np.fft.fft2(ph)
    H = np.abs(ph_fft)

    B = _clap_filter_kernel()
    H = np.fft.ifftshift(
        signal.convolve2d(np.fft.fftshift(H), B, mode="same", boundary="fill", fillvalue=0.0)
    )
    mean_h = float(np.median(H))
    if mean_h != 0.0:
        H = H / mean_h
    H = np.power(H, float(alpha))
    H = H - 1.0
    H[H < 0.0] = 0.0

    G = H * float(beta) + np.asarray(low_pass, dtype=np.float64)
    return np.fft.ifft2(ph_fft * G).astype(np.complex64)


def _clap_filt_grid(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
) -> np.ndarray:
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    if ph_arr.ndim != 2:
        raise PortedStageError("clap_filt_grid expects a 2-D complex grid")

    n_win_int = int(round(n_win))
    if n_win_int <= 0:
        n_win_int = 32
    n_pad_int = int(round(n_pad))
    n_i, n_j = ph_arr.shape
    ph_out = np.zeros_like(ph_arr)
    n_inc = max(1, n_win_int // 4)
    n_win_i = int(np.ceil(n_i / float(n_inc)) - 3)
    n_win_j = int(np.ceil(n_j / float(n_inc)) - 3)
    if n_win_i <= 0 or n_win_j <= 0:
        return ph_out

    x = np.arange(0, n_win_int // 2, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0) + 1e-6

    ph_arr[np.isnan(ph_arr)] = 0
    B = _clap_filter_kernel()
    n_win_ex = n_win_int + n_pad_int
    if low_pass is None:
        low_pass_use = np.zeros((n_win_ex, n_win_ex), dtype=np.float64)
    else:
        low_pass_use = np.asarray(low_pass, dtype=np.float64)
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex64)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_int
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_int
            wf = np.vstack((np.zeros((i_shift, n_win_int), dtype=np.float64), wf[: n_win_int - i_shift, :]))
        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_int
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_int
                wf2 = np.hstack((np.zeros((n_win_int, j_shift), dtype=np.float64), wf2[:, : n_win_int - j_shift]))

            ph_bit.fill(0)
            ph_bit[:n_win_int, :n_win_int] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = np.fft.ifftshift(
                signal.convolve2d(np.fft.fftshift(H), B, mode="same", boundary="fill", fillvalue=0.0)
            )
            mean_h = float(np.median(H))
            if mean_h != 0.0:
                H = H / mean_h
            H = np.power(H, float(alpha))
            H = H - 1.0
            H[H < 0.0] = 0.0
            G = H * float(beta) + low_pass_use
            ph_filt = np.fft.ifft2(ph_fft * G)
            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + (ph_filt[:n_win_int, :n_win_int] * wf2).astype(np.complex64)

    return ph_out.astype(np.complex64)


def _clap_filt_grid_stack(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
) -> np.ndarray:
    ph_arr = np.asarray(ph_stack, dtype=np.complex64).copy()
    if ph_arr.ndim != 3:
        raise PortedStageError("clap_filt_grid_stack expects a 3-D complex stack")

    n_win_int = int(round(n_win))
    if n_win_int <= 0:
        n_win_int = 32
    n_pad_int = int(round(n_pad))
    n_i, n_j, n_ifg = ph_arr.shape
    ph_out = np.zeros_like(ph_arr)
    n_inc = max(1, n_win_int // 4)
    n_win_i = int(np.ceil(n_i / float(n_inc)) - 3)
    n_win_j = int(np.ceil(n_j / float(n_inc)) - 3)
    if n_win_i <= 0 or n_win_j <= 0:
        return ph_out

    x = np.arange(0, n_win_int // 2, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0) + 1e-6

    ph_arr[np.isnan(ph_arr)] = 0
    B = _clap_filter_kernel()
    n_win_ex = n_win_int + n_pad_int
    if low_pass is None:
        low_pass_use = np.zeros((n_win_ex, n_win_ex), dtype=np.float64)
    else:
        low_pass_use = np.asarray(low_pass, dtype=np.float64)
    low_pass_stack = low_pass_use[:, :, None]
    ph_bit = np.zeros((n_win_ex, n_win_ex, n_ifg), dtype=np.complex64)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_int
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_int
            wf = np.vstack((np.zeros((i_shift, n_win_int), dtype=np.float64), wf[: n_win_int - i_shift, :]))
        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_int
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_int
                wf2 = np.hstack((np.zeros((n_win_int, j_shift), dtype=np.float64), wf2[:, : n_win_int - j_shift]))

            ph_bit.fill(0)
            ph_bit[:n_win_int, :n_win_int, :] = ph_arr[i1:i2, j1:j2, :]
            ph_fft = np.fft.fft2(ph_bit, axes=(0, 1))
            H = np.abs(ph_fft)
            H_smooth = np.empty_like(H, dtype=np.float64)
            for i_ifg in range(n_ifg):
                H_smooth[:, :, i_ifg] = np.fft.ifftshift(
                    signal.convolve2d(
                        np.fft.fftshift(H[:, :, i_ifg]),
                        B,
                        mode="same",
                        boundary="fill",
                        fillvalue=0.0,
                    )
                )
            H = H_smooth
            mean_h = np.median(H, axis=(0, 1), keepdims=True)
            H = np.divide(H, mean_h, out=H, where=mean_h != 0)
            H = np.power(H, float(alpha))
            H = H - 1.0
            H[H < 0.0] = 0.0
            G = H * float(beta) + low_pass_stack
            ph_filt = np.fft.ifft2(ph_fft * G, axes=(0, 1))
            ph_out[i1:i2, j1:j2, :] = ph_out[i1:i2, j1:j2, :] + (
                ph_filt[:n_win_int, :n_win_int, :] * wf2[:, :, None]
            ).astype(np.complex64)

    return ph_out.astype(np.complex64)


def _clap_filt_patch_stack(ph_stack: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
    ph_out = np.empty_like(np.asarray(ph_stack))
    for i in range(ph_stack.shape[2]):
        ph_out[:, :, i] = _clap_filt_patch(
            ph_stack[:, :, i],
            alpha=alpha,
            beta=beta,
            low_pass=low_pass,
        )
    return ph_out.astype(np.complex64)


def _gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    n_int = int(n)
    if n_int <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n_int == 1:
        return np.ones((1,), dtype=np.float64)
    alpha_f = float(alpha)
    if alpha_f <= 0:
        return np.ones((n_int,), dtype=np.float64)
    std = (n_int - 1) / (2.0 * alpha_f)
    return signal.windows.gaussian(n_int, std=std, sym=True).astype(np.float64)


def _matlab_interp(x: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    q = int(factor)
    if q <= 1 or arr.size == 0:
        return arr.copy()
    n = 4
    wc = 0.5
    y = np.zeros(arr.size * q + q * n + 1, dtype=np.float64)
    y[: arr.size * q : q] = arr
    b = signal.firwin(2 * q * n + 2, wc / q, window="hamming", scale=True).astype(np.float64)
    y = q * signal.lfilter(b, [1.0], y)
    return y[q * n + 1 :].astype(np.float64, copy=False)


def _stage2_weighting_snapshot_targets(patch_dir: Path) -> list[Path]:
    targets = [patch_dir / "stage2_weighting_snapshot.json"]
    if patch_dir.name == "PATCH_1":
        repo_root = Path(__file__).resolve().parents[2]
        repo_target = repo_root / _CANONICAL_STAGE2_WEIGHTING_SNAPSHOT
        if repo_target.parent.exists():
            targets.append(repo_target)
    return targets


def _stage2_psquare_weighting(
    Nr: np.ndarray,
    Na: np.ndarray,
    low_coh_thresh: int,
    nr_max_nz_ix: float | int,
    coh_ps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr = np.asarray(Nr, dtype=np.float64).reshape(-1)
    na = np.asarray(Na, dtype=np.float64).reshape(-1)
    coh = np.asarray(coh_ps, dtype=np.float64).reshape(-1)

    na_safe = na.copy()
    na_safe[na_safe == 0] = 1.0

    prand = nr / na_safe
    prand[: int(low_coh_thresh)] = 1.0
    prand[int(nr_max_nz_ix) :] = 0.0
    prand[prand > 1.0] = 1.0

    win = _gausswin(7)
    prand = signal.lfilter(win, [1.0], np.concatenate((np.ones(7, dtype=np.float64), prand))) / np.sum(win)
    prand = prand[7:]
    prand_hi = _matlab_interp(np.concatenate((np.ones(1, dtype=np.float64), prand)), 10)
    coh_ix = np.clip(np.round(coh * 1000).astype(np.int64), 0, prand_hi.size - 1)
    prand_ps = prand_hi[coh_ix]
    weighting = (1.0 - prand_ps) ** 2
    return prand, prand_hi, prand_ps, weighting


def _wrap_filt(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int | None = None,
    low_flag: str = "n",
) -> tuple[np.ndarray, np.ndarray | None]:
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    if ph_arr.ndim != 2:
        raise PortedStageError("wrap_filt expects a 2-D complex grid")

    n_i, n_j = ph_arr.shape
    n_win_i = int(round(n_win))
    if n_win_i <= 1:
        raise PortedStageError("wrap_filt window must be > 1")
    if n_pad is None:
        n_pad_i = int(round(n_win_i * 0.25))
    else:
        n_pad_i = int(round(n_pad))
    n_pad_i = max(0, n_pad_i)

    n_inc = int(np.floor(n_win_i / 2.0))
    if n_inc <= 0:
        n_inc = 1
    n_win_blocks_i = int(np.ceil(n_i / n_inc) - 1)
    n_win_blocks_j = int(np.ceil(n_j / n_inc) - 1)

    ph_out = np.zeros_like(ph_arr, dtype=np.complex64)
    want_low = str(low_flag).lower() == "y"
    ph_out_low = np.zeros_like(ph_arr, dtype=np.complex64) if want_low else None

    x = np.arange(1, n_win_i // 2 + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.concatenate((X, np.fliplr(X)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0).astype(np.float64)

    ph_arr[np.isnan(ph_arr)] = 0
    B = np.outer(_gausswin(7), _gausswin(7))
    ph_bit = np.zeros((n_win_i + n_pad_i, n_win_i + n_pad_i), dtype=np.complex64)

    L = None
    if want_low:
        g16 = _gausswin(n_win_i + n_pad_i, alpha=16.0)
        L = np.fft.ifftshift(np.outer(g16, g16))

    for ix1 in range(n_win_blocks_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_i
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_i
            wf = np.vstack((np.zeros((i_shift, n_win_i), dtype=np.float64), wf[: n_win_i - i_shift, :]))

        for ix2 in range(n_win_blocks_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_i
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_i
                wf2 = np.hstack((np.zeros((n_win_i, j_shift), dtype=np.float64), wf2[:, : n_win_i - j_shift]))

            ph_bit.fill(0)
            ph_bit[:n_win_i, :n_win_i] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = np.fft.ifftshift(
                signal.convolve2d(np.fft.fftshift(H), B, mode="same", boundary="fill", fillvalue=0.0)
            )
            mean_h = float(np.median(H))
            if mean_h != 0.0:
                H = H / mean_h
            H = np.power(H, float(alpha))

            ph_filt = np.fft.ifft2(ph_fft * H)
            ph_filt = ph_filt[:n_win_i, :n_win_i] * wf2
            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt.astype(np.complex64)

            if want_low and L is not None and ph_out_low is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * L)
                ph_filt_low = ph_filt_low[:n_win_i, :n_win_i] * wf2
                ph_out_low[i1:i2, j1:j2] = ph_out_low[i1:i2, j1:j2] + ph_filt_low.astype(np.complex64)

    ph_mag = np.abs(ph_arr).astype(np.float32)
    ph_out = (ph_mag * np.exp(1j * np.angle(ph_out))).astype(np.complex64)
    if ph_out_low is not None:
        ph_out_low = (ph_mag * np.exp(1j * np.angle(ph_out_low))).astype(np.complex64)
    return ph_out, ph_out_low


def _wrap_filt_global(
    ph: np.ndarray,
    alpha: float,
    low_flag: str = "n",
) -> tuple[np.ndarray, np.ndarray | None]:
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    if ph_arr.ndim != 2:
        raise PortedStageError("wrap_filt_global expects a 2-D complex grid")
    ph_arr[np.isnan(ph_arr)] = 0

    ph_fft = np.fft.fft2(ph_arr)
    H = np.abs(ph_fft)
    B = np.outer(_gausswin(7), _gausswin(7))
    H = np.fft.ifftshift(
        signal.convolve2d(np.fft.fftshift(H), B, mode="same", boundary="fill", fillvalue=0.0)
    )
    mean_h = float(np.median(H))
    if mean_h != 0.0:
        H = H / mean_h
    H = np.power(H, float(alpha))
    ph_filt = np.fft.ifft2(ph_fft * H)
    ph_out = (np.abs(ph_arr) * np.exp(1j * np.angle(ph_filt))).astype(np.complex64)

    ph_out_low = None
    if str(low_flag).lower() == "y":
        g_i = _gausswin(ph_arr.shape[0], alpha=16.0)
        g_j = _gausswin(ph_arr.shape[1], alpha=16.0)
        L = np.fft.ifftshift(np.outer(g_i, g_j))
        ph_low = np.fft.ifft2(ph_fft * L)
        ph_out_low = (np.abs(ph_arr) * np.exp(1j * np.angle(ph_low))).astype(np.complex64)

    return ph_out, ph_out_low


def _weighted_lstsq(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    Y = np.asarray(Y)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise PortedStageError("weighted_lstsq expects a 2-D design matrix")
    if X.shape[0] != w.size:
        raise PortedStageError("weighted_lstsq weights must match design rows")

    if Y.ndim == 1:
        Yw = Y * np.sqrt(w)
    elif Y.ndim == 2:
        Yw = Y * np.sqrt(w)[:, None]
    else:
        raise PortedStageError("weighted_lstsq expects 1-D or 2-D targets")

    Xw = X * np.sqrt(w)[:, None]
    coef, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    return coef


def _weighted_slope_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y)
    y_2d = y_arr.reshape(1, -1) if y_arr.ndim == 1 else y_arr
    if y_2d.shape[1] != x.size:
        raise PortedStageError("weighted_slope_fit target width must match x")
    if w.size != x.size:
        raise PortedStageError("weighted_slope_fit weights must match x")

    finite = np.isfinite(w)
    if not np.any(finite):
        out = np.zeros(y_2d.shape[0], dtype=np.complex128 if np.iscomplexobj(y_2d) else np.float64)
        return out if y_arr.ndim == 2 else out.reshape(-1)

    # MATLAB lscov effectively prioritizes infinite weights; mirror that
    # by solving on the infinite-weight subset when present.
    inf_mask = np.isinf(w)
    if np.any(inf_mask):
        x_use = x[inf_mask]
        y_use = y_2d[:, inf_mask]
        w_use = np.ones_like(x_use, dtype=np.float64)
    else:
        pos = finite & (w > 0)
        if not np.any(pos):
            out = np.zeros(y_2d.shape[0], dtype=np.complex128 if np.iscomplexobj(y_2d) else np.float64)
            return out if y_arr.ndim == 2 else out.reshape(-1)
        x_use = x[pos]
        y_use = y_2d[:, pos]
        w_use = w[pos]

    wx = w_use * x_use
    den = float(np.sum(wx * x_use))
    if den == 0.0:
        out = np.zeros(y_use.shape[0], dtype=np.complex128 if np.iscomplexobj(y_use) else np.float64)
    else:
        out = np.sum(y_use * wx[None, :], axis=1) / den
    return out if y_arr.ndim == 2 else out.reshape(-1)


def _weighted_affine_fit(time_diff: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_diff, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y_2d = np.asarray(y, dtype=np.float64)
    if y_2d.ndim != 2:
        raise PortedStageError("weighted_affine_fit expects a 2-D target matrix")
    if y_2d.shape[1] != t.size or w.size != t.size:
        raise PortedStageError("weighted_affine_fit dimensions must match time axis")

    s0 = float(np.sum(w))
    s1 = float(np.sum(w * t))
    s2 = float(np.sum(w * t * t))
    det = s0 * s2 - s1 * s1
    if det == 0.0:
        base = np.sum(y_2d * w[None, :], axis=1)
        intercept = np.divide(base, s0, out=np.zeros_like(base), where=s0 != 0)
        slope = np.zeros_like(intercept)
        return intercept, slope

    wy0 = np.sum(y_2d * w[None, :], axis=1)
    wy1 = np.sum(y_2d * (w * t)[None, :], axis=1)
    intercept = (wy0 * s2 - wy1 * s1) / det
    slope = (wy1 * s0 - wy0 * s1) / det
    return intercept, slope


def _stage7_mean_velocity_fit(
    ph_mean_v: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
) -> np.ndarray:
    day_f = np.asarray(day, dtype=np.float64).reshape(-1)
    if day_f.ndim != 1:
        raise PortedStageError("stage7 mean velocity fit expects a 1-D day vector")

    ph = np.asarray(ph_mean_v, dtype=np.float64)
    if ph.ndim != 2:
        raise PortedStageError("stage7 mean velocity fit expects a 2-D phase matrix")
    if ph.shape[1] != day_f.size:
        raise PortedStageError("stage7 mean velocity fit phase width must match day vector")

    std = np.asarray(ifg_std, dtype=np.float64).reshape(-1)
    if std.size != day_f.size:
        raise PortedStageError("stage7 mean velocity fit std vector must match day vector")

    master_zero = float(day_f[int(master_ix) - 1])
    time_diff = day_f - master_zero
    weights = np.divide(
        1.0,
        (std * np.pi / 180.0) ** 2,
        out=np.zeros_like(std, dtype=np.float64),
        where=std > 0,
    )
    intercept, slope = _weighted_affine_fit(time_diff, ph, weights)
    return np.vstack((intercept.astype(np.float32), slope.astype(np.float32)))


def _grid_neighbor_msd(ph_uw: np.ndarray, nzix: np.ndarray) -> np.ndarray:
    """Mirror uw_stat_costs.m MSD from neighboring unwrapped-grid jumps."""
    ph_uw_arr = np.asarray(ph_uw, dtype=np.float32)
    nzix_arr = np.asarray(nzix, dtype=bool)
    if ph_uw_arr.ndim != 2:
        raise PortedStageError("grid_neighbor_msd expects a 2-D unwrapped grid matrix")
    n_ps_grid, n_ifg = ph_uw_arr.shape
    if int(np.count_nonzero(nzix_arr)) != n_ps_grid:
        raise PortedStageError("grid_neighbor_msd nzix count must match grid rows")

    nrow, ncol = nzix_arr.shape
    msd = np.zeros((n_ifg,), dtype=np.float32)
    nz_flat = nzix_arr.reshape(-1, order="F")
    for i_ifg in range(n_ifg):
        ifguw = np.zeros((nrow, ncol), dtype=np.float32)
        flat = ifguw.reshape(-1, order="F")
        flat[nz_flat] = ph_uw_arr[:, i_ifg]
        diff1 = (ifguw[:-1, :] - ifguw[1:, :]).reshape(-1)
        diff1 = diff1[diff1 != 0]
        diff2 = (ifguw[:, :-1] - ifguw[:, 1:]).reshape(-1)
        diff2 = diff2[diff2 != 0]
        denom = diff1.size + diff2.size
        if denom > 0:
            num = float(np.sum(diff1.astype(np.float64) ** 2) + np.sum(diff2.astype(np.float64) ** 2))
            msd[i_ifg] = np.float32(num / denom)
    return msd


def _delaunay_edges(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    if n == 2:
        return np.asarray([[0, 1]], dtype=np.int64)

    try:
        tri = spatial.Delaunay(points)
        simp = np.asarray(tri.simplices, dtype=np.int64)
    except Exception:
        # Degenerate geometry fallback: connect to nearest neighbor.
        tree = spatial.cKDTree(points)
        _, nn = tree.query(points, k=2)
        edges = np.column_stack((np.arange(n, dtype=np.int64), nn[:, 1].astype(np.int64)))
        edges = np.sort(edges, axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        return np.unique(edges, axis=0)

    e1 = np.sort(simp[:, [0, 1]], axis=1)
    e2 = np.sort(simp[:, [1, 2]], axis=1)
    e3 = np.sort(simp[:, [0, 2]], axis=1)
    edges = np.vstack((e1, e2, e3))
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0).astype(np.int64)


def _load_triangle_edges(edge_path: Path, n_nodes: int) -> np.ndarray:
    if n_nodes < 2 or not edge_path.exists():
        return np.empty((0, 2), dtype=np.int64)
    raw = np.loadtxt(edge_path, skiprows=1, dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 3:
        return np.empty((0, 2), dtype=np.int64)

    edges = raw[:, 1:3].astype(np.int64) - 1
    edges = np.sort(edges, axis=1)
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < n_nodes)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < n_nodes)
        & (edges[:, 0] != edges[:, 1])
    )
    edges = edges[valid]
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.shape[0] > 1:
        _, keep = np.unique(edges, axis=0, return_index=True)
        edges = edges[np.sort(keep)]
    return edges.astype(np.int64)


def _adjacent_component_keep_mask(ij_cols23: np.ndarray, coh: np.ndarray) -> np.ndarray:
    ij = np.asarray(ij_cols23, dtype=np.int64)
    coh = np.asarray(coh, dtype=np.float64).reshape(-1)
    n_ps = ij.shape[0]
    if n_ps == 0:
        return np.zeros((0,), dtype=bool)

    ij_shift = ij + (np.asarray([2, 2], dtype=np.int64) - np.min(ij, axis=0))
    n_r = int(np.max(ij_shift[:, 0])) + 2
    n_c = int(np.max(ij_shift[:, 1])) + 2
    neigh_ix = np.zeros((n_r, n_c), dtype=np.int64)
    miss_middle = np.ones((3, 3), dtype=bool)
    miss_middle[1, 1] = False

    # Mirror MATLAB neighbor assignment logic in ps_weed.m.
    for i in range(n_ps):
        r = int(ij_shift[i, 0])
        c = int(ij_shift[i, 1])
        block = neigh_ix[r - 1 : r + 2, c - 1 : c + 2]
        fill = (block == 0) & miss_middle
        if np.any(fill):
            block = block.copy()
            block[fill] = i + 1  # MATLAB-style 1-based id
            neigh_ix[r - 1 : r + 2, c - 1 : c + 2] = block

    neigh_ps: list[list[int]] = [[] for _ in range(n_ps + 1)]
    for i in range(n_ps):
        r = int(ij_shift[i, 0])
        c = int(ij_shift[i, 1])
        my_neigh_ix = int(neigh_ix[r, c])
        if my_neigh_ix != 0:
            neigh_ps[my_neigh_ix].append(i + 1)

    ix_weed = np.ones(n_ps, dtype=bool)
    for i in range(1, n_ps + 1):
        if not neigh_ps[i]:
            continue
        same_ps = [i]
        i2 = 0
        while i2 < len(same_ps):
            ps_i = same_ps[i2]
            if neigh_ps[ps_i]:
                same_ps.extend(neigh_ps[ps_i])
                neigh_ps[ps_i] = []
            i2 += 1

        same = np.unique(np.asarray(same_ps, dtype=np.int64))
        coh_same = coh[same - 1]
        high_coh = int(np.argmax(coh_same))
        drop = np.ones(same.size, dtype=bool)
        drop[high_coh] = False
        ix_weed[same[drop] - 1] = False

    return ix_weed


def _write_stage4_debug(patch_dir: Path, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    (patch_dir / "stage4_debug.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_stage3_debug(patch_dir: Path, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    (patch_dir / "stage3_debug.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _coh_threshold_from_dist(
    coh_values: np.ndarray,
    D_A: np.ndarray,
    D_A_max: np.ndarray,
    coh_bins: np.ndarray,
    Nr_dist: np.ndarray,
    low_coh_thresh: int,
    max_percent_rand: float,
    select_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    min_coh = np.full(D_A_max.size - 1, np.nan, dtype=np.float64)
    D_A_mean = np.full(D_A_max.size - 1, np.nan, dtype=np.float64)

    for i in range(D_A_max.size - 1):
        bin_ix = (D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])
        if not np.any(bin_ix):
            continue
        coh_chunk = coh_values[bin_ix]
        coh_chunk = coh_chunk[np.isfinite(coh_chunk) & (coh_chunk != 0)]
        if coh_chunk.size == 0:
            continue

        D_A_mean[i] = float(np.mean(D_A[bin_ix]))
        Na = _hist_with_centers(coh_chunk, coh_bins)
        low_cut = min(low_coh_thresh, Na.size)
        denom = np.sum(Nr_dist[:low_cut])
        scale = np.sum(Na[:low_cut]) / denom if denom > 0 else 1.0
        Nr = Nr_dist * scale

        Na_safe = Na.copy()
        Na_safe[Na_safe == 0] = 1.0
        if select_method.upper() == "PERCENT":
            percent_rand = np.flip(np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na_safe)) * 100.0)
        else:
            percent_rand = np.flip(np.cumsum(np.flip(Nr)))
        ok_ix = np.where(percent_rand < max_percent_rand)[0]
        if ok_ix.size == 0:
            min_coh[i] = 1.0
            continue

        min_ok_1b = int(ok_ix.min()) + 1
        min_fit_ix = min_ok_1b - 3
        if min_fit_ix <= 0:
            min_coh[i] = np.nan
            continue
        max_fit_ix = min(min_ok_1b + 2, 100)
        xs = percent_rand[min_fit_ix - 1 : max_fit_ix]
        ys = np.arange(min_fit_ix, max_fit_ix + 1, dtype=np.float64) * 0.01
        if xs.size < 4:
            min_coh[i] = np.nan
            continue
        min_coh[i] = _polyfit_eval_centered(xs, ys, 3, max_percent_rand)

    valid = ~np.isnan(min_coh) & ~np.isnan(D_A_mean)
    if np.sum(valid) < 1:
        coh_thresh_all = np.full_like(coh_values, 0.3, dtype=np.float64)
        coh_thresh_coeffs = np.asarray([], dtype=np.float64)
    else:
        min_coh_valid = min_coh[valid]
        D_A_mean_valid = D_A_mean[valid]
        if min_coh_valid.size > 1:
            coeffs = np.polyfit(D_A_mean_valid, min_coh_valid, 1)
            if coeffs[0] > 0:
                coh_thresh_all = np.polyval(coeffs, D_A)
                coh_thresh_coeffs = coeffs.astype(np.float64)
            else:
                level = float(np.polyval(coeffs, 0.35))
                coh_thresh_all = np.full_like(coh_values, level, dtype=np.float64)
                coh_thresh_coeffs = np.asarray([], dtype=np.float64)
        else:
            coh_thresh_all = np.full_like(coh_values, float(min_coh_valid[0]), dtype=np.float64)
            coh_thresh_coeffs = np.asarray([], dtype=np.float64)
    coh_thresh_all[coh_thresh_all < 0] = 0.0
    return coh_thresh_all, coh_thresh_coeffs


def _ps_topofit_single(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float) -> tuple[float, float, float, np.ndarray]:
    cpxphase = np.asarray(cpxphase, dtype=np.complex128).reshape(-1)
    bperp = np.asarray(bperp, dtype=np.float64).reshape(-1)
    if cpxphase.size != bperp.size:
        raise PortedStageError("ps_topofit single expects vectors with matching lengths")

    phase_residual = np.zeros_like(cpxphase, dtype=np.complex64)
    valid = cpxphase != 0
    if not np.any(valid):
        return np.nan, np.nan, np.nan, phase_residual

    cpx = cpxphase[valid]
    bp = bperp[valid]

    trial_n = int(np.ceil(8.0 * float(n_trial_wraps)))
    trial_mult = np.arange(-trial_n, trial_n + 1, dtype=np.float64)
    bperp_range = float(np.max(bp) - np.min(bp))
    if bperp_range == 0.0:
        bperp_range = 1.0

    trial_phase = bp / bperp_range * (np.pi / 4.0)
    trial_phase_mat = np.exp(-1j * (trial_phase[:, None] * trial_mult[None, :])).astype(np.complex128)
    phaser_sum = np.sum(trial_phase_mat * cpx[:, None], axis=0, dtype=np.complex128)
    coh_trial = np.abs(phaser_sum).astype(np.float64)
    denom = float(np.sum(np.abs(cpx), dtype=np.float64))
    if denom == 0.0:
        denom = 1.0
    coh_trial = coh_trial / denom
    coh_high_max_ix = int(np.argmax(coh_trial))
    K0 = (np.pi / 4.0) / float(bperp_range) * float(trial_mult[coh_high_max_ix])

    bp64 = bp.astype(np.float64, copy=False)
    resphase = cpx * np.exp(-1j * (K0 * bp64))
    offset_phase = np.sum(resphase)
    resphase_angle = np.angle(resphase * np.conj(offset_phase))
    weighting = np.abs(cpx).astype(np.float64)
    wb = weighting * bp64
    den_lin = float(np.sum(wb * wb))
    if den_lin == 0.0:
        den_lin = 1.0
    mopt = float(np.sum(wb * (weighting * resphase_angle)) / den_lin)
    K0 = K0 + mopt

    valid_phase_residual = cpx * np.exp(-1j * (K0 * bp64))
    mean_phase_residual = np.sum(valid_phase_residual)
    C0 = float(np.angle(mean_phase_residual))
    denom2 = float(np.sum(np.abs(valid_phase_residual)))
    if denom2 == 0.0:
        denom2 = 1.0
    coh0 = float(np.abs(mean_phase_residual) / denom2)
    phase_residual[valid] = valid_phase_residual.astype(np.complex64)
    return float(K0), C0, coh0, phase_residual


def _ps_topofit_batch(
    cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float, _tie_refine: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cpxphase.ndim != 2 or bperp.ndim != 2 or cpxphase.shape != bperp.shape:
        raise PortedStageError("ps_topofit batch expects cpxphase and bperp with matching 2-D shapes")
    cpxphase = np.asarray(cpxphase, dtype=np.complex128)
    bperp = np.asarray(bperp, dtype=np.float64)
    n_row, n_col = cpxphase.shape
    if n_row == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, np.empty((0, cpxphase.shape[1]), dtype=np.complex64)

    trial_n = int(np.ceil(8.0 * float(n_trial_wraps)))
    trial_mult = np.arange(-trial_n, trial_n + 1, dtype=np.float64)
    bperp_range = np.max(bperp, axis=1) - np.min(bperp, axis=1)
    bperp_range[bperp_range == 0] = 1.0

    trial_phase = bperp / bperp_range[:, None] * (np.pi / 4.0)
    phaser_sum = np.zeros((n_row, trial_mult.size), dtype=np.complex128)
    for i, trial in enumerate(trial_mult):
        phaser_sum[:, i] = np.sum(np.exp(-1j * (trial_phase * trial)) * cpxphase, axis=1, dtype=np.complex128)
    coh_trial = np.abs(phaser_sum).astype(np.float64)
    denom = np.sum(np.abs(cpxphase), axis=1, dtype=np.float64)
    denom[denom == 0] = 1.0
    coh_trial = coh_trial / denom[:, None]

    coh_high_max_ix = np.argmax(coh_trial, axis=1)
    K0 = (np.pi / 4.0) / bperp_range.astype(np.float64) * trial_mult[coh_high_max_ix].astype(np.float64)

    bp64 = bperp.astype(np.float64, copy=False)
    resphase = cpxphase * np.exp(-1j * (K0[:, None] * bp64))
    offset_phase = np.sum(resphase, axis=1)
    resphase_angle = np.angle(resphase * np.conj(offset_phase[:, None]))
    weighting = np.abs(cpxphase).astype(np.float64)
    wb = weighting * bp64
    den_lin = np.sum(wb * wb, axis=1)
    den_lin[den_lin == 0] = 1.0
    mopt = np.sum(wb * (weighting * resphase_angle), axis=1) / den_lin
    K0 = K0 + mopt

    phase_residual = cpxphase * np.exp(-1j * (K0[:, None] * bp64))
    mean_phase_residual = np.sum(phase_residual, axis=1)
    C0 = np.angle(mean_phase_residual).astype(np.float64)
    coh0 = np.abs(mean_phase_residual).astype(np.float64)
    denom2 = np.sum(np.abs(phase_residual), axis=1)
    denom2[denom2 == 0] = 1.0
    coh0 = coh0 / denom2
    phase_residual = phase_residual.astype(np.complex64)

    # Match single-path handling when missing interferograms are present.
    zero_rows = np.any(cpxphase == 0, axis=1)
    if np.any(zero_rows):
        for row_ix in np.where(zero_rows)[0]:
            k, c, coh, ph_res = _ps_topofit_single(cpxphase[row_ix, :], bperp[row_ix, :], n_trial_wraps)
            K0[row_ix] = k
            C0[row_ix] = c
            coh0[row_ix] = coh
            phase_residual[row_ix, :] = ph_res

    return K0.astype(np.float64), C0.astype(np.float64), coh0.astype(np.float64), phase_residual.astype(np.complex64)


def _as_ps_matrix(values: Any, n_ps: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise PortedStageError(f"{name} must be a 2-D matrix")
    if arr.shape[0] == n_ps:
        return arr
    if arr.shape[1] == n_ps:
        return arr.T
    raise PortedStageError(f"{name} has incompatible shape {arr.shape} for n_ps={n_ps}")


def _as_ps_ifg_complex(values: Any, n_ps: int, name: str) -> np.ndarray:
    arr = _coerce_complex(values)
    if arr.ndim != 2:
        raise PortedStageError(f"{name} must be a 2-D matrix")
    if arr.shape[0] == n_ps:
        return arr.astype(np.complex64)
    if arr.shape[1] == n_ps:
        return arr.T.astype(np.complex64)
    raise PortedStageError(f"{name} has incompatible shape {arr.shape} for n_ps={n_ps}")


def _as_ps_vector(values: Any, n_ps: int, name: str) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size != n_ps:
        raise PortedStageError(f"{name} has incompatible length {arr.size} for n_ps={n_ps}")
    return arr


def _as_ps_dim(values: Any, n_ps: int, n_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise PortedStageError(f"{name} must be a 2-D matrix")
    if arr.shape == (n_ps, n_dim):
        return arr
    if arr.shape == (n_dim, n_ps):
        return arr.T
    raise PortedStageError(f"{name} has incompatible shape {arr.shape}; expected ({n_ps},{n_dim}) or ({n_dim},{n_ps})")


def _dedup_lonlat_keep_highest_coh(lonlat: np.ndarray, coh_ps: np.ndarray) -> np.ndarray:
    n = lonlat.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    key_dtype = np.dtype([("lon", lonlat.dtype), ("lat", lonlat.dtype)])
    keys = np.ascontiguousarray(lonlat).view(key_dtype).reshape(-1)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if np.all(counts == 1):
        return np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    dup_groups = np.where(counts > 1)[0]
    for group in dup_groups:
        idx = np.where(inverse == group)[0]
        if idx.size <= 1:
            continue
        best = idx[np.argmax(coh_ps[idx])]
        drop = idx[idx != best]
        keep[drop] = False
    return keep


def _intersect_rows_indices(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.size == 0 or b.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise PortedStageError("row intersection requires 2-D arrays with matching column counts")

    a_keys = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)
    b_keys = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1]))).reshape(-1)
    _, ia, ib = np.intersect1d(a_keys, b_keys, assume_unique=False, return_indices=True)
    return ia.astype(np.int64), ib.astype(np.int64)


def _ifg_index_for_selection(ps: dict[str, Any], parms: Parms) -> np.ndarray:
    n_ifg = int(round(_mat_scalar(ps.get("n_ifg", 0), 0)))
    drop = set(int(v) for v in parms.drop_ifg_index.tolist())
    ifg = [i for i in range(1, n_ifg + 1) if i not in drop]

    if parms.small_baseline_flag.lower() != "y":
        master_ix = int(round(_mat_scalar(ps.get("master_ix", 1), 1)))
        ifg = [i for i in ifg if i != master_ix]
        ifg = [i - 1 if i > master_ix else i for i in ifg]
    return np.asarray(ifg, dtype=np.float64)


def _ifg_index_for_weed(ps: dict[str, Any], parms: Parms) -> np.ndarray:
    n_ifg = int(round(_mat_scalar(ps.get("n_ifg", 0), 0)))
    drop = set(int(v) for v in parms.drop_ifg_index.tolist())
    ifg = [i for i in range(1, n_ifg + 1) if i not in drop]
    return np.asarray(ifg, dtype=np.float64)


def _yyyymmdd_to_ordinal(day_values: np.ndarray) -> np.ndarray:
    day_values = np.asarray(day_values, dtype=np.int64).reshape(-1)
    years = day_values // 10000
    months = (day_values % 10000) // 100
    days = day_values % 100

    ordinals = []
    for y, m, d in zip(years, months, days):
        ordinal = np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}").astype("datetime64[D]").astype(int)
        ordinals.append(float(ordinal) + 719529.0)  # MATLAB datenum offset from Unix epoch day count
    return np.asarray(ordinals, dtype=np.float64)


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    rounded = np.sign(arr) * np.floor(np.abs(arr) + 0.5)
    return rounded.astype(arr.dtype, copy=False)


def _local_xy_from_lonlat(
    lonlat: np.ndarray,
    heading_deg: float | None = None,
    origin_lonlat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ll0 = (
        np.asarray(origin_lonlat, dtype=np.float64)
        if origin_lonlat is not None
        else (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2.0
    )
    llh = np.asarray(lonlat, dtype=np.float64).T * (np.pi / 180.0)
    origin = np.asarray(ll0, dtype=np.float64) * (np.pi / 180.0)

    # WGS84 ellipsoid constants used by StaMPS llh2local.m
    a = 6378137.0
    e = 0.08209443794970

    lat = llh[1, :]
    z = lat != 0.0
    xy = np.zeros((2, llh.shape[1]), dtype=np.float64)

    if np.any(z):
        dlambda = llh[0, z] - origin[0]
        lat_z = lat[z]

        M = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat_z
            - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * lat_z)
            + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat_z)
            - (35 * e**6 / 3072) * np.sin(6 * lat_z)
        )
        M0 = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * origin[1]
            - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * origin[1])
            + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * origin[1])
            - (35 * e**6 / 3072) * np.sin(6 * origin[1])
        )
        N = a / np.sqrt(1 - e**2 * np.sin(lat_z) ** 2)
        E = dlambda * np.sin(lat_z)
        cot_lat = 1.0 / np.tan(lat_z)

        xy[0, z] = N * cot_lat * np.sin(E)
        xy[1, z] = M - M0 + N * cot_lat * (1 - np.cos(E))

    if np.any(~z):
        dlambda = llh[0, ~z] - origin[0]
        M0 = a * (
            (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * origin[1]
            - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * origin[1])
            + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * origin[1])
            - (35 * e**6 / 3072) * np.sin(6 * origin[1])
        )
        xy[0, ~z] = a * dlambda
        xy[1, ~z] = -M0

    xy = xy.T

    if heading_deg is not None:
        theta = (180.0 - float(heading_deg)) * np.pi / 180.0
        if theta > np.pi:
            theta = theta - 2.0 * np.pi
        rotm = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype=np.float64)
        xy_t = xy.T
        xy_rot = rotm @ xy_t
        if np.ptp(xy_rot[0, :]) < np.ptp(xy_t[0, :]) and np.ptp(xy_rot[1, :]) < np.ptp(xy_t[1, :]):
            xy = xy_rot.T

    return xy, ll0


def _select_reference_ps(ps: dict[str, Any], parms_raw: dict[str, Any]) -> np.ndarray:
    lonlat = _as_ps_dim(ps.get("lonlat"), int(round(_mat_scalar(ps.get("n_ps", 0), 0))), 2, "ps.lonlat").astype(np.float64)
    ref_lon = np.asarray(parms_raw.get("ref_lon", [-np.inf, np.inf]), dtype=np.float64).reshape(-1)
    ref_lat = np.asarray(parms_raw.get("ref_lat", [-np.inf, np.inf]), dtype=np.float64).reshape(-1)
    if ref_lon.size < 2:
        ref_lon = np.asarray([-np.inf, np.inf], dtype=np.float64)
    if ref_lat.size < 2:
        ref_lat = np.asarray([-np.inf, np.inf], dtype=np.float64)

    mask = (
        (lonlat[:, 0] > ref_lon[0])
        & (lonlat[:, 0] < ref_lon[1])
        & (lonlat[:, 1] > ref_lat[0])
        & (lonlat[:, 1] < ref_lat[1])
    )

    ref_radius = float(_mat_scalar(parms_raw.get("ref_radius", np.inf), np.inf))
    if ref_radius == -np.inf:
        return np.asarray([], dtype=np.int64)

    ref_ix = np.flatnonzero(mask)
    if np.isfinite(ref_radius) and ref_ix.size > 0:
        ref_center = np.asarray(parms_raw.get("ref_centre_lonlat", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        if ref_center.size >= 2:
            ll0 = np.asarray(ps.get("ll0"), dtype=np.float64).reshape(-1)
            origin = ll0[:2] if ll0.size >= 2 else ref_center[:2]
            ref_xy, _ = _local_xy_from_lonlat(ref_center[:2][None, :], origin_lonlat=origin)
            xy, _ = _local_xy_from_lonlat(lonlat[ref_ix], origin_lonlat=origin)
            dist_sq = np.sum((xy - ref_xy[0]) ** 2, axis=1)
            ref_ix = ref_ix[dist_sq <= ref_radius**2]

    if ref_ix.size == 0:
        ref_ix = np.arange(lonlat.shape[0], dtype=np.int64)
    return ref_ix


def _stage7_unwrap_ifg_sets(
    n_ifg: int,
    master_ix: int,
    drop_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    unwrap_ifg = np.asarray([i for i in range(1, n_ifg + 1) if i not in drop_set], dtype=np.int64)
    solve_ifg = unwrap_ifg[unwrap_ifg != master_ix]
    return unwrap_ifg, solve_ifg


def _center_to_reference(ph: np.ndarray, ref_ix: np.ndarray) -> np.ndarray:
    if ref_ix.size == 0:
        return ph
    ref_mean = np.nanmean(ph[ref_ix, :], axis=0, keepdims=True)
    return ph - ref_mean


def _deramp_unwrapped_phase(ps: dict[str, Any], ph_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_ps = int(round(_mat_scalar(ps.get("n_ps", 0), 0)))
    xy = _as_ps_dim(ps.get("xy"), n_ps, 3, "ps.xy").astype(np.float64)
    design = np.column_stack((xy[:, 1:3] / 1000.0, np.ones((n_ps, 1), dtype=np.float64)))
    ph = np.asarray(ph_all, dtype=np.float64)

    if not np.isnan(ph).any():
        coeffs, _, _, _ = np.linalg.lstsq(design, ph, rcond=None)
        ph_ramp = design @ coeffs
        return ph - ph_ramp, ph_ramp

    ph_ramp = np.full_like(ph, np.nan, dtype=np.float64)
    ph_out = ph.copy()
    for i in range(ph.shape[1]):
        valid = ~np.isnan(ph[:, i])
        if np.count_nonzero(valid) <= 5:
            continue
        coeffs, _, _, _ = np.linalg.lstsq(design[valid, :], ph[valid, i], rcond=None)
        ph_ramp[:, i] = design @ coeffs
        ph_out[valid, i] = ph[valid, i] - ph_ramp[valid, i]
    return ph_out, ph_ramp


def _weighted_lstsq_shared_design(G: np.ndarray, Y: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
    G64 = np.asarray(G, dtype=np.float64)
    Y64 = np.asarray(Y, dtype=np.float64)
    if cov is None:
        coeffs, _, _, _ = np.linalg.lstsq(G64, Y64, rcond=None)
        return coeffs

    cov64 = np.asarray(cov, dtype=np.float64)
    if cov64.ndim != 2 or cov64.shape[0] != cov64.shape[1] or cov64.shape[0] != G64.shape[0]:
        raise PortedStageError("weighted least-squares covariance has incompatible shape")

    if np.allclose(cov64, np.diag(np.diag(cov64))):
        scale = np.sqrt(np.diag(cov64))
        scale[scale == 0.0] = 1.0
        Gw = G64 / scale[:, None]
        Yw = Y64 / scale[:, None]
    else:
        jitter = 0.0
        eye = np.eye(cov64.shape[0], dtype=np.float64)
        while True:
            try:
                chol = np.linalg.cholesky(cov64 + jitter * eye)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
                if jitter > 1e-3:
                    raise
        Gw = np.linalg.solve(chol, G64)
        Yw = np.linalg.solve(chol, Y64)

    coeffs, _, _, _ = np.linalg.lstsq(Gw, Yw, rcond=None)
    return coeffs


def _load_complex_columns(path: Path, n_rows: int) -> np.ndarray:
    raw = _load_binary_float32(path, "phase")
    if raw.size % (2 * n_rows) != 0:
        raise PortedStageError(f"Unexpected binary size for phase file: {path}")

    n_cols = raw.size // (2 * n_rows)
    blocks = raw.reshape(n_cols, n_rows * 2)
    real = blocks[:, 0::2]
    imag = blocks[:, 1::2]
    return (real + 1j * imag).T.astype(np.complex64)


def _stage1_geometry(patch_dir: Path, ij: np.ndarray) -> tuple[float, float] | None:
    dataset_root = _stage1_dataset_root(patch_dir)
    if not (dataset_root / "diff0").exists() or not (dataset_root / "rslc").exists():
        return None
    try:
        records = _snap_ifg_records(dataset_root)
    except PortedStageError:
        return None
    master_days = sorted({master for master, _, _ in records})
    if len(master_days) != 1:
        return None
    try:
        rslc_par = _resolve_rslc_par(dataset_root, master_days[0])
    except PortedStageError:
        return None
    try:
        range_pixel_spacing = _read_named_scalar(rslc_par, "range_pixel_spacing")
        near_range_slc = _read_named_scalar(rslc_par, "near_range_slc")
        sar_to_earth_center = _read_named_scalar(rslc_par, "sar_to_earth_center")
        earth_radius_below_sensor = _read_named_scalar(rslc_par, "earth_radius_below_sensor")
        center_range_slc = _read_named_scalar(rslc_par, "center_range_slc")
    except PortedStageError:
        return None
    rg = near_range_slc + np.asarray(ij[:, 2], dtype=np.float64) * range_pixel_spacing
    inci_arg = (sar_to_earth_center**2 - earth_radius_below_sensor**2 - rg**2) / (2.0 * earth_radius_below_sensor * rg)
    inci = np.arccos(np.clip(inci_arg, -1.0, 1.0))
    return float(center_range_slc), float(np.mean(inci))


def _stage1_heading_deg(patch_dir: Path) -> float | None:
    dataset_root = _stage1_dataset_root(patch_dir)
    if not (dataset_root / "diff0").exists() or not (dataset_root / "rslc").exists():
        return None
    try:
        records = _snap_ifg_records(dataset_root)
    except PortedStageError:
        return None
    master_days = sorted({master for master, _, _ in records})
    if len(master_days) != 1:
        return None
    try:
        rslc_par = _resolve_rslc_par(dataset_root, master_days[0])
        return _read_named_scalar(rslc_par, "heading")
    except PortedStageError:
        return None


def stage1_load_initial(patch_dir: Path, backend: str = "auto") -> str:
    required = {
        "ij": patch_dir / "pscands.1.ij",
        "ph": patch_dir / "pscands.1.ph",
        "ll": patch_dir / "pscands.1.ll",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise PortedStageError(f"Missing stage-1 patch inputs: {', '.join(missing)}")

    ij = _load_text_matrix(required["ij"], dtype=np.float64)
    if ij.ndim == 1:
        ij = ij[None, :]
    n_ps = ij.shape[0]

    width_file = _resolve_file(patch_dir, "width.txt")
    len_file = _resolve_file(patch_dir, "len.txt")
    metadata_missing = [name for name, path in {"width.txt": width_file, "len.txt": len_file}.items() if path is None]
    if metadata_missing:
        raise PortedStageError("Stage 1 requires metadata files not found near patch: " + ", ".join(metadata_missing))

    metadata = resolve_stage1_metadata(patch_dir, ij)
    day_file = metadata.day_file
    master_day_file = metadata.master_day_file
    bperp_file = metadata.bperp_file

    day = _coerce_1d(_load_text_matrix(day_file, dtype=np.int64))
    master_day_yyyymmdd = float(_coerce_1d(_load_text_matrix(master_day_file, dtype=np.int64))[0])
    bperp = _coerce_1d(_load_text_matrix(bperp_file, dtype=np.float64))
    if day.size != bperp.size:
        raise PortedStageError(
            f"Stage 1 metadata mismatch: day.1.in has {day.size} rows but bperp.1.in has {bperp.size}"
        )

    slave_day = _yyyymmdd_to_ordinal(day)
    day_ix = np.argsort(slave_day)
    slave_day = slave_day[day_ix]
    master_day = _yyyymmdd_to_ordinal(np.asarray([master_day_yyyymmdd], dtype=np.int64))[0]
    master_ix = int(np.sum(slave_day < master_day)) + 1  # MATLAB-compatible 1-based index

    day_full = np.insert(slave_day, master_ix - 1, master_day)
    bperp_sorted = bperp[day_ix]
    bperp_full = np.insert(bperp_sorted, master_ix - 1, 0.0).astype(np.float32)

    ph = _load_complex_columns(required["ph"], n_ps)
    if ph.shape[1] != day_ix.size:
        raise PortedStageError(
            f"Stage 1 interferogram count mismatch: ph has {ph.shape[1]} columns but metadata has {day_ix.size} entries"
        )
    ph = ph[:, day_ix]
    ph = np.insert(ph, master_ix - 1, 1.0 + 0.0j, axis=1).astype(np.complex64)

    lonlat_raw = _load_binary_float32(required["ll"], "lonlat")
    lonlat = lonlat_raw.reshape(-1, 2).astype(np.float64)
    xy_local, ll0 = _local_xy_from_lonlat(lonlat, heading_deg=_stage1_heading_deg(patch_dir))

    xy_sort = np.asarray(xy_local, dtype=np.float32)
    sort_ix = np.lexsort((xy_sort[:, 0], xy_sort[:, 1]))
    ij_sorted = ij[sort_ix].copy()
    ij_sorted[:, 0] = np.arange(1, n_ps + 1)

    lonlat_sorted = lonlat[sort_ix]
    xy_sorted = xy_sort[sort_ix]
    xy_out = np.column_stack((np.arange(1, n_ps + 1), xy_sorted)).astype(np.float32)

    ph_sorted = ph[sort_ix]

    options = _build_stage_options(patch_dir)
    geometry = _stage1_geometry(patch_dir, ij)
    mean_range = float(options.mean_range)
    mean_incidence = float(options.mean_incidence)
    if geometry is not None:
        mean_range, mean_incidence = geometry

    ps_payload: dict[str, Any] = {
        "ij": ij_sorted.astype(np.float64),
        "lonlat": lonlat_sorted.astype(np.float64),
        "xy": xy_out,
        "bperp": bperp_full,
        "day": day_full.astype(np.float64),
        "master_day": np.asarray(master_day, dtype=np.float64),
        "master_ix": np.asarray(master_ix, dtype=np.float64),
        "n_ifg": np.asarray(ph_sorted.shape[1], dtype=np.float64),
        "n_image": np.asarray(ph_sorted.shape[1], dtype=np.float64),
        "n_ps": np.asarray(n_ps, dtype=np.float64),
        "sort_ix": (sort_ix + 1).astype(np.float64),
        "ll0": ll0.astype(np.float64),
        "mean_range": np.asarray(mean_range, dtype=np.float64),
        "mean_incidence": np.asarray(mean_incidence, dtype=np.float64),
    }

    write_mat(patch_dir / "ps1.mat", ps_payload)
    write_mat(patch_dir / "ph1.mat", {"ph": ph_sorted})
    write_mat(patch_dir / "psver.mat", {"psver": np.asarray(1, dtype=np.float64)})

    da_file = patch_dir / "pscands.1.da"
    if da_file.exists():
        da = _coerce_1d(_load_text_matrix(da_file, dtype=np.float64))[sort_ix]
        write_mat(patch_dir / "da1.mat", {"D_A": da.astype(np.float64)})

    hgt_file = patch_dir / "pscands.1.hgt"
    if hgt_file.exists():
        hgt = _load_binary_float32(hgt_file, "height").reshape(-1)[sort_ix]
        write_mat(patch_dir / "hgt1.mat", {"hgt": hgt.astype(np.float32)})

    if metadata.bperp_mat is not None:
        bperp_mat = np.asarray(metadata.bperp_mat[:, day_ix], dtype=np.float32)[sort_ix]
    else:
        no_master = np.arange(ph_sorted.shape[1]) != (master_ix - 1)
        bperp_nomaster = bperp_full[no_master]
        bperp_mat = np.tile(bperp_nomaster, (n_ps, 1)).astype(np.float32)
    write_mat(patch_dir / "bp1.mat", {"bperp_mat": bperp_mat})

    return f"Stage 1 created ps1/ph1 for {n_ps} candidates"


def _build_low_pass(options: StageOptions) -> np.ndarray:
    n_win = int(options.clap_win)
    if n_win <= 0:
        n_win = 32

    freq0 = 1.0 / float(options.clap_low_pass_wavelength)
    freq_i = np.arange(-n_win / 2, n_win / 2) / float(options.grid_size * n_win)
    butter = 1.0 / (1.0 + (freq_i / freq0) ** (2 * 5))
    low_pass = np.outer(butter, butter)
    return np.fft.fftshift(low_pass).astype(np.float64)


def stage2_estimate_gamma(patch_dir: Path, backend: str = "auto", debug: bool = False) -> str:
    stage2_t0 = time.perf_counter()
    ps = read_mat(patch_dir / "ps1.mat")
    parms_file = _resolve_file(patch_dir, "parms.mat")
    parms_raw = read_mat(parms_file) if parms_file is not None else {}
    parms = _load_parms(patch_dir)
    n_ps = int(round(_mat_scalar(ps.get("n_ps", 0), 0)))
    if n_ps <= 0:
        raise PortedStageError("ps1.mat missing valid n_ps")

    ph = read_mat(patch_dir / "ph1.mat").get("ph")
    if ph is None:
        raise PortedStageError("ph1.mat missing 'ph' variable")
    ph = _as_ps_ifg_complex(ph, n_ps, "ph1.ph")

    bp_file = patch_dir / "bp1.mat"
    if bp_file.exists():
        bp = read_mat(bp_file)
        bperp_mat = _as_ps_matrix(bp.get("bperp_mat"), n_ps, "bp1.bperp_mat").astype(np.float64)
    else:
        bperp = np.asarray(ps.get("bperp"), dtype=np.float64).reshape(-1)
        master_ix = int(round(_mat_scalar(ps.get("master_ix", 1), 1)))
        no_master = np.arange(bperp.size) != (master_ix - 1)
        bperp_mat = np.tile(bperp[no_master], (ph.shape[0], 1)).astype(np.float64)
        write_mat(bp_file, {"bperp_mat": bperp_mat.astype(np.float32)})

    n_ps, n_ifg_full = ph.shape
    master_ix = int(round(_mat_scalar(ps.get("master_ix", 1), 1)))
    if parms.small_baseline_flag.lower() == "y":
        ph_nm = ph.astype(np.complex64, copy=False)
        bperp_nm = np.asarray(ps.get("bperp"), dtype=np.float64).reshape(-1)
    else:
        no_master = np.arange(n_ifg_full) != (master_ix - 1)
        ph_nm = ph[:, no_master].astype(np.complex64, copy=False)
        bperp_nm = np.asarray(ps.get("bperp"), dtype=np.float64).reshape(-1)[no_master]

    amp = np.abs(ph_nm).astype(np.float32)
    amp[amp == 0] = 1.0
    ph_nm = np.divide(ph_nm, amp, out=np.zeros_like(ph_nm), where=amp != 0).astype(np.complex64)
    n_ifg = ph_nm.shape[1]

    da_file = patch_dir / "da1.mat"
    if da_file.exists():
        D_A = np.asarray(read_mat(da_file).get("D_A"), dtype=np.float64).reshape(-1)
    else:
        D_A = np.ones(n_ps, dtype=np.float64)
    if D_A.size != n_ps:
        D_A = np.ones(n_ps, dtype=np.float64)

    options = _build_stage_options(patch_dir)
    grid_size = float(_mat_scalar(parms_raw.get("filter_grid_size", options.grid_size), options.grid_size))
    filter_weighting = str(parms_raw.get("filter_weighting", "P-square"))
    gamma_change_convergence = float(
        _mat_scalar(parms_raw.get("gamma_change_convergence", 1e-4), 1e-4)
    )
    gamma_max_iterations = int(round(_mat_scalar(parms_raw.get("gamma_max_iterations", 25.0), 25.0)))
    clap_window = int(round(options.clap_win * 0.75))
    clap_pad = int(round(options.clap_win * 0.25))

    xy = _as_ps_dim(ps.get("xy"), n_ps, 3, "ps1.xy").astype(np.float64)
    x = xy[:, 1]
    y = xy[:, 2]
    grid_i = np.ceil((y - np.min(y) + 1e-6) / grid_size).astype(np.int64)
    grid_j = np.ceil((x - np.min(x) + 1e-6) / grid_size).astype(np.int64)
    if np.max(grid_i) > 1:
        grid_i[grid_i == np.max(grid_i)] = np.max(grid_i) - 1
    if np.max(grid_j) > 1:
        grid_j[grid_j == np.max(grid_j)] = np.max(grid_j) - 1
    grid_i[grid_i < 1] = 1
    grid_j[grid_j < 1] = 1
    grid_ij = np.column_stack((grid_i, grid_j)).astype(np.float32)
    n_i = int(np.max(grid_i))
    n_j = int(np.max(grid_j))
    grid_rows = grid_i - 1
    grid_cols = grid_j - 1
    grid_lin = np.ravel_multi_index((grid_rows, grid_cols), (n_i, n_j))

    low_pass = _build_low_pass(options)
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    low_coh_thresh = 15 if parms.small_baseline_flag.lower() == "y" else 31

    debug_payload: dict[str, Any] | None = None
    if debug:
        debug_payload = {
            "patch": patch_dir.name,
            "backend": backend,
            "status": "started",
            "phase": "setup",
            "small_baseline_flag": str(parms.small_baseline_flag),
            "filter_weighting": filter_weighting,
            "gamma_change_convergence": gamma_change_convergence,
            "gamma_max_iterations": gamma_max_iterations,
            "n_rand": int(n_rand) if "n_rand" in locals() else None,
            "clap_window": int(clap_window),
            "clap_pad": int(clap_pad),
            "random_mode": "small_baseline_diff" if parms.small_baseline_flag.lower() == "y" else "iid_ifg",
            "n_ps": int(n_ps),
            "n_ifg": int(n_ifg),
            "ph_shape": [int(v) for v in ph.shape],
            "ph_nm_shape": [int(v) for v in ph_nm.shape],
            "bperp_mat_shape": [int(v) for v in bperp_mat.shape],
            "grid_ij_shape": [int(v) for v in grid_ij.shape],
            "grid_shape": [int(n_i), int(n_j)],
            "iteration": 0,
            "pm1_written": False,
        }

        def _emit_stage2(
            phase: str,
            *,
            status: str = "running",
            iteration: int = 0,
            timings: dict[str, float] | None = None,
            extra: dict[str, Any] | None = None,
        ) -> None:
            assert debug_payload is not None
            debug_payload["status"] = status
            debug_payload["phase"] = phase
            debug_payload["iteration"] = int(iteration)
            debug_payload["updated_at_epoch_sec"] = time.time()
            if timings is not None:
                debug_payload["timings_sec"] = timings
            if extra:
                debug_payload.update(extra)
            (patch_dir / "stage2_debug.json").write_text(
                json.dumps(debug_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    else:
        def _emit_stage2(
            phase: str,
            *,
            status: str = "running",
            iteration: int = 0,
            timings: dict[str, float] | None = None,
            extra: dict[str, Any] | None = None,
        ) -> None:
            return

    _emit_stage2("setup_complete", timings={"total": time.perf_counter() - stage2_t0})

    n_rand = 300000
    if debug and debug_payload is not None:
        debug_payload["n_rand"] = int(n_rand)
    mean_range = _mat_scalar(ps.get("mean_range", options.mean_range), options.mean_range)
    mean_inc = _mat_scalar(ps.get("mean_incidence", options.mean_incidence), options.mean_incidence)
    max_k = options.max_topo_err / (options.lambda_m * mean_range * np.sin(mean_inc) / (4 * np.pi))
    n_trial_wraps = float((np.max(bperp_nm) - np.min(bperp_nm)) * max_k / (2 * np.pi))

    rng = _MatlabV5UniformRNG(2005)
    coh_rand = np.zeros(n_rand, dtype=np.float64)
    rand_chunk = 5000
    rand_bp = np.tile(bperp_nm.astype(np.float64), (rand_chunk, 1))
    if parms.small_baseline_flag.lower() == "y":
        ifgday_ix_raw = np.asarray(ps.get("ifgday_ix"))
        if ifgday_ix_raw.size == 0:
            raise PortedStageError("ps1.mat missing ifgday_ix required for small-baseline stage-2 random statistics")
        ifgday_ix = np.asarray(ifgday_ix_raw, dtype=np.int64)
        if ifgday_ix.ndim != 2:
            raise PortedStageError(f"ps1.ifgday_ix must be a 2-D matrix, got shape {ifgday_ix.shape}")
        if ifgday_ix.shape[0] != n_ifg and ifgday_ix.shape[1] == n_ifg:
            ifgday_ix = ifgday_ix.T
        if ifgday_ix.shape != (n_ifg, 2):
            raise PortedStageError(
                f"ps1.ifgday_ix has incompatible shape {ifgday_ix.shape} for n_ifg={n_ifg}"
            )
        n_image = int(np.max(ifgday_ix))
        if n_image <= 0:
            raise PortedStageError("ps1.ifgday_ix does not define a valid image count")
    for start in range(0, n_rand, rand_chunk):
        chunk = min(rand_chunk, n_rand - start)
        if parms.small_baseline_flag.lower() == "y":
            rand_image = rng.uniform((chunk, n_image)) * (2 * np.pi)
            rand_ifg = np.zeros((chunk, n_ifg), dtype=np.float64)
            for i_ifg in range(n_ifg):
                rand_ifg[:, i_ifg] = (
                    rand_image[:, ifgday_ix[i_ifg, 1] - 1] - rand_image[:, ifgday_ix[i_ifg, 0] - 1]
                )
            rand_phase = np.exp(1j * rand_ifg).astype(np.complex64)
        else:
            rand_phase = np.exp(1j * (rng.uniform((chunk, n_ifg)) * (2 * np.pi))).astype(np.complex64)
        _rk, _rc, coh_chunk, _res = _ps_topofit_batch(rand_phase, rand_bp[:chunk, :], n_trial_wraps)
        coh_rand[start : start + chunk] = coh_chunk.astype(np.float64)
    Nr = _hist_with_centers(coh_rand, coh_bins).astype(np.float64)
    nonzero_bins = np.where(Nr > 0)[0]
    Nr_max_nz_ix = float(nonzero_bins[-1] + 1) if nonzero_bins.size > 0 else 1.0

    weighting = np.divide(1.0, D_A, out=np.zeros_like(D_A, dtype=np.float64), where=D_A != 0)
    weighting_save = weighting.copy()
    gamma_change_save = 0.0
    coh_ps_save = np.zeros(n_ps, dtype=np.float64)
    K_ps = np.zeros(n_ps, dtype=np.float64)
    C_ps = np.zeros(n_ps, dtype=np.float64)
    coh_ps = np.zeros(n_ps, dtype=np.float64)
    N_opt = np.zeros(n_ps, dtype=np.float64)
    ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_patch = np.zeros((n_ps, n_ifg), dtype=np.complex64)
    ph_weight = np.zeros((n_ps, n_ifg), dtype=np.complex64)
    ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
    i_loop = 1
    last_gamma_change_change = np.nan

    def _stage2_pm_payload(loop_value: int) -> dict[str, Any]:
        return {
            "K_ps": _matlab_col(K_ps, np.float64),
            "C_ps": _matlab_col(C_ps, np.float64),
            "coh_ps": _matlab_col(coh_ps, np.float64),
            "N_opt": _matlab_col(N_opt, np.float64),
            "ph_res": ph_res,
            "ph_patch": ph_patch,
            "step_number": np.asarray(1.0, dtype=np.float64),
            "ph_grid": ph_grid,
            "n_trial_wraps": np.asarray(n_trial_wraps, dtype=np.float32),
            "grid_ij": grid_ij,
            "grid_size": np.asarray(grid_size, dtype=np.float64),
            "low_pass": low_pass,
            "i_loop": np.asarray(float(loop_value), dtype=np.float64),
            "ph_weight": ph_weight,
            "Nr": _matlab_row(Nr, np.float64),
            "Nr_max_nz_ix": np.asarray(Nr_max_nz_ix, dtype=np.float64),
            "coh_bins": _matlab_row(coh_bins, np.float64),
            "coh_ps_save": _matlab_col(coh_ps_save.copy(), np.float64),
            "gamma_change_save": np.asarray(gamma_change_save, dtype=np.float64),
        }

    def _write_stage2_pm(loop_value: int) -> None:
        write_mat(patch_dir / "pm1.mat", _stage2_pm_payload(loop_value))

    def _write_stage2_weighting_snapshot(
        iteration: int,
        Nr_curr: np.ndarray,
        Na_curr: np.ndarray,
        low_coh_thresh_curr: int,
        nr_max_nz_ix_curr: float,
        coh_ps_curr: np.ndarray,
        prand_curr: np.ndarray,
        prand_hi_curr: np.ndarray,
        prand_ps_curr: np.ndarray,
        weighting_curr: np.ndarray,
    ) -> None:
        if not debug:
            return
        payload = {
            "patch": patch_dir.name,
            "iteration": int(iteration),
            "filter_weighting": filter_weighting,
            "inputs": {
                "Nr": np.asarray(Nr_curr, dtype=np.float64).reshape(-1).tolist(),
                "Na": np.asarray(Na_curr, dtype=np.float64).reshape(-1).tolist(),
                "low_coh_thresh": int(low_coh_thresh_curr),
                "Nr_max_nz_ix": float(nr_max_nz_ix_curr),
                "coh_ps": np.asarray(coh_ps_curr, dtype=np.float64).reshape(-1).tolist(),
            },
            "outputs": {
                "prand": np.asarray(prand_curr, dtype=np.float64).reshape(-1).tolist(),
                "prand_hi": np.asarray(prand_hi_curr, dtype=np.float64).reshape(-1).tolist(),
                "prand_ps": np.asarray(prand_ps_curr, dtype=np.float64).reshape(-1).tolist(),
                "weighting": np.asarray(weighting_curr, dtype=np.float64).reshape(-1).tolist(),
            },
        }
        snapshot_text = json.dumps(payload, indent=2)
        for target in _stage2_weighting_snapshot_targets(patch_dir):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(snapshot_text, encoding="utf-8")

    while True:
        iter_t0 = time.perf_counter()
        phase_ramp = np.exp(-1j * (bperp_mat * K_ps[:, None])).astype(np.complex64)
        ph_weight = (ph_nm * phase_ramp * weighting[:, None]).astype(np.complex64)

        grid_t0 = time.perf_counter()
        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
        for i_ifg in range(n_ifg):
            flat = ph_grid[:, :, i_ifg].reshape(-1)
            np.add.at(flat, grid_lin, ph_weight[:, i_ifg])
        grid_dt = time.perf_counter() - grid_t0
        _emit_stage2(
            "grid_accumulated",
            iteration=i_loop,
            timings={
                "grid_accumulate": grid_dt,
                "total": time.perf_counter() - stage2_t0,
            },
        )

        filt_t0 = time.perf_counter()
        _emit_stage2(
            "clap_filter_in_progress",
            iteration=i_loop,
            extra={"filter_completed_ifg": 0},
            timings={
                "grid_accumulate": grid_dt,
                "clap_filter": 0.0,
                "total": time.perf_counter() - stage2_t0,
            },
        )
        ph_filt = _clap_filt_grid_stack(
            ph_grid,
            options.clap_alpha,
            options.clap_beta,
            int(round(options.clap_win * 0.75)),
            int(round(options.clap_win * 0.25)),
            low_pass,
        )
        _emit_stage2(
            "clap_filter_in_progress",
            iteration=i_loop,
            extra={"filter_completed_ifg": int(n_ifg)},
            timings={
                "grid_accumulate": grid_dt,
                "clap_filter": time.perf_counter() - filt_t0,
                "total": time.perf_counter() - stage2_t0,
            },
        )
        filt_dt = time.perf_counter() - filt_t0

        patch_t0 = time.perf_counter()
        ph_patch = ph_filt[grid_rows, grid_cols, :].astype(np.complex64, copy=False)
        nonzero = ph_patch != 0
        ph_patch = np.divide(ph_patch, np.abs(ph_patch), out=np.zeros_like(ph_patch), where=nonzero)
        patch_dt = time.perf_counter() - patch_t0

        topofit_t0 = time.perf_counter()
        psdph = (ph_nm * np.conj(ph_patch)).astype(np.complex64)
        valid_rows = np.all(psdph != 0, axis=1)
        K_new = np.full(n_ps, np.nan, dtype=np.float64)
        C_new = np.zeros(n_ps, dtype=np.float64)
        coh_new = np.zeros(n_ps, dtype=np.float64)
        N_new = np.zeros(n_ps, dtype=np.float64)
        ph_res_new = np.zeros((n_ps, n_ifg), dtype=np.float32)
        if np.any(valid_rows):
            K_chunk, C_chunk, coh_chunk, phase_residual = _ps_topofit_batch(
                psdph[valid_rows].astype(np.complex128),
                bperp_mat[valid_rows, :].astype(np.float64),
                n_trial_wraps,
            )
            K_new[valid_rows] = K_chunk
            C_new[valid_rows] = C_chunk
            coh_new[valid_rows] = coh_chunk
            N_new[valid_rows] = 1.0
            ph_res_new[valid_rows, :] = np.angle(phase_residual).astype(np.float32)
        topofit_dt = time.perf_counter() - topofit_t0

        K_ps = K_new
        C_ps = C_new
        coh_ps = coh_new
        N_opt = N_new
        ph_res = ph_res_new

        gamma_change_rms = float(np.sqrt(np.sum((coh_ps - coh_ps_save) ** 2) / max(1, n_ps)))
        gamma_change_change = gamma_change_rms - gamma_change_save
        gamma_change_save = gamma_change_rms
        coh_ps_save = coh_ps.copy()

        _emit_stage2(
            "iteration_complete",
            iteration=i_loop,
            timings={
                "grid_accumulate": grid_dt,
                "clap_filter": filt_dt,
                "patch_extract": patch_dt,
                "topofit": topofit_dt,
                "iteration_total": time.perf_counter() - iter_t0,
                "total": time.perf_counter() - stage2_t0,
            },
            extra={
                "valid_topofit_count": int(np.sum(valid_rows)),
                "invalid_topofit_count": int(n_ps - np.sum(valid_rows)),
                "coh_ps_nan_count": int(np.isnan(coh_ps).sum()),
                "coh_ps_zero_count": int(np.sum(coh_ps == 0)),
                "coh_ps_mean": float(np.nanmean(coh_ps)) if coh_ps.size else 0.0,
                "gamma_change_save": float(gamma_change_save),
                "gamma_change_change": float(gamma_change_change),
                "pm1_written": False,
            },
        )
        last_gamma_change_change = float(gamma_change_change)
        should_stop = abs(gamma_change_change) < gamma_change_convergence or i_loop >= gamma_max_iterations

        if not should_stop:
            weight_t0 = time.perf_counter()
            if filter_weighting.lower() == "p-square":
                Na = _hist_with_centers(coh_ps, coh_bins).astype(np.float64)
                denom = np.sum(Nr[:low_coh_thresh])
                scale = np.sum(Na[:low_coh_thresh]) / denom if denom > 0 else 1.0
                Nr = Nr * scale
                _prand, _prand_hi, _prand_ps, weighting = _stage2_psquare_weighting(
                    Nr,
                    Na,
                    low_coh_thresh,
                    Nr_max_nz_ix,
                    coh_ps,
                )
                _write_stage2_weighting_snapshot(
                    i_loop,
                    Nr,
                    Na,
                    low_coh_thresh,
                    Nr_max_nz_ix,
                    coh_ps,
                    _prand,
                    _prand_hi,
                    _prand_ps,
                    weighting,
                )
            else:
                g = np.mean(amp * np.cos(ph_res), axis=1)
                sigma_n = np.sqrt(0.5 * (np.mean(amp**2, axis=1) - g**2))
                weighting = np.zeros_like(g, dtype=np.float64)
                nz = sigma_n != 0
                weighting[nz] = g[nz] / sigma_n[nz]
            weight_dt = time.perf_counter() - weight_t0
            _emit_stage2(
                "weighting_updated",
                iteration=i_loop,
                timings={
                    "grid_accumulate": grid_dt,
                    "clap_filter": filt_dt,
                    "patch_extract": patch_dt,
                    "topofit": topofit_dt,
                    "weighting_update": weight_dt,
                    "iteration_total": time.perf_counter() - iter_t0,
                    "total": time.perf_counter() - stage2_t0,
                },
                extra={
                    "weighting_min": float(np.nanmin(weighting)) if weighting.size else 0.0,
                    "weighting_mean": float(np.nanmean(weighting)) if weighting.size else 0.0,
                    "weighting_max": float(np.nanmax(weighting)) if weighting.size else 0.0,
                    "gamma_change_change": float(gamma_change_change),
                    "pm1_written": False,
                },
            )
            i_loop += 1

        _write_stage2_pm(i_loop)
        _emit_stage2(
            "pm1_checkpoint_written",
            iteration=i_loop,
            timings={"total": time.perf_counter() - stage2_t0},
            extra={
                "pm1_written": True,
                "gamma_change_save": float(gamma_change_save),
                "gamma_change_change": float(last_gamma_change_change),
            },
        )

        if should_stop:
            break

    if debug:
        _emit_stage2(
            "completed",
            status="completed",
            iteration=i_loop,
            timings={"total": time.perf_counter() - stage2_t0},
            extra={
                "iterations_completed": i_loop,
                "ph_grid_shape": [int(v) for v in ph_grid.shape],
                "ifg_count": int(n_ifg),
                "coh_ps_nan_count": int(np.isnan(coh_ps).sum()),
                "coh_ps_zero_count": int(np.sum(coh_ps == 0)),
                "coh_ps_min": float(np.nanmin(coh_ps)) if coh_ps.size else 0.0,
                "coh_ps_mean": float(np.nanmean(coh_ps)) if coh_ps.size else 0.0,
                "coh_ps_max": float(np.nanmax(coh_ps)) if coh_ps.size else 0.0,
                "K_ps_nan_count": int(np.isnan(K_ps).sum()),
                "C_ps_nan_count": int(np.isnan(C_ps).sum()),
                "Nr_sum": float(np.sum(Nr)),
                "coh_bins_len": int(coh_bins.size),
                "gamma_change_change": float(last_gamma_change_change),
                "pm1_written": True,
            },
        )
    return f"Stage 2 computed coherence for {n_ps} candidates in {i_loop} iterations"


def stage3_select_ps(patch_dir: Path, backend: str = "auto") -> str:
    pm = read_mat(patch_dir / "pm1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    parms = _load_parms(patch_dir)
    debug_payload: dict[str, Any] = {
        "patch": patch_dir.name,
        "reestimate_used": False,
        "reestimate_status": "not_attempted",
        "reestimate_exception": None,
    }
    n_ps = int(round(_mat_scalar(ps.get("n_ps", 0), 0)))
    if n_ps <= 0:
        raise PortedStageError("ps1.mat missing valid n_ps")

    coh_ps = _as_ps_vector(pm.get("coh_ps"), n_ps, "pm1.coh_ps").astype(np.float64)
    if coh_ps.size == 0:
        raise PortedStageError("pm1.mat has empty coh_ps")

    coh_bins = np.asarray(pm.get("coh_bins"), dtype=np.float64).reshape(-1)
    Nr_dist = np.asarray(pm.get("Nr"), dtype=np.float64).reshape(-1)
    if coh_bins.size == 0:
        coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    if Nr_dist.size == 0:
        Nr_dist = np.ones(coh_bins.size, dtype=np.float64)

    da_file = patch_dir / "da1.mat"
    if da_file.exists():
        D_A = np.asarray(read_mat(da_file).get("D_A"), dtype=np.float64).reshape(-1)
    else:
        D_A = np.ones_like(coh_ps, dtype=np.float64)

    if D_A.size >= 10000:
        D_A_sort = np.sort(D_A)
        bin_size = 10000 if D_A.size >= 50000 else 2000
        D_A_max = np.concatenate(([0.0], D_A_sort[bin_size : D_A.size - bin_size : bin_size], [D_A_sort[-1]]))
    else:
        D_A_max = np.asarray([0.0, 1.0], dtype=np.float64)
        D_A = np.ones_like(coh_ps, dtype=np.float64)

    low_coh_thresh = 15 if parms.small_baseline_flag.lower() == "y" else 31

    if parms.select_method.upper() == "PERCENT":
        max_percent_rand = float(parms.percent_rand)
    else:
        xy = _as_ps_dim(ps.get("xy"), n_ps, 3, "ps1.xy").astype(np.float64)
        if xy.size == 0:
            patch_area = 1.0
        else:
            patch_area = np.prod(np.max(xy[:, 1:3], axis=0) - np.min(xy[:, 1:3], axis=0)) / 1e6
            if patch_area <= 0:
                patch_area = 1.0
        max_percent_rand = float(parms.density_rand) * patch_area / max(1, (D_A_max.size - 1))

    coh_thresh_all, coh_thresh_coeffs = _coh_threshold_from_dist(
        coh_values=coh_ps,
        D_A=D_A,
        D_A_max=D_A_max,
        coh_bins=coh_bins,
        Nr_dist=Nr_dist,
        low_coh_thresh=low_coh_thresh,
        max_percent_rand=max_percent_rand,
        select_method=parms.select_method,
    )
    debug_payload["initial_coh_thresh_coeffs"] = np.asarray(coh_thresh_coeffs, dtype=np.float64).reshape(-1).tolist()

    ix_mask = coh_ps > coh_thresh_all
    ix = np.where(ix_mask)[0] + 1  # MATLAB-style 1-based indices
    ix0 = ix - 1
    ifg_index = _ifg_index_for_selection(ps, parms)
    ifg_index_ix = np.asarray(ifg_index, dtype=np.int64).reshape(-1) - 1

    ph_patch = _as_ps_ifg_complex(pm.get("ph_patch"), n_ps, "pm1.ph_patch").astype(np.complex64)
    ph_res = _as_ps_matrix(pm.get("ph_res"), n_ps, "pm1.ph_res").astype(np.float32)
    K_ps = _as_ps_vector(pm.get("K_ps"), n_ps, "pm1.K_ps").astype(np.float64)
    C_ps = _as_ps_vector(pm.get("C_ps"), n_ps, "pm1.C_ps").astype(np.float64)

    ph_patch2 = ph_patch[ix0, :].astype(np.complex64, copy=True)
    ph_res2 = ph_res[ix0, :].astype(np.float32, copy=True)
    K_ps2 = K_ps[ix0].astype(np.float64, copy=True)
    C_ps2 = C_ps[ix0].astype(np.float64, copy=True)
    coh_ps2 = coh_ps[ix0].astype(np.float64, copy=True)
    keep_ix = np.ones(ix.size, dtype=bool)

    if parms.gamma_stdev_reject > 0 and ix.size > 0 and ifg_index_ix.size > 0:
        ph_res_cpx = np.exp(1j * ph_res[:, ifg_index_ix])
        coh_std = np.zeros(ix.size, dtype=np.float64)
        rng = np.random.default_rng(0)
        for row_i, ps_i in enumerate(ix0):
            sample = ph_res_cpx[ps_i, :]
            n_sample = sample.size
            if n_sample == 0:
                coh_std[row_i] = np.inf
                continue
            draw_ix = rng.integers(0, n_sample, size=(100, n_sample))
            boot = sample[draw_ix]
            coh_boot = np.abs(np.sum(boot, axis=1)) / float(n_sample)
            coh_std[row_i] = float(np.std(coh_boot))
        ix_mask_reject = coh_std < float(parms.gamma_stdev_reject)
        ix = ix[ix_mask_reject]
        ix0 = ix - 1

    if ix.size > 0:
        reestimate_ok = True
        ph_grid = _coerce_complex(pm.get("ph_grid")).astype(np.complex64)
        if ph_grid.ndim != 3 or ph_grid.shape[0] < 2 or ph_grid.shape[1] < 2:
            reestimate_ok = False

        try:
            grid_ij = _as_ps_dim(pm.get("grid_ij"), n_ps, 2, "pm1.grid_ij").astype(np.int64)
            if grid_ij.size == 0:
                reestimate_ok = False
        except Exception:
            reestimate_ok = False
            grid_ij = np.empty((0, 2), dtype=np.int64)

        bp1_file = patch_dir / "bp1.mat"
        if not bp1_file.exists():
            reestimate_ok = False

        if reestimate_ok:
            try:
                debug_payload["reestimate_status"] = "running"
                ph_all = _as_ps_ifg_complex(read_mat(patch_dir / "ph1.mat").get("ph"), n_ps, "ph1.ph").astype(np.complex64)
                bperp_full = np.asarray(ps.get("bperp"), dtype=np.float64).reshape(-1)
                if parms.small_baseline_flag.lower() == "y":
                    ph_work = ph_all
                    bperp_work = bperp_full
                else:
                    master_ix = int(round(_mat_scalar(ps.get("master_ix", 1), 1)))
                    no_master_ix = np.arange(ph_all.shape[1]) != (master_ix - 1)
                    ph_work = ph_all[:, no_master_ix]
                    bperp_work = bperp_full[no_master_ix]

                n_ifg_work = ph_work.shape[1]
                ifg_index_ix = ifg_index_ix[(ifg_index_ix >= 0) & (ifg_index_ix < n_ifg_work)]
                if ifg_index_ix.size == 0:
                    reestimate_ok = False
                else:
                    ph_patch2 = ph_patch[ix0, :].astype(np.complex64, copy=True)
                    ph_res2 = np.zeros((ix.size, n_ifg_work), dtype=np.float32)
                    K_ps2 = np.zeros(ix.size, dtype=np.float64)
                    C_ps2 = np.zeros(ix.size, dtype=np.float64)
                    coh_ps2 = np.zeros(ix.size, dtype=np.float64)
                    keep_ix = np.ones(ix.size, dtype=bool)

                    options = _build_stage_options(patch_dir)
                    n_win = int(round(options.clap_win))
                    if n_win <= 0:
                        n_win = 32
                    half_win = n_win // 2
                    alpha = float(options.clap_alpha)
                    beta = float(options.clap_beta)
                    low_pass = np.asarray(pm.get("low_pass"), dtype=np.float64)
                    if low_pass.shape != (n_win, n_win):
                        low_pass = _build_low_pass(options)

                    n_i = int(np.max(grid_ij[:, 0]))
                    n_j = int(np.max(grid_ij[:, 1]))
                    slc_osf = max(1, int(round(float(parms.slc_osf))))

                    grid_sel = grid_ij[ix0, :]
                    unique_cells, inverse = np.unique(grid_sel, axis=0, return_inverse=True)
                    for cell_id, (gi, gj) in enumerate(unique_cells):
                        ps_ij_i = int(gi)
                        ps_ij_j = int(gj)

                        i_min = max(ps_ij_i - half_win, 1)
                        i_max = i_min + n_win - 1
                        if i_max > n_i:
                            i_min = i_min - i_max + n_i
                            i_max = n_i
                        j_min = max(ps_ij_j - half_win, 1)
                        j_max = j_min + n_win - 1
                        if j_max > n_j:
                            j_min = j_min - j_max + n_j
                            j_max = n_j

                        rows = np.where(inverse == cell_id)[0]
                        if i_min < 1 or j_min < 1:
                            ph_patch2[rows, :] = 0
                            continue

                        ps_bit_i = ps_ij_i - i_min + 1
                        ps_bit_j = ps_ij_j - j_min + 1
                        ph_bit = ph_grid[i_min - 1 : i_max, j_min - 1 : j_max, :].copy()
                        ph_bit[ps_bit_i - 1, ps_bit_j - 1, :] = 0

                        rad = slc_osf - 1
                        ii = np.arange(ps_bit_i - rad, ps_bit_i + rad + 1, dtype=np.int64)
                        jj = np.arange(ps_bit_j - rad, ps_bit_j + rad + 1, dtype=np.int64)
                        ii = ii[(ii > 0) & (ii <= ph_bit.shape[0])] - 1
                        jj = jj[(jj > 0) & (jj <= ph_bit.shape[1])] - 1
                        if ii.size and jj.size:
                            ph_bit[np.ix_(ii, jj)] = 0

                        ph_filt_stack = _clap_filt_patch_stack(ph_bit, alpha, beta, low_pass)
                        ph_center = ph_filt_stack[ps_bit_i - 1, ps_bit_j - 1, :]
                        ph_patch2[rows, :] = ph_center[None, :]

                    psdph = ph_work[ix0, :] * np.conj(ph_patch2)
                    valid_rows = np.all(psdph != 0, axis=1)
                    bad_rows = ~valid_rows
                    if np.any(bad_rows):
                        K_ps2[bad_rows] = np.nan
                        coh_ps2[bad_rows] = np.nan

                    bperp_mat = _as_ps_matrix(read_mat(bp1_file).get("bperp_mat"), n_ps, "bp1.bperp_mat").astype(np.float64)
                    n_trial_wraps = float(_mat_scalar(pm.get("n_trial_wraps", 0.0), 0.0))
                    rows = np.where(valid_rows)[0]
                    if rows.size > 0:
                        chunk_size = 4000
                        for start in range(0, rows.size, chunk_size):
                            row_ix = rows[start : start + chunk_size]
                            cpx = psdph[row_ix][:, ifg_index_ix]
                            cpx = np.divide(cpx, np.abs(cpx), out=np.zeros_like(cpx), where=np.abs(cpx) != 0)
                            bp = bperp_mat[ix0[row_ix], :][:, ifg_index_ix]
                            K_chunk, C_chunk, coh_chunk, phase_residual = _ps_topofit_batch(cpx, bp, n_trial_wraps)
                            K_ps2[row_ix] = K_chunk
                            C_ps2[row_ix] = C_chunk
                            coh_ps2[row_ix] = coh_chunk
                            ph_res2[np.ix_(row_ix, ifg_index_ix)] = np.angle(phase_residual).astype(np.float32)

                    coh_for_threshold = coh_ps.copy()
                    coh_for_threshold[ix0] = coh_ps2
                    coh_thresh_re_all, _ = _coh_threshold_from_dist(
                        coh_values=coh_for_threshold,
                        D_A=D_A,
                        D_A_max=D_A_max,
                        coh_bins=coh_bins,
                        Nr_dist=Nr_dist,
                        low_coh_thresh=low_coh_thresh,
                        max_percent_rand=max_percent_rand,
                        select_method=parms.select_method,
                    )
                    coh_thresh_sel = coh_thresh_re_all[ix0]
                    coh_thresh_sel[coh_thresh_sel < 0] = 0
                    coh_thresh_all[ix0] = coh_thresh_sel

                    bperp_range = float(np.max(bperp_work) - np.min(bperp_work))
                    if bperp_range <= 0:
                        bperp_range = 1.0
                    eps_keep = 1e-6
                    keep_ix = (coh_ps2 > (coh_thresh_sel + eps_keep)) & (
                        np.abs(K_ps[ix0] - K_ps2) < (2 * np.pi / bperp_range)
                    )
                    debug_payload["reestimate_used"] = True
                    debug_payload["reestimate_status"] = "completed"
            except Exception as exc:
                reestimate_ok = False
                debug_payload["reestimate_status"] = "failed"
                debug_payload["reestimate_exception"] = f"{type(exc).__name__}: {exc}"

        if not reestimate_ok:
            ph_patch2 = ph_patch[ix0, :].astype(np.complex64, copy=True)
            ph_res2 = ph_res[ix0, :].astype(np.float32, copy=True)
            K_ps2 = K_ps[ix0].astype(np.float64, copy=True)
            C_ps2 = C_ps[ix0].astype(np.float64, copy=True)
            coh_ps2 = coh_ps[ix0].astype(np.float64, copy=True)
            keep_ix = np.ones(ix.size, dtype=bool)
    else:
        ph_patch2 = np.empty((0, ph_patch.shape[1]), dtype=np.complex64)
        ph_res2 = np.empty((0, ph_res.shape[1]), dtype=np.float32)
        K_ps2 = np.empty((0,), dtype=np.float64)
        C_ps2 = np.empty((0,), dtype=np.float64)
        coh_ps2 = np.empty((0,), dtype=np.float64)
        keep_ix = np.empty((0,), dtype=bool)
    payload: dict[str, Any] = {
        "ix": _matlab_col(ix, np.float64),
        "keep_ix": _matlab_col(keep_ix, np.bool_),
        "ph_patch2": ph_patch2,
        "ph_res2": ph_res2,
        "K_ps2": _matlab_col(K_ps2, np.float64),
        "C_ps2": _matlab_col(C_ps2, np.float64),
        "coh_ps2": _matlab_col(coh_ps2, np.float64),
        "coh_thresh": _matlab_col(coh_thresh_all[ix0], np.float64),
        "coh_thresh_coeffs": coh_thresh_coeffs,
        "clap_alpha": np.asarray(_build_stage_options(patch_dir).clap_alpha, dtype=np.float64),
        "clap_beta": np.asarray(_build_stage_options(patch_dir).clap_beta, dtype=np.float64),
        "n_win": np.asarray(_build_stage_options(patch_dir).clap_win, dtype=np.float64),
        "max_percent_rand": np.asarray(max_percent_rand, dtype=np.float32),
        "gamma_stdev_reject": np.asarray(parms.gamma_stdev_reject, dtype=np.float64),
        "small_baseline_flag": _matlab_char_row(parms.small_baseline_flag),
        "ifg_index": _matlab_row(ifg_index, np.float64),
    }

    write_mat(patch_dir / "select1.mat", payload)
    debug_payload.update(
        {
            "ix_count": int(ix.size),
            "keep_true_count": int(np.count_nonzero(keep_ix)),
            "coh_thresh_coeffs": np.asarray(coh_thresh_coeffs, dtype=np.float64).reshape(-1).tolist(),
            "max_percent_rand": float(max_percent_rand),
            "gamma_stdev_reject": float(parms.gamma_stdev_reject),
        }
    )
    _write_stage3_debug(patch_dir, debug_payload)
    return f"Stage 3 selected {ix.size} PS"


def _stage4_checkpoint(
    patch_dir: Path,
    payload: dict[str, Any] | None,
    *,
    status: str | None = None,
    phase: str | None = None,
    last_completed_ifg: int | None = None,
    timings: dict[str, float] | None = None,
) -> None:
    if payload is None:
        return
    if status is not None:
        payload["status"] = status
    if phase is not None:
        payload["phase"] = phase
    if last_completed_ifg is not None:
        payload["last_completed_ifg"] = int(last_completed_ifg)
    payload["updated_at_epoch_sec"] = time.time()
    if timings is not None:
        payload["timings_sec"] = timings
    _write_stage4_debug(patch_dir, payload)


def stage4_weed_ps(
    patch_dir: Path,
    backend: str = "auto",
    debug: bool = False,
    strict_reference: bool = False,
) -> str:
    stage4_t0 = time.perf_counter()
    sel = read_mat(patch_dir / "select1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    parms = _load_parms(patch_dir)
    debug_payload: dict[str, Any] | None = None
    if debug:
        debug_payload = {
            "patch": patch_dir.name,
            "backend": backend,
            "small_baseline_flag": str(parms.small_baseline_flag),
            "weed_neighbours": str(parms.weed_neighbours),
            "weed_zero_elevation": str(parms.weed_zero_elevation),
            "weed_standard_dev": float(parms.weed_standard_dev),
            "weed_max_noise": float(parms.weed_max_noise),
            "strict_reference": bool(strict_reference),
            "status": "started",
            "phase": "load_inputs",
            "last_completed_ifg": 0,
        }
    n_ps_total = int(round(_mat_scalar(ps.get("n_ps", 0), 0)))
    if n_ps_total <= 0:
        raise PortedStageError("ps1.mat missing valid n_ps")

    ix = np.asarray(sel.get("ix"), dtype=np.int64).reshape(-1)
    if ix.size == 0:
        raise PortedStageError("select1.mat has empty ix")

    keep_ix = np.asarray(sel.get("keep_ix", np.ones(ix.size, dtype=bool))).reshape(-1).astype(bool)
    if keep_ix.size != ix.size:
        keep_ix = np.ones(ix.size, dtype=bool)
    ix2 = ix[keep_ix]  # MATLAB 1-based
    if debug_payload is not None:
        debug_payload["selected_input_count"] = int(ix.size)
        debug_payload["selected_keep_count"] = int(ix2.size)
        _stage4_checkpoint(
            patch_dir,
            debug_payload,
            phase="selected_inputs_ready",
            timings={"total": time.perf_counter() - stage4_t0},
        )

    if ix2.size == 0:
        payload = {
            "ifg_index": _matlab_row(_ifg_index_for_weed(ps, parms), np.float64),
            "ix_weed": np.empty((0, 1), dtype=np.uint8),
            "ix_weed2": np.empty((0, 1), dtype=np.uint8),
            "ps_max": np.empty((0, 1), dtype=np.float32),
            "ps_std": np.empty((0, 1), dtype=np.float32),
        }
        write_mat(patch_dir / "weed1.mat", payload)
        if debug_payload is not None:
            debug_payload["count_after_adjacency"] = 0
            debug_payload["count_after_zero_elevation"] = 0
            debug_payload["count_after_duplicate_removal"] = 0
            debug_payload["count_before_noise_filter"] = 0
            debug_payload["count_after_noise_filter"] = 0
            debug_payload["final_retained_count"] = 0
            debug_payload["edge_source"] = "none"
            debug_payload["edge_count"] = 0
            debug_payload["ifg_count_used"] = 0
            _stage4_checkpoint(
                patch_dir,
                debug_payload,
                status="completed",
                phase="completed",
                timings={"total": time.perf_counter() - stage4_t0},
            )
        return "Stage 4 retained 0/0 selected PS"

    coh_ps2 = _as_ps_vector(sel.get("coh_ps2"), ix.size, "select1.coh_ps2").astype(np.float64)[keep_ix]
    K_ps2 = _as_ps_vector(sel.get("K_ps2"), ix.size, "select1.K_ps2").astype(np.float64)[keep_ix]
    C_ps2 = _as_ps_vector(sel.get("C_ps2"), ix.size, "select1.C_ps2").astype(np.float64)[keep_ix]

    ij_all = _as_ps_dim(ps.get("ij"), n_ps_total, 3, "ps1.ij").astype(np.float64)
    xy_all = _as_ps_dim(ps.get("xy"), n_ps_total, 3, "ps1.xy").astype(np.float64)
    ij2 = ij_all[ix2 - 1, :]
    xy2 = xy_all[ix2 - 1, :]

    n_ps = ix2.size
    ix_weed = np.ones(n_ps, dtype=bool)

    adjacency_t0 = time.perf_counter()
    if parms.weed_neighbours.lower() == "y":
        keep_adj = _adjacent_component_keep_mask(ij2[:, 1:3].astype(np.int64), coh_ps2)
        ix_weed &= keep_adj
    adjacency_dt = time.perf_counter() - adjacency_t0
    if debug_payload is not None:
        debug_payload["count_after_adjacency"] = int(np.sum(ix_weed))
        _stage4_checkpoint(
            patch_dir,
            debug_payload,
            phase="adjacency_done",
            timings={"adjacency": adjacency_dt, "total": time.perf_counter() - stage4_t0},
        )

    zero_elev_t0 = time.perf_counter()
    if parms.weed_zero_elevation.lower() == "y":
        hgt_file = patch_dir / "hgt1.mat"
        if hgt_file.exists():
            hgt = np.asarray(read_mat(hgt_file).get("hgt"), dtype=np.float32).reshape(-1)
            hgt2 = hgt[ix2 - 1]
            ix_weed[hgt2 < 1e-6] = False
    zero_elev_dt = time.perf_counter() - zero_elev_t0
    if debug_payload is not None:
        debug_payload["count_after_zero_elevation"] = int(np.sum(ix_weed))
        _stage4_checkpoint(
            patch_dir,
            debug_payload,
            phase="zero_elevation_done",
            timings={
                "adjacency": adjacency_dt,
                "zero_elevation": zero_elev_dt,
                "total": time.perf_counter() - stage4_t0,
            },
        )

    # Remove duplicate lon/lat among currently weeded-in points only.
    duplicate_t0 = time.perf_counter()
    if np.any(ix_weed):
        ix_weed_num = np.where(ix_weed)[0]
        xy_weed = xy2[ix_weed, :]
        _, inverse, counts = np.unique(xy_weed[:, 1:3], axis=0, return_inverse=True, return_counts=True)
        dup_groups = np.where(counts > 1)[0]
        for grp in dup_groups:
            loc = np.where(inverse == grp)[0]
            if loc.size <= 1:
                continue
            orig_ix = ix_weed_num[loc]
            best = orig_ix[np.argmax(coh_ps2[orig_ix])]
            drop = orig_ix[orig_ix != best]
            ix_weed[drop] = False
    duplicate_dt = time.perf_counter() - duplicate_t0
    if debug_payload is not None:
        debug_payload["count_after_duplicate_removal"] = int(np.sum(ix_weed))
        _stage4_checkpoint(
            patch_dir,
            debug_payload,
            phase="duplicates_done",
            timings={
                "adjacency": adjacency_dt,
                "zero_elevation": zero_elev_dt,
                "duplicate_removal": duplicate_dt,
                "total": time.perf_counter() - stage4_t0,
            },
        )

    n_pre_noise = int(np.sum(ix_weed))
    ix_weed2 = np.ones(n_pre_noise, dtype=bool)
    # MATLAB carries the edge statistics in double precision and only
    # quantizes at save time; reducing in float32 shifts the retained minima.
    ps_std = np.zeros(n_pre_noise, dtype=np.float64)
    ps_max = np.zeros(n_pre_noise, dtype=np.float64)
    edge_source = "none"
    edge_count = 0
    ifg_count_used = 0
    edge_build_dt = 0.0
    ph_prep_dt = 0.0
    smooth_dt = 0.0
    edge_reduce_dt = 0.0

    no_weed_noisy = bool(parms.weed_standard_dev >= np.pi and parms.weed_max_noise >= np.pi)
    if not no_weed_noisy and n_pre_noise > 0:
        ph2 = _as_ps_ifg_complex(read_mat(patch_dir / "ph1.mat").get("ph"), n_ps_total, "ph1.ph")[ix2 - 1, :].astype(
            np.complex128
        )
        bperp = np.asarray(ps.get("bperp"), dtype=np.float64).reshape(-1)
        ifg_index = _ifg_index_for_weed(ps, parms)
        ifg_index_ix = np.asarray(ifg_index, dtype=np.int64).reshape(-1) - 1
        ifg_index_ix = ifg_index_ix[(ifg_index_ix >= 0) & (ifg_index_ix < ph2.shape[1])]
        ifg_count_used = int(ifg_index_ix.size)

        xy_weed = xy2[ix_weed, :]
        edge_file = patch_dir / "psweed.2.edge"
        edge_t0 = time.perf_counter()
        edges = _load_triangle_edges(edge_file, n_pre_noise)
        if edges.size > 0:
            edge_source = "triangle_file"
        if edges.size == 0:
            if strict_reference:
                if debug_payload is not None:
                    debug_payload["count_before_noise_filter"] = int(n_pre_noise)
                    debug_payload["edge_source"] = "missing_or_invalid_triangle_file"
                    debug_payload["edge_count"] = 0
                    debug_payload["ifg_count_used"] = int(ifg_count_used)
                    _stage4_checkpoint(
                        patch_dir,
                        debug_payload,
                        status="failed",
                        phase="edge_build_failed",
                        timings={
                            "adjacency": adjacency_dt,
                            "zero_elevation": zero_elev_dt,
                            "duplicate_removal": duplicate_dt,
                            "edge_build": time.perf_counter() - edge_t0,
                            "total": time.perf_counter() - stage4_t0,
                        },
                    )
                raise PortedStageError(
                    "Strict reference parity requires valid psweed.2.edge; Delaunay fallback is disabled"
                )
            edges = _delaunay_edges(xy_weed[:, 1:3].astype(np.float64))
            edge_source = "delaunay_fallback"
        edge_build_dt = time.perf_counter() - edge_t0
        n_edge = edges.shape[0]
        edge_count = int(n_edge)
        if debug_payload is not None:
            debug_payload["count_before_noise_filter"] = int(n_pre_noise)
            debug_payload["edge_source"] = edge_source
            debug_payload["edge_count"] = edge_count
            debug_payload["ifg_count_used"] = int(ifg_count_used)
            _stage4_checkpoint(
                patch_dir,
                debug_payload,
                phase="edge_build_done",
                timings={
                    "adjacency": adjacency_dt,
                    "zero_elevation": zero_elev_dt,
                    "duplicate_removal": duplicate_dt,
                    "edge_build": edge_build_dt,
                    "total": time.perf_counter() - stage4_t0,
                },
            )
        ps_std = np.full(n_pre_noise, np.inf, dtype=np.float64)
        ps_max = np.full(n_pre_noise, np.inf, dtype=np.float64)

        if n_edge > 0 and ifg_index_ix.size > 0:
            ph_prep_t0 = time.perf_counter()
            ph_weed = ph2[ix_weed, :] * np.exp(-1j * (K_ps2[ix_weed][:, None] * bperp[None, :]))
            ph_weed = np.divide(ph_weed, np.abs(ph_weed), out=np.zeros_like(ph_weed), where=np.abs(ph_weed) != 0)
            if parms.small_baseline_flag.lower() != "y":
                master_ix = int(round(_mat_scalar(ps.get("master_ix", 1), 1)))
                ph_weed[:, master_ix - 1] = np.exp(1j * C_ps2[ix_weed])
            ph_prep_dt = time.perf_counter() - ph_prep_t0
            if debug_payload is not None:
                _stage4_checkpoint(
                    patch_dir,
                    debug_payload,
                    phase="phase_prep_done",
                    timings={
                        "adjacency": adjacency_dt,
                        "zero_elevation": zero_elev_dt,
                        "duplicate_removal": duplicate_dt,
                        "edge_build": edge_build_dt,
                        "ph_prepare": ph_prep_dt,
                        "total": time.perf_counter() - stage4_t0,
                    },
                )

            dph_space = ph_weed[edges[:, 1], :] * np.conj(ph_weed[edges[:, 0], :])
            dph_space = dph_space[:, ifg_index_ix]
            n_use = dph_space.shape[1]
            b_use = bperp[ifg_index_ix].astype(np.float64)

            if parms.small_baseline_flag.lower() != "y":
                day = np.asarray(ps.get("day"), dtype=np.float64).reshape(-1)
                time_win = max(float(parms.weed_time_win), 1e-6)
                day_use = day[ifg_index_ix].astype(np.float64)
                time_diff_all = day_use[:, None] - day_use[None, :]
                weight_all = np.exp(-(time_diff_all**2) / (2.0 * time_win**2))
                weight_sums = np.sum(weight_all, axis=1, keepdims=True)
                zero_rows = weight_sums[:, 0] <= 0
                if np.any(zero_rows):
                    weight_all[zero_rows, :] = 1.0 / float(max(1, n_use))
                    weight_sums = np.sum(weight_all, axis=1, keepdims=True)
                weight_all = weight_all / weight_sums
                diag_weights = np.diag(weight_all).copy()
                dph_smooth = dph_space @ weight_all.T
                dph_smooth2 = dph_smooth - (dph_space * diag_weights[None, :])
                checkpoint_every = max(1, n_use // 20)
                if debug_payload is not None:
                    debug_payload["smoothing_ifg_count"] = int(n_use)
                    debug_payload["smoothing_checkpoint_every"] = int(checkpoint_every)
                    _stage4_checkpoint(
                        patch_dir,
                        debug_payload,
                        phase="smoothing_started",
                        timings={
                            "adjacency": adjacency_dt,
                            "zero_elevation": zero_elev_dt,
                            "duplicate_removal": duplicate_dt,
                            "edge_build": edge_build_dt,
                            "ph_prepare": ph_prep_dt,
                            "total": time.perf_counter() - stage4_t0,
                        },
                    )
                smooth_t0 = time.perf_counter()
                max_workers = min(4, max(1, os.cpu_count() or 1), n_use)

                def _smooth_one_ifg(i1: int) -> int:
                    time_diff = time_diff_all[i1]
                    weight = weight_all[i1]
                    dph_mean = dph_smooth[:, i1].copy()
                    dph_mean_adj = np.angle(dph_space * np.conj(dph_mean)[:, None])
                    m0, m1 = _weighted_affine_fit(time_diff, dph_mean_adj, weight)
                    detrended = dph_mean_adj - (m0[:, None] + m1[:, None] * time_diff[None, :])
                    dph_mean_adj2 = np.angle(np.exp(1j * detrended))
                    m20, _m21 = _weighted_affine_fit(time_diff, dph_mean_adj2, weight)
                    dph_smooth[:, i1] = dph_mean * np.exp(1j * (m0 + m20))
                    return i1 + 1

                completed_ifg = 0
                if max_workers <= 1:
                    for i1 in range(n_use):
                        completed_ifg = _smooth_one_ifg(i1)
                        if debug_payload is not None and (
                            (completed_ifg % checkpoint_every) == 0 or completed_ifg == n_use
                        ):
                            _stage4_checkpoint(
                                patch_dir,
                                debug_payload,
                                phase="smoothing_in_progress",
                                last_completed_ifg=completed_ifg,
                                timings={
                                    "adjacency": adjacency_dt,
                                    "zero_elevation": zero_elev_dt,
                                    "duplicate_removal": duplicate_dt,
                                    "edge_build": edge_build_dt,
                                    "ph_prepare": ph_prep_dt,
                                    "smoothing": time.perf_counter() - smooth_t0,
                                    "total": time.perf_counter() - stage4_t0,
                                },
                            )
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        for completed_ifg in executor.map(_smooth_one_ifg, range(n_use)):
                            if debug_payload is not None and (
                                (completed_ifg % checkpoint_every) == 0 or completed_ifg == n_use
                            ):
                                _stage4_checkpoint(
                                    patch_dir,
                                    debug_payload,
                                    phase="smoothing_in_progress",
                                    last_completed_ifg=completed_ifg,
                                    timings={
                                        "adjacency": adjacency_dt,
                                        "zero_elevation": zero_elev_dt,
                                        "duplicate_removal": duplicate_dt,
                                        "edge_build": edge_build_dt,
                                        "ph_prepare": ph_prep_dt,
                                        "smoothing": time.perf_counter() - smooth_t0,
                                        "total": time.perf_counter() - stage4_t0,
                                    },
                                )
                smooth_dt = time.perf_counter() - smooth_t0

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
                K_edge = _weighted_slope_fit(b_use, dph_noise, w_ifg.astype(np.float64))
                dph_noise = dph_noise - K_edge[:, None] * b_use[None, :]
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
                K_edge = _weighted_slope_fit(b_use, dph_space, w_ifg.astype(np.float64))
                dph_adj = dph_space - K_edge[:, None] * b_use[None, :]
                ang = np.angle(dph_adj)
                ddof = 1 if n_use > 1 else 0
                edge_std = np.std(ang, axis=1, ddof=ddof)
                edge_max = np.max(np.abs(ang), axis=1)

            reduce_t0 = time.perf_counter()
            np.minimum.at(ps_std, edges[:, 0], edge_std)
            np.minimum.at(ps_std, edges[:, 1], edge_std)
            np.minimum.at(ps_max, edges[:, 0], edge_max)
            np.minimum.at(ps_max, edges[:, 1], edge_max)
            edge_reduce_dt = time.perf_counter() - reduce_t0
            if debug_payload is not None:
                _stage4_checkpoint(
                    patch_dir,
                    debug_payload,
                    phase="edge_reduce_done",
                    timings={
                        "adjacency": adjacency_dt,
                        "zero_elevation": zero_elev_dt,
                        "duplicate_removal": duplicate_dt,
                        "edge_build": edge_build_dt,
                        "ph_prepare": ph_prep_dt,
                        "smoothing": smooth_dt,
                        "edge_reduce": edge_reduce_dt,
                        "total": time.perf_counter() - stage4_t0,
                    },
                )

        ix_weed2 = (ps_std < float(parms.weed_standard_dev)) & (ps_max < float(parms.weed_max_noise))
        ix_weed_idx = np.where(ix_weed)[0]
        ix_weed[ix_weed_idx] = ix_weed2

    ifg_index = _ifg_index_for_weed(ps, parms)
    payload = {
        "ifg_index": _matlab_row(ifg_index, np.float64),
        "ix_weed": _matlab_col(ix_weed.astype(np.uint8), np.uint8),
        "ix_weed2": _matlab_col(ix_weed2.astype(np.uint8), np.uint8),
        "ps_max": _matlab_col(ps_max.astype(np.float32), np.float32),
        "ps_std": _matlab_col(ps_std.astype(np.float32), np.float32),
    }

    write_mat(patch_dir / "weed1.mat", payload)
    if debug_payload is not None:
        debug_payload["count_before_noise_filter"] = int(n_pre_noise)
        debug_payload["count_after_noise_filter"] = int(np.sum(ix_weed2))
        debug_payload["final_retained_count"] = int(np.sum(ix_weed))
        debug_payload["edge_source"] = edge_source
        debug_payload["edge_count"] = edge_count
        debug_payload["ifg_count_used"] = ifg_count_used
        _stage4_checkpoint(
            patch_dir,
            debug_payload,
            status="completed",
            phase="completed",
            timings={
                "adjacency": adjacency_dt,
                "zero_elevation": zero_elev_dt,
                "duplicate_removal": duplicate_dt,
                "edge_build": edge_build_dt,
                "ph_prepare": ph_prep_dt,
                "smoothing": smooth_dt,
                "edge_reduce": edge_reduce_dt,
                "total": time.perf_counter() - stage4_t0,
            },
        )
    return f"Stage 4 retained {int(np.sum(ix_weed))}/{ix_weed.size} selected PS"


def stage5_correct_and_promote(patch_dir: Path, backend: str = "auto") -> str:
    ps1 = read_mat(patch_dir / "ps1.mat")
    pm1 = read_mat(patch_dir / "pm1.mat")
    sel = read_mat(patch_dir / "select1.mat")
    weed = read_mat(patch_dir / "weed1.mat")
    parms = _load_parms(patch_dir)

    n_ps1 = int(round(_mat_scalar(ps1.get("n_ps", 0), 0)))
    if n_ps1 <= 0:
        raise PortedStageError("ps1.mat missing valid n_ps")

    ph1 = _as_ps_ifg_complex(read_mat(patch_dir / "ph1.mat").get("ph"), n_ps1, "ph1.ph")
    ij1 = _as_ps_dim(ps1.get("ij"), n_ps1, 3, "ps1.ij").astype(np.float64)
    lonlat1 = _as_ps_dim(ps1.get("lonlat"), n_ps1, 2, "ps1.lonlat").astype(np.float64)
    xy1 = _as_ps_dim(ps1.get("xy"), n_ps1, 3, "ps1.xy").astype(np.float32)

    ix = np.asarray(sel.get("ix"), dtype=np.int64).reshape(-1)  # 1-based
    if ix.size == 0:
        raise PortedStageError("select1.mat has empty ix")

    keep_ix = np.asarray(sel.get("keep_ix", np.ones(ix.size, dtype=bool))).reshape(-1).astype(bool)
    if keep_ix.size != ix.size:
        keep_ix = np.ones(ix.size, dtype=bool)
    ix2 = ix[keep_ix]  # MATLAB stage4 input indices

    ix_weed = np.asarray(weed.get("ix_weed"), dtype=bool).reshape(-1)
    if ix_weed.size == ix2.size:
        final_ix1 = ix2[ix_weed]
    else:
        final_ix1 = ix2
    final_ix = (final_ix1 - 1).astype(np.int64)

    ps2: dict[str, Any] = {
        "bperp": _matlab_col(np.asarray(ps1.get("bperp"), dtype=np.float32), np.float32),
        "day": _matlab_col(np.asarray(ps1.get("day"), dtype=np.float64), np.float64),
        "ij": ij1[final_ix, :],
        "ll0": np.asarray(ps1.get("ll0"), dtype=np.float64),
        "lonlat": lonlat1[final_ix, :],
        "master_day": np.asarray(ps1.get("master_day"), dtype=np.float64),
        "master_ix": np.asarray(ps1.get("master_ix"), dtype=np.float64),
        "n_ifg": np.asarray(ps1.get("n_ifg"), dtype=np.float64),
        "n_image": np.asarray(ps1.get("n_image"), dtype=np.float64),
        "n_ps": np.asarray(final_ix.size, dtype=np.float64),
        "xy": xy1[final_ix, :],
    }
    if "mean_incidence" in ps1:
        ps2["mean_incidence"] = np.asarray(ps1.get("mean_incidence"), dtype=np.float64)
    if "mean_range" in ps1:
        ps2["mean_range"] = np.asarray(ps1.get("mean_range"), dtype=np.float64)

    ph2 = ph1[final_ix, :].astype(np.complex64)

    K_ps2 = _as_ps_vector(sel.get("K_ps2"), ix.size, "select1.K_ps2").astype(np.float64)[keep_ix]
    C_ps2 = _as_ps_vector(sel.get("C_ps2"), ix.size, "select1.C_ps2").astype(np.float64)[keep_ix]
    coh_ps2 = _as_ps_vector(sel.get("coh_ps2"), ix.size, "select1.coh_ps2").astype(np.float64)[keep_ix]
    ph_res2_all = _as_ps_matrix(sel.get("ph_res2"), ix.size, "select1.ph_res2").astype(np.float32)[keep_ix, :]

    ph_patch_all = _as_ps_ifg_complex(pm1.get("ph_patch"), n_ps1, "pm1.ph_patch")
    ph_patch2 = ph_patch_all[ix2 - 1, :]
    if ix_weed.size == ix2.size:
        K_ps = K_ps2[ix_weed]
        C_ps = C_ps2[ix_weed]
        coh_ps = coh_ps2[ix_weed]
        ph_patch = ph_patch2[ix_weed, :]
        ph_res = ph_res2_all[ix_weed, :]
    else:
        K_ps = K_ps2
        C_ps = C_ps2
        coh_ps = coh_ps2
        ph_patch = ph_patch2
        ph_res = ph_res2_all

    pm2 = {
        "K_ps": _matlab_col(K_ps.astype(np.float64), np.float64),
        "C_ps": _matlab_col(C_ps.astype(np.float64), np.float64),
        "coh_ps": _matlab_col(coh_ps.astype(np.float64), np.float64),
        "ph_patch": ph_patch.astype(np.complex64),
        "ph_res": ph_res.astype(np.float32),
    }

    write_mat(patch_dir / "ps2.mat", ps2)
    write_mat(patch_dir / "ph2.mat", {"ph": ph2})
    write_mat(patch_dir / "pm2.mat", pm2)
    write_mat(patch_dir / "psver.mat", {"psver": np.asarray(2, dtype=np.float64)})

    hgt1 = patch_dir / "hgt1.mat"
    if hgt1.exists():
        hgt = _as_ps_vector(read_mat(hgt1).get("hgt"), n_ps1, "hgt1.hgt").astype(np.float32)
        write_mat(patch_dir / "hgt2.mat", {"hgt": _matlab_col(hgt[final_ix], np.float32)})

    la1 = patch_dir / "la1.mat"
    if la1.exists():
        la = _as_ps_vector(read_mat(la1).get("la"), n_ps1, "la1.la").astype(np.float64)
        write_mat(patch_dir / "la2.mat", {"la": _matlab_col(la[final_ix], np.float64)})

    bp1 = patch_dir / "bp1.mat"
    bperp_mat2: np.ndarray | None = None
    if bp1.exists():
        bperp_mat = _as_ps_matrix(read_mat(bp1).get("bperp_mat"), n_ps1, "bp1.bperp_mat").astype(np.float32)
        bperp_mat2 = bperp_mat[final_ix, :]
        write_mat(patch_dir / "bp2.mat", {"bperp_mat": bperp_mat2})

    da1 = patch_dir / "da1.mat"
    if da1.exists():
        da = _as_ps_vector(read_mat(da1).get("D_A"), n_ps1, "da1.D_A").astype(np.float64)
        write_mat(patch_dir / "da2.mat", {"D_A": _matlab_col(da[final_ix], np.float64)})

    master_ix = int(round(_mat_scalar(ps2.get("master_ix", 1), 1)))
    if bperp_mat2 is None:
        bperp_mat2 = np.zeros((final_ix.size, max(1, ph2.shape[1] - 1)), dtype=np.float32)

    if parms.small_baseline_flag.lower() == "y":
        ph_rc = ph2.astype(np.complex128) * np.exp(-1j * (K_ps[:, None] * bperp_mat2.astype(np.float64)))
        write_mat(patch_dir / "rc2.mat", {"ph_rc": ph_rc.astype(np.complex64)})
    else:
        bperp_full = np.concatenate(
            [
                bperp_mat2[:, : master_ix - 1].astype(np.float64),
                np.zeros((final_ix.size, 1), dtype=np.float64),
                bperp_mat2[:, master_ix - 1 :].astype(np.float64),
            ],
            axis=1,
        )
        ph_rc = ph2.astype(np.complex128) * np.exp(-1j * (K_ps[:, None] * bperp_full + C_ps[:, None]))
        ph_reref = np.concatenate(
            [
                ph_patch[:, : master_ix - 1],
                np.ones((final_ix.size, 1), dtype=np.complex64),
                ph_patch[:, master_ix - 1 :],
            ],
            axis=1,
        )
        write_mat(
            patch_dir / "rc2.mat",
            {"ph_rc": ph_rc.astype(np.complex64), "ph_reref": ph_reref.astype(np.complex64)},
        )

    return f"Stage 5 promoted {final_ix.size} PS to version 2"


def _discover_patch_dirs(dataset_root: Path) -> list[Path]:
    patch_list = dataset_root / "patch.list"
    if patch_list.exists():
        names = [line.strip() for line in patch_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [dataset_root / name for name in names if (dataset_root / name).is_dir()]
    return sorted([p for p in dataset_root.glob("PATCH_*") if p.is_dir()])


def _load_stage5_patch_bundle(patch: Path) -> Stage5PatchBundle:
    ps_file = patch / "ps2.mat"
    ph_file = patch / "ph2.mat"
    pm_file = patch / "pm2.mat"
    if not (ps_file.exists() and ph_file.exists() and pm_file.exists()):
        raise PortedStageError(f"Patch missing stage-5 outputs: {patch.name}")

    ps = read_mat(ps_file)
    ph = read_mat(ph_file)
    pm = read_mat(pm_file)
    n_ps_patch = int(round(_mat_scalar(ps.get("n_ps", 0), 0)))
    if n_ps_patch <= 0:
        raise PortedStageError(f"{patch.name}/ps2.mat missing valid n_ps")

    ij_patch = _as_ps_dim(ps["ij"], n_ps_patch, 3, f"{patch.name}.ps2.ij").astype(np.float64)
    lonlat_patch = _as_ps_dim(ps["lonlat"], n_ps_patch, 2, f"{patch.name}.ps2.lonlat").astype(np.float64)
    ph_patch2 = _as_ps_ifg_complex(ph["ph"], n_ps_patch, f"{patch.name}.ph2.ph").astype(np.complex64)
    k_patch = _as_ps_vector(pm["K_ps"], n_ps_patch, f"{patch.name}.pm2.K_ps").astype(np.float64)
    c_patch = _as_ps_vector(pm["C_ps"], n_ps_patch, f"{patch.name}.pm2.C_ps").astype(np.float64)
    coh_patch = _as_ps_vector(pm["coh_ps"], n_ps_patch, f"{patch.name}.pm2.coh_ps").astype(np.float64)
    ph_patch_patch = _as_ps_ifg_complex(pm["ph_patch"], n_ps_patch, f"{patch.name}.pm2.ph_patch").astype(np.complex64)
    ph_res_patch = _as_ps_matrix(pm["ph_res"], n_ps_patch, f"{patch.name}.pm2.ph_res").astype(np.float32)
    ij_cols = np.rint(ij_patch[:, 1:3]).astype(np.int64)

    patch_bounds: tuple[int, int, int, int] | None = None
    patch_noover_file = patch / "patch_noover.in"
    if patch_noover_file.exists():
        bounds = _coerce_1d(_load_text_matrix(patch_noover_file, dtype=np.int64))
        if bounds.size >= 4:
            patch_bounds = tuple(int(v) for v in bounds[:4])

    bp_patch: np.ndarray | None = None
    bp_file = patch / "bp2.mat"
    if bp_file.exists():
        bp_patch = _as_ps_matrix(read_mat(bp_file)["bperp_mat"], n_ps_patch, f"{patch.name}.bp2.bperp_mat").astype(np.float32)

    hgt_patch: np.ndarray | None = None
    hgt_file = patch / "hgt2.mat"
    if hgt_file.exists():
        hgt_patch = _as_ps_vector(read_mat(hgt_file).get("hgt"), n_ps_patch, f"{patch.name}.hgt2.hgt").astype(np.float64)

    la_patch: np.ndarray | None = None
    la_file = patch / "la2.mat"
    if la_file.exists():
        la_patch = _as_ps_vector(read_mat(la_file).get("la"), n_ps_patch, f"{patch.name}.la2.la").astype(np.float64)

    rc_patch: np.ndarray | None = None
    rc_file = patch / "rc2.mat"
    if rc_file.exists():
        rc_payload = read_mat(rc_file)
        rc = rc_payload.get("ph_rc", rc_payload.get("rc"))
        if rc is not None:
            rc_arr = np.asarray(rc)
            if rc_arr.ndim == 2:
                rc_patch = _as_ps_ifg_complex(rc_arr, n_ps_patch, f"{patch.name}.rc2.ph_rc").astype(np.complex64)
            else:
                rc_patch = rc_arr.reshape(-1).astype(np.float32)

    return Stage5PatchBundle(
        patch=patch,
        ps=ps,
        n_ps_patch=n_ps_patch,
        ij_patch=ij_patch,
        lonlat_patch=lonlat_patch,
        ph_patch2=ph_patch2,
        k_patch=k_patch,
        c_patch=c_patch,
        coh_patch=coh_patch,
        ph_patch_patch=ph_patch_patch,
        ph_res_patch=ph_res_patch,
        ij_cols=ij_cols,
        ij_keys=_row_keys(ij_cols),
        patch_bounds=patch_bounds,
        bp_patch=bp_patch,
        hgt_patch=hgt_patch,
        la_patch=la_patch,
        rc_patch=rc_patch,
    )


def _compute_patch_keep_mask(
    ij_cols: np.ndarray,
    ij_keys: list[bytes],
    patch_bounds: tuple[int, int, int, int] | None,
    merged_index_by_key: dict[bytes, int],
) -> tuple[np.ndarray, list[int]]:
    keep_patch = np.ones(ij_cols.shape[0], dtype=bool)
    if patch_bounds is not None:
        row_min, row_max, col_min, col_max = patch_bounds
        keep_patch = (
            (ij_cols[:, 0] >= col_min - 1)
            & (ij_cols[:, 0] <= col_max - 1)
            & (ij_cols[:, 1] >= row_min - 1)
            & (ij_cols[:, 1] <= row_max - 1)
        )

    remove_ix: list[int] = []
    for idx in np.flatnonzero(keep_patch):
        merged_ix = merged_index_by_key.get(ij_keys[idx])
        if merged_ix is not None:
            remove_ix.append(int(merged_ix))

    ix_ex = np.ones(ij_cols.shape[0], dtype=bool)
    for idx, key in enumerate(ij_keys):
        if key in merged_index_by_key:
            ix_ex[idx] = False
    keep_patch[ix_ex] = True

    return keep_patch, remove_ix


def _concat_rows(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def stage5_merge_and_ifgstd(
    dataset_root: Path,
    backend: str = "auto",
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
    enable_mat_cache: bool = True,
) -> str:
    patch_dirs = _discover_patch_dirs(dataset_root)
    if not patch_dirs:
        raise PortedStageError("No patch directories found for merged stage-5 processing")

    cache = {} if mat_cache is None else mat_cache
    heading_deg = 0.0
    parms_file = _resolve_file(dataset_root, "parms.mat")
    if parms_file is not None:
        try:
            parms_raw = _read_mat_cached(parms_file, cache, enabled=enable_mat_cache)
            heading_deg = _mat_scalar(parms_raw.get("heading", 0.0), 0.0)
        except Exception:
            heading_deg = 0.0

    load_workers = _resolve_io_workers(io_workers, len(patch_dirs))
    if len(patch_dirs) > 1 and load_workers > 1:
        with ThreadPoolExecutor(max_workers=load_workers, thread_name_prefix="pystamps-stage5") as pool:
            bundles = list(pool.map(_load_stage5_patch_bundle, patch_dirs))
    else:
        bundles = [_load_stage5_patch_bundle(patch) for patch in patch_dirs]

    ps_chunks: list[dict[str, np.ndarray]] = []
    ph_chunks: list[np.ndarray] = []
    pm_k: list[np.ndarray] = []
    pm_c: list[np.ndarray] = []
    pm_coh: list[np.ndarray] = []
    pm_patch: list[np.ndarray] = []
    pm_res: list[np.ndarray] = []
    bp_chunks: list[np.ndarray] = []
    hgt_chunks: list[np.ndarray] = []
    la_chunks: list[np.ndarray] = []
    rc_chunks: list[np.ndarray] = []
    remove_ix: list[int] = []
    merged_index_by_key: dict[bytes, int] = {}
    merged_count = 0
    base_ps: dict[str, Any] | None = None

    for bundle in bundles:
        base_ps = bundle.ps
        keep_patch, remove_patch_ix = _compute_patch_keep_mask(
            bundle.ij_cols,
            bundle.ij_keys,
            bundle.patch_bounds,
            merged_index_by_key,
        )
        if remove_patch_ix:
            remove_ix.extend(remove_patch_ix)
        if not np.any(keep_patch):
            continue

        kept_ix = np.flatnonzero(keep_patch)
        ps_chunks.append({"ij": bundle.ij_patch[keep_patch, :], "lonlat": bundle.lonlat_patch[keep_patch, :]})
        ph_chunks.append(bundle.ph_patch2[keep_patch, :])
        pm_k.append(bundle.k_patch[keep_patch])
        pm_c.append(bundle.c_patch[keep_patch])
        pm_coh.append(bundle.coh_patch[keep_patch])
        pm_patch.append(bundle.ph_patch_patch[keep_patch, :])
        pm_res.append(bundle.ph_res_patch[keep_patch, :])
        if bundle.bp_patch is not None:
            bp_chunks.append(bundle.bp_patch[keep_patch, :])
        if bundle.hgt_patch is not None:
            hgt_chunks.append(bundle.hgt_patch[keep_patch])
        if bundle.la_patch is not None:
            la_chunks.append(bundle.la_patch[keep_patch])
        if bundle.rc_patch is not None:
            rc_chunks.append(np.asarray(bundle.rc_patch)[keep_patch, ...])

        for offset, idx in enumerate(kept_ix.tolist()):
            merged_index_by_key.setdefault(bundle.ij_keys[idx], merged_count + offset)
        merged_count += kept_ix.size

    if base_ps is None:
        raise PortedStageError("No patch PS data available for merge")

    ij = _concat_rows([chunk["ij"] for chunk in ps_chunks]).astype(np.float64)
    lonlat = _concat_rows([chunk["lonlat"] for chunk in ps_chunks]).astype(np.float64)
    ij[:, 0] = np.arange(1, ij.shape[0] + 1)

    ph2 = _concat_rows(ph_chunks).astype(np.complex64)
    K_ps = _concat_rows(pm_k).astype(np.float64)
    C_ps = _concat_rows(pm_c).astype(np.float64)
    coh_ps = _concat_rows(pm_coh).astype(np.float64)
    ph_patch = _concat_rows(pm_patch).astype(np.complex64)
    ph_res = _concat_rows(pm_res).astype(np.float32)
    bp2_all = _concat_rows(bp_chunks).astype(np.float32) if bp_chunks else None
    hgt2_all = _concat_rows(hgt_chunks).astype(np.float64) if hgt_chunks else None
    la2_all = _concat_rows(la_chunks).astype(np.float64) if la_chunks else None
    rc2_all = _concat_rows([np.asarray(r) for r in rc_chunks]) if rc_chunks else None

    if remove_ix:
        keep_overlap = np.ones(ij.shape[0], dtype=bool)
        keep_overlap[np.asarray(remove_ix, dtype=np.int64)] = False
        ij, lonlat, ph2, K_ps, C_ps, coh_ps, ph_patch, ph_res, bp2_all, hgt2_all, la2_all, rc2_all = _apply_selector_all(
            keep_overlap,
            ij,
            lonlat,
            ph2,
            K_ps,
            C_ps,
            coh_ps,
            ph_patch,
            ph_res,
            bp2_all,
            hgt2_all,
            la2_all,
            rc2_all,
        )

    keep = _dedup_lonlat_keep_highest_coh(lonlat, coh_ps)
    if keep.size == lonlat.shape[0] and not np.all(keep):
        ij, lonlat, ph2, K_ps, C_ps, coh_ps, ph_patch, ph_res, bp2_all, hgt2_all, la2_all, rc2_all = _apply_selector_all(
            keep,
            ij,
            lonlat,
            ph2,
            K_ps,
            C_ps,
            coh_ps,
            ph_patch,
            ph_res,
            bp2_all,
            hgt2_all,
            la2_all,
            rc2_all,
        )

    if lonlat.shape[0] > 0:
        xy_local, ll0_xy = _local_xy_from_lonlat(lonlat, heading_deg=heading_deg)
        xy_sort_key = np.asarray(xy_local, dtype=np.float32)
        sort_ix = np.lexsort((xy_sort_key[:, 0], xy_sort_key[:, 1]))
        ij, lonlat, ph2, K_ps, C_ps, coh_ps, ph_patch, ph_res, bp2_all, hgt2_all, la2_all, rc2_all = _apply_selector_all(
            sort_ix,
            ij,
            lonlat,
            ph2,
            K_ps,
            C_ps,
            coh_ps,
            ph_patch,
            ph_res,
            bp2_all,
            hgt2_all,
            la2_all,
            rc2_all,
        )
        xy_local = xy_sort_key[sort_ix, :]
    else:
        ll0_xy = np.asarray(base_ps.get("ll0", [0.0, 0.0]), dtype=np.float64).reshape(-1)[:2]
        xy_local = np.zeros((0, 2), dtype=np.float32)

    ll0_out = np.asarray(base_ps.get("ll0", ll0_xy), dtype=np.float64).reshape(-1)
    ij[:, 0] = np.arange(1, ij.shape[0] + 1)
    xy_scaled = xy_local * np.float32(1000.0)
    xy_mm_even = np.round(xy_scaled)
    xy_mm_away = _round_half_away_from_zero(xy_scaled)
    frac = np.abs(xy_scaled) - np.floor(np.abs(xy_scaled))
    tie_mask = frac == np.float32(0.5)
    xy_mm = (np.where(tie_mask, xy_mm_away, xy_mm_even) / np.float32(1000.0)).astype(np.float32)
    xy = np.column_stack((np.arange(1, ij.shape[0] + 1, dtype=np.float32), xy_mm)).astype(np.float32)

    ps2_payload: dict[str, Any] = {
        "bperp": _matlab_col(np.asarray(base_ps["bperp"], dtype=np.float32), np.float32),
        "day": _matlab_col(np.asarray(base_ps["day"], dtype=np.float64), np.float64),
        "ij": ij,
        "ll0": ll0_out,
        "lonlat": lonlat,
        "master_day": np.asarray(base_ps["master_day"], dtype=np.float64),
        "master_ix": np.asarray(base_ps["master_ix"], dtype=np.float64),
        "n_ifg": np.asarray(base_ps["n_ifg"], dtype=np.float64),
        "n_image": np.asarray(base_ps["n_image"], dtype=np.float64),
        "n_ps": np.asarray(ij.shape[0], dtype=np.float64),
        "xy": xy,
    }
    if "mean_incidence" in base_ps:
        ps2_payload["mean_incidence"] = np.asarray(base_ps["mean_incidence"], dtype=np.float64)
    if "mean_range" in base_ps:
        ps2_payload["mean_range"] = np.asarray(base_ps["mean_range"], dtype=np.float64)

    pm2_payload = {
        "K_ps": _matlab_col(K_ps, np.float64),
        "C_ps": _matlab_col(C_ps, np.float64),
        "coh_ps": _matlab_col(coh_ps, np.float64),
        "ph_patch": ph_patch,
        "ph_res": ph_res,
    }

    write_mat(dataset_root / "ps2.mat", ps2_payload)
    _cache_mat_payload(dataset_root / "ps2.mat", ps2_payload, cache, enabled=enable_mat_cache)
    write_mat(dataset_root / "ph2.mat", {"ph": ph2})
    _cache_mat_payload(dataset_root / "ph2.mat", {"ph": ph2}, cache, enabled=enable_mat_cache)
    write_mat(dataset_root / "pm2.mat", pm2_payload)
    _cache_mat_payload(dataset_root / "pm2.mat", pm2_payload, cache, enabled=enable_mat_cache)
    write_mat(dataset_root / "psver.mat", {"psver": np.asarray(2, dtype=np.float64)})

    if bp2_all is not None:
        write_mat(dataset_root / "bp2.mat", {"bperp_mat": bp2_all})
        _cache_mat_payload(dataset_root / "bp2.mat", {"bperp_mat": bp2_all}, cache, enabled=enable_mat_cache)
    if hgt2_all is not None:
        hgt2_payload = {"hgt": _matlab_col(hgt2_all, np.float64)}
        write_mat(dataset_root / "hgt2.mat", hgt2_payload)
        _cache_mat_payload(dataset_root / "hgt2.mat", hgt2_payload, cache, enabled=enable_mat_cache)
    if la2_all is not None:
        la2_payload = {"la": _matlab_col(la2_all, np.float64)}
        write_mat(dataset_root / "la2.mat", la2_payload)
        _cache_mat_payload(dataset_root / "la2.mat", la2_payload, cache, enabled=enable_mat_cache)
    if rc2_all is not None:
        rc2_payload = np.asarray(rc2_all)
        if np.iscomplexobj(rc2_payload):
            nz = rc2_payload != 0
            rc2_payload = rc2_payload.astype(np.complex64, copy=True)
            rc2_payload[nz] = rc2_payload[nz] / np.abs(rc2_payload[nz])
        write_mat(dataset_root / "rc2.mat", {"ph_rc": rc2_payload})

    parms = _load_parms(dataset_root)
    n_ps = ph2.shape[0]
    if bp2_all is not None:
        bp = np.asarray(bp2_all, dtype=np.float32)
    else:
        bp = _as_ps_matrix(
            _read_mat_cached(dataset_root / "bp2.mat", cache, enabled=enable_mat_cache)["bperp_mat"],
            n_ps,
            "bp2.bperp_mat",
        ).astype(np.float32)

    if parms.small_baseline_flag.lower() == "y":
        ph_diff = np.angle(
            ph2.astype(np.complex128) * np.conj(ph_patch.astype(np.complex128)) * np.exp(-1j * (K_ps[:, None] * bp))
        )
    else:
        master_ix = int(round(_mat_scalar(ps2_payload.get("master_ix", 1), 1)))
        bperp_full = np.concatenate(
            [bp[:, : master_ix - 1], np.zeros((n_ps, 1), dtype=np.float64), bp[:, master_ix - 1 :]],
            axis=1,
        )
        ph_patch_full = np.concatenate(
            [
                ph_patch[:, : master_ix - 1],
                np.ones((n_ps, 1), dtype=np.complex64),
                ph_patch[:, master_ix - 1 :],
            ],
            axis=1,
        )
        ph_diff = np.angle(
            ph2.astype(np.complex128)
            * np.conj(ph_patch_full.astype(np.complex128))
            * np.exp(-1j * (K_ps[:, None] * bperp_full + C_ps[:, None]))
        )
    ifg_std = (np.sqrt(np.sum(ph_diff**2, axis=0) / max(1, n_ps)) * 180.0 / np.pi).astype(np.float32)
    ifgstd_payload = {"ifg_std": _matlab_col(ifg_std, np.float32)}
    write_mat(dataset_root / "ifgstd2.mat", ifgstd_payload)
    _cache_mat_payload(dataset_root / "ifgstd2.mat", ifgstd_payload, cache, enabled=enable_mat_cache)

    return f"Merged {len(patch_dirs)} patches into {ij.shape[0]} PS records"


def stage6_unwrap(
    dataset_root: Path,
    backend: str = "auto",
    io_workers: int = 0,
    enable_mat_cache: bool = True,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
) -> str:
    cache = {} if mat_cache is None else mat_cache
    if not (dataset_root / "ps2.mat").exists() or not (dataset_root / "ph2.mat").exists():
        stage5_merge_and_ifgstd(
            dataset_root,
            backend=backend,
            io_workers=io_workers,
            mat_cache=cache,
            enable_mat_cache=enable_mat_cache,
        )

    ps2 = _read_mat_cached(dataset_root / "ps2.mat", cache, enabled=enable_mat_cache)
    n_ps = int(round(_mat_scalar(ps2.get("n_ps", 0), 0)))
    if n_ps <= 0:
        raise PortedStageError("ps2.mat missing valid n_ps")
    ph2 = _as_ps_ifg_complex(
        _read_mat_cached(dataset_root / "ph2.mat", cache, enabled=enable_mat_cache)["ph"], n_ps, "ph2.ph"
    )
    n_ps, n_ifg = ph2.shape
    master_ix = int(round(_mat_scalar(ps2.get("master_ix", 1), 1)))

    parms_raw: dict[str, Any] = {}
    parms_file = _resolve_file(dataset_root, "parms.mat")
    if parms_file is not None:
        try:
            parms_raw = _read_mat_cached(parms_file, cache, enabled=enable_mat_cache)
        except Exception:
            parms_raw = {}

    small_baseline = _mat_text(parms_raw.get("small_baseline_flag", "n"), "n").lower() == "y"
    unwrap_patch_phase = _mat_text(parms_raw.get("unwrap_patch_phase", "n"), "n").lower() == "y"
    unwrap_method = _mat_text(parms_raw.get("unwrap_method", "3D"), "3D")
    drop_ifg = _normalize_drop_index(parms_raw.get("drop_ifg_index", None))
    drop_set = set(int(v) for v in drop_ifg.tolist())
    unwrap_ifg = np.asarray([i for i in range(1, n_ifg + 1) if i not in drop_set], dtype=np.int64)
    if not small_baseline:
        unwrap_ifg = unwrap_ifg[unwrap_ifg != master_ix]
    if unwrap_ifg.size == 0:
        raise PortedStageError("No interferograms available for stage-6 unwrapping")
    unwrap_ifg_ix = unwrap_ifg - 1
    effective_unwrap_method = unwrap_method
    lowfilt_flag = False
    if unwrap_method.upper() in {"3D", "3D_NEW"}:
        if small_baseline:
            # Multi-master stacks follow uw_3d.m and enable low-pass support.
            lowfilt_flag = True
        else:
            # Single-master stacks are promoted to 3D_FULL without low-pass filtering.
            effective_unwrap_method = "3D_FULL"

    # Build wrapped phase input as in ps_unwrap.m.
    ph_w: np.ndarray
    if unwrap_patch_phase:
        pm2_for_patch = _read_mat_cached(dataset_root / "pm2.mat", cache, enabled=enable_mat_cache)
        ph_patch = _as_ps_ifg_complex(pm2_for_patch["ph_patch"], n_ps, "pm2.ph_patch").astype(np.complex64)
        patch_abs = np.abs(ph_patch)
        ph_patch = np.divide(ph_patch, patch_abs, out=np.zeros_like(ph_patch), where=patch_abs != 0)
        if not small_baseline:
            ph_w = np.concatenate(
                [
                    ph_patch[:, : master_ix - 1],
                    np.ones((n_ps, 1), dtype=np.complex64),
                    ph_patch[:, master_ix - 1 :],
                ],
                axis=1,
            )
        else:
            ph_w = ph_patch
    else:
        rc2_file = dataset_root / "rc2.mat"
        if rc2_file.exists():
            rc2 = _read_mat_cached(rc2_file, cache, enabled=enable_mat_cache)
            ph_w = _as_ps_ifg_complex(rc2.get("ph_rc"), n_ps, "rc2.ph_rc").astype(np.complex64)
        else:
            ph_w = ph2.astype(np.complex64)

        pm2 = _read_mat_cached(dataset_root / "pm2.mat", cache, enabled=enable_mat_cache)
        k_ps_raw = pm2.get("K_ps")
        if k_ps_raw is not None:
            K_ps = _as_ps_vector(k_ps_raw, n_ps, "pm2.K_ps").astype(np.float32)
            bp2_file = dataset_root / "bp2.mat"
            if bp2_file.exists():
                bp_nm = _as_ps_matrix(
                    _read_mat_cached(bp2_file, cache, enabled=enable_mat_cache).get("bperp_mat"),
                    n_ps,
                    "bp2.bperp_mat",
                ).astype(np.float32)
                if not small_baseline:
                    bperp_mat = np.concatenate(
                        [
                            bp_nm[:, : master_ix - 1],
                            np.zeros((n_ps, 1), dtype=np.float32),
                            bp_nm[:, master_ix - 1 :],
                        ],
                        axis=1,
                    )
                else:
                    bperp_mat = bp_nm
            else:
                bperp_vec = _as_ps_vector(ps2.get("bperp"), n_ifg, "ps2.bperp").astype(np.float32)
                bperp_mat = np.tile(bperp_vec[None, :], (n_ps, 1))
            ph_w = ph_w * np.exp(1j * (K_ps[:, None] * bperp_mat))

    if not small_baseline:
        scla_path = dataset_root / "scla_smooth2.mat"
        if scla_path.exists():
            scla = _read_mat_cached(scla_path, cache, enabled=enable_mat_cache)
            k_ps_uw = scla.get("K_ps_uw")
            if k_ps_uw is not None:
                K_ps_uw = _as_ps_vector(k_ps_uw, n_ps, "scla_smooth2.K_ps_uw").astype(np.float32)
                bp2_file = dataset_root / "bp2.mat"
                if bp2_file.exists():
                    bp_nm = _as_ps_matrix(
                        _read_mat_cached(bp2_file, cache, enabled=enable_mat_cache).get("bperp_mat"),
                        n_ps,
                        "bp2.bperp_mat",
                    ).astype(np.float32)
                    bperp_mat = np.concatenate(
                        [
                            bp_nm[:, : master_ix - 1],
                            np.zeros((n_ps, 1), dtype=np.float32),
                            bp_nm[:, master_ix - 1 :],
                        ],
                        axis=1,
                    )
                    ph_w = ph_w * np.exp(-1j * (K_ps_uw[:, None] * bperp_mat))
            c_ps_uw = scla.get("C_ps_uw")
            if c_ps_uw is not None:
                C_ps_uw = _as_ps_vector(c_ps_uw, n_ps, "scla_smooth2.C_ps_uw").astype(np.float32)
                ph_w = ph_w * np.exp(-1j * C_ps_uw[:, None])
            ph_ramp = scla.get("ph_ramp")
            if ph_ramp is not None:
                ph_ramp_arr = _as_ps_matrix(ph_ramp, n_ps, "scla_smooth2.ph_ramp").astype(np.float32)
                if ph_ramp_arr.shape == ph_w.shape:
                    ph_w = ph_w * np.exp(-1j * ph_ramp_arr)

    nz = ph_w != 0
    ph_w[nz] = ph_w[nz] / np.abs(ph_w[nz])

    if not (dataset_root / "uw_grid.mat").exists():
        pix_size = float(_mat_scalar(parms_raw.get("unwrap_grid_size", 20.0), 20.0))
        prefilt_win = int(round(_mat_scalar(parms_raw.get("unwrap_gold_n_win", 32.0), 32.0)))
        if prefilt_win <= 0:
            prefilt_win = 32
        gold_alpha = float(_mat_scalar(parms_raw.get("unwrap_gold_alpha", 0.8), 0.8))
        goldfilt_flag = _mat_text(parms_raw.get("unwrap_prefilter_flag", "y"), "y").lower() == "y"
        if pix_size <= 0:
            pix_size = 20.0

        xy_in = _as_ps_dim(ps2.get("xy"), n_ps, 3, "ps2.xy").astype(np.float32)
        x = xy_in[:, 1]
        y = xy_in[:, 2]
        pix_size32 = np.float32(pix_size)
        grid_x_min = float(np.min(x))
        grid_y_min = float(np.min(y))

        grid_i = np.ceil((y - np.float32(grid_y_min) + np.float32(1e-3)) / pix_size32).astype(np.int64)
        grid_j = np.ceil((x - np.float32(grid_x_min) + np.float32(1e-3)) / pix_size32).astype(np.int64)
        if grid_i.size > 0 and int(np.max(grid_i)) > 1:
            max_i = int(np.max(grid_i))
            grid_i[grid_i == max_i] = max_i - 1
        if grid_j.size > 0 and int(np.max(grid_j)) > 1:
            max_j = int(np.max(grid_j))
            grid_j[grid_j == max_j] = max_j - 1

        n_i = int(np.max(grid_i)) if grid_i.size > 0 else 1
        n_j = int(np.max(grid_j)) if grid_j.size > 0 else 1
        grid_ij = np.column_stack((grid_i, grid_j)).astype(np.float64)

        ph_in = ph_w[:, unwrap_ifg_ix].astype(np.complex64)
        lin0 = ((grid_j - 1) * n_i + (grid_i - 1)).astype(np.int64)
        n_ifg_nm = ph_in.shape[1]
        group_lin, grouped_cols = _group_reduce_by_index(ph_in, lin0)
        ph_grid_flat0 = _accumulate_grid_column(group_lin, grouped_cols[:, 0], n_i * n_j)
        nz_flat = ph_grid_flat0 != 0
        n_ps_grid = int(np.sum(nz_flat))
        if n_ps_grid <= 0:
            raise PortedStageError("uw_grid has no non-zero points in first interferogram")
        nz_lin = np.flatnonzero(nz_flat).astype(np.int64)
        nz_i = (nz_lin % n_i) + 1
        nz_j = (nz_lin // n_i) + 1

        if (goldfilt_flag or lowfilt_flag) and min(n_i, n_j) < prefilt_win:
            raise PortedStageError(
                f"Minimum resampled grid dimension ({min(n_i, n_j)}) is smaller than prefilter window ({prefilt_win})"
            )

        if goldfilt_flag or lowfilt_flag:
            ph_grid_vals = np.zeros((n_ps_grid, n_ifg_nm), dtype=np.complex64)
            ph_lowpass_vals = np.zeros((n_ps_grid, n_ifg_nm), dtype=np.complex64) if lowfilt_flag else np.empty((0, 0), dtype=np.complex64)

            def _compute_grid_column(i_ifg: int) -> tuple[int, np.ndarray, np.ndarray | None]:
                ph_grid_flat = _accumulate_grid_column(group_lin, grouped_cols[:, i_ifg], n_i * n_j)
                ph_grid_2d = ph_grid_flat.reshape((n_i, n_j), order="F")
                ph_gold, _ph_low = _wrap_filt_global(
                    ph_grid_2d,
                    alpha=gold_alpha,
                    low_flag="y" if lowfilt_flag else "n",
                )
                if goldfilt_flag:
                    col = ph_gold.reshape(-1, order="F")[nz_flat]
                else:
                    col = ph_grid_2d.reshape(-1, order="F")[nz_flat]
                low_col = _ph_low.reshape(-1, order="F")[nz_flat] if lowfilt_flag else None
                return i_ifg, np.asarray(col, dtype=np.complex64), (
                    np.asarray(low_col, dtype=np.complex64) if low_col is not None else None
                )

            worker_count = _resolve_io_workers(io_workers, n_ifg_nm)
            if n_ifg_nm > 1 and worker_count > 1:
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pystamps-stage6") as pool:
                    for i_ifg, col, low_col in pool.map(_compute_grid_column, range(n_ifg_nm)):
                        ph_grid_vals[:, i_ifg] = col
                        if lowfilt_flag and low_col is not None:
                            ph_lowpass_vals[:, i_ifg] = low_col
            else:
                for i_ifg in range(n_ifg_nm):
                    _, col, low_col = _compute_grid_column(i_ifg)
                    ph_grid_vals[:, i_ifg] = col
                    if lowfilt_flag and low_col is not None:
                        ph_lowpass_vals[:, i_ifg] = low_col
        else:
            keep_group = grouped_cols[:, 0] != 0
            ph_grid_vals = grouped_cols[keep_group, :].astype(np.complex64, copy=False)
            ph_lowpass_vals = np.empty((0, 0), dtype=np.complex64)

        nzix = nz_flat.reshape((n_i, n_j), order="F")
        n_ps_grid = int(ph_grid_vals.shape[0])

        xy_grid = np.column_stack(
            (
                np.arange(1, n_ps_grid + 1, dtype=np.float64),
                (nz_j.astype(np.float64) - 0.5) * pix_size,
                (nz_i.astype(np.float64) - 0.5) * pix_size,
            )
        )
        ij_grid = np.column_stack((nz_i, nz_j)).astype(np.float64)

        uw_grid_payload = {
            "ph": ph_grid_vals,
            "ph_in": ph_in,
            "ph_lowpass": ph_lowpass_vals,
            "ph_uw_predef": np.empty((0, 0), dtype=np.float32),
            "ph_in_predef": np.empty((0, 0), dtype=np.float32),
            "xy": xy_grid,
            "ij": ij_grid,
            "nzix": nzix,
            "grid_x_min": np.asarray(grid_x_min, dtype=np.float32),
            "grid_y_min": np.asarray(grid_y_min, dtype=np.float32),
            "n_i": np.asarray(n_i, dtype=np.float32),
            "n_j": np.asarray(n_j, dtype=np.float32),
            "n_ifg": np.asarray(ph_in.shape[1], dtype=np.float64),
            "n_ps": np.asarray(n_ps_grid, dtype=np.float64),
            "grid_ij": grid_ij,
            "pix_size": np.asarray(pix_size, dtype=np.float64),
        }
        write_mat(dataset_root / "uw_grid.mat", uw_grid_payload)
        _cache_mat_payload(dataset_root / "uw_grid.mat", uw_grid_payload, cache, enabled=enable_mat_cache)

    uw_grid_payload = _read_mat_cached(dataset_root / "uw_grid.mat", cache, enabled=enable_mat_cache)
    n_ps_grid = int(round(_mat_scalar(uw_grid_payload.get("n_ps", 0), 0)))
    if n_ps_grid <= 0:
        raise PortedStageError("uw_grid.mat missing valid n_ps")
    uw_ph = _as_ps_ifg_complex(uw_grid_payload.get("ph"), n_ps_grid, "uw_grid.ph").astype(np.complex64)
    ph_uw_some = np.unwrap(np.angle(uw_ph), axis=1).astype(np.float32)
    msd_some = _grid_neighbor_msd(ph_uw_some, np.asarray(uw_grid_payload.get("nzix"), dtype=bool)).astype(np.float64)
    uw_phaseuw_payload = {"ph_uw": ph_uw_some, "msd": _matlab_col(msd_some, np.float64)}
    write_mat(dataset_root / "uw_phaseuw.mat", uw_phaseuw_payload)
    _cache_mat_payload(dataset_root / "uw_phaseuw.mat", uw_phaseuw_payload, cache, enabled=enable_mat_cache)

    nzix = np.asarray(uw_grid_payload.get("nzix"), dtype=bool)
    grid_ij = _as_ps_dim(uw_grid_payload.get("grid_ij"), n_ps, 2, "uw_grid.grid_ij").astype(np.int64)
    n_i_grid, n_j_grid = nzix.shape
    if grid_ij.shape[0] != n_ps:
        raise PortedStageError("uw_grid.grid_ij has incompatible length for ps2")
    if np.any(grid_ij[:, 0] < 1) or np.any(grid_ij[:, 0] > n_i_grid) or np.any(grid_ij[:, 1] < 1) or np.any(grid_ij[:, 1] > n_j_grid):
        raise PortedStageError("uw_grid.grid_ij contains out-of-range indices")

    gridix_flat = np.zeros(n_i_grid * n_j_grid, dtype=np.int64)
    nz_flat_f = np.flatnonzero(nzix.reshape(-1, order="F"))
    gridix_flat[nz_flat_f] = np.arange(1, n_ps_grid + 1, dtype=np.int64)
    gridix = gridix_flat.reshape((n_i_grid, n_j_grid), order="F")

    ps_grid_idx = gridix[grid_ij[:, 0] - 1, grid_ij[:, 1] - 1]
    ph_in_sel = ph_w[:, unwrap_ifg_ix].astype(np.complex64)
    ph_uw_sel = np.full((n_ps, unwrap_ifg_ix.size), np.nan, dtype=np.float32)
    valid = ps_grid_idx > 0
    if np.any(valid):
        ph_uw_pix = ph_uw_some[ps_grid_idx[valid] - 1, :].astype(np.float32)
        ph_uw_sel[valid, :] = ph_uw_pix + np.angle(
            ph_in_sel[valid, :] * np.exp(-1j * ph_uw_pix.astype(np.float32))
        ).astype(np.float32)

    ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)
    msd = np.zeros((n_ifg,), dtype=np.float32)
    ph_uw[:, unwrap_ifg_ix] = ph_uw_sel
    msd[unwrap_ifg_ix] = msd_some.astype(np.float32)
    phuw2_payload = {"ph_uw": ph_uw, "msd": _matlab_col(msd, np.float32)}
    write_mat(dataset_root / "phuw2.mat", phuw2_payload)
    _cache_mat_payload(dataset_root / "phuw2.mat", phuw2_payload, cache, enabled=enable_mat_cache)

    if not (dataset_root / "uw_interp.mat").exists():
        uw_grid = uw_grid_payload
        nzix = np.asarray(uw_grid.get("nzix"), dtype=bool)
        n_ps_grid = int(round(_mat_scalar(uw_grid.get("n_ps", 0), 0)))
        if n_ps_grid <= 0:
            raise PortedStageError("uw_grid.mat missing valid n_ps")

        nrow, ncol = nzix.shape
        lin_true = np.flatnonzero(nzix.reshape(-1, order="F"))
        y_nodes = (lin_true % nrow) + 1
        x_nodes = (lin_true // nrow) + 1
        if y_nodes.size != n_ps_grid:
            raise PortedStageError("uw_grid.nzix and uw_grid.n_ps are inconsistent")

        pts = np.column_stack((x_nodes.astype(np.float64), y_nodes.astype(np.float64)))
        tri = spatial.Delaunay(pts)
        simplices = tri.simplices.astype(np.int64)

        edges = np.vstack((simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]))
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0) + 1  # MATLAB-style 1-based node indices
        n_edge = int(edges.shape[0])
        edgs = np.column_stack((np.arange(1, n_edge + 1, dtype=np.int64), edges)).astype(np.float64)

        X, Y = np.meshgrid(np.arange(1, ncol + 1), np.arange(1, nrow + 1))
        q = np.column_stack((X.reshape(-1, order="F"), Y.reshape(-1, order="F")))
        tree = spatial.cKDTree(pts)
        k_nn = min(8, pts.shape[0])
        d_nn, z_nn = tree.query(q, k=k_nn)
        if k_nn == 1:
            z_idx = z_nn.astype(np.int64) + 1
        else:
            # MATLAB dsearchn tie behavior is closest to selecting the
            # highest-index point among exact nearest-neighbor ties.
            d_nn = np.asarray(d_nn, dtype=np.float64)
            z_nn = np.asarray(z_nn, dtype=np.int64)
            d0 = d_nn[:, [0]]
            tie_mask = np.isclose(d_nn, d0, rtol=0.0, atol=1e-12)
            z_choose = np.max(np.where(tie_mask, z_nn, -1), axis=1)
            z_idx = z_choose.astype(np.int64) + 1
        Z = z_idx.reshape((nrow, ncol), order="F").astype(np.float64)

        z_vec = Z.reshape(-1, order="F")
        grid_edges = np.column_stack((z_vec[: -nrow], z_vec[nrow:]))
        z_vec_t = Z.T.reshape(-1, order="F")
        grid_edges = np.vstack((grid_edges, np.column_stack((z_vec_t[: -ncol], z_vec_t[ncol:]))))

        sort_edges = np.sort(grid_edges, axis=1)
        i_sort = np.argsort(grid_edges, axis=1)
        edge_sign = i_sort[:, 1] - i_sort[:, 0]

        all_edges, inv1 = np.unique(sort_edges, axis=0, return_inverse=True)
        sameix = all_edges[:, 0] == all_edges[:, 1]
        all_edges[sameix, :] = 0
        uniq_edges, inv2 = np.unique(all_edges, axis=0, return_inverse=True)

        n_edge_grid = int(uniq_edges.shape[0] - 1)
        edgs_grid = np.column_stack((np.arange(1, n_edge_grid + 1), uniq_edges[1:, :])).astype(np.float64)
        grid_edge_ix = (inv2[inv1] * edge_sign).astype(np.float64)
        colix = grid_edge_ix[: nrow * (ncol - 1)].reshape((nrow, ncol - 1), order="F")
        rowix = grid_edge_ix[nrow * (ncol - 1) :].reshape((ncol, nrow - 1), order="F").T

        uw_interp_payload = {
            "edgs": edgs_grid,
            "n_edge": np.asarray(n_edge_grid, dtype=np.float64),
            "rowix": rowix.astype(np.float64),
            "colix": colix.astype(np.float64),
            "Z": Z.astype(np.float64),
        }
        write_mat(dataset_root / "uw_interp.mat", uw_interp_payload)
        _cache_mat_payload(dataset_root / "uw_interp.mat", uw_interp_payload, cache, enabled=enable_mat_cache)

    return f"Stage 6 unwrapped {n_ps} PS across {n_ifg} interferograms"


def stage7_calc_scla(
    dataset_root: Path,
    backend: str = "auto",
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
) -> str:
    cache = {} if mat_cache is None else mat_cache
    if not (dataset_root / "phuw2.mat").exists():
        stage6_unwrap(
            dataset_root,
            backend=backend,
            io_workers=io_workers,
            enable_mat_cache=enable_mat_cache,
            mat_cache=cache,
        )

    ps2_file = dataset_root / "ps2.mat"
    if not ps2_file.exists():
        raise PortedStageError("Missing required artifact: ps2.mat (stage-5 merged output) before stage 7")
    ps2 = _read_mat_cached(ps2_file, cache, enabled=enable_mat_cache)
    phuw = _read_mat_cached(dataset_root / "phuw2.mat", cache, enabled=enable_mat_cache)
    n_ps = int(round(_mat_scalar(ps2.get("n_ps", 0), 0)))
    if n_ps <= 0:
        raise PortedStageError("ps2.mat missing valid n_ps")
    ph_uw = _as_ps_matrix(phuw["ph_uw"], n_ps, "phuw2.ph_uw").astype(np.float32)
    n_ps, n_ifg = ph_uw.shape

    master_ix = int(round(_mat_scalar(ps2.get("master_ix", 1), 1)))
    no_master = np.arange(n_ifg) != (master_ix - 1)

    parms_raw: dict[str, Any] = {}
    parms_file = _resolve_file(dataset_root, "parms.mat")
    if parms_file is not None:
        try:
            parms_raw = _read_mat_cached(parms_file, cache, enabled=enable_mat_cache)
        except Exception:
            parms_raw = {}

    small_baseline = _mat_text(parms_raw.get("small_baseline_flag", "n"), "n").lower() == "y"
    if small_baseline:
        raise PortedStageError("stage7_calc_scla parity path currently supports single-master stacks only")

    bp2_file = dataset_root / "bp2.mat"
    if bp2_file.exists():
        bp_nm = _as_ps_matrix(
            _read_mat_cached(bp2_file, cache, enabled=enable_mat_cache)["bperp_mat"], n_ps, "bp2.bperp_mat"
        ).astype(np.float64)
    else:
        bperp = _as_ps_vector(ps2.get("bperp"), n_ifg, "ps2.bperp").astype(np.float64)
        bp_nm = np.tile(bperp[no_master][None, :], (n_ps, 1))
        write_mat(bp2_file, {"bperp_mat": bp_nm.astype(np.float32)})
        _cache_mat_payload(bp2_file, {"bperp_mat": bp_nm.astype(np.float32)}, cache, enabled=enable_mat_cache)
    bperp_mat = np.concatenate(
        [
            bp_nm[:, : master_ix - 1],
            np.zeros((n_ps, 1), dtype=np.float64),
            bp_nm[:, master_ix - 1 :],
        ],
        axis=1,
    )

    ref_ix = _select_reference_ps(ps2, parms_raw)
    ph_raw = ph_uw.astype(np.float64)
    if _mat_text(parms_raw.get("scla_deramp", "y"), "y").lower() == "y":
        ph_deramped, ph_ramp = _deramp_unwrapped_phase(ps2, ph_raw)
    else:
        ph_deramped = ph_raw
        ph_ramp = np.empty((0, 0), dtype=np.float64)
    ph_proc = _center_to_reference(ph_deramped, ref_ix)
    ph_mean_v = _center_to_reference(ph_raw, ref_ix)

    drop_ifg = _normalize_drop_index(parms_raw.get("drop_ifg_index", None))
    scla_drop_ifg = _normalize_drop_index(parms_raw.get("scla_drop_index", None))
    drop_set = set(int(v) for v in drop_ifg.tolist()) | set(int(v) for v in scla_drop_ifg.tolist())
    unwrap_ifg, solve_ifg = _stage7_unwrap_ifg_sets(n_ifg, master_ix, drop_set)
    if solve_ifg.size < 2:
        raise PortedStageError("stage7_calc_scla requires at least two non-master interferograms")
    unwrap_ix = unwrap_ifg - 1
    solve_ix = solve_ifg - 1

    day = np.asarray(ps2["day"], dtype=np.float64).reshape(-1)
    ph_seq = np.diff(ph_proc[:, unwrap_ix], axis=1)
    bperp_seq = np.diff(bperp_mat[:, unwrap_ix], axis=1)
    day_seq = np.diff(day[unwrap_ix])
    coest_mean_vel = solve_ifg.size >= 4

    mean_bperp = np.mean(bperp_seq, axis=0)
    if coest_mean_vel:
        G_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp, day_seq))
    else:
        G_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp))
    coeffs_seq = _weighted_lstsq_shared_design(G_seq, ph_seq.T, cov=None)
    K_ps_uw = coeffs_seq[1, :].astype(np.float64)
    ph_scla = (K_ps_uw[:, None] * bperp_mat).astype(np.float32)

    ifgstd = _read_mat_cached(dataset_root / "ifgstd2.mat", cache, enabled=enable_mat_cache)
    ifg_std = _as_ps_vector(ifgstd.get("ifg_std"), n_ifg, "ifgstd2.ifg_std").astype(np.float64)
    ifg_vcm = np.diag((ifg_std * np.pi / 180.0) ** 2)

    resid_full = ph_proc[:, solve_ix] - ph_scla[:, solve_ix].astype(np.float64)
    if coest_mean_vel:
        G_c = np.column_stack((np.ones(solve_ifg.size, dtype=np.float64), day[solve_ix] - day[master_ix - 1]))
        coeffs_c = _weighted_lstsq_shared_design(G_c, resid_full.T, cov=ifg_vcm[np.ix_(solve_ix, solve_ix)])
        C_ps_uw = coeffs_c[0, :].astype(np.float32)
    else:
        C_ps_uw = np.mean(resid_full, axis=1).astype(np.float32)

    mean_v_m = _stage7_mean_velocity_fit(ph_mean_v, day, master_ix, ifg_std)
    mean_v = mean_v_m[1, :].astype(np.float32)

    payload = {
        "K_ps_uw": _matlab_col(K_ps_uw, np.float32),
        "C_ps_uw": _matlab_col(C_ps_uw, np.float32),
        "ph_scla": ph_scla,
        "ph_ramp": ph_ramp.astype(np.float64),
        "ifg_vcm": ifg_vcm.astype(np.float64),
    }
    write_mat(dataset_root / "scla2.mat", payload)
    _cache_mat_payload(dataset_root / "scla2.mat", payload, cache, enabled=enable_mat_cache)
    write_mat(
        dataset_root / "scla_smooth2.mat",
        {k: payload[k] for k in ("K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp")},
    )
    _cache_mat_payload(
        dataset_root / "scla_smooth2.mat",
        {k: payload[k] for k in ("K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp")},
        cache,
        enabled=enable_mat_cache,
    )

    m = mean_v_m
    write_mat(dataset_root / "mean_v.mat", {"m": m})
    _cache_mat_payload(dataset_root / "mean_v.mat", {"m": m}, cache, enabled=enable_mat_cache)
    write_mat(
        dataset_root / "mv2.mat",
        {
            "mean_v": mean_v,
            "mean_v_std": np.zeros_like(mean_v, dtype=np.float32),
            "n_boot": np.asarray(0.0, dtype=np.float64),
            "subtract_switches": "do",
        },
    )
    _cache_mat_payload(
        dataset_root / "mv2.mat",
        {
            "mean_v": mean_v,
            "mean_v_std": np.zeros_like(mean_v, dtype=np.float32),
            "n_boot": np.asarray(0.0, dtype=np.float64),
            "subtract_switches": "do",
        },
        cache,
        enabled=enable_mat_cache,
    )

    return f"Stage 7 estimated SCLA and mean velocity for {n_ps} PS"


def stage8_filter_scn(
    dataset_root: Path,
    backend: str = "auto",
    chunk_edges: int = 0,
    chunk_ps: int = 0,
    enable_mat_cache: bool = True,
    io_workers: int = 0,
    mat_cache: dict[Path, dict[str, Any]] | None = None,
) -> str:
    cache = {} if mat_cache is None else mat_cache
    if not (dataset_root / "scla2.mat").exists():
        stage7_calc_scla(
            dataset_root,
            backend=backend,
            chunk_ps=chunk_ps,
            enable_mat_cache=enable_mat_cache,
            io_workers=io_workers,
            mat_cache=cache,
        )

    ps2_file = dataset_root / "ps2.mat"
    if not ps2_file.exists():
        raise PortedStageError("Missing required artifact: ps2.mat (stage-5 merged output) before stage 8")
    ps2 = _read_mat_cached(ps2_file, cache, enabled=enable_mat_cache)
    n_ps = int(round(_mat_scalar(ps2.get("n_ps", 0), 0)))
    if n_ps <= 0:
        raise PortedStageError("ps2.mat missing valid n_ps")

    if not (dataset_root / "uw_grid.mat").exists() or not (dataset_root / "uw_interp.mat").exists():
        stage6_unwrap(
            dataset_root,
            backend=backend,
            io_workers=io_workers,
            enable_mat_cache=enable_mat_cache,
            mat_cache=cache,
        )

    uw_grid = _read_mat_cached(dataset_root / "uw_grid.mat", cache, enabled=enable_mat_cache)
    uw_interp = _read_mat_cached(dataset_root / "uw_interp.mat", cache, enabled=enable_mat_cache)
    n_grid_ps = int(round(_mat_scalar(uw_grid.get("n_ps", 0), 0)))
    if n_grid_ps <= 0:
        raise PortedStageError("uw_grid.mat missing valid n_ps")

    uw_ph = _as_ps_ifg_complex(uw_grid.get("ph"), n_grid_ps, "uw_grid.ph").astype(np.complex64)
    n_grid_ps, n_ifg_nm = uw_ph.shape

    edgs_raw = np.asarray(uw_interp.get("edgs", np.empty((0, 3), dtype=np.float64)))
    if edgs_raw.size == 0:
        edgs = np.empty((0, 3), dtype=np.int64)
    elif edgs_raw.ndim == 1:
        if edgs_raw.size % 3 != 0:
            raise PortedStageError(f"uw_interp.edgs has incompatible shape {edgs_raw.shape}")
        edgs = edgs_raw.reshape(-1, 3).astype(np.int64)
    elif edgs_raw.ndim == 2 and edgs_raw.shape[1] == 3:
        edgs = edgs_raw.astype(np.int64)
    elif edgs_raw.ndim == 2 and edgs_raw.shape[0] == 3:
        edgs = edgs_raw.T.astype(np.int64)
    else:
        raise PortedStageError(f"uw_interp.edgs has incompatible shape {edgs_raw.shape}")

    node_a = edgs[:, 1] - 1
    node_b = edgs[:, 2] - 1
    valid = (node_a >= 0) & (node_a < n_grid_ps) & (node_b >= 0) & (node_b < n_grid_ps)
    node_a = node_a[valid]
    node_b = node_b[valid]
    n_edge = int(node_a.size)

    try:
        edge_payload = run_stage8_edge_noise_kernel(
            uw_ph=uw_ph, node_a=node_a, node_b=node_b, backend=backend, chunk_edges=chunk_edges
        )
    except BackendUnavailableError as exc:
        raise PortedStageError(str(exc)) from exc
    dph_noise = edge_payload["dph_noise"]
    dph_space_uw = edge_payload["dph_space_uw"]

    n_ifg = int(round(_mat_scalar(ps2.get("n_ifg", 0), 0)))
    master_ix = int(round(_mat_scalar(ps2.get("master_ix", 1), 1)))
    G = np.zeros((n_ifg - 1, n_ifg), dtype=np.float64)
    cols = np.delete(np.arange(n_ifg, dtype=np.int64), master_ix - 1)
    rows = np.arange(cols.size, dtype=np.int64)
    G[rows, master_ix - 1] = -1.0
    G[rows, cols] = 1.0

    payload = {
        "G": G,
        "dph_noise": dph_noise,
        "dph_space_uw": dph_space_uw,
        "spread": sparse.csc_matrix((n_edge, n_ifg_nm), dtype=np.float64),
        "ifreq_ij": np.empty((0, 0), dtype=np.float64),
        "jfreq_ij": np.empty((0, 0), dtype=np.float64),
        "shaky_ix": np.empty((0, 0), dtype=np.float64),
        "predef_ix": np.empty((0, 0), dtype=np.float64),
    }
    write_mat(dataset_root / "uw_space_time.mat", payload)
    _cache_mat_payload(dataset_root / "uw_space_time.mat", payload, cache, enabled=enable_mat_cache)
    return f"Stage 8 produced space-time noise model for {n_edge} arcs"
