#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from pystamps.io.dataset import discover_authoritative_patch_paths
from pystamps.io.mat import read_mat
from pystamps.pipeline import ported


DEFAULT_ROWS = (22182, 22100, 12693, 22181, 22098)
DEFAULT_OUTPUT = "inputs_and_outputs/validation_runs/stage2_prior_state_drift_report.json"
SCENARIOS = (
    "current_K_current_weight",
    "oracle_K_current_weight",
    "current_K_oracle_weight",
    "oracle_K_oracle_weight",
)
REPORT_KEYS = ("C_ps", "K_ps", "coh_ps", "ph_weight", "ph_patch", "ph_grid")
PIPELINE_INTERMEDIATES = ("ph_weight", "ph_grid", "ph_patch", "C_ps")


@dataclass(frozen=True)
class PriorState:
    k_ps: np.ndarray
    weighting: np.ndarray


def _as_patch_dir(root: Path, patch: str) -> Path:
    return root if root.name == patch else root / patch


def _bperp_matrix(context: Any) -> np.ndarray:
    if context.bperp_mat is not None:
        return np.asarray(context.bperp_mat, dtype=np.float64)
    return np.broadcast_to(np.asarray(context.bperp_nm, dtype=np.float64), context.ph_nm.shape)


def infer_prior_state_from_ph_weight(
    ph_nm: np.ndarray,
    ph_weight: np.ndarray,
    bperp: np.ndarray,
) -> PriorState:
    ph = np.asarray(ph_nm, dtype=np.complex64)
    weight_ph = np.asarray(ph_weight, dtype=np.complex64)
    bp = np.asarray(bperp, dtype=np.float64)
    if ph.shape != weight_ph.shape or ph.shape != bp.shape:
        raise ValueError("ph_nm, ph_weight, and bperp must have matching shapes")

    valid = np.isfinite(ph) & np.isfinite(weight_ph) & (ph != 0) & (weight_ph != 0)
    ratio = np.ones(ph.shape, dtype=np.complex128)
    np.divide(weight_ph, ph, out=ratio, where=valid)
    magnitude = np.where(valid, np.abs(ratio), np.nan)
    weighting = np.nanmedian(magnitude, axis=1)
    weighting = np.where(np.isfinite(weighting), weighting, 0.0)

    phase = np.unwrap(np.angle(ratio), axis=1).astype(np.float64, copy=False)
    k_ps = np.zeros(ph.shape[0], dtype=np.float64)
    full_rows = np.all(valid, axis=1)
    if np.any(full_rows):
        bp_full = bp[full_rows]
        phase_full = phase[full_rows]
        bp_centered = bp_full - np.mean(bp_full, axis=1, keepdims=True)
        phase_centered = phase_full - np.mean(phase_full, axis=1, keepdims=True)
        denom = np.sum(bp_centered * bp_centered, axis=1)
        slope = np.divide(
            np.sum(bp_centered * phase_centered, axis=1),
            denom,
            out=np.zeros(np.sum(full_rows), dtype=np.float64),
            where=denom != 0,
        )
        k_ps[full_rows] = -slope

    for row in np.flatnonzero(~full_rows):
        row_valid = valid[row]
        if np.sum(row_valid) < 2:
            continue
        x = bp[row, row_valid]
        y = np.unwrap(np.angle(ratio[row, row_valid])).astype(np.float64)
        x_centered = x - np.mean(x)
        denom = float(np.sum(x_centered * x_centered))
        if denom != 0.0:
            k_ps[row] = -float(np.sum(x_centered * (y - np.mean(y))) / denom)

    return PriorState(k_ps=k_ps, weighting=weighting.astype(np.float64, copy=False))


def synthesize_ph_weight_from_prior(
    ph_nm: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    weighting: np.ndarray,
    *,
    row_chunk: int = 20000,
) -> np.ndarray:
    ph = np.asarray(ph_nm, dtype=np.complex64)
    bp = np.asarray(bperp, dtype=np.float64)
    k = np.asarray(k_ps, dtype=np.float64).reshape(-1)
    w = np.asarray(weighting, dtype=np.float64).reshape(-1)
    if ph.shape != bp.shape or ph.shape[0] != k.size or k.size != w.size:
        raise ValueError("prior state inputs have incompatible shapes")

    out = np.empty(ph.shape, dtype=np.complex64)
    for start in range(0, ph.shape[0], row_chunk):
        stop = min(start + row_chunk, ph.shape[0])
        out[start:stop, :] = ported._stage2_ph_weight_block(
            ph[start:stop, :],
            bp[start:stop, :],
            k[start:stop],
            w[start:stop],
        )
    return out


def _clap_filter_grid_samples(
    context: Any,
    ph_grid: np.ndarray,
    row_ix: np.ndarray,
) -> tuple[np.ndarray, int]:
    prepared = context.clap_prepared
    ph_arr = np.asarray(ph_grid, dtype=np.complex64)
    sample_rows = np.asarray(context.grid_rows, dtype=np.int64)[row_ix]
    sample_cols = np.asarray(context.grid_cols, dtype=np.int64)[row_ix]
    samples = list(zip(sample_rows.tolist(), sample_cols.tolist()))
    out = np.zeros((len(samples), ph_arr.shape[2]), dtype=np.complex128)
    ph_bit = np.zeros_like(prepared.ph_bit)
    h_smooth = np.empty_like(prepared.h_smooth)
    windows_used = 0

    for window in prepared.windows:
        hits = [
            (sample_ix, grid_row - window.i1, grid_col - window.j1)
            for sample_ix, (grid_row, grid_col) in enumerate(samples)
            if window.i1 <= grid_row < window.i2 and window.j1 <= grid_col < window.j2
        ]
        if not hits:
            continue
        windows_used += 1
        ph_bit.fill(0)
        ph_bit[: prepared.n_win_int, : prepared.n_win_int, :] = ph_arr[
            window.i1 : window.i2,
            window.j1 : window.j2,
            :,
        ]
        ph_fft = np.fft.fft2(ph_bit, axes=(0, 1))
        h_smooth[:, :, :] = np.fft.ifftshift(
            ndimage.convolve(
                np.fft.fftshift(np.abs(ph_fft), axes=(0, 1)),
                prepared.kernel[:, :, None],
                mode="constant",
                cval=0.0,
            ),
            axes=(0, 1),
        )
        mean_h = np.median(h_smooth, axis=(0, 1), keepdims=True)
        np.divide(h_smooth, mean_h, out=h_smooth, where=mean_h != 0)
        np.power(h_smooth, float(context.clap_alpha), out=h_smooth)
        h_smooth -= 1.0
        h_smooth[h_smooth < 0.0] = 0.0
        gain = h_smooth * float(context.clap_beta) + prepared.low_pass_stack
        ph_filt = np.fft.ifft2(ph_fft * gain, axes=(0, 1))
        for sample_ix, local_i, local_j in hits:
            out[sample_ix, :] += ph_filt[local_i, local_j, :] * window.weight[local_i, local_j]

    return out.astype(np.complex64), windows_used


def _topofit_selected_rows(
    context: Any,
    ph_patch: np.ndarray,
    row_ix: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    kernel_backend: str,
    native_threads: int,
) -> dict[str, np.ndarray]:
    selected = np.asarray(row_ix, dtype=np.int64)
    psdph = np.conjugate(np.asarray(ph_patch, dtype=np.complex64))
    psdph *= np.asarray(context.ph_nm, dtype=np.complex64)[selected, :]
    valid = np.any(psdph != 0, axis=1)
    k_ps = np.full(selected.size, np.nan, dtype=np.float64)
    c_ps = np.zeros(selected.size, dtype=np.float64)
    coh_ps = np.zeros(selected.size, dtype=np.float64)
    if np.any(valid):
        k_fit, c_fit, coh_fit, _phase_residual = ported._ps_topofit_batch(
            psdph[valid].astype(np.complex128),
            np.asarray(bperp, dtype=np.float64)[selected, :][valid],
            n_trial_wraps,
            kernel_backend=kernel_backend,
            native_threads=native_threads,
        )
        out_ix = np.flatnonzero(valid)
        k_ps[out_ix] = k_fit
        c_ps[out_ix] = c_fit
        coh_ps[out_ix] = coh_fit
    return {"K_ps": k_ps, "C_ps": c_ps, "coh_ps": coh_ps}


def _replay_selected_rows(
    context: Any,
    pm_payload: dict[str, Any],
    ph_weight: np.ndarray,
    row_ix: np.ndarray,
    bperp: np.ndarray,
    *,
    kernel_backend: str,
    native_threads: int,
) -> dict[str, Any]:
    ph_grid = ported._stage2_grid_accumulate_matlab(
        ph_weight,
        context.grid_lin,
        context.n_i,
        context.n_j,
    )
    ph_grid_samples = ph_grid[context.grid_rows[row_ix], context.grid_cols[row_ix], :].copy()
    ph_patch, windows_used = _clap_filter_grid_samples(context, ph_grid, row_ix)
    ported._normalize_complex_unit_magnitude_inplace(ph_patch)
    topofit = _topofit_selected_rows(
        context,
        ph_patch,
        row_ix,
        bperp,
        float(np.asarray(pm_payload["n_trial_wraps"]).reshape(-1)[0]),
        kernel_backend=kernel_backend,
        native_threads=native_threads,
    )
    topofit.update(
        {
            "ph_weight": np.asarray(ph_weight, dtype=np.complex64)[row_ix, :].copy(),
            "ph_grid": ph_grid_samples,
            "ph_patch": ph_patch,
            "windows_used": int(windows_used),
        }
    )
    return topofit


def _row_metric(observed: np.ndarray, expected: np.ndarray, key: str) -> np.ndarray:
    obs = np.asarray(observed)
    exp = np.asarray(expected)
    if key in {"C_ps", "K_ps", "coh_ps"}:
        return np.abs(obs.reshape(-1) - exp.reshape(-1)).astype(np.float64)
    return np.max(np.abs(obs - exp), axis=1).astype(np.float64)


def _key_tolerance(key: str) -> float:
    if key == "K_ps":
        return 1e-10
    if key in {"C_ps", "coh_ps"}:
        return 1e-8
    return 5e-7


def _first_divergent_intermediate(row_diffs: dict[str, dict[str, float]]) -> str:
    for key in PIPELINE_INTERMEDIATES:
        if float(row_diffs[key]["current_K_current_weight"]) > _key_tolerance(key):
            return key
    return "matched"


def _classify_source(key: str, diffs: dict[str, float]) -> str:
    baseline = float(diffs["current_K_current_weight"])
    oracle_k = float(diffs["oracle_K_current_weight"])
    oracle_weight = float(diffs["current_K_oracle_weight"])
    oracle_both = float(diffs["oracle_K_oracle_weight"])
    tol = _key_tolerance(key)
    if baseline <= tol:
        return "matched"
    if oracle_both > max(tol, baseline * 0.25):
        if key in {"C_ps", "K_ps", "coh_ps"}:
            return "topofit_output"
        if key == "ph_grid":
            return "grid_accumulation"
        if key == "ph_patch":
            return "grid_filter_or_patch_extraction"
        return "unexplained_prior_state"

    k_improvement = baseline - oracle_k
    weight_improvement = baseline - oracle_weight
    if oracle_k <= max(tol, baseline * 0.1):
        return "prior_K_state"
    if oracle_weight <= max(tol, baseline * 0.1):
        return "prior_weighting_state"
    if k_improvement > 0 and k_improvement >= max(weight_improvement, 0.0) * 1.25 and oracle_k <= baseline * 0.75:
        return "prior_K_state"
    if (
        weight_improvement > 0
        and weight_improvement >= max(k_improvement, 0.0) * 1.25
        and oracle_weight <= baseline * 0.75
    ):
        return "prior_weighting_state"
    if oracle_both <= max(tol, baseline * 0.1):
        return "combined_prior_K_and_weighting_state"
    return "mixed_or_unresolved_prior_state"


def _first_mismatch_for_key(
    run_payload: dict[str, Any],
    golden_payload: dict[str, Any],
    key: str,
    *,
    row_base: int,
    rtol: float,
    atol: float,
    max_rows: int,
) -> dict[str, Any] | None:
    lhs = np.asarray(run_payload[key])
    rhs = np.asarray(golden_payload[key])
    if lhs.shape != rhs.shape:
        return {
            "key": key,
            "failure_kind": "shape_mismatch",
            "shape_run": list(lhs.shape),
            "shape_oracle": list(rhs.shape),
        }

    close = np.isclose(lhs, rhs, rtol=float(rtol), atol=float(atol), equal_nan=True)
    bad = np.flatnonzero(~close.reshape(-1))
    if bad.size == 0:
        return None

    diff = np.abs(np.asarray(lhs) - np.asarray(rhs))
    first_flat = int(bad[0])
    first_index = tuple(int(v) for v in np.unravel_index(first_flat, lhs.shape))
    row_zero_based = first_index[0] if first_index else first_flat
    flat_diff = diff.reshape(-1)
    max_flat = int(np.nanargmax(flat_diff))
    max_index = tuple(int(v) for v in np.unravel_index(max_flat, lhs.shape))
    max_row_zero_based = max_index[0] if max_index else max_flat
    row_numbers = [int(np.unravel_index(int(ix), lhs.shape)[0]) + int(row_base) for ix in bad[:max_rows]]
    max_row = int(max_row_zero_based + int(row_base))
    if max_row not in row_numbers:
        row_numbers.append(max_row)
    return _jsonable(
        {
            "key": key,
            "failure_kind": "value_mismatch",
            "index": list(first_index),
            "row": int(row_zero_based + int(row_base)),
            "zero_based_row": int(row_zero_based),
            "run_value": np.asarray(lhs).reshape(-1)[first_flat],
            "golden_value": np.asarray(rhs).reshape(-1)[first_flat],
            "abs_diff": diff.reshape(-1)[first_flat],
            "max_abs": flat_diff[max_flat],
            "max_abs_index": list(max_index),
            "max_abs_row": max_row,
            "max_abs_zero_based_row": int(max_row_zero_based),
            "max_abs_run_value": np.asarray(lhs).reshape(-1)[max_flat],
            "max_abs_golden_value": np.asarray(rhs).reshape(-1)[max_flat],
            "sample_rows": row_numbers,
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    return value


def build_report(
    run_root: Path,
    golden_root: Path,
    *,
    patch: str,
    rows: tuple[int, ...],
    row_base: int,
    kernel_backend: str,
    native_threads: int,
) -> dict[str, Any]:
    run_patch = _as_patch_dir(run_root, patch)
    golden_patch = _as_patch_dir(golden_root, patch)
    current_pm = read_mat(run_patch / "pm1.mat")
    oracle_pm = read_mat(golden_patch / "pm1.mat")
    context = ported._stage2_prepare_replay_context(
        run_patch,
        kernel_backend=kernel_backend,
        native_threads=native_threads,
    )
    row_ix = np.asarray(rows, dtype=np.int64) - int(row_base)
    if np.any(row_ix < 0) or np.any(row_ix >= context.ph_nm.shape[0]):
        raise ValueError("requested row is outside the Stage 2 candidate range")

    bperp = _bperp_matrix(context)
    current_state = infer_prior_state_from_ph_weight(context.ph_nm, current_pm["ph_weight"], bperp)
    oracle_state = infer_prior_state_from_ph_weight(context.ph_nm, oracle_pm["ph_weight"], bperp)
    scenario_ph_weight = {
        "current_K_current_weight": np.asarray(current_pm["ph_weight"], dtype=np.complex64),
        "oracle_K_current_weight": synthesize_ph_weight_from_prior(
            context.ph_nm,
            bperp,
            oracle_state.k_ps,
            current_state.weighting,
        ),
        "current_K_oracle_weight": synthesize_ph_weight_from_prior(
            context.ph_nm,
            bperp,
            current_state.k_ps,
            oracle_state.weighting,
        ),
        "oracle_K_oracle_weight": np.asarray(oracle_pm["ph_weight"], dtype=np.complex64),
    }
    oracle_selected = {
        "C_ps": np.asarray(oracle_pm["C_ps"], dtype=np.float64).reshape(-1)[row_ix],
        "K_ps": np.asarray(oracle_pm["K_ps"], dtype=np.float64).reshape(-1)[row_ix],
        "coh_ps": np.asarray(oracle_pm["coh_ps"], dtype=np.float64).reshape(-1)[row_ix],
        "ph_weight": np.asarray(oracle_pm["ph_weight"], dtype=np.complex64)[row_ix, :],
        "ph_patch": np.asarray(oracle_pm["ph_patch"], dtype=np.complex64)[row_ix, :],
        "ph_grid": np.asarray(oracle_pm["ph_grid"], dtype=np.complex64)[
            context.grid_rows[row_ix],
            context.grid_cols[row_ix],
            :,
        ],
    }

    scenario_outputs: dict[str, dict[str, Any]] = {}
    scenario_metrics: dict[str, dict[str, list[float]]] = {name: {} for name in SCENARIOS}
    scenario_timings: dict[str, float] = {}
    for scenario in SCENARIOS:
        start = time.perf_counter()
        replay = _replay_selected_rows(
            context,
            current_pm,
            scenario_ph_weight[scenario],
            row_ix,
            bperp,
            kernel_backend=kernel_backend,
            native_threads=native_threads,
        )
        scenario_timings[scenario] = time.perf_counter() - start
        scenario_outputs[scenario] = replay
        for key in REPORT_KEYS:
            scenario_metrics[scenario][key] = _row_metric(replay[key], oracle_selected[key], key).tolist()

    aggregate_metrics: dict[str, dict[str, float]] = {}
    dominant_sources: dict[str, str] = {}
    for key in REPORT_KEYS:
        aggregate_metrics[key] = {
            scenario: float(np.max(scenario_metrics[scenario][key]))
            for scenario in SCENARIOS
        }
        dominant_sources[key] = _classify_source(key, aggregate_metrics[key])

    row_reports = []
    for pos, row_number in enumerate(rows):
        row_diffs = {
            key: {
                scenario: float(scenario_metrics[scenario][key][pos])
                for scenario in SCENARIOS
            }
            for key in REPORT_KEYS
        }
        row_reports.append(
            {
                "row": int(row_number),
                "zero_based_row": int(row_ix[pos]),
                "grid_ij": np.asarray(context.grid_ij[row_ix[pos]], dtype=np.int64).tolist(),
                "current_prior_K": float(current_state.k_ps[row_ix[pos]]),
                "oracle_prior_K": float(oracle_state.k_ps[row_ix[pos]]),
                "delta_prior_K": float(current_state.k_ps[row_ix[pos]] - oracle_state.k_ps[row_ix[pos]]),
                "current_prior_weight": float(current_state.weighting[row_ix[pos]]),
                "oracle_prior_weight": float(oracle_state.weighting[row_ix[pos]]),
                "delta_prior_weight": float(current_state.weighting[row_ix[pos]] - oracle_state.weighting[row_ix[pos]]),
                "diffs": row_diffs,
                "first_divergent_intermediate": _first_divergent_intermediate(row_diffs),
                "dominant_sources": {
                    key: _classify_source(key, row_diffs[key])
                    for key in REPORT_KEYS
                },
            }
        )

    return _jsonable(
        {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_root": str(run_root.resolve()),
            "golden_root": str(golden_root.resolve()),
            "patch": patch,
            "rows": list(rows),
            "row_base": int(row_base),
            "scenarios": list(SCENARIOS),
            "kernel_backend": kernel_backend,
            "native_threads": int(native_threads),
            "replay_timings_sec": scenario_timings,
            "localized_clap_windows": {
                scenario: int(scenario_outputs[scenario]["windows_used"])
                for scenario in SCENARIOS
            },
            "dominant_sources": dominant_sources,
            "aggregate_metrics": aggregate_metrics,
            "rows_detail": row_reports,
            "interpretation": (
                "Prior K is inferred from the saved ph_weight phase ramp; prior weighting is inferred "
                "from its scalar magnitude. Each scenario rebuilds ph_weight, ph_grid, localized "
                "ph_patch samples, and topofit outputs for the selected final-iteration rows."
            ),
        }
    )


def _patch_names_for_report(run_root: Path, golden_root: Path) -> list[str]:
    names = [patch.name for patch in discover_authoritative_patch_paths(run_root)]
    if names:
        return names
    return [patch.name for patch in discover_authoritative_patch_paths(golden_root)]


def build_auto_report(
    run_root: Path,
    golden_root: Path,
    *,
    patch: str,
    failure_key: str,
    row_base: int,
    rtol: float,
    atol: float,
    max_failing_rows: int,
    kernel_backend: str,
    native_threads: int,
) -> dict[str, Any]:
    run_patch = _as_patch_dir(run_root, patch)
    golden_patch = _as_patch_dir(golden_root, patch)
    current_pm = read_mat(run_patch / "pm1.mat")
    oracle_pm = read_mat(golden_patch / "pm1.mat")
    first_failure = _first_mismatch_for_key(
        current_pm,
        oracle_pm,
        failure_key,
        row_base=row_base,
        rtol=rtol,
        atol=atol,
        max_rows=max_failing_rows,
    )
    if first_failure is None:
        return {
            "patch": patch,
            "first_failure": None,
            "status": "matched",
        }
    if first_failure["failure_kind"] != "value_mismatch":
        return {
            "patch": patch,
            "first_failure": first_failure,
            "status": "blocked",
        }

    report = build_report(
        run_root,
        golden_root,
        patch=patch,
        rows=tuple(int(row) for row in first_failure["sample_rows"]),
        row_base=row_base,
        kernel_backend=kernel_backend,
        native_threads=native_threads,
    )
    report["first_failure"] = first_failure
    report["status"] = "failed"
    return report


def build_all_patch_auto_report(
    run_root: Path,
    golden_root: Path,
    *,
    failure_key: str,
    row_base: int,
    rtol: float,
    atol: float,
    max_failing_rows: int,
    kernel_backend: str,
    native_threads: int,
) -> dict[str, Any]:
    patches: dict[str, Any] = {}
    for patch_name in _patch_names_for_report(run_root, golden_root):
        patches[patch_name] = build_auto_report(
            run_root,
            golden_root,
            patch=patch_name,
            failure_key=failure_key,
            row_base=row_base,
            rtol=rtol,
            atol=atol,
            max_failing_rows=max_failing_rows,
            kernel_backend=kernel_backend,
            native_threads=native_threads,
        )

    return _jsonable(
        {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_root": str(run_root.resolve()),
            "golden_root": str(golden_root.resolve()),
            "patch_mode": "all_authoritative",
            "failure_key": failure_key,
            "row_base": int(row_base),
            "rtol": float(rtol),
            "atol": float(atol),
            "max_failing_rows": int(max_failing_rows),
            "patch_count": len(patches),
            "failed_patches": [
                patch_name
                for patch_name, patch_report in patches.items()
                if patch_report.get("first_failure") is not None
            ],
            "patches": patches,
            "interpretation": (
                "Each patch auto-selects the first failing C_ps row under the requested tolerance, "
                "then replays ph_weight, ph_grid, ph_patch, and topofit output at that row."
            ),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage 2 final-iteration prior-state drift.")
    parser.add_argument(
        "--run",
        default="inputs_and_outputs/validation_runs/notebook_stage_by_stage/fresh_run",
        help="Current run root or patch directory.",
    )
    parser.add_argument(
        "--golden",
        default="inputs_and_outputs/InSAR_dataset_test_stage8diag_hl",
        help="Oracle/golden run root or patch directory.",
    )
    parser.add_argument("--patch", default="PATCH_1")
    parser.add_argument("--all-patches", action="store_true")
    parser.add_argument("--auto-first-failing", action="store_true")
    parser.add_argument("--failure-key", default="C_ps")
    parser.add_argument("--max-failing-rows", type=int, default=1)
    parser.add_argument("--rows", nargs="+", type=int, default=list(DEFAULT_ROWS))
    parser.add_argument("--row-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--kernel-backend", default="python")
    parser.add_argument("--native-threads", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run).expanduser()
    golden_root = Path(args.golden).expanduser()
    if bool(args.all_patches):
        report = build_all_patch_auto_report(
            run_root,
            golden_root,
            failure_key=str(args.failure_key),
            row_base=int(args.row_base),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_failing_rows=int(args.max_failing_rows),
            kernel_backend=str(args.kernel_backend),
            native_threads=int(args.native_threads),
        )
    elif bool(args.auto_first_failing):
        report = build_auto_report(
            run_root,
            golden_root,
            patch=str(args.patch),
            failure_key=str(args.failure_key),
            row_base=int(args.row_base),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_failing_rows=int(args.max_failing_rows),
            kernel_backend=str(args.kernel_backend),
            native_threads=int(args.native_threads),
        )
    else:
        report = build_report(
            run_root,
            golden_root,
            patch=str(args.patch),
            rows=tuple(int(row) for row in args.rows),
            row_base=int(args.row_base),
            kernel_backend=str(args.kernel_backend),
            native_threads=int(args.native_threads),
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    if bool(args.all_patches):
        for patch_name, patch_report in report["patches"].items():
            first_failure = patch_report.get("first_failure")
            if first_failure is None:
                print(f"{patch_name} matched {args.failure_key}")
                continue
            if patch_report.get("rows_detail"):
                first_detail = patch_report["rows_detail"][0]
                first_intermediate = first_detail["first_divergent_intermediate"]
                source = first_detail["dominant_sources"].get(first_intermediate, "matched")
            else:
                first_intermediate = "unavailable"
                source = "unavailable"
            print(
                f"{patch_name} key={first_failure['key']} row={first_failure.get('row')} "
                f"zero_based={first_failure.get('zero_based_row')} "
                f"run={first_failure.get('run_value')} golden={first_failure.get('golden_value')} "
                f"abs_diff={first_failure.get('abs_diff')} max_abs={first_failure.get('max_abs')} "
                f"max_abs_row={first_failure.get('max_abs_row')} "
                f"first_divergent={first_intermediate} source={source}"
            )
        return 0
    if bool(args.auto_first_failing):
        first_failure = report.get("first_failure")
        if first_failure is None:
            print(f"{args.patch} matched {args.failure_key}")
        elif report.get("rows_detail"):
            first_detail = report["rows_detail"][0]
            first_intermediate = first_detail["first_divergent_intermediate"]
            source = first_detail["dominant_sources"].get(first_intermediate, "matched")
            print(
                f"{args.patch} key={first_failure['key']} row={first_failure.get('row')} "
                f"zero_based={first_failure.get('zero_based_row')} "
                f"run={first_failure.get('run_value')} golden={first_failure.get('golden_value')} "
                f"abs_diff={first_failure.get('abs_diff')} max_abs={first_failure.get('max_abs')} "
                f"max_abs_row={first_failure.get('max_abs_row')} "
                f"first_divergent={first_intermediate} source={source}"
            )
        else:
            print(f"{args.patch} key={first_failure['key']} failure_kind={first_failure['failure_kind']}")
        return 0
    for key in REPORT_KEYS:
        metrics = report["aggregate_metrics"][key]
        print(
            key,
            report["dominant_sources"][key],
            "baseline=",
            f"{metrics['current_K_current_weight']:.6g}",
            "oracleK=",
            f"{metrics['oracle_K_current_weight']:.6g}",
            "oracleWeight=",
            f"{metrics['current_K_oracle_weight']:.6g}",
            "oracleBoth=",
            f"{metrics['oracle_K_oracle_weight']:.6g}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
