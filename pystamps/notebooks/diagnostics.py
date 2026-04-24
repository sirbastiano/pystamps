from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pystamps.io.mat import read_mat


@dataclass(slots=True)
class VelocityDiagnostics:
    root: Path
    plot_mode: str
    lonlat: np.ndarray
    day: np.ndarray
    master_ix: int
    time_axis_days: np.ndarray
    coherence: np.ndarray
    velocity: np.ndarray
    velocity_source: str
    stability: np.ndarray
    stability_source: str
    intercept: np.ndarray
    slope: np.ndarray


@dataclass(slots=True)
class MatchedDiagnostics:
    run_indices: np.ndarray
    stamps_indices: np.ndarray
    decimals: int


def _as_vector(value: Any, *, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _as_points(value: Any) -> np.ndarray:
    lonlat = np.asarray(value, dtype=float)
    if lonlat.ndim != 2:
        return np.empty((0, 2), dtype=float)
    if lonlat.shape[1] == 2:
        return lonlat
    if lonlat.shape[0] == 2:
        return lonlat.T
    return np.empty((0, 2), dtype=float)


def plot_mode_from_step8(apply_step8: bool, override: str | None = None) -> str:
    if override is None or str(override).strip().lower() in {"", "auto"}:
        return "dos" if apply_step8 else "do"
    value = str(override).strip().lower()
    if value not in {"do", "dos"}:
        raise ValueError("plot_mode override must be 'do', 'dos', or None/'auto'")
    return value


def plot_name(prefix: str, plot_mode: str) -> str:
    return f"{prefix}-{plot_mode}"


def _mean_v_coefficients(payload: dict[str, Any], n_ps: int) -> tuple[np.ndarray, np.ndarray]:
    coeffs = np.asarray(payload.get("m", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if coeffs.ndim == 1:
        slope = coeffs.reshape(-1)[:n_ps]
        intercept = np.zeros_like(slope, dtype=np.float32)
        return intercept, slope.astype(np.float32)
    if coeffs.ndim != 2:
        return np.zeros(n_ps, dtype=np.float32), np.zeros(n_ps, dtype=np.float32)
    if coeffs.shape[0] == 2:
        intercept = coeffs[0, :n_ps]
        slope = coeffs[1, :n_ps]
        return intercept.astype(np.float32), slope.astype(np.float32)
    if coeffs.shape[1] == 2:
        intercept = coeffs[:n_ps, 0]
        slope = coeffs[:n_ps, 1]
        return intercept.astype(np.float32), slope.astype(np.float32)
    slope = coeffs.reshape(-1)[:n_ps]
    intercept = np.zeros_like(slope, dtype=np.float32)
    return intercept, slope.astype(np.float32)


def load_velocity_diagnostics(
    root: str | Path,
    *,
    apply_step8: bool = True,
    plot_mode: str | None = None,
    load_coherence: bool = False,
) -> VelocityDiagnostics:
    dataset_root = Path(root).expanduser().resolve()
    ps2 = read_mat(dataset_root / "ps2.mat")
    mean_v_payload = read_mat(dataset_root / "mean_v.mat")
    mv2_path = dataset_root / "mv2.mat"
    mv2 = read_mat(mv2_path) if mv2_path.exists() else {}

    lonlat = _as_points(ps2.get("lonlat", np.empty((0, 2), dtype=np.float64)))
    n_ps = lonlat.shape[0]
    day = _as_vector(ps2.get("day", np.empty((0,), dtype=np.float64)), dtype=float)
    master_ix = int(round(float(np.asarray(ps2.get("master_ix", 1.0)).reshape(-1)[0]))) if day.size else 1
    master_ix = min(max(master_ix, 1), max(1, day.size))
    master_day = float(day[master_ix - 1]) if day.size else 0.0
    time_axis_days = day - master_day
    coherence = np.ones(n_ps, dtype=float)
    if load_coherence:
        pm2 = read_mat(dataset_root / "pm2.mat")
        coherence = _as_vector(pm2.get("coh_ps", np.ones(n_ps, dtype=np.float64)), dtype=float)[:n_ps]

    velocity = _as_vector(mv2.get("mean_v", np.empty((0,), dtype=np.float32)), dtype=float)[:n_ps]
    velocity_source = "mv2.mean_v"
    if velocity.size != n_ps:
        intercept, slope = _mean_v_coefficients(mean_v_payload, n_ps)
        velocity = slope.astype(float)
        velocity_source = "mean_v.m slope fallback"
    else:
        intercept, slope = _mean_v_coefficients(mean_v_payload, n_ps)

    mean_v_std = _as_vector(mv2.get("mean_v_std", np.empty((0,), dtype=np.float32)), dtype=float)[:n_ps]
    if mean_v_std.size == n_ps and np.any(np.abs(mean_v_std) > 0):
        stability = mean_v_std
        stability_source = "mv2.mean_v_std"
    else:
        scla2 = read_mat(dataset_root / "scla2.mat")
        stability = np.abs(_as_vector(scla2.get("C_ps_uw", np.zeros(n_ps, dtype=np.float32)), dtype=float)[:n_ps])
        stability_source = "abs(scla2.C_ps_uw) stability proxy"

    return VelocityDiagnostics(
        root=dataset_root,
        plot_mode=plot_mode_from_step8(apply_step8, plot_mode),
        lonlat=lonlat,
        day=day,
        master_ix=master_ix,
        time_axis_days=time_axis_days,
        coherence=coherence,
        velocity=velocity.astype(np.float32),
        velocity_source=velocity_source,
        stability=stability.astype(np.float32),
        stability_source=stability_source,
        intercept=intercept.astype(np.float32),
        slope=slope.astype(np.float32),
    )


def build_comparison_masks(
    run_diag: VelocityDiagnostics,
    stamps_diag: VelocityDiagnostics | None = None,
    *,
    coherence_threshold: float | None = None,
    masking_strategy: str = "none",
) -> tuple[np.ndarray, np.ndarray | None]:
    run_mask = np.ones(run_diag.velocity.size, dtype=bool)
    stamps_mask = None if stamps_diag is None else np.ones(stamps_diag.velocity.size, dtype=bool)

    strategy = str(masking_strategy).strip().lower()
    if coherence_threshold is None or strategy == "none":
        return run_mask, stamps_mask
    if strategy == "coherence":
        run_mask &= run_diag.coherence[: run_mask.size] >= float(coherence_threshold)
        if stamps_diag is not None and stamps_mask is not None:
            stamps_mask &= stamps_diag.coherence[: stamps_mask.size] >= float(coherence_threshold)
        return run_mask, stamps_mask
    if strategy == "common_coherence":
        if stamps_diag is None or stamps_mask is None:
            run_mask &= run_diag.coherence[: run_mask.size] >= float(coherence_threshold)
            return run_mask, None
        size = min(run_mask.size, stamps_mask.size, run_diag.coherence.size, stamps_diag.coherence.size)
        common = (run_diag.coherence[:size] >= float(coherence_threshold)) & (
            stamps_diag.coherence[:size] >= float(coherence_threshold)
        )
        run_mask[:size] &= common
        run_mask[size:] = False
        stamps_mask[:size] &= common
        stamps_mask[size:] = False
        return run_mask, stamps_mask
    raise ValueError("masking_strategy must be 'none', 'coherence', or 'common_coherence'")


def match_diagnostic_points(
    run_diag: VelocityDiagnostics,
    stamps_diag: VelocityDiagnostics | None,
    *,
    run_mask: np.ndarray | None = None,
    stamps_mask: np.ndarray | None = None,
    decimals: int = 10,
) -> MatchedDiagnostics:
    if stamps_diag is None:
        return MatchedDiagnostics(np.empty((0,), dtype=int), np.empty((0,), dtype=int), int(decimals))

    run_valid = np.flatnonzero(np.asarray(run_mask, dtype=bool)) if run_mask is not None else np.arange(run_diag.lonlat.shape[0])
    stamps_valid = (
        np.flatnonzero(np.asarray(stamps_mask, dtype=bool))
        if stamps_mask is not None
        else np.arange(stamps_diag.lonlat.shape[0])
    )

    if run_valid.size == 0 or stamps_valid.size == 0:
        return MatchedDiagnostics(np.empty((0,), dtype=int), np.empty((0,), dtype=int), int(decimals))

    stamps_points = np.round(stamps_diag.lonlat[stamps_valid], decimals=int(decimals))
    stamps_lookup = {tuple(point.tolist()): int(ix) for point, ix in zip(stamps_points, stamps_valid, strict=False)}

    matched_run: list[int] = []
    matched_stamps: list[int] = []
    run_points = np.round(run_diag.lonlat[run_valid], decimals=int(decimals))
    for point, run_ix in zip(run_points, run_valid, strict=False):
        stamps_ix = stamps_lookup.get(tuple(point.tolist()))
        if stamps_ix is None:
            continue
        matched_run.append(int(run_ix))
        matched_stamps.append(stamps_ix)

    return MatchedDiagnostics(np.asarray(matched_run, dtype=int), np.asarray(matched_stamps, dtype=int), int(decimals))


def _filtered(values: np.ndarray, *, mask: np.ndarray | None = None, outlier_filter: str = "none") -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool).reshape(-1)[: array.size]
    data = array[finite]
    if data.size == 0:
        return data

    method = str(outlier_filter).strip().lower()
    if method in {"", "none"}:
        return data
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return data
        keep = (data >= (q1 - 1.5 * iqr)) & (data <= (q3 + 1.5 * iqr))
        return data[keep]
    if method == "z-score":
        mean = float(np.mean(data))
        std = float(np.std(data))
        if std == 0:
            return data
        keep = np.abs((data - mean) / std) <= 3.0
        return data[keep]
    raise ValueError("outlier_filter must be 'none', 'iqr', or 'z-score'")


def compute_field_statistics(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    compute_percentiles: bool = True,
    percentiles: tuple[float, float] = (5.0, 95.0),
    outlier_filter: str = "none",
) -> dict[str, float]:
    data = _filtered(values, mask=mask, outlier_filter=outlier_filter)
    if data.size == 0:
        stats = {"count": 0.0, "mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    else:
        stats = {
            "count": float(data.size),
            "mean": float(np.mean(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "std": float(np.std(data)),
        }
    if compute_percentiles:
        low, high = percentiles
        if data.size == 0:
            stats[f"p{int(low)}"] = np.nan
            stats[f"p{int(high)}"] = np.nan
        else:
            stats[f"p{int(low)}"] = float(np.percentile(data, low))
            stats[f"p{int(high)}"] = float(np.percentile(data, high))
    return stats


def compare_fields(
    run_values: np.ndarray,
    stamps_values: np.ndarray,
    *,
    run_mask: np.ndarray | None = None,
    stamps_mask: np.ndarray | None = None,
    metrics: tuple[str, ...] = ("RMSE", "MAE", "bias"),
) -> dict[str, float]:
    run = np.asarray(run_values, dtype=float).reshape(-1)
    stamps = np.asarray(stamps_values, dtype=float).reshape(-1)
    size = min(run.size, stamps.size)
    run = run[:size]
    stamps = stamps[:size]
    valid = np.isfinite(run) & np.isfinite(stamps)
    if run_mask is not None:
        valid &= np.asarray(run_mask, dtype=bool).reshape(-1)[:size]
    if stamps_mask is not None:
        valid &= np.asarray(stamps_mask, dtype=bool).reshape(-1)[:size]
    run = run[valid]
    stamps = stamps[valid]
    diff = run - stamps
    out: dict[str, float] = {"count": float(diff.size)}
    if diff.size == 0:
        for metric in metrics:
            out[metric.lower()] = np.nan
        return out
    metric_set = {metric.upper() for metric in metrics}
    if "BIAS" in metric_set:
        out["bias"] = float(np.mean(diff))
    if "RMSE" in metric_set:
        out["rmse"] = float(np.sqrt(np.mean(diff**2)))
    if "MAE" in metric_set:
        out["mae"] = float(np.mean(np.abs(diff)))
    return out


def _interpret_comparison(
    bias: float,
    rmse: float,
    reference_row: dict[str, Any] | None,
) -> str:
    if reference_row is None:
        return "No STAMPS spread reference was available for interpretation."

    spread = np.nan
    low = reference_row.get("p5", np.nan)
    high = reference_row.get("p95", np.nan)
    low = float(low) if low is not None else np.nan
    high = float(high) if high is not None else np.nan
    if np.isfinite(low) and np.isfinite(high):
        spread = high - low
    if not np.isfinite(spread) or spread <= 0:
        spread = float(reference_row.get("std", np.nan))
    if not np.isfinite(spread) or spread <= 0:
        return "The STAMPS reference spread is too small or undefined for a stable interpretation."

    bias_ratio = abs(float(bias)) / spread
    rmse_ratio = abs(float(rmse)) / spread

    if bias_ratio <= 0.05:
        bias_note = "low systematic bias"
    elif bias_ratio <= 0.2:
        bias_note = "moderate systematic bias"
    else:
        bias_note = "strong systematic bias"

    if rmse_ratio <= 0.1:
        spread_note = "tight agreement with the STAMPS spread"
    elif rmse_ratio <= 0.25:
        spread_note = "visible but limited spread difference"
    else:
        spread_note = "broad disagreement relative to the STAMPS spread"

    return f"{bias_note}; {spread_note}"


def build_velocity_report(
    run_diag: VelocityDiagnostics,
    stamps_diag: VelocityDiagnostics | None = None,
    *,
    run_mask: np.ndarray | None = None,
    stamps_mask: np.ndarray | None = None,
    compute_percentiles: bool = True,
    percentiles: tuple[float, float] = (5.0, 95.0),
    outlier_filter: str = "none",
    comparison_metrics: tuple[str, ...] = ("RMSE", "MAE", "bias"),
) -> dict[str, Any]:
    stats_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    matched = match_diagnostic_points(run_diag, stamps_diag, run_mask=run_mask, stamps_mask=stamps_mask)

    for dataset_name, diag, mask in (
        ("pySTAMPS", run_diag, run_mask),
        ("STAMPS", stamps_diag, stamps_mask),
    ):
        if diag is None:
            continue
        for field_name, values, source in (
            (plot_name("v", diag.plot_mode), diag.velocity, diag.velocity_source),
            (plot_name("vs", diag.plot_mode), diag.stability, diag.stability_source),
        ):
            stats = compute_field_statistics(
                values,
                mask=mask,
                compute_percentiles=compute_percentiles,
                percentiles=percentiles,
                outlier_filter=outlier_filter,
            )
            stats_rows.append({"dataset": dataset_name, "field": field_name, "source": source, **stats})

    stats_index = {(row["dataset"], row["field"]): row for row in stats_rows}

    if stamps_diag is not None:
        for field_name, run_values, stamps_values in (
            (plot_name("v", run_diag.plot_mode), run_diag.velocity, stamps_diag.velocity),
            (plot_name("vs", run_diag.plot_mode), run_diag.stability, stamps_diag.stability),
        ):
            metrics = compare_fields(
                run_values[matched.run_indices],
                stamps_values[matched.stamps_indices],
                metrics=comparison_metrics,
            )
            comparison_rows.append(
                {
                    "field": field_name,
                    "alignment": f"lonlat@{matched.decimals}dp",
                    **metrics,
                    "interpretation": _interpret_comparison(
                        metrics.get("bias", np.nan),
                        metrics.get("rmse", np.nan),
                        stats_index.get(("STAMPS", field_name)),
                    ),
                }
            )

    summary_lines = [
        f"plot mode: {run_diag.plot_mode}",
        f"pySTAMPS velocity source: {run_diag.velocity_source}",
        f"pySTAMPS stability source: {run_diag.stability_source}",
    ]
    if stamps_diag is not None:
        summary_lines.append(
            f"matched comparison subset: {matched.run_indices.size} PS aligned by lon/lat rounded to {matched.decimals} decimals"
        )
        for row in comparison_rows:
            bias = row.get("bias", np.nan)
            rmse = row.get("rmse", np.nan)
            summary_lines.append(
                f"{row['field']}: bias={bias:.6g} rmse={rmse:.6g} compared on {int(row.get('count', 0))} PS"
            )
            summary_lines.append(f"{row['field']} interpretation: {row['interpretation']}")
    return {
        "stats_rows": stats_rows,
        "comparison_rows": comparison_rows,
        "summary": "\n".join(summary_lines),
    }


def select_diagnostic_indices(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    rule: str = "high_velocity",
    count: int = 5,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=int)
    array = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)[: array.size]
    ix = np.flatnonzero(valid)
    if ix.size == 0:
        return np.empty((0,), dtype=int)
    mode = str(rule).strip().lower()
    if mode in {"high_velocity", "high_variance"}:
        order = np.argsort(np.abs(array[ix]))[::-1]
        return ix[order[:count]]
    if mode == "evenly_spaced":
        if ix.size <= count:
            return ix
        sample_ix = np.linspace(0, ix.size - 1, num=count, dtype=int)
        return ix[sample_ix]
    raise ValueError("selection rule must be 'high_velocity', 'high_variance', or 'evenly_spaced'")


def fitted_velocity_series(diag: VelocityDiagnostics, indices: np.ndarray) -> np.ndarray:
    ix = np.asarray(indices, dtype=int).reshape(-1)
    ix = ix[(ix >= 0) & (ix < diag.slope.size)]
    if ix.size == 0:
        return np.empty((0, diag.time_axis_days.size), dtype=np.float32)
    return diag.intercept[ix, None] + diag.slope[ix, None] * diag.time_axis_days[None, :]


def stability_series(diag: VelocityDiagnostics, indices: np.ndarray) -> np.ndarray:
    ix = np.asarray(indices, dtype=int).reshape(-1)
    ix = ix[(ix >= 0) & (ix < diag.stability.size)]
    if ix.size == 0:
        return np.empty((0, diag.time_axis_days.size), dtype=np.float32)
    return np.repeat(diag.stability[ix, None], diag.time_axis_days.size, axis=1)


def export_diagnostics_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    stats_path = target / "velocity_stats.csv"
    if report.get("stats_rows"):
        with stats_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report["stats_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(report["stats_rows"])

    comparison_path = target / "velocity_comparison.csv"
    if report.get("comparison_rows"):
        with comparison_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report["comparison_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(report["comparison_rows"])

    summary_path = target / "velocity_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return {
        "stats_csv": stats_path,
        "comparison_csv": comparison_path,
        "summary_json": summary_path,
    }
