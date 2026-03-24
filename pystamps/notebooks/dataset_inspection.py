from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any

import numpy as np

from pystamps.io.dataset import discover_dataset
from pystamps.pipeline.ported import PortedStageError, _load_complex_columns, _resolve_file, _snap_ifg_records


DATE_RE = re.compile(r"^(?P<date>\d{8})\.rslc(?:\.par)?$")


def _load_text_matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return np.atleast_2d(values)


def _load_text_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _binary_float32_endian(path: Path) -> str:
    sample_count = min(max(32, path.stat().st_size // 4), 512)
    sample_le = np.fromfile(path, dtype="<f4", count=sample_count)
    sample_be = np.fromfile(path, dtype=">f4", count=sample_count)

    def _score(array: np.ndarray) -> tuple[float, float]:
        finite = np.isfinite(array)
        if not finite.any():
            return (-1.0, float("-inf"))
        values = np.asarray(array[finite], dtype=np.float64)
        finite_ratio = float(np.mean(finite))
        abs_values = np.abs(values)
        plausible = np.logical_or(abs_values == 0.0, np.logical_and(abs_values >= 1e-12, abs_values <= 1e12))
        return (finite_ratio + float(np.mean(plausible)), -float(np.nanmedian(abs_values)))

    return ">f4" if _score(sample_be) > _score(sample_le) else "<f4"


def _load_lonlat(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=_binary_float32_endian(path)).astype(np.float64, copy=False)
    if values.size % 2:
        raise ValueError(f"{path} does not contain an even number of float32 values")
    lonlat = values.reshape(-1, 2)
    return lonlat[np.isfinite(lonlat).all(axis=1)]


def _acquisition_dates(dataset_root: Path) -> list[date]:
    dates: list[date] = []
    for path in sorted((dataset_root / "rslc").glob("*.rslc*")):
        match = DATE_RE.match(path.name)
        if match is not None:
            dates.append(datetime.strptime(match.group("date"), "%Y%m%d").date())
    return sorted(dict.fromkeys(dates))


def _inspect_patch(patch_dir: Path) -> dict[str, object]:
    ij = _load_text_matrix(patch_dir / "pscands.1.ij")
    da = _load_text_vector(patch_dir / "pscands.1.da")
    lonlat = _load_lonlat(patch_dir / "pscands.1.ll")

    candidate_count = int(ij.shape[0])
    if da.size != candidate_count:
        raise ValueError(f"{patch_dir.name}: expected {candidate_count} D_A values, found {da.size}")
    if lonlat.shape[0] != candidate_count:
        raise ValueError(f"{patch_dir.name}: expected {candidate_count} lon/lat rows, found {lonlat.shape[0]}")

    return {
        "name": patch_dir.name,
        "count": candidate_count,
        "ij": ij,
        "da": da,
        "lonlat": lonlat,
        "row_min": float(np.min(ij[:, 1])),
        "row_max": float(np.max(ij[:, 1])),
        "col_min": float(np.min(ij[:, 2])),
        "col_max": float(np.max(ij[:, 2])),
        "da_min": float(np.min(da)),
        "da_median": float(np.median(da)),
        "da_mean": float(np.mean(da)),
        "da_max": float(np.max(da)),
        "lon_min": float(np.min(lonlat[:, 0])),
        "lon_max": float(np.max(lonlat[:, 0])),
        "lat_min": float(np.min(lonlat[:, 1])),
        "lat_max": float(np.max(lonlat[:, 1])),
    }


def _display_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_scalar_text(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    values = np.loadtxt(path)
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def _read_vector_text(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.array([], dtype=np.float64)
    values = np.loadtxt(path)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _complex_column_count(path: Path, n_rows: int) -> int:
    if n_rows <= 0:
        return 0
    bytes_per_complex_column = n_rows * 8
    return int(path.stat().st_size // bytes_per_complex_column)


def _sample_phase_preview(path: Path, n_rows: int, max_rows: int = 256, max_cols: int = 8) -> dict[str, np.ndarray]:
    ph = _load_complex_columns(path, n_rows)
    row_ix = np.linspace(0, ph.shape[0] - 1, num=min(max_rows, ph.shape[0]), dtype=int)
    col_ix = np.linspace(0, ph.shape[1] - 1, num=min(max_cols, ph.shape[1]), dtype=int)
    sample = ph[np.ix_(row_ix, col_ix)]
    return {
        "row_ix": row_ix,
        "col_ix": col_ix,
        "angle": np.angle(sample).astype(np.float32),
        "magnitude": np.abs(sample).astype(np.float32),
    }


def _status_label(ok: bool, pending: bool = False) -> str:
    if ok:
        return "ok"
    if pending:
        return "derived at runtime"
    return "warning"


def inspect_stage1_inputs(dataset_root: str | Path, patch_name: str = "PATCH_1") -> dict[str, Any]:
    """Return a notebook-friendly, read-only summary of the true stage-1 inputs."""

    root = Path(dataset_root).expanduser().resolve()
    patch_dir = root / patch_name
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory does not exist: {patch_dir}")

    ij_path = patch_dir / "pscands.1.ij"
    ph_path = patch_dir / "pscands.1.ph"
    ll_path = patch_dir / "pscands.1.ll"
    da_path = patch_dir / "pscands.1.da"
    hgt_path = patch_dir / "pscands.1.hgt"

    required_missing = [name for name, path in {"pscands.1.ij": ij_path, "pscands.1.ph": ph_path, "pscands.1.ll": ll_path}.items() if not path.exists()]
    if required_missing:
        raise FileNotFoundError(f"Missing Stage-1 raw inputs in {patch_dir.name}: {', '.join(required_missing)}")

    ij = _load_text_matrix(ij_path)
    lonlat = _load_lonlat(ll_path)
    n_candidates = int(ij.shape[0])
    if lonlat.shape[0] != n_candidates:
        raise ValueError(f"{patch_dir.name}: expected {n_candidates} lon/lat rows, found {lonlat.shape[0]}")

    if da_path.exists():
        da = _load_text_vector(da_path)
        if da.size != n_candidates:
            raise ValueError(f"{patch_dir.name}: expected {n_candidates} D_A values, found {da.size}")
    else:
        da = np.full(n_candidates, np.nan, dtype=np.float64)

    patch = {
        "name": patch_name,
        "count": n_candidates,
        "ij": ij,
        "da": da,
        "lonlat": lonlat,
        "row_min": float(np.min(ij[:, 1])),
        "row_max": float(np.max(ij[:, 1])),
        "col_min": float(np.min(ij[:, 2])),
        "col_max": float(np.max(ij[:, 2])),
        "da_min": float(np.nanmin(da)) if np.isfinite(da).any() else float("nan"),
        "da_median": float(np.nanmedian(da)) if np.isfinite(da).any() else float("nan"),
        "da_mean": float(np.nanmean(da)) if np.isfinite(da).any() else float("nan"),
        "da_max": float(np.nanmax(da)) if np.isfinite(da).any() else float("nan"),
        "lon_min": float(np.min(lonlat[:, 0])),
        "lon_max": float(np.max(lonlat[:, 0])),
        "lat_min": float(np.min(lonlat[:, 1])),
        "lat_max": float(np.max(lonlat[:, 1])),
    }

    width_path = _resolve_file(patch_dir, "width.txt")
    len_path = _resolve_file(patch_dir, "len.txt")
    day_path = _resolve_file(patch_dir, "day.1.in")
    master_day_path = _resolve_file(patch_dir, "master_day.1.in")
    bperp_path = _resolve_file(patch_dir, "bperp.1.in")

    direct_metadata = day_path is not None and master_day_path is not None and bperp_path is not None
    snap_records: list[tuple[str, str, Path]] = []
    snap_ready = False
    try:
        snap_records = _snap_ifg_records(root)
        snap_ready = (root / "rslc").exists()
    except PortedStageError:
        snap_records = []
        snap_ready = False

    if direct_metadata:
        metadata_mode = "direct patch metadata files"
    elif snap_ready:
        metadata_mode = "derived from diff0 base files and rslc metadata at stage-1 runtime"
    else:
        metadata_mode = "missing"

    phase_preview = _sample_phase_preview(ph_path, n_candidates)
    hgt = (
        np.fromfile(hgt_path, dtype=_binary_float32_endian(hgt_path)).astype(np.float64, copy=False)
        if hgt_path is not None and hgt_path.exists()
        else np.array([], dtype=np.float64)
    )

    acquisition_dates = _acquisition_dates(root)
    day_values = _read_vector_text(day_path)
    bperp_values = _read_vector_text(bperp_path)
    master_day = _read_scalar_text(master_day_path)

    warnings: list[str] = []
    if width_path is None:
        warnings.append("Stage 1 requires width.txt, but no resolvable width.txt was found near the patch.")
    if len_path is None:
        warnings.append("Stage 1 requires len.txt, but no resolvable len.txt was found near the patch.")
    if not da_path.exists():
        warnings.append("pscands.1.da is absent; Stage 1 can still run, but D_A-based inspection is limited.")
    if not hgt_path.exists():
        warnings.append("pscands.1.hgt is absent; height-prior inspection is unavailable for this patch.")
    if not direct_metadata and snap_ready:
        warnings.append(
            "day.1.in, master_day.1.in, and bperp.1.in are absent; pySTAMPS will derive them from diff0/rslc metadata."
        )
    if not direct_metadata and not snap_ready:
        warnings.append(
            "No direct day/master_day/bperp metadata was found, and diff0/rslc synthesis is unavailable. Stage 1 cannot build the time axis."
        )
    if hgt.size and hgt.size != n_candidates:
        warnings.append(
            f"pscands.1.hgt has {hgt.size} values but pscands.1.ij has {n_candidates} candidates. Stage 1 would fail when sorting heights."
        )
    n_ifg = _complex_column_count(ph_path, n_candidates)
    if day_values.size and day_values.size != n_ifg:
        warnings.append(
            f"day.1.in has {day_values.size} entries but pscands.1.ph encodes {n_ifg} interferograms. Stage 1 would reject this mismatch."
        )
    if bperp_values.size and bperp_values.size != n_ifg:
        warnings.append(
            f"bperp.1.in has {bperp_values.size} entries but pscands.1.ph encodes {n_ifg} interferograms. Stage 1 would reject this mismatch."
        )
    if day_values.size and bperp_values.size and day_values.size != bperp_values.size:
        warnings.append(
            f"day.1.in and bperp.1.in disagree ({day_values.size} vs {bperp_values.size} rows). Stage 1 would reject this mismatch."
        )

    input_rows = [
        {
            "role": "candidate indices",
            "file": "pscands.1.ij",
            "location": _display_path(patch_dir / "pscands.1.ij", root),
            "status": "present",
            "required": "yes",
            "shape_or_value": f"{patch['ij'].shape[0]} rows x {patch['ij'].shape[1]} cols",
            "contains": "candidate id, azimuth row, range column",
        },
        {
            "role": "complex phase stack",
            "file": "pscands.1.ph",
            "location": _display_path(ph_path, root),
            "status": "present" if ph_path.exists() else "missing",
            "required": "yes",
            "shape_or_value": f"{n_candidates} candidates x {_complex_column_count(ph_path, n_candidates)} interferograms",
            "contains": "one complex phase value per candidate and interferogram",
        },
        {
            "role": "candidate longitude/latitude",
            "file": "pscands.1.ll",
            "location": _display_path(patch_dir / "pscands.1.ll", root),
            "status": "present",
            "required": "yes",
            "shape_or_value": f"{patch['lonlat'].shape[0]} rows x 2 cols",
            "contains": "longitude and latitude for each candidate",
        },
        {
            "role": "candidate stability metric",
            "file": "pscands.1.da",
            "location": _display_path(da_path, root),
            "status": "present" if da_path.exists() else "missing",
            "required": "no",
            "shape_or_value": f"{patch['da'].size} values" if da_path.exists() else "",
            "contains": "D_A stability values used for QC and plotting",
        },
        {
            "role": "candidate height prior",
            "file": "pscands.1.hgt",
            "location": _display_path(hgt_path, root),
            "status": "present" if hgt.size else "missing",
            "required": "no",
            "shape_or_value": f"{hgt.size} values" if hgt.size else "",
            "contains": "height values used if stage 1 needs them later",
        },
        {
            "role": "patch width",
            "file": "width.txt",
            "location": _display_path(width_path, root),
            "status": "present" if width_path is not None else "missing",
            "required": "yes",
            "shape_or_value": "" if width_path is None else str(int(round(_read_scalar_text(width_path) or 0.0))),
            "contains": "range width of the patch raster",
        },
        {
            "role": "patch length",
            "file": "len.txt",
            "location": _display_path(len_path, root),
            "status": "present" if len_path is not None else "missing",
            "required": "yes",
            "shape_or_value": "" if len_path is None else str(int(round(_read_scalar_text(len_path) or 0.0))),
            "contains": "azimuth length of the patch raster",
        },
        {
            "role": "slave acquisition days",
            "file": "day.1.in",
            "location": _display_path(day_path, root) or "derived from diff0/*.base and rslc/*.rslc.par",
            "status": "present" if day_path is not None else "derived at stage 1" if snap_ready else "missing",
            "required": "yes",
            "shape_or_value": f"{day_values.size} values" if day_values.size else f"{len(snap_records)} values" if snap_records else "",
            "contains": "slave acquisition dates used to build the time axis",
        },
        {
            "role": "master acquisition day",
            "file": "master_day.1.in",
            "location": _display_path(master_day_path, root) or "derived from diff0/*.base and rslc/*.rslc.par",
            "status": "present" if master_day_path is not None else "derived at stage 1" if snap_ready else "missing",
            "required": "yes",
            "shape_or_value": "" if master_day is None else str(int(round(master_day))),
            "contains": "the single master date inserted into the image timeline",
        },
        {
            "role": "perpendicular baseline summary",
            "file": "bperp.1.in",
            "location": _display_path(bperp_path, root) or "derived from diff0/*.base and rslc/*.rslc.par",
            "status": "present" if bperp_path is not None else "derived at stage 1" if snap_ready else "missing",
            "required": "yes",
            "shape_or_value": f"{bperp_values.size} values" if bperp_values.size else f"{len(snap_records)} values" if snap_records else "",
            "contains": "one baseline value per interferogram",
        },
    ]

    consistency_rows = [
        {
            "check": "candidate rows in pscands.1.ij",
            "observed": n_candidates,
            "expected": "base count",
            "status": "ok",
            "why_it_matters": "all other candidate arrays must agree with this count",
        },
        {
            "check": "candidate rows in pscands.1.ll",
            "observed": lonlat.shape[0],
            "expected": n_candidates,
            "status": _status_label(lonlat.shape[0] == n_candidates),
            "why_it_matters": "each candidate needs one longitude/latitude pair",
        },
        {
            "check": "candidate rows in pscands.1.da",
            "observed": int(da.size) if da_path.exists() else "missing",
            "expected": n_candidates if da_path.exists() else "optional",
            "status": _status_label(da.size == n_candidates, pending=not da_path.exists()),
            "why_it_matters": "if present, D_A should align with the candidate table",
        },
        {
            "check": "candidate rows in pscands.1.hgt",
            "observed": int(hgt.size) if hgt.size else "missing",
            "expected": n_candidates if hgt.size else "optional",
            "status": _status_label(hgt.size == n_candidates, pending=not hgt.size),
            "why_it_matters": "if present, heights should align with the candidate table",
        },
        {
            "check": "interferogram columns in pscands.1.ph",
            "observed": n_ifg,
            "expected": "base count",
            "status": "ok",
            "why_it_matters": "timing and baseline vectors must match this column count",
        },
        {
            "check": "entries in day.1.in",
            "observed": int(day_values.size) if day_values.size else "derived from diff0/rslc" if snap_ready else "missing",
            "expected": n_ifg if day_values.size else n_ifg if snap_ready else "required",
            "status": _status_label(day_values.size == n_ifg, pending=not day_values.size and snap_ready),
            "why_it_matters": "Stage 1 uses one slave acquisition date per interferogram",
        },
        {
            "check": "entries in bperp.1.in",
            "observed": int(bperp_values.size) if bperp_values.size else "derived from diff0/rslc" if snap_ready else "missing",
            "expected": n_ifg if bperp_values.size else n_ifg if snap_ready else "required",
            "status": _status_label(bperp_values.size == n_ifg, pending=not bperp_values.size and snap_ready),
            "why_it_matters": "Stage 1 uses one baseline value per interferogram",
        },
        {
            "check": "master_day.1.in",
            "observed": int(round(master_day)) if master_day is not None else "derived from diff0/rslc" if snap_ready else "missing",
            "expected": "single scalar",
            "status": _status_label(master_day is not None, pending=master_day is None and snap_ready),
            "why_it_matters": "the master acquisition is inserted into the Stage-1 time axis",
        },
    ]

    overview_rows = [
        {"metric": "dataset", "value": root.name, "meaning": "dataset that stage 1 would read"},
        {"metric": "patch", "value": patch_name, "meaning": "single patch inspected in this notebook"},
        {"metric": "candidate count", "value": n_candidates, "meaning": "number of persistent-scatterer candidates"},
        {
            "metric": "interferogram count",
            "value": n_ifg,
            "meaning": "number of phase columns in pscands.1.ph",
        },
        {
            "metric": "metadata mode",
            "value": metadata_mode,
            "meaning": "whether time/baseline metadata is stored directly or derived at runtime",
        },
        {
            "metric": "acquisition count",
            "value": len(acquisition_dates),
            "meaning": "number of unique acquisition dates seen in rslc/",
        },
    ]

    preview_rows: list[dict[str, float]] = []
    max_preview = min(8, n_candidates)
    for idx in range(max_preview):
        row = {
            "candidate_id": int(patch["ij"][idx, 0]),
            "azimuth_row": int(patch["ij"][idx, 1]),
            "range_col": int(patch["ij"][idx, 2]),
            "lon": float(patch["lonlat"][idx, 0]),
            "lat": float(patch["lonlat"][idx, 1]),
            "D_A": float(patch["da"][idx]),
        }
        if hgt.size >= n_candidates:
            row["hgt"] = float(hgt[idx])
        preview_rows.append(row)

    acquisition_rows = [
        {"acquisition_index": index + 1, "date": value.isoformat()}
        for index, value in enumerate(acquisition_dates)
    ]
    interferogram_rows = [
        {
            "interferogram_index": index + 1,
            "master_date": master,
            "slave_date": slave,
            "base_file": base_file.name,
        }
        for index, (master, slave, base_file) in enumerate(snap_records)
    ]

    return {
        "dataset_root": root,
        "patch_dir": patch_dir,
        "patch_name": patch_name,
        "overview_rows": overview_rows,
        "input_rows": input_rows,
        "consistency_rows": consistency_rows,
        "preview_rows": preview_rows,
        "acquisition_rows": acquisition_rows,
        "interferogram_rows": interferogram_rows,
        "patch": patch,
        "phase_preview": phase_preview,
        "height_values": hgt,
        "metadata_mode": metadata_mode,
        "warnings": warnings,
    }


def inspect_dataset(dataset_root: str | Path) -> dict[str, object]:
    """Return a notebook-friendly summary of a StaMPS-style dataset."""

    root = Path(dataset_root).expanduser().resolve()
    layout = discover_dataset(root)
    patches = [_inspect_patch(patch_dir) for patch_dir in layout.patches]
    acquisition_dates = _acquisition_dates(root)
    all_da = np.concatenate([patch["da"] for patch in patches]) if patches else np.array([], dtype=np.float64)
    all_lonlat = (
        np.concatenate([patch["lonlat"] for patch in patches]) if patches else np.empty((0, 2), dtype=np.float64)
    )

    return {
        "root": root,
        "patches": patches,
        "patch_count": len(patches),
        "candidate_count": int(sum(int(patch["count"]) for patch in patches)),
        "acquisition_dates": acquisition_dates,
        "all_da": all_da,
        "all_lonlat": all_lonlat,
    }
