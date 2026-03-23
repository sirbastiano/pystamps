from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import numpy as np

from pystamps.io.dataset import discover_dataset


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
