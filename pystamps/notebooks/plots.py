from __future__ import annotations

import numpy as np


MAX_SCATTER_POINTS = 25_000
MAX_VECTOR_POINTS = 100_000
MAX_MATRIX_ROWS = 1_800
MAX_MATRIX_COLS = 12


def scalar(value) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def normalize_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        return pts
    if pts.shape[1] == 2:
        return pts
    if pts.shape[0] == 2:
        return pts.T
    return pts


def point_count(points) -> int:
    pts = normalize_points(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return 0
    return int(pts.shape[0])


def select_points(points, indices) -> np.ndarray:
    pts = normalize_points(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return pts
    ix = np.asarray(indices, dtype=int).reshape(-1)
    ix = ix[(ix >= 0) & (ix < pts.shape[0])]
    return pts[ix]


def stride_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length)
    return np.linspace(0, length - 1, num=limit, dtype=int)


def sample_vector(values, limit: int = MAX_VECTOR_POINTS) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    return array[stride_indices(array.size, limit)]


def sample_points(points, values=None, limit: int = MAX_SCATTER_POINTS):
    pts = normalize_points(points)
    if pts.ndim != 2:
        return pts, None if values is None else np.asarray(values).reshape(-1)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return pts, None if values is None else np.asarray(values).reshape(-1)

    if values is None:
        ix = stride_indices(pts.shape[0], limit)
        return pts[ix], None

    flat_values = np.asarray(values).reshape(-1)
    size = min(pts.shape[0], flat_values.shape[0])
    pts = pts[:size]
    flat_values = flat_values[:size]
    ix = stride_indices(size, limit)
    return pts[ix], flat_values[ix]


def footprint_compare(ax_run, ax_stamps, lonlat_run, lonlat_stamps, title: str, cmap: str = "viridis") -> None:
    run_values = np.arange(point_count(lonlat_run), dtype=float)
    stamps_values = np.arange(point_count(lonlat_stamps), dtype=float)
    scatter_compare(
        ax_run,
        ax_stamps,
        lonlat_run,
        run_values,
        lonlat_stamps,
        stamps_values,
        title,
        cmap=cmap,
    )


def sample_matrix(matrix, max_rows: int = MAX_MATRIX_ROWS, max_cols: int = MAX_MATRIX_COLS) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    row_ix = stride_indices(array.shape[0], min(max_rows, array.shape[0]))
    col_ix = stride_indices(array.shape[1], min(max_cols, array.shape[1]))
    return array[np.ix_(row_ix, col_ix)]


def scatter_compare(
    ax_run,
    ax_stamps,
    lonlat_run,
    values_run,
    lonlat_stamps,
    values_stamps,
    title: str,
    cmap: str = "viridis",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pts_run, val_run = sample_points(lonlat_run, values_run)
    pts_stamps, val_stamps = sample_points(lonlat_stamps, values_stamps)

    sc_run = None
    sc_stamps = None
    if pts_run.ndim == 2 and pts_run.shape[0] > 0:
        sc_run = ax_run.scatter(pts_run[:, 1], pts_run[:, 0], c=val_run, s=3, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax_run.text(0.5, 0.5, "No pySTAMPS points to plot", ha="center", va="center", transform=ax_run.transAxes)

    if pts_stamps.ndim == 2 and pts_stamps.shape[0] > 0:
        sc_stamps = ax_stamps.scatter(
            pts_stamps[:, 1],
            pts_stamps[:, 0],
            c=val_stamps,
            s=3,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        ax_stamps.text(0.5, 0.5, "No STAMPS points to plot", ha="center", va="center", transform=ax_stamps.transAxes)

    ax_run.set_title(f"pySTAMPS {title}")
    ax_stamps.set_title(f"STAMPS {title}")
    ax_run.set_xlabel("lon")
    ax_stamps.set_xlabel("lon")
    ax_run.set_ylabel("lat")

    fig = ax_run.figure
    if sc_run is not None:
        fig.colorbar(sc_run, ax=ax_run, fraction=0.046, pad=0.04)
    if sc_stamps is not None:
        fig.colorbar(sc_stamps, ax=ax_stamps, fraction=0.046, pad=0.04)


def hist_compare(ax, run_values, stamps_values, title: str, bins: int = 60) -> None:
    run_sample = sample_vector(run_values)
    stamps_sample = sample_vector(stamps_values)
    arrays = [array for array in (run_sample, stamps_sample) if array.size]
    bin_edges = bins if not arrays else np.histogram_bin_edges(np.concatenate(arrays), bins=bins)
    ax.hist(stamps_sample, bins=bin_edges, histtype="step", linewidth=1.8, linestyle="--", color="tab:red", label="STAMPS")
    ax.hist(run_sample, bins=bin_edges, histtype="step", linewidth=1.8, color="tab:green", label="pySTAMPS")
    if run_sample.size and stamps_sample.size and np.array_equal(run_sample, stamps_sample):
        ax.text(0.02, 0.98, "sampled values match exactly", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.set_title(title)
    ax.legend()


def heatmap_compare(ax_run, ax_stamps, run_matrix, stamps_matrix, title: str, cmap: str = "viridis") -> None:
    run_sample = sample_matrix(run_matrix)
    stamps_sample = sample_matrix(stamps_matrix)
    im_run = ax_run.imshow(run_sample, aspect="auto", cmap=cmap)
    im_stamps = ax_stamps.imshow(stamps_sample, aspect="auto", cmap=cmap)
    ax_run.set_title(f"pySTAMPS {title}")
    ax_stamps.set_title(f"STAMPS {title}")
    ax_run.set_xlabel("sampled columns")
    ax_stamps.set_xlabel("sampled columns")
    ax_run.set_ylabel("sampled rows")

    fig = ax_run.figure
    fig.colorbar(im_run, ax=ax_run, fraction=0.046, pad=0.04)
    fig.colorbar(im_stamps, ax=ax_stamps, fraction=0.046, pad=0.04)
