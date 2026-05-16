from __future__ import annotations

from pathlib import Path

import numpy as np

from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline import ported


def _write_edge_file(path: Path, rows: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(rows)} 1\n")
        for idx, (a, b) in enumerate(rows, start=1):
            handle.write(f"{idx} {a} {b} 0\n")


def test_resolve_stage4_edges_regenerates_triangle_from_current_nodes(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    # Seed a stale edge file that omits the third node.
    _write_edge_file(patch_dir / "psweed.2.edge", [(1, 2)])

    xy_weed = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    seen: dict[str, str | None] = {}

    def fake_resolve(tool_name: str, configured_path: str | None = None) -> str:
        seen["tool_name"] = tool_name
        seen["configured_path"] = configured_path
        return "/fake/triangle"

    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", fake_resolve)

    def fake_run_external_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
        assert cmd == ["/fake/triangle", "-e", "psweed.1.node"]
        log_path.write_text("triangle ok\n", encoding="utf-8")
        _write_edge_file(cwd / "psweed.2.edge", [(2, 1), (3, 2)])

    monkeypatch.setattr(ported, "_run_external_command", fake_run_external_command)

    edges, source = ported._resolve_stage4_edges(
        patch_dir,
        xy_weed,
        strict_reference=False,
        triangle_path="/configured/triangle",
    )

    assert source == "triangle_regenerated"
    assert seen == {"tool_name": "triangle", "configured_path": "/configured/triangle"}
    np.testing.assert_array_equal(edges, np.asarray([[1, 0], [2, 1]], dtype=np.int64))
    node_lines = (patch_dir / "psweed.1.node").read_text(encoding="utf-8").splitlines()
    assert node_lines[0] == "3 2 0 0"
    assert node_lines[1] == "1 0.000000 0.000000"


def test_resolve_stage4_edges_uses_existing_file_without_triangle(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    _write_edge_file(patch_dir / "psweed.2.edge", [(1, 2), (2, 3)])

    xy_weed = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", lambda *args, **kwargs: None)

    edges, source = ported._resolve_stage4_edges(patch_dir, xy_weed, strict_reference=False)

    assert source == "triangle_file"
    np.testing.assert_array_equal(edges, np.asarray([[0, 1], [1, 2]], dtype=np.int64))


def test_resolve_stage4_edges_preserves_existing_triangle_orientation(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    _write_edge_file(patch_dir / "psweed.2.edge", [(2, 1), (3, 2)])

    xy_weed = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", lambda *args, **kwargs: None)

    edges, source = ported._resolve_stage4_edges(patch_dir, xy_weed, strict_reference=False)

    assert source == "triangle_file"
    np.testing.assert_array_equal(edges, np.asarray([[1, 0], [2, 1]], dtype=np.int64))


def test_stage4_duplicate_noise_masks_keep_matlab_shapes(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    write_mat(
        tmp_path / "parms.mat",
        {
            "weed_neighbours": "n",
            "weed_zero_elevation": "n",
            "weed_standard_dev": np.asarray(1.0, dtype=np.float64),
            "weed_max_noise": np.asarray(np.inf, dtype=np.float64),
            "weed_time_win": np.asarray(360.0, dtype=np.float64),
            "small_baseline_flag": "n",
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(4.0, dtype=np.float64),
            "n_ifg": np.asarray(3.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "day": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "xy": np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [4.0, 2.0, 2.0],
                ],
                dtype=np.float64,
            ),
            "ij": np.asarray(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 2.0],
                    [3.0, 1.0, 3.0],
                    [4.0, 1.0, 4.0],
                ],
                dtype=np.float64,
            ),
        },
    )
    write_mat(
        patch_dir / "select1.mat",
        {
            "ix": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "keep_ix": np.ones((4, 1), dtype=np.bool_),
            "K_ps2": np.zeros((4, 1), dtype=np.float64),
            "C_ps2": np.zeros((4, 1), dtype=np.float64),
            "coh_ps2": np.asarray([0.5, 0.1, 0.9, 0.8], dtype=np.float64).reshape(-1, 1),
        },
    )
    write_mat(patch_dir / "ph1.mat", {"ph": np.ones((4, 3), dtype=np.complex64)})

    def fake_resolve_edges(patch: Path, xy_weed: np.ndarray, **kwargs):
        assert patch == patch_dir
        np.testing.assert_array_equal(xy_weed[:, 0], np.asarray([1.0, 3.0, 4.0]))
        return np.asarray([[1, 0], [2, 1]], dtype=np.int64), "triangle_regenerated"

    def fake_edge_stats_kernel(**kwargs):
        assert kwargs["ph_weed"].shape == (3, 3)
        np.testing.assert_array_equal(kwargs["node_a"], np.asarray([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(kwargs["node_b"], np.asarray([0, 1], dtype=np.int64))
        return {
            "ps_std": np.asarray([0.5, 1.5, 0.2], dtype=np.float64),
            "ps_max": np.zeros(3, dtype=np.float64),
        }

    monkeypatch.setattr(ported, "_resolve_stage4_edges", fake_resolve_edges)
    monkeypatch.setattr(ported, "run_stage4_edge_stats_kernel", fake_edge_stats_kernel)

    details = ported.stage4_weed_ps(patch_dir, backend="python", triangle_path="/configured/triangle")
    payload = read_mat(patch_dir / "weed1.mat")

    assert details == "Stage 4 retained 2/4 selected PS"
    np.testing.assert_array_equal(np.asarray(payload["ix_weed"]).reshape(-1).astype(bool), [True, False, False, True])
    np.testing.assert_array_equal(np.asarray(payload["ix_weed2"]).reshape(-1).astype(bool), [True, False, True])
    assert np.asarray(payload["ps_std"]).reshape(-1).shape == (3,)
    assert np.asarray(payload["ps_max"]).reshape(-1).shape == (3,)
