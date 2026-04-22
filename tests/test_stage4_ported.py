from __future__ import annotations

from pathlib import Path

import numpy as np

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

    monkeypatch.setattr(ported, "_maybe_resolve_external_tool", lambda *args, **kwargs: "/fake/triangle")

    def fake_run_external_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
        assert cmd == ["/fake/triangle", "-e", "psweed.1.node"]
        log_path.write_text("triangle ok\n", encoding="utf-8")
        _write_edge_file(cwd / "psweed.2.edge", [(1, 2), (2, 3)])

    monkeypatch.setattr(ported, "_run_external_command", fake_run_external_command)

    edges, source = ported._resolve_stage4_edges(patch_dir, xy_weed, strict_reference=False)

    assert source == "triangle_regenerated"
    np.testing.assert_array_equal(edges, np.asarray([[0, 1], [1, 2]], dtype=np.int64))
    assert (patch_dir / "psweed.1.node").read_text(encoding="utf-8").splitlines()[0] == "3 2 0 0"


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
