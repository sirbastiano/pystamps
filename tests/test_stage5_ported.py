from pathlib import Path

import numpy as np

from pystamps.pipeline.ported import (
    _build_uw_interp_payload,
    _discover_patch_dirs,
    _format_merged_rc2_payload,
)


def test_discover_patch_dirs_prefers_patch_list_when_present(tmp_path: Path) -> None:
    (tmp_path / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    for name in ["PATCH_1", "PATCH_2", "PATCH_3"]:
        (tmp_path / name).mkdir()

    patch_dirs = _discover_patch_dirs(tmp_path)

    assert [path.name for path in patch_dirs] == ["PATCH_1"]


def test_build_uw_interp_payload_prefers_lower_index_on_equal_distance(monkeypatch, tmp_path: Path) -> None:
    uw_grid_payload = {
        "nzix": np.asarray([[True, False, True]], dtype=bool),
        "n_ps": np.asarray(2.0),
    }

    monkeypatch.setattr("pystamps.pipeline.ported._maybe_resolve_external_tool", lambda *args, **kwargs: None)
    payload = _build_uw_interp_payload(tmp_path, uw_grid_payload, triangle_path=None)

    assert int(np.asarray(payload["Z"])[0, 1]) == 1


def test_format_merged_rc2_payload_normalizes_and_transposes() -> None:
    rc2_all = np.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, -2.0j],
            [1.0 - 1.0j, 2.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex64,
    )

    payload = _format_merged_rc2_payload(rc2_all)

    assert payload.shape == (3, 2)
    np.testing.assert_allclose(payload[:, 0], np.asarray([0.6 + 0.8j, 0.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex64))
    np.testing.assert_allclose(
        payload[:, 1],
        np.asarray([(1.0 - 1.0j) / np.sqrt(2.0), 1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64),
        rtol=1e-6,
        atol=1e-6,
    )
