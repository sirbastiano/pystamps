from pathlib import Path

import numpy as np

from pystamps.pipeline.ported import (
    Stage5PatchBundle,
    _build_uw_interp_payload,
    _discover_patch_dirs,
    _format_merged_rc2_payload,
    _stage5_best_coherence_keep_masks,
    _stage5_weed_mask,
)


def test_discover_patch_dirs_prefers_legacy_patch_list_when_present(tmp_path: Path) -> None:
    (tmp_path / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (tmp_path / "patch.list_old").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    for name in ["PATCH_1", "PATCH_2", "PATCH_3"]:
        patch_dir = tmp_path / name
        patch_dir.mkdir()
        if name in {"PATCH_1", "PATCH_2"}:
            for filename in ("ps2.mat", "ph2.mat", "pm2.mat"):
                (patch_dir / filename).write_text("stub", encoding="utf-8")

    patch_dirs = _discover_patch_dirs(tmp_path)

    assert [path.name for path in patch_dirs] == ["PATCH_1", "PATCH_2"]


def test_build_uw_interp_payload_prefers_lower_index_on_equal_distance(monkeypatch, tmp_path: Path) -> None:
    uw_grid_payload = {
        "nzix": np.asarray([[True, False, True]], dtype=bool),
        "n_ps": np.asarray(2.0),
    }

    monkeypatch.setattr("pystamps.pipeline.ported._maybe_resolve_external_tool", lambda *args, **kwargs: None)
    payload = _build_uw_interp_payload(tmp_path, uw_grid_payload, triangle_path=None)

    assert int(np.asarray(payload["Z"])[0, 1]) == 1


def test_format_merged_rc2_payload_transposes_without_normalizing() -> None:
    rc2_all = np.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, -2.0j],
            [1.0 - 1.0j, 2.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex64,
    )

    payload = _format_merged_rc2_payload(rc2_all)

    assert payload.shape == (3, 2)
    np.testing.assert_allclose(payload[:, 0], np.asarray([3.0 + 4.0j, 0.0 + 0.0j, -2.0j], dtype=np.complex64))
    np.testing.assert_allclose(
        payload[:, 1],
        np.asarray([1.0 - 1.0j, 2.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_stage5_best_coherence_keep_masks_preserve_patch_order() -> None:
    def bundle(keys: list[bytes], coh: list[float]) -> Stage5PatchBundle:
        n_ps = len(keys)
        return Stage5PatchBundle(
            patch=Path("."),
            ps={},
            n_ps_patch=n_ps,
            ij_patch=np.zeros((n_ps, 3)),
            lonlat_patch=np.zeros((n_ps, 2)),
            xy_patch=np.zeros((n_ps, 3), dtype=np.float32),
            ph_patch2=np.zeros((n_ps, 1), dtype=np.complex64),
            k_patch=np.zeros(n_ps),
            c_patch=np.zeros(n_ps),
            coh_patch=np.asarray(coh),
            ph_patch_patch=np.zeros((n_ps, 1), dtype=np.complex64),
            ph_res_patch=np.zeros((n_ps, 1), dtype=np.float32),
            ij_cols=np.zeros((n_ps, 2), dtype=np.int64),
            ij_keys=keys,
            patch_bounds=None,
        )

    keep_a, keep_b = _stage5_best_coherence_keep_masks(
        [
            bundle([b"a", b"b", b"c"], [0.1, 0.9, 0.2]),
            bundle([b"a", b"c", b"d"], [0.8, 0.1, 0.3]),
        ]
    )

    np.testing.assert_array_equal(keep_a, np.asarray([False, True, True]))
    np.testing.assert_array_equal(keep_b, np.asarray([True, False, True]))


def test_stage5_weed_mask_pads_short_legacy_mask_with_false() -> None:
    mask = _stage5_weed_mask({"ix_weed": np.asarray([True, False, True], dtype=bool)}, 5)

    np.testing.assert_array_equal(mask, np.asarray([True, False, True, False, False]))


def test_stage5_weed_mask_uses_reference_ij_for_short_legacy_mask() -> None:
    candidate_ij = np.asarray(
        [
            [1, 10, 10],
            [2, 11, 10],
            [3, 12, 10],
            [4, 13, 10],
        ],
        dtype=np.float64,
    )
    reference_ij = np.asarray([[1, 10, 10], [2, 12, 10]], dtype=np.float64)

    mask = _stage5_weed_mask(
        {"ix_weed": np.asarray([True, True, False], dtype=bool)},
        4,
        candidate_ij=candidate_ij,
        reference_ij=reference_ij,
    )

    np.testing.assert_array_equal(mask, np.asarray([True, False, True, False]))
