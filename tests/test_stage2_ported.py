from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pystamps.config import ConfigError, load_config
from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline import ported

_STAGE2_RANDOM_HIST_CALLS = int(np.ceil(300000 / 250))


def test_load_config_parses_stage2_checkpoint_settings(tmp_path: Path) -> None:
    cfg_path = tmp_path / "stage2.yaml"
    cfg_path.write_text(
        "runtime:\n"
        "  stage2_kernel_backend: native\n"
        "  stage2_patch_backend_overrides:\n"
        "    PATCH_1: python\n"
        "    PATCH_2: native\n"
        "  stage2_native_threads: 3\n"
        "  stage2_checkpoint_mode: periodic\n"
        "  stage2_checkpoint_interval: 4\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.runtime.stage2_kernel_backend == "native"
    assert cfg.runtime.stage2_patch_backend_overrides == {"PATCH_1": "python", "PATCH_2": "native"}
    assert cfg.runtime.stage2_native_threads == 3
    assert cfg.runtime.stage2_checkpoint_mode == "periodic"
    assert cfg.runtime.stage2_checkpoint_interval == 4


def test_load_config_parses_generic_kernel_backend_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "kernel-overrides.yaml"
    cfg_path.write_text(
        "runtime:\n"
        "  kernel_backend_overrides:\n"
        "    stage2_topofit: native\n"
        "    stage7_scla: gpu\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.runtime.kernel_backend_overrides == {
        "stage2_topofit": "native",
        "stage7_scla": "cuda",
    }


def test_load_config_rejects_non_mapping_kernel_backend_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "kernel-overrides-bad.yaml"
    cfg_path.write_text(
        "runtime:\n"
        "  kernel_backend_overrides:\n"
        "    - native\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="runtime.kernel_backend_overrides"):
        load_config(cfg_path)


def test_stage2_override_map_rejects_cuda_for_stage2_kernels() -> None:
    with pytest.raises(ported.PortedStageError, match="Unsupported stage-2 kernel backend"):
        ported._normalize_kernel_backend_override_map({"stage2_topofit": "cuda"})


def test_stage2_override_map_allows_generic_backend_for_non_stage2_kernels() -> None:
    assert ported._normalize_kernel_backend_override_map({"stage7_scla": "gpu"}) == {"stage7_scla": "cuda"}


def test_stage2_estimate_gamma_routes_per_kernel_overrides(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0], [25.0, 40.0]], dtype=np.float64)}
    parms_payload = {"gamma_max_iterations": np.asarray(1.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    backends: dict[str, list[str]] = {
        "topofit": [],
        "rowcoh": [],
        "histogram": [],
    }

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(
        ported,
        "_stage2_random_phase_chunks",
        lambda *args, **kwargs: [np.ones((2, 2), dtype=np.complex64)],
    )
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        backends["rowcoh"].append(backend)
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        backends["topofit"].append(kernel_backend)
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    def fake_histogram(
        values: np.ndarray,
        centers: np.ndarray,
        *,
        backend: str = "auto",
    ) -> np.ndarray:
        backends["histogram"].append(backend)
        return np.ones(np.asarray(centers).reshape(-1).size, dtype=np.float64)

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
    monkeypatch.setattr(ported, "run_stage2_histogram_kernel", fake_histogram)

    result = ported.stage2_estimate_gamma(
        patch_dir,
        kernel_backend="python",
        kernel_backend_overrides={
            "stage2_topofit": "native",
            "stage2_topofit_coh_row_invariant": "native",
            "stage2_histogram": "python",
        },
        debug=False,
    )

    assert result.startswith("Stage 2 computed coherence for 2 candidates in ")
    assert backends["rowcoh"] == ["native"]
    assert backends["topofit"]
    assert set(backends["topofit"]) == {"native"}
    assert backends["histogram"]
    assert set(backends["histogram"]) == {"python"}


def test_resolve_stage1_metadata_uses_existing_ps1_when_text_metadata_missing(tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    ij = np.asarray([[1.0, 10.0, 20.0], [2.0, 30.0, 40.0]], dtype=np.float64)
    write_mat(
        patch_dir / "ps1.mat",
        {
            "day": np.asarray([738949.0, 738961.0, 738973.0], dtype=np.float64),
            "master_day": np.asarray(738949.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, -354.25, -148.0], dtype=np.float32),
        },
    )
    write_mat(
        patch_dir / "bp1.mat",
        {
            "bperp_mat": np.asarray(
                [
                    [-354.25, -148.0],
                    [-354.125, -147.875],
                ],
                dtype=np.float32,
            )
        },
    )

    metadata = ported.resolve_stage1_metadata(patch_dir, ij)

    assert metadata.day_file is None
    assert metadata.master_day_file is None
    assert metadata.bperp_file is None
    np.testing.assert_array_equal(metadata.day_full, np.asarray([738949.0, 738961.0, 738973.0], dtype=np.float64))
    assert metadata.master_day == 738949.0
    assert metadata.master_ix == 1
    np.testing.assert_array_equal(metadata.bperp_full, np.asarray([0.0, -354.25, -148.0], dtype=np.float64))
    np.testing.assert_array_equal(
        metadata.bperp_mat,
        np.asarray(
            [
                [-354.25, -148.0],
                [-354.125, -147.875],
            ],
            dtype=np.float32,
        ),
    )


def test_stage1_load_initial_uses_existing_ps1_metadata_without_in_files(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    for name in ("pscands.1.ij", "pscands.1.ph", "pscands.1.ll", "width.txt", "len.txt"):
        (patch_dir / name).write_bytes(b"")

    write_mat(
        patch_dir / "ps1.mat",
        {
            "day": np.asarray([738949.0, 738961.0, 738973.0], dtype=np.float64),
            "master_day": np.asarray(738949.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, -354.25, -148.0], dtype=np.float32),
        },
    )
    write_mat(
        patch_dir / "bp1.mat",
        {
            "bperp_mat": np.asarray(
                [
                    [-354.25, -148.0],
                    [-354.125, -147.875],
                ],
                dtype=np.float32,
            )
        },
    )

    writes: dict[str, dict[str, object]] = {}

    def fake_load_text_matrix(path: Path, dtype=float) -> np.ndarray:
        if path.name == "pscands.1.ij":
            return np.asarray([[1.0, 10.0, 20.0], [2.0, 30.0, 40.0]], dtype=np.float64)
        raise AssertionError(f"unexpected stage-1 metadata text load: {path}")

    monkeypatch.setattr(ported, "_load_text_matrix", fake_load_text_matrix)
    monkeypatch.setattr(
        ported,
        "_load_complex_columns",
        lambda path, n_rows: np.asarray(
            [
                [0.8 + 0.2j, 0.6 + 0.4j],
                [0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        ),
    )
    monkeypatch.setattr(
        ported,
        "_load_binary_float32",
        lambda path, kind: np.asarray([12.0, 45.0, 13.0, 46.0], dtype=np.float32),
    )
    monkeypatch.setattr(ported, "_local_xy_from_lonlat", lambda lonlat, heading_deg=None: (lonlat.copy(), np.zeros(2)))
    monkeypatch.setattr(ported, "_quantize_xy_millimeters", lambda xy: np.asarray(xy, dtype=np.float32))
    monkeypatch.setattr(ported, "_stage1_heading_deg", lambda patch_dir: None)
    monkeypatch.setattr(ported, "_stage1_geometry", lambda patch_dir, ij: None)
    monkeypatch.setattr(ported, "_build_stage_options", lambda patch_dir: SimpleNamespace(mean_range=830000.0, mean_incidence=0.4))
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: writes.__setitem__(Path(path).name, payload))

    result = ported.stage1_load_initial(patch_dir)

    assert result == "Stage 1 created ps1/ph1 for 2 candidates"
    ps1 = writes["ps1.mat"]
    np.testing.assert_array_equal(np.asarray(ps1["day"]), np.asarray([738949.0, 738961.0, 738973.0], dtype=np.float64))
    assert float(np.asarray(ps1["master_ix"]).reshape(-1)[0]) == 1.0
    np.testing.assert_array_equal(np.asarray(ps1["bperp"]), np.asarray([0.0, -354.25, -148.0], dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(writes["bp1.mat"]["bperp_mat"]),
        np.asarray(
            [
                [-354.25, -148.0],
                [-354.125, -147.875],
            ],
            dtype=np.float32,
        ),
    )


def test_ps_topofit_batch_row_invariant_matches_generic() -> None:
    rng = np.random.default_rng(7)
    phase = rng.normal(size=(32, 6))
    amp = rng.uniform(0.5, 1.5, size=(32, 6))
    cpxphase = (amp * np.exp(1j * phase)).astype(np.complex128)
    bperp = np.tile(np.asarray([-120.0, -40.0, 0.0, 55.0, 90.0, 130.0], dtype=np.float64), (32, 1))

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=1.5)
    observed = ported._ps_topofit_batch_row_invariant(cpxphase, bperp, n_trial_wraps=1.5)

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-6, rtol=0.0)


def test_ps_topofit_near_max_trial_indices_keep_local_peaks_only() -> None:
    coh_trial = np.asarray([0.1, 0.5, 0.2, 0.49985, 0.4, 0.5e-1], dtype=np.float64)

    observed = ported._ps_topofit_near_max_trial_indices(coh_trial)

    np.testing.assert_array_equal(observed, np.asarray([1, 3], dtype=np.int64))

def test_ps_topofit_select_candidate_keeps_endpoint_symmetric_coarse_peak() -> None:
    observed = ported._ps_topofit_select_candidate(
        np.asarray([0, 12], dtype=np.int64),
        np.asarray([0.4999, 0.5], dtype=np.float64),
        np.asarray([0.8, 0.81], dtype=np.float64),
        13,
    )

    assert observed == 12


def test_ps_topofit_select_candidate_prefers_refined_winner_for_non_endpoint_peaks() -> None:
    observed = ported._ps_topofit_select_candidate(
        np.asarray([5, 11], dtype=np.int64),
        np.asarray([0.4805803620005628, 0.4805406966462037], dtype=np.float64),
        np.asarray([0.48055193847810107, 0.4806269793017278], dtype=np.float64),
        13,
    )

    assert observed == 11


def test_ps_topofit_select_candidate_prefers_refined_oracle_backed_winner() -> None:
    observed = ported._ps_topofit_select_candidate(
        np.asarray([5, 12], dtype=np.int64),
        np.asarray([0.07544345285859379, 0.07534473474841559], dtype=np.float64),
        np.asarray([0.07550967603443018, 0.10078124974475239], dtype=np.float64),
        13,
    )

    assert observed == 12


def test_ps_topofit_select_candidate_prefers_refined_peak_when_coarse_worse() -> None:
    observed = ported._ps_topofit_select_candidate(
        np.asarray([7, 11], dtype=np.int64),
        np.asarray([0.24294833618216508, 0.24314597798524543], dtype=np.float64),
        np.asarray([0.24287533848242288, 0.24271840306689843], dtype=np.float64),
        13,
    )

    assert observed == 7


def test_ps_topofit_single_routes_through_near_max_selector(monkeypatch) -> None:
    trial_mult = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    refine_calls: list[float] = []

    monkeypatch.setattr(ported, "_stage2_trial_values", lambda n_trial_wraps: trial_mult)
    monkeypatch.setattr(
        ported,
        "_ps_topofit_near_max_trial_indices",
        lambda coh_trial: np.asarray([0, 2], dtype=np.int64),
    )

    def fake_refine(cpx, bp64, weighting, wb, den_lin, coarse_k0):
        refine_calls.append(float(coarse_k0))
        if coarse_k0 < 0:
            return -10.0, -1.0, 0.2, np.full(cpx.shape, 1.0 + 0.0j, dtype=np.complex64)
        return 10.0, 1.0, 0.9, np.full(cpx.shape, 0.0 + 1.0j, dtype=np.complex64)

    monkeypatch.setattr(ported, "_ps_topofit_refine_candidate", fake_refine)

    observed = ported._ps_topofit_single(
        np.asarray([1.0 + 0.0j, 0.5 + 0.5j], dtype=np.complex128),
        np.asarray([-2.0, 2.0], dtype=np.float64),
        n_trial_wraps=0.125,
    )

    assert refine_calls == [-(np.pi / 16.0), np.pi / 16.0]
    assert observed[0] == 10.0
    assert observed[1] == 1.0
    assert observed[2] == 0.9
    np.testing.assert_allclose(
        observed[3],
        np.asarray([0.0 + 1.0j, 0.0 + 1.0j], dtype=np.complex64),
        rtol=0.0,
        atol=0.0,
    )


def test_ps_topofit_single_refines_argmax_coarse_candidate() -> None:
    cpxphase = np.asarray(
        [
            (0.9982544183731079 - 0.05906030535697937j),
            (0.8873175978660583 + 0.4611586332321167j),
            (0.29619884490966797 + 0.9551264047622681j),
            (-0.9712775945663452 + 0.23794908821582794j),
            (-0.2978763282299042 - 0.9546045064926147j),
            (0.5448974967002869 + 0.8385025262832642j),
            (-0.792346715927124 + 0.6100711226463318j),
            (-0.9800438284873962 - 0.19878160953521729j),
            (-0.07463288307189941 + 0.9972109794616699j),
            (-0.48219722509384155 + 0.87606281042099j),
            (-0.9998970031738281 - 0.014354166574776173j),
            (0.9977385401725769 - 0.06721504032611847j),
            (-0.09485628455877304 + 0.9954910278320312j),
            (-0.3512025773525238 + 0.9362995624542236j),
            (0.8777415752410889 + 0.4791341722011566j),
            (0.9811630249023438 + 0.19318196177482605j),
            (-0.11611422151327133 - 0.9932358860969543j),
            (0.8924177885055542 + 0.45121023058891296j),
            (0.09596839547157288 + 0.9953843355178833j),
            (0.9769843220710754 + 0.21331138908863068j),
            (-0.021037235856056213 + 0.9997786283493042j),
            (-0.8792001605033875 + 0.47645264863967896j),
            (-0.1141195148229599 - 0.9934670925140381j),
            (0.3829892873764038 - 0.9237527847290039j),
            (-0.9088584780693054 - 0.41710495948791504j),
            (-0.010275483131408691 - 0.9999473094940186j),
            (-0.9824351072311401 + 0.18660499155521393j),
            (-0.17854323983192444 - 0.9839321374893188j),
            (0.1761409044265747 - 0.984364926815033j),
            (0.3011814057826996 - 0.9535667300224304j),
            (0.9816787242889404 - 0.19054442644119263j),
            (0.7752916216850281 - 0.6316033005714417j),
            (0.9490712285041809 - 0.315062016248703j),
            (0.03413497656583786 + 0.9994171261787415j),
            (0.1550622135400772 + 0.9879047274589539j),
            (0.8024176955223083 + 0.5967625975608826j),
            (0.11122504621744156 + 0.9937950968742371j),
            (0.6744176149368286 + 0.7383500933647156j),
            (-0.9529107213020325 - 0.3032509386539459j),
            (0.6581352949142456 - 0.7528997659683228j),
            (-0.4477686285972595 + 0.8941494822502136j),
            (0.9615651369094849 - 0.27457690238952637j),
            (-0.6289923787117004 - 0.7774114012718201j),
            (0.8930615782737732 - 0.4499346911907196j),
            (0.08623586595058441 + 0.9962747693061829j),
            (-0.9984840154647827 - 0.055043138563632965j),
            (0.58107590675354 - 0.8138492703437805j),
            (0.5663127899169922 + 0.8241905570030212j),
            (-0.7192481756210327 - 0.6947533488273621j),
            (0.3608364164829254 + 0.9326291680335999j),
            (0.15415889024734497 - 0.9880460500717163j),
            (0.9999878406524658 - 0.004955730866640806j),
            (-0.874171793460846 + 0.4856167435646057j),
            (-0.8551686406135559 - 0.5183498859405518j),
            (-0.7429763674736023 - 0.669317364692688j),
            (-0.8908007740974426 - 0.45439407229423523j),
            (-0.5406952500343323 + 0.8412185907363892j),
            (0.5624164342880249 - 0.8268542289733887j),
            (0.7658286690711975 - 0.6430449485778809j),
            (0.688852071762085 - 0.7249021530151367j),
            (-0.6209292411804199 - 0.7838664650917053j),
            (0.10671449452638626 - 0.9942896366119385j),
            (-0.7070095539093018 - 0.7072041630744934j),
            (0.2713291347026825 + 0.9624866843223572j),
            (0.3476226329803467 - 0.9376345276832581j),
            (0.1296563446521759 + 0.991558849811554j),
            (0.3715232312679291 + 0.9284237027168274j),
            (0.993870735168457 + 0.11054952442646027j),
            (0.8484829664230347 - 0.5292226672172546j),
            (-0.5058281421661377 - 0.8626341819763184j),
            (0.3247547447681427 - 0.9457983374595642j),
            (0.9948955774307251 - 0.10090969502925873j),
            (-0.30419793725013733 - 0.9526088833808899j),
            (0.9999990463256836 + 0.0013774563558399677j),
            (0.9995426535606384 - 0.03023890033364296j),
        ],
        dtype=np.complex128,
    )
    bperp = np.asarray(
        [
            -354.87603759765625,
            -148.41204833984375,
            11.849981307983398,
            -143.1541290283203,
            -32.72481155395508,
            -25.145904541015625,
            -78.75508117675781,
            -264.9224548339844,
            -76.25702667236328,
            -51.60499954223633,
            -27.05449104309082,
            -60.93518829345703,
            -136.94781494140625,
            -36.86498260498047,
            -70.62962341308594,
            -105.08844757080078,
            -55.63603210449219,
            -184.31182861328125,
            -77.8426742553711,
            -59.77418899536133,
            -166.5838623046875,
            -227.78106689453125,
            -140.2783966064453,
            -80.2271728515625,
            -75.05155944824219,
            -99.26610565185547,
            -116.754150390625,
            -187.2490692138672,
            -142.78953552246094,
            -70.37977600097656,
            -94.35164642333984,
            -80.41920471191406,
            -104.22224426269531,
            -335.36883544921875,
            24.38962745666504,
            187.44175720214844,
            94.51117706298828,
            177.63790893554688,
            155.34625244140625,
            113.44317626953125,
            213.68711853027344,
            103.27958679199219,
            121.45433044433594,
            -64.13392639160156,
            -43.7103271484375,
            -86.97090148925781,
            -50.98783874511719,
            -162.53964233398438,
            -298.62713623046875,
            -383.17041015625,
            -103.06913757324219,
            50.384891510009766,
            -6.7172770500183105,
            -41.386409759521484,
            -65.24752044677734,
            -100.1313705444336,
            -9.320809364318848,
            45.66149139404297,
            2.6101229190826416,
            -203.9463653564453,
            -195.20115661621094,
            -32.57182693481445,
            65.24066162109375,
            -14.017067909240723,
            -36.665836334228516,
            49.27800369262695,
            149.13890075683594,
            230.8185577392578,
            268.7464599609375,
            203.97015380859375,
            179.36856079101562,
            143.6444549560547,
            121.03768920898438,
            110.94869995117188,
            -41.953399658203125,
        ],
        dtype=np.float64,
    )

    K0, C0, coh0, _ = ported._ps_topofit_single(cpxphase, bperp, n_trial_wraps=0.725669801235199)
    trial_mult = ported._stage2_trial_values(0.725669801235199)
    bperp_range = float(np.max(bperp) - np.min(bperp))
    trial_phase = bperp / bperp_range * (np.pi / 4.0)
    trial_phase_mat = np.exp(-1j * (trial_phase[:, None] * trial_mult[None, :])).astype(np.complex128)
    phaser_sum = np.sum(trial_phase_mat * cpxphase[:, None], axis=0, dtype=np.complex128)
    coh_trial = np.abs(phaser_sum).astype(np.float64)
    coh_trial /= float(np.sum(np.abs(cpxphase), dtype=np.float64))
    candidate_ix = ported._ps_topofit_near_max_trial_indices(coh_trial)
    weighting = np.abs(cpxphase).astype(np.float64)
    wb = weighting * bperp
    den_lin = float(np.sum(wb * wb))
    refined = []
    for trial_ix in candidate_ix:
        coarse_k0 = (np.pi / 4.0) / bperp_range * float(trial_mult[int(trial_ix)])
        refined.append(
            ported._ps_topofit_refine_candidate(
                cpxphase,
                bperp,
                weighting,
                wb,
                den_lin,
                coarse_k0,
            )
        )
    selected_trial_ix = ported._ps_topofit_select_candidate(
        candidate_ix,
        coh_trial[candidate_ix],
        np.asarray([result[2] for result in refined], dtype=np.float64),
        trial_mult.size,
    )
    expected_local_ix = int(np.flatnonzero(candidate_ix == selected_trial_ix)[0])
    expected_K0, expected_C0, expected_coh0, _ = refined[expected_local_ix]

    assert candidate_ix.size > 1
    np.testing.assert_allclose(K0, expected_K0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(C0, expected_C0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(coh0, expected_coh0, rtol=0.0, atol=1e-12)


def test_ps_topofit_single_uses_selected_near_max_candidate(monkeypatch) -> None:
    cpxphase = np.asarray([1.0 + 0.0j, 0.5 + 0.5j, 0.25 - 0.75j], dtype=np.complex128)
    bperp = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)

    monkeypatch.setattr(ported, "_ps_topofit_near_max_trial_indices", lambda coh_trial: np.asarray([0, 2], dtype=np.int64))

    refine_calls: list[float] = []

    def fake_refine(
        cpx: np.ndarray,
        bp64: np.ndarray,
        weighting: np.ndarray,
        wb: np.ndarray,
        den_lin: float,
        coarse_k0: float,
    ) -> tuple[float, float, float, np.ndarray]:
        refine_calls.append(float(coarse_k0))
        if len(refine_calls) == 1:
            return (-1.5, 0.1, 0.49, np.full(cpx.shape, 2 + 0j, dtype=np.complex128))
        return (2.5, -0.2, 0.61, np.full(cpx.shape, 3 + 0j, dtype=np.complex128))

    monkeypatch.setattr(ported, "_ps_topofit_refine_candidate", fake_refine)
    monkeypatch.setattr(ported, "_ps_topofit_select_candidate", lambda *_args: 2)

    K0, C0, coh0, phase_residual = ported._ps_topofit_single(cpxphase, bperp, n_trial_wraps=0.2)

    assert len(refine_calls) == 2
    np.testing.assert_allclose(K0, 2.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(C0, -0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(coh0, 0.61, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(phase_residual, np.full(cpxphase.shape, 3 + 0j, dtype=np.complex64), rtol=0.0, atol=0.0)


def test_ps_topofit_single_refines_argmax_candidate_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cpxphase = np.asarray([1.0 + 0.0j, 0.5 + 0.5j, 0.25 - 0.75j], dtype=np.complex128)
    bperp = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)

    refine_calls: list[float] = []

    def fake_refine(
        cpx: np.ndarray,
        bp64: np.ndarray,
        weighting: np.ndarray,
        wb: np.ndarray,
        den_lin: float,
        coarse_k0: float,
    ) -> tuple[float, float, float, np.ndarray]:
        refine_calls.append(float(coarse_k0))
        return (2.5, -0.2, 0.61, np.full(cpx.shape, 3 + 0j, dtype=np.complex128))

    monkeypatch.setattr(ported, "_ps_topofit_refine_candidate", fake_refine)

    K0, C0, coh0, phase_residual = ported._ps_topofit_single(cpxphase, bperp, n_trial_wraps=0.2)

    trial_mult = ported._stage2_trial_values(0.2)
    bperp_range = float(np.max(bperp) - np.min(bperp))
    trial_phase = bperp / bperp_range * (np.pi / 4.0)
    trial_phase_mat = np.exp(-1j * (trial_phase[:, None] * trial_mult[None, :])).astype(np.complex128)
    phaser_sum = np.sum(trial_phase_mat * cpxphase[:, None], axis=0, dtype=np.complex128)
    coh_trial = np.abs(phaser_sum).astype(np.float64)
    coh_trial /= float(np.sum(np.abs(cpxphase), dtype=np.float64))
    coarse_ix = int(np.argmax(coh_trial))
    expected_coarse_k0 = (np.pi / 4.0) / bperp_range * float(trial_mult[coarse_ix])

    assert len(refine_calls) == 1
    np.testing.assert_allclose(refine_calls[0], expected_coarse_k0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(K0, 2.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(C0, -0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(coh0, 0.61, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(phase_residual, np.full(cpxphase.shape, 3 + 0j, dtype=np.complex64), rtol=0.0, atol=0.0)


def test_stage2_trial_values_match_stamps_topofit_trial_semantics() -> None:
    values = ported._stage2_trial_values(0.725669801235199)

    assert values.shape == (13,)
    np.testing.assert_array_equal(values, np.arange(-6.0, 7.0, dtype=np.float64))


def test_ps_topofit_batch_dispatches_row_invariant_fast_path(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fast(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str,
        threads: int,
        cpu_fallback: object | None,
    ):
        calls.append("fast")
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.zeros((n_row, n_col), dtype=np.complex64),
        )

    def fake_generic(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("generic")
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.zeros((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_fast)
    monkeypatch.setattr(ported, "_ps_topofit_batch_generic", fake_generic)

    cpxphase = np.ones((4, 3), dtype=np.complex128)
    invariant_bp = np.tile(np.asarray([1.0, 2.0, 3.0], dtype=np.float64), (4, 1))
    varied_bp = invariant_bp.copy()
    varied_bp[1, 0] = 9.0

    ported._ps_topofit_batch(cpxphase, invariant_bp, n_trial_wraps=1.0)
    ported._ps_topofit_batch(cpxphase, varied_bp, n_trial_wraps=1.0)

    assert calls == ["fast", "generic"]


def test_ps_topofit_batch_uses_stage2_kernel_backend(monkeypatch) -> None:
    calls: list[str] = []

    def fake_kernel(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str,
        threads: int,
        cpu_fallback: object | None,
    ):
        calls.append(backend)
        assert threads == 0
        assert cpu_fallback is ported._ps_topofit_batch_generic
        n_row, n_col = cpxphase.shape
        return (
            np.full(n_row, 1.5, dtype=np.float64),
            np.full(n_row, 2.5, dtype=np.float64),
            np.full(n_row, 0.5, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_kernel", fake_kernel)

    cpxphase = np.ones((4, 3), dtype=np.complex128)
    varied_bp = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [9.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [1.0, 2.0, 7.0],
        ],
        dtype=np.float64,
    )

    K0, C0, coh0, phase_residual = ported._ps_topofit_batch(
        cpxphase,
        varied_bp,
        n_trial_wraps=1.0,
        kernel_backend="native",
    )

    assert calls == ["native"]
    np.testing.assert_allclose(K0, np.full(4, 1.5))
    np.testing.assert_allclose(C0, np.full(4, 2.5))
    np.testing.assert_allclose(coh0, np.full(4, 0.5))
    np.testing.assert_allclose(phase_residual, np.ones((4, 3), dtype=np.complex64))


def test_stage2_ph_weight_block_uses_double_precision_phase_ramp() -> None:
    ph_nm = np.asarray(
        [
            [1.0 + 1.0j, -0.25 + 0.75j],
            [-0.5 + 0.125j, 0.875 - 0.375j],
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray(
        [
            [123.456789, -234.567891],
            [345.678912, -456.789123],
        ],
        dtype=np.float64,
    )
    k_ps = np.asarray([0.0123456789, -0.0234567891], dtype=np.float64)
    weighting = np.asarray([0.3456789012, 0.987654321], dtype=np.float64)

    observed = ported._stage2_ph_weight_block(ph_nm, bperp, k_ps, weighting)

    expected = (
        ph_nm.astype(np.complex128)
        * np.exp(-1j * (bperp * k_ps.reshape(-1, 1)))
        * weighting.reshape(-1, 1)
    ).astype(np.complex64)

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    assert observed.dtype == np.complex64


def test_stage2_ph_weight_block_preserve_precision_matches_double_phase_ramp() -> None:
    ph_nm = np.asarray(
        [
            [1.0 + 1.0j, -0.25 + 0.75j],
            [-0.5 + 0.125j, 0.875 - 0.375j],
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray(
        [
            [123.456789, -234.567891],
            [345.678912, -456.789123],
        ],
        dtype=np.float64,
    )
    k_ps = np.asarray([0.0123456789, -0.0234567891], dtype=np.float64)
    weighting = np.asarray([0.3456789012, 0.987654321], dtype=np.float64)

    observed = ported._stage2_ph_weight_block(ph_nm, bperp, k_ps, weighting, preserve_precision=True)

    expected = (
        ph_nm.astype(np.complex128)
        * np.exp(-1j * (bperp * k_ps.reshape(-1, 1)))
        * weighting.reshape(-1, 1)
    )

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0.0)
    assert observed.dtype == np.complex128

def test_stage2_grid_accumulate_matlab_keeps_single_precision_addition() -> None:
    ph_weight = np.asarray(
        [
            [1.0e-6 + 2.0e-6j, -3.0e-6 + 4.0e-6j],
            [5.0e-6 - 6.0e-6j, 7.0e-6 + 8.0e-6j],
            [9.0e-6 + 1.0e-6j, -2.0e-6 - 3.0e-6j],
        ],
        dtype=np.complex64,
    )
    grid_lin = np.asarray([0, 0, 0], dtype=np.int64)

    observed = ported._stage2_grid_accumulate_matlab(ph_weight, grid_lin, 1, 1)
    expected = np.zeros((1, 1, 2), dtype=np.complex64)
    flat = expected.reshape(-1, 2)
    for row in range(ph_weight.shape[0]):
        np.add(flat[0], ph_weight[row], out=flat[0], casting="unsafe")

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    assert observed.dtype == np.complex64


def test_clap_filt_grid_stack_prepared_matches_per_ifg_reference() -> None:
    rng = np.random.default_rng(0)
    ph = (
        rng.standard_normal((37, 53, 3)) + 1j * rng.standard_normal((37, 53, 3))
    ).astype(np.complex64)
    low_pass = np.full((32, 32), 0.01, dtype=np.float64)

    prepared = ported._prepare_clap_filt_grid_stack(ph.shape, n_win=24, n_pad=8, low_pass=low_pass)
    observed = ported._clap_filt_grid_stack_prepared(ph, alpha=1.0, beta=0.3, prepared=prepared)
    expected = np.empty_like(observed)
    for ifg_ix in range(ph.shape[2]):
        expected[:, :, ifg_ix] = ported._clap_filt_grid(
            ph[:, :, ifg_ix],
            alpha=1.0,
            beta=0.3,
            n_win=24,
            n_pad=8,
            low_pass=low_pass,
        )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-10)


def test_clap_filt_grid_matches_matlab_padded_buffer_reuse() -> None:
    rng = np.random.default_rng(1)
    ph = (
        rng.standard_normal((37, 53)) + 1j * rng.standard_normal((37, 53))
    ).astype(np.complex64)
    low_pass = np.full((32, 32), 0.01, dtype=np.float64)

    def matlab_reference(ph_grid: np.ndarray) -> np.ndarray:
        n_win = 24
        n_pad = 8
        n_i, n_j = ph_grid.shape
        ph_out = np.zeros((n_i, n_j), dtype=np.complex128)
        n_inc = n_win // 4
        n_win_i = int(np.ceil(n_i / float(n_inc)) - 3)
        n_win_j = int(np.ceil(n_j / float(n_inc)) - 3)
        x = np.arange(0, n_win / 2, dtype=np.float64)
        X, Y = np.meshgrid(x, x, indexing="xy")
        wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
        wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0) + 1.0e-6
        ph_arr = np.asarray(ph_grid, dtype=np.complex64).copy()
        ph_arr[np.isnan(ph_arr)] = 0
        kernel = ported._clap_filter_kernel()
        ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=np.complex128)

        for ix1 in range(n_win_i):
            wf = wind_func.copy()
            i1 = ix1 * n_inc
            i2 = i1 + n_win
            if i2 > n_i:
                i_shift = i2 - n_i
                i2 = n_i
                i1 = n_i - n_win
                wf = np.vstack((np.zeros((i_shift, n_win), dtype=np.float64), wf[: n_win - i_shift, :]))

            for ix2 in range(n_win_j):
                wf2 = wf.copy()
                j1 = ix2 * n_inc
                j2 = j1 + n_win
                if j2 > n_j:
                    j_shift = j2 - n_j
                    j2 = n_j
                    j1 = n_j - n_win
                    wf2 = np.hstack((np.zeros((n_win, j_shift), dtype=np.float64), wf2[:, : n_win - j_shift]))

                ph_bit[:n_win, :n_win] = ph_arr[i1:i2, j1:j2]
                ph_fft = np.fft.fft2(ph_bit)
                H = np.abs(ph_fft)
                H = np.fft.ifftshift(
                    ported.signal.convolve2d(np.fft.fftshift(H), kernel, mode="same", boundary="fill", fillvalue=0.0)
                )
                mean_h = float(np.median(H))
                if mean_h != 0.0:
                    H = H / mean_h
                H = np.power(H, 1.0)
                H = H - 1.0
                H[H < 0.0] = 0.0
                G = H * 0.3 + low_pass
                ph_filt = np.fft.ifft2(ph_fft * G)
                ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + (ph_filt[:n_win, :n_win] * wf2)

        return ph_out.astype(np.complex64)

    observed = ported._clap_filt_grid(ph, alpha=1.0, beta=0.3, n_win=24, n_pad=8, low_pass=low_pass)
    expected = matlab_reference(ph)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-10)


def test_stage2_ph_weight_block_differs_from_legacy_single_precision_ramp() -> None:
    ph_nm = np.asarray(
        [
            [0.7 + 0.2j, -0.3 + 0.9j, 0.1 - 0.8j],
            [0.6 - 0.4j, -0.2 + 0.5j, -0.9 - 0.1j],
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray(
        [
            [12345.678, -9876.543, 5432.1],
            [-22222.25, 11111.75, 3333.333],
        ],
        dtype=np.float64,
    )
    k_ps = np.asarray([0.000123456789, -0.000987654321], dtype=np.float64)
    weighting = np.asarray([0.25, 0.75], dtype=np.float64)

    observed = ported._stage2_ph_weight_block(ph_nm, bperp, k_ps, weighting)
    legacy_expected = (
        ph_nm
        * np.exp(
            np.asarray(-1j, dtype=np.complex64)
            * (bperp.astype(np.float32) * k_ps.astype(np.float32).reshape(-1, 1))
        ).astype(np.complex64)
        * weighting.astype(np.float32).reshape(-1, 1)
    ).astype(np.complex64)
    higher_precision = (
        ph_nm.astype(np.complex128)
        * np.exp(-1j * (bperp * k_ps.reshape(-1, 1)))
        * weighting.reshape(-1, 1)
    ).astype(np.complex64)

    np.testing.assert_allclose(observed, higher_precision, rtol=0.0, atol=0.0)
    assert np.max(np.abs(higher_precision - legacy_expected)) > 0.0


def test_normalize_complex_unit_magnitude_inplace_matches_legacy_single_precision_divide() -> None:
    values = np.asarray(
        [
            [0.9709364 + 0.06327464j, -0.82275736 - 0.08478022j, 0.0 + 0.0j],
            [0.9851158 + 0.17893803j, 0.4308287 + 0.01955994j, -0.5815857 + 0.10361151j],
        ],
        dtype=np.complex64,
    )

    observed = values.copy()
    ported._normalize_complex_unit_magnitude_inplace(observed)

    higher_precision = values.astype(np.complex128)
    higher_precision_abs = np.abs(values).astype(np.float64, copy=False)
    np.divide(higher_precision, higher_precision_abs, out=higher_precision, where=higher_precision_abs != 0)
    higher_precision = higher_precision.astype(np.complex64)

    legacy = values.copy()
    legacy_abs = np.abs(legacy).astype(np.float32, copy=False)
    np.divide(legacy, legacy_abs, out=legacy, where=legacy_abs != 0)

    np.testing.assert_allclose(observed, legacy, rtol=0.0, atol=0.0)
    assert np.max(np.abs(higher_precision - legacy)) > 0.0


def test_stage2_checkpoint_modes(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    cache_root = tmp_path / "stage2_random_hist_cache"

    ps_payload = {
        "n_ps": np.asarray(3.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 100.0, 100.0],
                [3.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
                [1.0 + 0.0j, 0.6 + 0.4j, 0.4 + 0.6j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([15.0, 30.0], dtype=np.float64), (3, 1))}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )

    def fake_clap(
        ph_stack: np.ndarray,
        alpha: float,
        beta: float,
        prepared: object,
        out: np.ndarray | None = None,
        workers: int = 1,
        preserve_precision: bool = False,
    ):
        if out is None:
            return np.asarray(ph_stack, dtype=np.complex64).copy()
        out[...] = np.asarray(ph_stack, dtype=np.complex64)
        return out

    monkeypatch.setattr(ported, "_clap_filt_grid_stack_prepared", fake_clap)
    monkeypatch.setattr(
        ported._MatlabV5UniformRNG,
        "uniform",
        lambda self, size: np.zeros(size, dtype=np.float64),
    )
    monkeypatch.setattr(ported, "_stage2_random_hist_cache_root", lambda: cache_root)

    def run_for_mode(mode: str, interval: int) -> list[int]:
        writes: list[int] = []
        topofit_calls = {"count": 0}
        random_hist_calls = {"count": 0}
        if cache_root.exists():
            for cache_file in cache_root.glob("*.npz"):
                cache_file.unlink()

        def fake_write_mat(path: Path, payload: dict[str, object]) -> None:
            if Path(path).name != "pm1.mat":
                return
            loop_value = int(float(np.asarray(payload["i_loop"]).reshape(-1)[0]))
            writes.append(loop_value)

        def fake_row_invariant_coh(
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            *,
            backend: str = "python",
            threads: int = 0,
            cpu_fallback: object | None = None,
        ) -> np.ndarray:
            random_hist_calls["count"] += 1
            assert np.asarray(bperp).ndim == 1
            return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

        def fake_topofit(
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            *,
            kernel_backend: str = "python",
            native_threads: int = 0,
        ):
            topofit_calls["count"] += 1
            n_row, n_col = cpxphase.shape
            return (
                np.zeros(n_row, dtype=np.float64),
                np.zeros(n_row, dtype=np.float64),
                np.full(n_row, 0.6, dtype=np.float64),
                np.ones((n_row, n_col), dtype=np.complex64),
            )

        def fake_row_invariant_topofit(
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            *,
            backend: str = "python",
            threads: int = 0,
            cpu_fallback: object | None = None,
        ):
            bperp_mat = np.broadcast_to(np.asarray(bperp, dtype=np.float64), cpxphase.shape)
            return fake_topofit(cpxphase, bperp_mat, n_trial_wraps)

        monkeypatch.setattr(ported, "write_mat", fake_write_mat)
        monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
        monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
        monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_row_invariant_topofit)

        result = ported.stage2_estimate_gamma(
            patch_dir,
            checkpoint_mode=mode,
            checkpoint_interval=interval,
            debug=False,
        )

        assert result == "Stage 2 computed coherence for 3 candidates in 3 iterations"
        assert topofit_calls["count"] == 3
        assert random_hist_calls["count"] == _STAGE2_RANDOM_HIST_CALLS
        return writes

    assert run_for_mode("final", 2) == [3]
    assert run_for_mode("periodic", 2) == [2, 3]
    assert run_for_mode("always", 1) == [2, 3, 3]


def test_stage2_saved_ph_weight_matches_saved_ph_grid(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0], [25.0, 40.0]], dtype=np.float64)}
    parms_payload = {"gamma_max_iterations": np.asarray(1.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    saved: dict[str, object] = {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: (np.ones(100), 43.0))
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.asarray([0.15, -0.2], dtype=np.float64)[:n_row],
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
    monkeypatch.setattr(
        ported,
        "write_mat",
        lambda path, payload: saved.update({Path(path).name: payload}) if Path(path).name == "pm1.mat" else None,
    )

    ported.stage2_estimate_gamma(patch_dir, debug=False)

    payload = saved["pm1.mat"]
    ph_weight = np.asarray(payload["ph_weight"], dtype=np.complex64)
    ph_grid = np.asarray(payload["ph_grid"], dtype=np.complex64)
    grid_ij = np.asarray(payload["grid_ij"], dtype=np.int64)
    replay = ported._stage2_grid_accumulate_matlab(
        ph_weight,
        np.ravel_multi_index((grid_ij[:, 0] - 1, grid_ij[:, 1] - 1), ph_grid.shape[:2]),
        ph_grid.shape[0],
        ph_grid.shape[1],
    )

    np.testing.assert_allclose(ph_grid, replay, atol=2e-7, rtol=0.0)


def test_stage2_estimate_gamma_uses_legacy_precision_path(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    (patch_dir / "parms.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0], [25.0, 40.0]], dtype=np.float64)}
    parms_payload = {"gamma_max_iterations": np.asarray(1.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    flags: dict[str, list[bool]] = {
        "ph_weight": [],
        "grid": [],
        "clap": [],
        "normalize": [],
    }

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(ported, "_build_stage_options", lambda patch: ported.StageOptions())
    monkeypatch.setattr(ported, "_load_parms", lambda patch: ported.Parms())
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: (np.ones(100), 43.0))
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)

    def fake_ph_weight(
        ph_nm: np.ndarray,
        bperp: np.ndarray,
        k_ps: np.ndarray,
        weighting: np.ndarray,
        *,
        preserve_precision: bool = False,
    ) -> np.ndarray:
        flags["ph_weight"].append(bool(preserve_precision))
        return np.asarray(ph_nm, dtype=np.complex64).copy()

    def fake_grid_accumulate(
        ph_weight: np.ndarray,
        grid_lin: np.ndarray,
        n_i: int,
        n_j: int,
        *,
        out: np.ndarray | None = None,
        preserve_precision: bool = False,
    ) -> np.ndarray:
        flags["grid"].append(bool(preserve_precision))
        result = np.zeros((n_i, n_j, ph_weight.shape[1]), dtype=np.complex64)
        flat = result.reshape(-1, ph_weight.shape[1])
        for row, idx in enumerate(np.asarray(grid_lin, dtype=np.int64).reshape(-1)):
            flat[idx, :] += np.asarray(ph_weight[row], dtype=np.complex64)
        if out is not None:
            out[...] = result
            return out
        return result

    def fake_clap(
        ph_stack: np.ndarray,
        alpha: float,
        beta: float,
        prepared: object,
        out: np.ndarray | None = None,
        workers: int = 1,
        preserve_precision: bool = False,
    ):
        flags["clap"].append(bool(preserve_precision))
        filt = np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None:
            return filt
        out[...] = filt
        return out

    def fake_normalize(values: np.ndarray, *, preserve_precision: bool = False) -> np.ndarray:
        flags["normalize"].append(bool(preserve_precision))
        return np.asarray(values, dtype=np.complex64)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "_stage2_ph_weight_block", fake_ph_weight)
    monkeypatch.setattr(ported, "_stage2_grid_accumulate_matlab", fake_grid_accumulate)
    monkeypatch.setattr(ported, "_clap_filt_grid_stack_prepared", fake_clap)
    monkeypatch.setattr(ported, "_normalize_complex_unit_magnitude_inplace", fake_normalize)
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
    monkeypatch.setattr(
        ported,
        "run_stage2_topofit_coh_row_invariant_kernel",
        lambda cpxphase, bperp, n_trial_wraps, **kwargs: np.full(cpxphase.shape[0], 0.25, dtype=np.float64),
    )

    result = ported.stage2_estimate_gamma(patch_dir, debug=False)

    assert result == "Stage 2 computed coherence for 2 candidates in 1 iterations"
    assert flags["ph_weight"] == [False]
    assert flags["grid"] == [False]
    assert flags["clap"] == [False]
    assert flags["normalize"] == [False]


def test_stage2_saved_nr_matches_scaled_histogram(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0], [25.0, 40.0]], dtype=np.float64)}
    parms_payload = {"gamma_max_iterations": np.asarray(2.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    saved: dict[str, object] = {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: (np.ones(100), 43.0))
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.1, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
    monkeypatch.setattr(
        ported,
        "write_mat",
        lambda path, payload: saved.update({Path(path).name: payload}) if Path(path).name == "pm1.mat" else None,
    )

    ported.stage2_estimate_gamma(patch_dir, debug=False)

    payload = saved["pm1.mat"]
    coh_bins = np.asarray(payload["coh_bins"], dtype=np.float64).reshape(-1)
    na = ported.run_stage2_histogram_kernel(np.full(2, 0.1, dtype=np.float64), coh_bins, backend="python").astype(np.float64)
    scale = float(np.sum(na[:31]) / 31.0)
    expected_nr = np.ones(100, dtype=np.float64) * scale

    np.testing.assert_allclose(np.asarray(payload["Nr"], dtype=np.float64).reshape(-1), expected_nr, atol=1e-15, rtol=0.0)


def test_stage2_saved_nr_preserves_previous_histogram_when_last_iteration_changes(
    monkeypatch, tmp_path: Path
) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0], [25.0, 40.0]], dtype=np.float64)}
    parms_payload = {"gamma_max_iterations": np.asarray(2.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    saved: dict[str, object] = {}
    topofit_calls = {"count": 0}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: (np.ones(100), 43.0))
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        topofit_calls["count"] += 1
        coh = 0.1 if topofit_calls["count"] == 1 else 0.2
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, coh, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)
    monkeypatch.setattr(
        ported,
        "write_mat",
        lambda path, payload: saved.update({Path(path).name: payload}) if Path(path).name == "pm1.mat" else None,
    )

    ported.stage2_estimate_gamma(patch_dir, debug=False)

    payload = saved["pm1.mat"]
    coh_bins = np.asarray(payload["coh_bins"], dtype=np.float64).reshape(-1)
    coh_ps = np.asarray(payload["coh_ps"], dtype=np.float64).reshape(-1)
    na = ported.run_stage2_histogram_kernel(np.full(2, 0.1, dtype=np.float64), coh_bins, backend="python").astype(
        np.float64
    )
    scale = float(np.sum(na[:31]) / 31.0)
    expected_nr = np.ones(100, dtype=np.float64) * scale

    np.testing.assert_allclose(coh_ps, np.full(2, 0.2, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(payload["Nr"], dtype=np.float64).reshape(-1), expected_nr, atol=1e-15, rtol=0.0)


def test_stage2_replay_iteration_can_target_specific_rows(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 150.0, 150.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([15.0, 30.0], dtype=np.float64), (2, 1))}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )

    def fake_row_invariant_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.full(n_row, 0.25, dtype=np.float64),
            np.full(n_row, 0.75, dtype=np.float64),
            np.full(n_row, 0.5, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_row_invariant_topofit)

    context = ported._stage2_prepare_replay_context(patch_dir, kernel_backend="python", native_threads=0)
    pm_payload = {
        "ph_weight": np.asarray([[1.0 + 0.0j, 0.4 + 0.1j], [0.5 + 0.5j, 0.2 + 0.3j]], dtype=np.complex64),
        "coh_bins": np.arange(0.005, 1.0, 0.01, dtype=np.float64),
        "Nr": np.ones(100, dtype=np.float64),
        "Nr_max_nz_ix": np.asarray(43.0, dtype=np.float64),
        "n_trial_wraps": np.asarray(1.0, dtype=np.float32),
    }

    replay = ported._stage2_replay_iteration_from_payload(
        context,
        pm_payload,
        row_ix=np.asarray([1], dtype=np.int64),
        compute_weighting=False,
    )

    np.testing.assert_array_equal(replay["row_ix"], np.asarray([1], dtype=np.int64))
    np.testing.assert_allclose(replay["grid_ij"], np.asarray([[2, 2]], dtype=np.int64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["ph_grid_samples"], pm_payload["ph_weight"][1:2, :], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["K_ps"], np.asarray([0.25], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["C_ps"], np.asarray([0.75], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["coh_ps"], np.asarray([0.5], dtype=np.float64), rtol=0.0, atol=0.0)


def test_stage2_replay_iteration_keeps_partially_zero_rows(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(1.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {"ph": np.asarray([[1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex64)}
    bp_payload = {"bperp_mat": np.asarray([[15.0, 30.0]], dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        return {}

    calls: list[np.ndarray] = []

    def fake_batch(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        _tie_refine: bool = True,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        calls.append(np.asarray(cpxphase).copy())
        n_row, n_col = cpxphase.shape
        return (
            np.full(n_row, 0.25, dtype=np.float64),
            np.full(n_row, 0.75, dtype=np.float64),
            np.full(n_row, 0.5, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_batch)

    context = ported._stage2_prepare_replay_context(patch_dir, kernel_backend="python", native_threads=0)
    pm_payload = {
        "ph_weight": np.asarray([[1.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex64),
        "coh_bins": np.arange(0.005, 1.0, 0.01, dtype=np.float64),
        "Nr": np.ones(100, dtype=np.float64),
        "Nr_max_nz_ix": np.asarray(43.0, dtype=np.float64),
        "n_trial_wraps": np.asarray(1.0, dtype=np.float32),
    }

    replay = ported._stage2_replay_iteration_from_payload(context, pm_payload, compute_weighting=False)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.asarray([[0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128))
    np.testing.assert_allclose(replay["K_ps"], np.asarray([0.25], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["C_ps"], np.asarray([0.75], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(replay["coh_ps"], np.asarray([0.5], dtype=np.float64), rtol=0.0, atol=0.0)


def test_stage2_random_hist_cache_reuses_deterministic_histogram(
    monkeypatch,
    tmp_path: Path,
) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(3.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 50.0, 50.0],
                [3.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
                [1.0 + 0.0j, 0.6 + 0.4j, 0.4 + 0.6j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([15.0, 30.0], dtype=np.float64), (3, 1))}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(
        ported._MatlabV5UniformRNG,
        "uniform",
        lambda self, size: np.zeros(size, dtype=np.float64),
    )
    monkeypatch.setattr(ported, "_stage2_random_hist_cache_root", lambda: tmp_path / "cache")

    random_hist_calls = {"count": 0}

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        random_hist_calls["count"] += 1
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)

    result_1 = ported.stage2_estimate_gamma(patch_dir, debug=False)
    assert result_1 == "Stage 2 computed coherence for 3 candidates in 3 iterations"
    assert random_hist_calls["count"] == _STAGE2_RANDOM_HIST_CALLS
    assert list((tmp_path / "cache").glob("*.npz"))

    result_2 = ported.stage2_estimate_gamma(patch_dir, debug=False)
    assert result_2 == result_1
    assert random_hist_calls["count"] == _STAGE2_RANDOM_HIST_CALLS


def test_stage2_does_not_reuse_scaled_pm1_histogram_as_random_baseline(
    monkeypatch, tmp_path: Path
) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    (patch_dir / "pm1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray([[1.0, 0.0, 0.0], [2.0, 50.0, 50.0]], dtype=np.float64),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {"bperp_mat": np.tile(np.asarray([15.0, 30.0], dtype=np.float64), (2, 1))}
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    master_ix = int(round(ported._mat_scalar(ps_payload["master_ix"], 1)))
    bperp_nm = np.asarray(ps_payload["bperp"], dtype=np.float64).reshape(-1)
    bperp_nm = bperp_nm[np.arange(bperp_nm.size) != (master_ix - 1)]
    mean_inc = float(ported._mat_scalar(ps_payload["mean_incidence"], np.deg2rad(23.0)))
    rho = 830000.0
    options = ported.StageOptions()
    max_k = options.max_topo_err / (options.lambda_m * rho * np.sin(mean_inc) / (4 * np.pi))
    expected_wraps = float((np.max(bperp_nm) - np.min(bperp_nm)) * max_k / (2 * np.pi))
    pm_payload = {
        "Nr": np.full(coh_bins.size, 7.0, dtype=np.float64),
        "Nr_max_nz_ix": np.asarray(43.0, dtype=np.float64),
        "coh_bins": coh_bins + np.finfo(np.float64).eps,
        "n_trial_wraps": np.asarray(np.float32(expected_wraps), dtype=np.float32),
    }
    parms_payload = {"gamma_max_iterations": np.asarray(1.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "pm1.mat":
            return pm_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    saved: dict[str, object] = {}
    random_hist_calls = {"count": 0}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "_load_stage2_random_hist_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(ported, "_write_stage2_random_hist_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        random_hist_calls["count"] += 1
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_row_invariant_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ):
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "run_stage2_topofit_row_invariant_kernel", fake_row_invariant_topofit)
    monkeypatch.setattr(
        ported,
        "write_mat",
        lambda path, payload: saved.update({Path(path).name: payload}) if Path(path).name == "pm1.mat" else None,
    )

    result = ported.stage2_estimate_gamma(patch_dir, debug=False)

    assert result.startswith("Stage 2 computed coherence for 2 candidates in ")
    assert random_hist_calls["count"] == _STAGE2_RANDOM_HIST_CALLS
    assert float(np.asarray(saved["pm1.mat"]["Nr_max_nz_ix"], dtype=np.float64).reshape(-1)[0]) > 0.0


def test_stage2_uses_bp1_matrix_for_non_small_baseline(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 200.0, 200.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {
        "bperp_mat": np.asarray(
            [
                [15.0, 30.0],
                [16.0, 31.0],
            ],
            dtype=np.float64,
        )
    }

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_grid_stack_prepared",
        lambda ph_stack, alpha, beta, prepared, out=None, workers=1, preserve_precision=False: np.asarray(ph_stack, dtype=np.complex64).copy()
        if out is None
        else np.copyto(out, np.asarray(ph_stack, dtype=np.complex64)) or out,
    )
    monkeypatch.setattr(ported, "_stage2_random_hist_cache_root", lambda: tmp_path / "cache")

    seen_bperp: list[np.ndarray] = []
    seen_random_bperp: list[np.ndarray] = []

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        seen_random_bperp.append(np.asarray(bperp, dtype=np.float64).copy())
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        seen_bperp.append(np.asarray(bperp, dtype=np.float64).copy())
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)

    result = ported.stage2_estimate_gamma(patch_dir, debug=False)

    assert result == "Stage 2 computed coherence for 2 candidates in 3 iterations"
    assert seen_bperp
    assert seen_random_bperp
    np.testing.assert_allclose(seen_bperp[0], bp_payload["bperp_mat"])
    np.testing.assert_allclose(seen_random_bperp[0], np.asarray([15.0, 30.0], dtype=np.float64))


def test_stage2_reprocesses_partial_zero_rows_for_generic_topofit(monkeypatch, tmp_path: Path) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    (patch_dir / "bp1.mat").touch()
    (patch_dir / "parms.mat").touch()

    ps_payload = {
        "n_ps": np.asarray(2.0, dtype=np.float64),
        "master_ix": np.asarray(1.0, dtype=np.float64),
        "bperp": np.asarray([0.0, 15.0, 30.0], dtype=np.float64),
        "xy": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        ),
        "mean_range": np.asarray(830000.0, dtype=np.float64),
        "mean_incidence": np.asarray(np.deg2rad(23.0), dtype=np.float64),
    }
    ph_payload = {
        "ph": np.asarray(
            [
                [1.0 + 0.0j, 0.8 + 0.2j, 0.6 + 0.4j],
                [1.0 + 0.0j, 0.7 + 0.3j, 0.5 + 0.5j],
            ],
            dtype=np.complex64,
        )
    }
    bp_payload = {
        "bperp_mat": np.asarray(
            [
                [15.0, 30.0],
                [16.0, 31.0],
            ],
            dtype=np.float64,
        )
    }
    parms_payload = {"gamma_max_iterations": np.asarray(1.0, dtype=np.float64)}

    def fake_read_mat(path: Path):
        name = Path(path).name
        if name == "ps1.mat":
            return ps_payload
        if name == "ph1.mat":
            return ph_payload
        if name == "bp1.mat":
            return bp_payload
        if name == "parms.mat":
            return parms_payload
        return {}

    monkeypatch.setattr(ported, "read_mat", fake_read_mat)
    monkeypatch.setattr(ported, "write_mat", lambda path, payload: None)
    monkeypatch.setattr(ported, "_build_stage_options", lambda patch: ported.StageOptions())
    monkeypatch.setattr(ported, "_load_parms", lambda patch: ported.Parms())
    monkeypatch.setattr(
        ported,
        "_prepare_clap_filt_grid_stack",
        lambda shape, n_win, n_pad, low_pass: SimpleNamespace(n_i=shape[0], n_j=shape[1], n_ifg=shape[2]),
    )

    def fake_clap(
        ph_stack: np.ndarray,
        alpha: float,
        beta: float,
        prepared: object,
        out: np.ndarray | None = None,
        workers: int = 1,
        preserve_precision: bool = False,
    ):
        filt = np.asarray(ph_stack, dtype=np.complex64).copy()
        filt[-1, -1, 0] = 0.0
        if out is None:
            return filt
        out[...] = filt
        return out

    monkeypatch.setattr(ported, "_clap_filt_grid_stack_prepared", fake_clap)
    monkeypatch.setattr(ported, "_stage2_random_hist_cache_root", lambda: tmp_path / "cache")

    def fake_row_invariant_coh(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "python",
        threads: int = 0,
        cpu_fallback: object | None = None,
    ) -> np.ndarray:
        return np.full(cpxphase.shape[0], 0.25, dtype=np.float64)

    seen_cpxphase: list[np.ndarray] = []

    def fake_topofit(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        kernel_backend: str = "python",
        native_threads: int = 0,
    ):
        seen_cpxphase.append(np.asarray(cpxphase, dtype=np.complex128).copy())
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.full(n_row, 0.6, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_coh_row_invariant_kernel", fake_row_invariant_coh)
    monkeypatch.setattr(ported, "_ps_topofit_batch", fake_topofit)

    result = ported.stage2_estimate_gamma(patch_dir, debug=False)

    assert result == "Stage 2 computed coherence for 2 candidates in 1 iterations"
    assert len(seen_cpxphase) == 1
    np.testing.assert_allclose(
        seen_cpxphase[0],
        np.asarray(
            [
                [0.0 + 0.0j, 0.99513334 - 0.09853761j],
                [0.0 + 0.0j, 0.99513328 + 0.09853766j],
            ],
            dtype=np.complex128,
        ),
        atol=1e-7,
        rtol=0.0,
    )
