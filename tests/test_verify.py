from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pystamps.io.mat import write_mat
from pystamps.tolerance_manifest import load_artifact_tolerance_manifest
from pystamps.verify import (
    DEFAULT_GLOBS,
    FileComparison,
    VerificationReport,
    classify_failures,
    summarize_failures,
    verify_run_against_golden,
)


def test_classify_failures_groups_downstream_residuals() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison("PATCH_1/select1.mat", False, "Value mismatch for key 'C_ps2', max_abs=2.79e-05"),
            FileComparison("phuw2.mat", False, "Value mismatch for key 'msd', max_abs=14.9361"),
            FileComparison("uw_space_time.mat", False, "Wrap mismatch for key 'dph_noise', wrapped_max_abs=6.26338"),
            FileComparison("uw_interp.mat", True, "Matched 1 numeric keys"),
        ]
    )

    failures = classify_failures(report)

    assert [failure.failure_class for failure in failures] == [
        "stage3_patch_boundary",
        "unwrap_smoothing",
        "unwrapped_noise_statistics",
    ]
    assert [failure.failing_key for failure in failures] == ["C_ps2", "msd", "dph_noise"]


def test_summarize_failures_includes_trace_guidance() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison("ifgstd2.mat", False, "Value mismatch for key 'ifg_std', max_abs=0.125"),
            FileComparison("mean_v.mat", False, "Value mismatch for key 'm', max_abs=8.3154"),
        ]
    )

    summary = summarize_failures(report)

    assert summary["failed"] == 2
    assert [group["failure_class"] for group in summary["groups"]] == [
        "unwrap_smoothing",
        "unwrapped_noise_statistics",
    ]
    assert summary["first_boundary_failure"]["path"] == "ifgstd2.mat"
    assert summary["trace"]["stage3_4_residual_present"] is False
    assert summary["trace"]["stage3_4_coupling_evidence_present"] is False


def test_summarize_failures_prioritizes_earliest_stage_boundary() -> None:
    report = VerificationReport(
        comparisons=[
            FileComparison(
                "uw_space_time.mat",
                False,
                "Shape mismatch for key 'dph_noise': (3, 4) != (5, 4)",
                failure_kind="shape_mismatch",
                failing_key="dph_noise",
                shape_run=(3, 4),
                shape_oracle=(5, 4),
            ),
            FileComparison(
                "PATCH_1/pm1.mat",
                False,
                "Value mismatch for key 'C_ps', max_abs=1.25",
                failure_kind="value_mismatch",
                failing_key="C_ps",
                shape_run=(2,),
                shape_oracle=(2,),
                max_abs=1.25,
            ),
        ]
    )

    summary = summarize_failures(report)

    assert summary["first_boundary_failure"] == {
        "path": "PATCH_1/pm1.mat",
        "message": "Value mismatch for key 'C_ps', max_abs=1.25",
        "stage_scope": "stage2",
        "failure_class": "stage2_patch_boundary",
        "label": "Stage 2 patch boundary",
        "failing_key": "C_ps",
        "failure_kind": "value_mismatch",
        "shape_run": [2],
        "shape_oracle": [2],
        "max_abs": 1.25,
        "guidance": (
            "pm1.mat diverges before later patch stages; fix stage-2 parity before changing stage-3/4 or "
            "downstream code."
        ),
    }
    assert summary["trace"]["stage2_residual_present"] is True


def test_verify_uses_patch_list_old_when_patch_list_is_subset(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    (golden / "PATCH_1").mkdir(parents=True)
    (golden / "PATCH_2").mkdir(parents=True)
    (run / "PATCH_1").mkdir(parents=True)

    (golden / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (golden / "patch.list_old").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")
    (run / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (golden / "PATCH_1" / "artifact.txt").write_text("same", encoding="utf-8")
    (run / "PATCH_1" / "artifact.txt").write_text("same", encoding="utf-8")
    (golden / "PATCH_2" / "artifact.txt").write_text("missing-from-run", encoding="utf-8")

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-6, atol=1e-8, wrap_equivalence=False),
        patterns=("PATCH_*/artifact.txt",),
    )

    assert not report.ok
    assert any(comparison.failure_kind == "patch_manifest_mismatch" for comparison in report.comparisons)
    assert any(comparison.relative_path == "PATCH_2/artifact.txt" for comparison in report.comparisons)


def test_tolerance_manifest_covers_core_artifacts_and_modes() -> None:
    manifest = load_artifact_tolerance_manifest()
    paths = {spec.path for spec in manifest.artifacts}

    assert {
        "PATCH_*/ps1.mat",
        "PATCH_*/pm1.mat",
        "PATCH_*/select1.mat",
        "PATCH_*/weed1.mat",
        "PATCH_*/ps2.mat",
        "PATCH_*/ph2.mat",
        "PATCH_*/pm2.mat",
        "PATCH_*/bp2.mat",
        "ps2.mat",
        "ph2.mat",
        "pm2.mat",
        "bp2.mat",
        "ifgstd2.mat",
        "phuw2.mat",
        "uw_grid.mat",
        "uw_interp.mat",
        "scla2.mat",
        "mean_v.mat",
        "uw_space_time.mat",
    }.issubset(paths)
    assert set(DEFAULT_GLOBS) == paths

    modes = {rule.comparison_mode for spec in manifest.artifacts for rule in spec.rules}
    assert {"exact_structural", "numeric_f32", "numeric_f64", "phase_modulo_f32", "sparse_exact"}.issubset(modes)

    ph2_rule = manifest.spec_for_path("ph2.mat").rule_for_key("ph")
    assert ph2_rule is not None
    assert ph2_rule.dtype == "complex64"
    assert ph2_rule.comparison_mode == "phase_modulo_f32"
    assert ph2_rule.atol == 0.0001
    assert manifest.spec_for_path("ph2.mat").shape_policy == "exact"


def test_verify_reports_tolerance_rule_id_for_manifest_numeric_failure(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    golden.mkdir()
    run.mkdir()
    write_mat(golden / "ph2.mat", {"ph": np.asarray([[1.0 + 0.0j]], dtype=np.complex64)})
    write_mat(run / "ph2.mat", {"ph": np.asarray([[np.exp(1j * 0.01)]], dtype=np.complex64)})

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-12, atol=1e-12, wrap_equivalence=False),
        patterns=("ph2.mat",),
    )

    assert not report.ok
    failure = report.failures[0]
    assert failure.failing_key == "ph"
    assert failure.failure_kind == "wrap_mismatch"
    assert failure.tolerance_rule_id == "merged_ph2.ph.phase_modulo_f32"
    assert failure.comparison_mode == "phase_modulo_f32"
    assert "merged_ph2.ph.phase_modulo_f32" in failure.message


def _write_uw_space_time(
    path,
    *,
    include_ifreq: bool = True,
    matrix_shape: tuple[int, int] = (1, 1),
    spread_shape: tuple[int, int] = (1, 1),
) -> None:
    payload = {
        "G": np.zeros(matrix_shape, dtype=np.float64),
        "dph_noise": np.zeros(matrix_shape, dtype=np.float32),
        "dph_space_uw": np.zeros(matrix_shape, dtype=np.float32),
        "jfreq_ij": np.empty((0, 0), dtype=np.float64),
        "predef_ix": np.empty((0, 0), dtype=np.float64),
        "shaky_ix": np.empty((0, 0), dtype=np.float64),
        "spread": sparse.csc_matrix(spread_shape, dtype=np.float64),
    }
    if include_ifreq:
        payload["ifreq_ij"] = np.empty((0, 0), dtype=np.float64)
    write_mat(path, payload)


def _write_hdf5_uw_space_time(
    path,
    *,
    matrix_shape: tuple[int, int] = (1, 1),
    spread_shape: tuple[int, int] = (1, 1),
) -> None:
    h5py = pytest.importorskip("h5py")

    with h5py.File(path, "w") as file:
        for key, values in {
            "G": np.zeros(matrix_shape, dtype=np.float64),
            "dph_noise": np.zeros(matrix_shape, dtype=np.float32),
            "dph_space_uw": np.zeros(matrix_shape, dtype=np.float32),
            "ifreq_ij": np.empty((0, 0), dtype=np.float64),
            "jfreq_ij": np.empty((0, 0), dtype=np.float64),
            "predef_ix": np.empty((0, 0), dtype=np.float64),
            "shaky_ix": np.empty((0, 0), dtype=np.float64),
        }.items():
            dataset = file.create_dataset(key, data=values)
            dataset.attrs["PY_STAMPS_row_major"] = np.uint8(1)

        spread = file.create_group("spread")
        spread.create_dataset("data", data=np.array([], dtype=np.float64))
        spread.create_dataset("ir", data=np.array([], dtype=np.int32))
        spread.create_dataset("jc", data=np.zeros(spread_shape[1] + 1, dtype=np.int32))
        spread.create_dataset("shape", data=np.array(spread_shape, dtype=np.uint64))


def test_verify_manifest_missing_uw_space_time_key_fails_even_when_numeric_values_match(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    golden.mkdir()
    run.mkdir()
    _write_uw_space_time(golden / "uw_space_time.mat")
    _write_uw_space_time(run / "uw_space_time.mat", include_ifreq=False)

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-12, atol=1e-12, wrap_equivalence=False),
        patterns=("uw_space_time.mat",),
    )

    assert not report.ok
    failure = report.failures[0]
    assert failure.failure_kind == "missing_required_keys"
    assert failure.failing_key == "ifreq_ij"
    assert failure.tolerance_rule_id == "merged_uw_space_time.ifreq_ij.exact_structural"


def test_verify_manifest_enforces_sparse_structural_parity(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    golden.mkdir()
    run.mkdir()
    _write_uw_space_time(golden / "uw_space_time.mat", spread_shape=(1, 1))
    _write_uw_space_time(run / "uw_space_time.mat", spread_shape=(2, 1))

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-12, atol=1e-12, wrap_equivalence=False),
        patterns=("uw_space_time.mat",),
    )

    assert not report.ok
    failure = report.failures[0]
    assert failure.failure_kind == "shape_mismatch"
    assert failure.failing_key == "spread"
    assert failure.tolerance_rule_id == "merged_uw_space_time.spread.sparse_exact"


def test_verify_manifest_rejects_dense_placeholder_for_sparse_spread(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    golden.mkdir()
    run.mkdir()
    _write_uw_space_time(golden / "uw_space_time.mat", spread_shape=(2, 3))
    payload = {
        "G": np.zeros((1, 1), dtype=np.float64),
        "dph_noise": np.zeros((1, 1), dtype=np.float32),
        "dph_space_uw": np.zeros((1, 1), dtype=np.float32),
        "ifreq_ij": np.empty((0, 0), dtype=np.float64),
        "jfreq_ij": np.empty((0, 0), dtype=np.float64),
        "predef_ix": np.empty((0, 0), dtype=np.float64),
        "shaky_ix": np.empty((0, 0), dtype=np.float64),
        "spread": np.zeros((2, 3), dtype=np.float64),
    }
    write_mat(run / "uw_space_time.mat", payload)

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-12, atol=1e-12, wrap_equivalence=False),
        patterns=("uw_space_time.mat",),
    )

    assert not report.ok
    failure = report.failures[0]
    assert failure.failure_kind == "sparse_structure_mismatch"
    assert failure.failing_key == "spread"
    assert failure.tolerance_rule_id == "merged_uw_space_time.spread.sparse_exact"


def test_verify_manifest_accepts_hdf5_sparse_spread_and_empty_keys(tmp_path) -> None:
    golden = tmp_path / "golden"
    run = tmp_path / "run"
    golden.mkdir()
    run.mkdir()
    _write_uw_space_time(golden / "uw_space_time.mat", matrix_shape=(2, 3), spread_shape=(2, 3))
    _write_hdf5_uw_space_time(run / "uw_space_time.mat", matrix_shape=(2, 3), spread_shape=(2, 3))

    report = verify_run_against_golden(
        run,
        golden,
        SimpleNamespace(rtol=1e-12, atol=1e-12, wrap_equivalence=False),
        patterns=("uw_space_time.mat",),
    )

    assert report.ok
