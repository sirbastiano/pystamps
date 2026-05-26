from types import SimpleNamespace

from pystamps.verify import (
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
