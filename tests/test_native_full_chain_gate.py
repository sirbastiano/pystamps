from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_gate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "native_full_chain_gate.py"
    spec = importlib.util.spec_from_file_location("native_full_chain_gate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_run_copy_restores_legacy_patch_list_and_cleans_outputs(tmp_path: Path) -> None:
    module = _load_gate_module()
    dataset = tmp_path / "dataset"
    run_root = tmp_path / "run"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        patch = dataset / name
        patch.mkdir(parents=True)
        (patch / "pscands.1.ij").write_text("input", encoding="utf-8")
        (patch / "ps1.mat").write_text("generated", encoding="utf-8")
        (patch / "pm1.mat").write_text("generated", encoding="utf-8")
        (patch / "select1.mat").write_text("generated", encoding="utf-8")
        (patch / "weed1.mat").write_text("generated", encoding="utf-8")
        (patch / "ps2.mat").write_text("generated", encoding="utf-8")
    (dataset / "phuw2.mat").write_text("generated", encoding="utf-8")
    (dataset / "scla2.mat").write_text("generated", encoding="utf-8")
    (dataset / "mean_v.mat").write_text("generated", encoding="utf-8")
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")
    (dataset / "patch.list_old").write_text("PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n", encoding="utf-8")

    setup = module.prepare_run_copy(dataset, run_root, 1, 8)

    assert setup["patch_manifest_source"] == "patch.list_old"
    assert (run_root / "patch.list").read_text(encoding="utf-8") == "PATCH_1\nPATCH_2\nPATCH_3\nPATCH_4\n"
    assert (dataset / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    assert (run_root / "PATCH_2").exists()
    assert (run_root / "PATCH_1" / "pscands.1.ij").exists()
    assert not (run_root / "PATCH_1" / "ps1.mat").exists()
    assert not (run_root / "PATCH_1" / "pm1.mat").exists()
    assert not (run_root / "PATCH_1" / "select1.mat").exists()
    assert not (run_root / "PATCH_1" / "weed1.mat").exists()
    assert not (run_root / "PATCH_1" / "ps2.mat").exists()
    assert not (run_root / "phuw2.mat").exists()
    assert not (run_root / "scla2.mat").exists()
    assert not (run_root / "mean_v.mat").exists()


def test_authoritative_patch_manifest_rejects_subset_without_legacy_manifest(tmp_path: Path) -> None:
    module = _load_gate_module()
    dataset = tmp_path / "dataset"
    for name in ("PATCH_1", "PATCH_2", "PATCH_3", "PATCH_4"):
        (dataset / name).mkdir(parents=True)
    (dataset / "patch.list").write_text("PATCH_1\n", encoding="utf-8")

    with pytest.raises(module.GateError, match="patch.list lists 1 patch"):
        module.authoritative_patch_manifest(dataset)


def test_performance_budget_manifest_is_packaged_and_release_capped() -> None:
    module = _load_gate_module()

    manifest = module.load_performance_budget_manifest(module.DEFAULT_BUDGET_MANIFEST)

    assert manifest["dataset"] == "inputs_and_outputs/InSAR_dataset_test"
    assert manifest["release"]["max_total_duration_sec"] == 600.0
    assert {(entry["stage"], entry["scope"]) for entry in manifest["stages"]} >= {
        (1, "patch"),
        (5, "merged"),
        (6, "merged"),
        (8, "merged"),
    }
    assert all("max_duration_sec" in entry and "max_peak_rss_bytes" in entry for entry in manifest["stages"])


def test_budget_evaluation_fails_release_runtime_without_waiver() -> None:
    module = _load_gate_module()
    manifest = {
        "release": {"max_total_duration_sec": 600.0, "temporary_waiver": None},
        "stages": [],
    }

    report = module.evaluate_performance_budgets(
        manifest,
        601.0,
        [],
        now=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert report["violations"][0]["kind"] == "release_runtime"


def test_budget_evaluation_accepts_documented_temporary_runtime_waiver() -> None:
    module = _load_gate_module()
    manifest = {
        "release": {
            "max_total_duration_sec": 600.0,
            "temporary_waiver": {
                "reason": "validation VM maintenance window",
                "owner": "native-parity",
                "expires_at_utc": "2026-05-27T00:00:00+00:00",
            },
        },
        "stages": [],
    }

    report = module.evaluate_performance_budgets(
        manifest,
        601.0,
        [],
        now=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert report["ok"] is True
    assert report["waivers"][0]["kind"] == "release_runtime"


def test_budget_evaluation_fails_slow_or_memory_heavy_stage() -> None:
    module = _load_gate_module()
    manifest = {
        "release": {"max_total_duration_sec": 600.0, "temporary_waiver": None},
        "stages": [
            {
                "stage": 6,
                "scope": "merged",
                "target": "*",
                "max_duration_sec": 10.0,
                "max_peak_rss_bytes": 100,
                "temporary_waiver": None,
            }
        ],
    }

    report = module.evaluate_performance_budgets(
        manifest,
        20.0,
        [
            {
                "stage": 6,
                "scope": "merged",
                "target": "native-full-chain",
                "status": "completed",
                "duration_sec": 11.0,
                "memory_peak_bytes": 101,
            }
        ],
        now=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert {violation["kind"] for violation in report["violations"]} == {"stage_duration", "stage_memory"}


def test_stage_duration_rows_preserve_native_telemetry_fields() -> None:
    module = _load_gate_module()

    rows = module._stage_durations(
        [
            {
                "stage": 6,
                "scope": "merged",
                "target": "run",
                "status": "completed",
                "duration_sec": 1.0,
                "input_artifact_count": 5,
                "output_artifact_count": 4,
                "rows_processed": 10,
                "memory_peak_bytes": 4096,
                "n_grid_ps": 4,
                "n_grid_rows": 2,
                "n_grid_cols": 3,
                "n_edges": 5,
            }
        ]
    )

    assert rows == [
        {
            "stage": 6,
            "scope": "merged",
            "target": "run",
            "status": "completed",
            "duration_sec": 1.0,
            "input_artifact_count": 5,
            "output_artifact_count": 4,
            "rows_processed": 10,
            "memory_peak_bytes": 4096,
            "n_grid_ps": 4,
            "n_grid_rows": 2,
            "n_grid_cols": 3,
            "n_edges": 5,
        }
    ]


def test_verifier_waiver_evaluation_blocks_unapproved_shape_mismatch(tmp_path: Path) -> None:
    module = _load_gate_module()
    manifest = tmp_path / "artifact_tolerances.json"
    manifest.write_text('{"manifest_version": 1, "artifacts": [], "waivers": []}', encoding="utf-8")

    report = module.evaluate_verifier_tolerance_waivers(
        {
            "ok": False,
            "checked": 1,
            "failed": [
                {
                    "path": "ps2.mat",
                    "message": "Shape mismatch for key 'ij'",
                    "failure_kind": "shape_mismatch",
                    "failing_key": "ij",
                    "tolerance_rule_id": "merged_ps2.ij.exact_structural",
                }
            ],
        },
        manifest,
        returncode=1,
        now=datetime(2026, 5, 28, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert report["unapproved_failures"][0]["path"] == "ps2.mat"
    assert report["waived_failures"] == []


def test_verifier_waiver_evaluation_accepts_documented_manifest_waiver(tmp_path: Path) -> None:
    module = _load_gate_module()
    manifest = tmp_path / "artifact_tolerances.json"
    manifest.write_text(
        """
{
  "manifest_version": 1,
  "artifacts": [],
  "waivers": [
    {
      "path": "uw_space_time.mat",
      "key": "spread",
      "failure_kind": "sparse_structure_mismatch",
      "scientific_reason": "Legacy STAMPS sparse encoding differs but data vectors are analytically equivalent.",
      "owner": "native-parity",
      "expires_at_utc": "2026-06-30T00:00:00Z"
    }
  ]
}
""",
        encoding="utf-8",
    )

    report = module.evaluate_verifier_tolerance_waivers(
        {
            "ok": False,
            "checked": 1,
            "failed": [
                {
                    "path": "uw_space_time.mat",
                    "message": "Sparse structure mismatch for key 'spread'",
                    "failure_kind": "sparse_structure_mismatch",
                    "failing_key": "spread",
                    "tolerance_rule_id": "merged_uw_space_time.spread.sparse_exact",
                }
            ],
        },
        manifest,
        returncode=1,
        now=datetime(2026, 5, 28, tzinfo=timezone.utc),
    )

    assert report["ok"] is True
    assert report["unapproved_failures"] == []
    assert report["waived_failures"][0]["waiver"]["scientific_reason"].startswith("Legacy STAMPS")


def test_certification_payload_contains_release_evidence(tmp_path: Path) -> None:
    module = _load_gate_module()
    manifest = tmp_path / "artifact_tolerances.json"
    manifest.write_text('{"manifest_version": 1, "artifacts": [], "waivers": []}', encoding="utf-8")
    run_report = {
        "ok": True,
        "elapsed_sec": 42.5,
        "setup": {
            "dataset": "/data/InSAR_dataset_test",
            "run_root": "/runs/native-full-chain",
            "start_step": 1,
            "end_step": 8,
        },
        "coverage": {"command": ["target/release/pystamps-native", "coverage"]},
        "command": ["target/release/pystamps-native", "run", "--native-only"],
        "results": [
            {
                "stage": 8,
                "scope": "merged",
                "target": "native-full-chain",
                "status": "completed",
                "duration_sec": 12.25,
                "memory_peak_bytes": 4096,
            }
        ],
        "performance_budget": {"ok": True, "violations": [], "waivers": []},
    }
    verify_report = {
        "ok": True,
        "status": "passed",
        "command": [sys.executable, "-m", "pystamps.cli", "verify"],
        "returncode": 0,
        "verifier": {"ok": True, "checked": 7, "failed": []},
    }

    payload = module.build_certification_payload(
        run_report,
        verify_report,
        golden_root=tmp_path / "golden",
        tolerance_manifest=manifest,
        commit_sha="abc123",
        now=datetime(2026, 5, 28, tzinfo=timezone.utc),
    )

    assert payload["ok"] is True
    assert payload["commit_sha"] == "abc123"
    assert payload["dataset_path"] == "/data/InSAR_dataset_test"
    assert payload["total_runtime_sec"] == 42.5
    assert payload["peak_memory_bytes"] == 4096
    assert payload["per_stage_runtime"][0]["stage"] == 8
    assert payload["verifier_result"]["ok"] is True
    assert payload["tolerance_waiver_list"] == []
    assert "native_run" in payload["command_lines"]


def test_native_command_requires_native_only_policy(tmp_path: Path) -> None:
    module = _load_gate_module()
    native_bin = tmp_path / "pystamps-native"
    native_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    args = SimpleNamespace(native_bin=str(native_bin), start_step=1, end_step=8, threads=0)

    command = module._native_command(args, tmp_path / "run")

    assert "--native-only" in command
    module.validate_native_only_command(command)


def test_native_only_command_rejects_bridge_and_python_backends() -> None:
    module = _load_gate_module()

    with pytest.raises(module.GateError, match="forbids bridge/external execution"):
        module.validate_native_only_command(["uv", "run", "python", "-m", "pystamps.cli"])

    with pytest.raises(module.GateError, match="requires --backend native"):
        module.validate_native_only_command(
            [
                "target/release/pystamps-native",
                "run",
                "--native-only",
                "--dataset",
                "run",
                "--backend",
                "python",
                "--stage2-kernel-backend",
                "native",
            ]
        )

    with pytest.raises(module.GateError, match="requires --stage2-kernel-backend native"):
        module.validate_native_only_command(
            [
                "target/release/pystamps-native",
                "run",
                "--native-only",
                "--dataset",
                "run",
                "--backend",
                "native",
                "--stage2-kernel-backend",
                "python",
            ]
        )

    with pytest.raises(module.GateError, match="forbids shelling out"):
        module.validate_native_only_command(
            [
                "target/release/pystamps-native",
                "run",
                "--native-only",
                "--dataset",
                "run",
                "--backend",
                "native",
                "--stage2-kernel-backend",
                "native",
                "matlab",
            ]
        )


def test_native_coverage_evaluation_requires_native_certified_metadata() -> None:
    module = _load_gate_module()

    report = module.evaluate_native_coverage(
        [
            {
                "stage": 3,
                "scope": "patch",
                "target": "PATCH_*",
                "native_stage": False,
                "parity_certified": False,
                "not_parity_certified_reason": "story gate has not passed",
                "unsupported_modes": [{"mode": "python", "reason": "not native"}],
            }
        ]
    )

    assert report["ok"] is False
    assert {violation["kind"] for violation in report["violations"]} >= {
        "not_native_stage",
        "not_parity_certified",
        "missing_unsupported_modes",
    }


def test_native_coverage_evaluation_accepts_full_native_metadata() -> None:
    module = _load_gate_module()

    report = module.evaluate_native_coverage(
        [
            {
                "stage": 8,
                "scope": "merged",
                "target": "dataset root",
                "native_stage": True,
                "parity_certified": True,
                "disabled": False,
                "unsupported_modes": [
                    {"mode": "python", "reason": "not native"},
                    {"mode": "matlab", "reason": "not native"},
                    {"mode": "octave", "reason": "not native"},
                    {"mode": "bridge", "reason": "not native"},
                ],
            }
        ]
    )

    assert report == {"ok": True, "checked_scope_count": 1, "violations": []}
