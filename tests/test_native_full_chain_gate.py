from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

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
