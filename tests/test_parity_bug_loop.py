from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_parity_bug_loop_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "parity_bug_loop.py"
    spec = importlib.util.spec_from_file_location("parity_bug_loop", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_divergence_sets_includes_all_and_group_sets() -> None:
    module = _load_parity_bug_loop_module()
    audit = {
        "failures": [
            {"path": "uw_grid.mat"},
            {"path": "uw_interp.mat"},
            {"path": "uw_grid.mat"},
        ],
        "groups": [
            {
                "failure_class": "unwrap_smoothing",
                "label": "Unwrap / smoothing",
                "stage_scope": "stage5_6",
                "guidance": "focus stage 5/6",
                "paths": ["uw_grid.mat", "uw_interp.mat"],
            }
        ],
    }

    divergence_sets = module._build_divergence_sets(audit)

    assert [item["name"] for item in divergence_sets] == ["all", "unwrap_smoothing"]
    assert divergence_sets[0]["patterns"] == ["uw_grid.mat", "uw_interp.mat"]
    assert divergence_sets[1]["patterns"] == ["uw_grid.mat", "uw_interp.mat"]


def test_evaluate_audit_ranks_next_target(monkeypatch) -> None:
    module = _load_parity_bug_loop_module()
    audit = {
        "workflow": "InSAR_dataset_test_audit",
        "dataset": "InSAR_dataset_test",
        "run_root": "/tmp/run",
        "golden_root": "/tmp/golden",
        "ok": False,
        "failed": 3,
        "failures": [
            {
                "path": "PATCH_1/select1.mat",
                "message": "Value mismatch for key 'C_ps2', max_abs=1",
                "stage_scope": "stage3_4",
                "failure_class": "upstream_patch_residual",
                "label": "Upstream patch residual",
                "failing_key": "C_ps2",
                "guidance": "fix upstream",
            },
            {
                "path": "uw_grid.mat",
                "message": "Value mismatch for key 'ph', max_abs=2",
                "stage_scope": "stage5_6",
                "failure_class": "unwrap_smoothing",
                "label": "Unwrap / smoothing",
                "failing_key": "ph",
                "guidance": "fix unwrap",
            },
        ],
        "groups": [
            {
                "failure_class": "unwrap_smoothing",
                "label": "Unwrap / smoothing",
                "stage_scope": "stage5_6",
                "guidance": "fix unwrap",
                "paths": ["uw_grid.mat"],
            },
            {
                "failure_class": "upstream_patch_residual",
                "label": "Upstream patch residual",
                "stage_scope": "stage3_4",
                "guidance": "fix upstream",
                "paths": ["PATCH_1/select1.mat"],
            },
        ],
    }

    def fake_verify(run_root, golden_root, tolerance, patterns):
        failures = []
        if "PATCH_1/select1.mat" in patterns:
            failures.append(SimpleNamespace(relative_path="PATCH_1/select1.mat", ok=False, message="bad select"))
        if "uw_grid.mat" in patterns:
            failures.append(SimpleNamespace(relative_path="uw_grid.mat", ok=False, message="bad grid"))
        return SimpleNamespace(ok=not failures, comparisons=failures)

    def fake_summarize(report):
        failures = [
            {
                "path": failure.relative_path,
                "message": failure.message,
                "stage_scope": "stage3_4" if "select1" in failure.relative_path else "stage5_6",
                "failure_class": "upstream_patch_residual" if "select1" in failure.relative_path else "unwrap_smoothing",
                "label": "x",
                "failing_key": None,
                "guidance": "x",
            }
            for failure in report.comparisons
        ]
        return {
            "ok": report.ok,
            "checked": len(report.comparisons),
            "failed": len(report.comparisons),
            "failures": failures,
            "groups": [],
            "trace": {},
        }

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)
    monkeypatch.setattr(module, "summarize_failures", fake_summarize)

    payload = module._evaluate_audit(audit)

    assert payload["next_target"]["failure_class"] == "upstream_patch_residual"
    assert payload["next_target"]["patterns"] == ["PATCH_1/select1.mat"]


def test_evaluate_audit_prefers_saved_first_boundary_trace(monkeypatch) -> None:
    module = _load_parity_bug_loop_module()
    audit = {
        "workflow": "InSAR_dataset_test_audit",
        "dataset": "InSAR_dataset_test",
        "run_root": "/tmp/run",
        "golden_root": "/tmp/golden",
        "ok": False,
        "failed": 1,
        "failures": [
            {
                "path": "uw_space_time.mat",
                "message": "downstream",
                "stage_scope": "stage7_8",
                "failure_class": "unwrapped_noise_statistics",
                "label": "Unwrapped-noise / statistics",
                "failing_key": "dph_noise",
                "guidance": "fix downstream",
            }
        ],
        "groups": [
            {
                "failure_class": "unwrapped_noise_statistics",
                "label": "Unwrapped-noise / statistics",
                "stage_scope": "stage7_8",
                "guidance": "fix downstream",
                "paths": ["uw_space_time.mat"],
            }
        ],
        "trace": {
            "first_divergent_boundary": {
                "artifact_path": "PATCH_1/select1.mat",
                "stage_scope": "stage3",
                "failure_class": "stage3_patch_boundary",
                "label": "Stage 3 patch boundary",
                "guidance": "fix stage 3",
                "failure_kind": "value_mismatch",
                "failing_key": "C_ps2",
                "shape_run": [10],
                "shape_oracle": [10],
                "max_abs": 2.5,
                "message": "select mismatch",
            }
        },
    }

    monkeypatch.setattr(
        module,
        "_compare_patterns",
        lambda run_root, golden_root, patterns: {
            "ok": False,
            "checked": 1,
            "failed": 1,
            "patterns": patterns,
            "first_failure": {
                "path": "uw_space_time.mat",
                "message": "downstream",
                "stage_scope": "stage7_8",
                "failure_class": "unwrapped_noise_statistics",
                "label": "Unwrapped-noise / statistics",
                "failing_key": "dph_noise",
                "guidance": "fix downstream",
            },
            "groups": [],
            "failures": audit["failures"],
            "trace": {},
        },
    )

    payload = module._evaluate_audit(audit)

    assert payload["next_target"]["failure_class"] == "stage3_patch_boundary"
    assert payload["next_target"]["patterns"] == ["PATCH_1/select1.mat"]
    assert payload["next_target"]["source"] == "stage_boundary_trace"


def test_main_writes_loop_payload(monkeypatch, tmp_path: Path) -> None:
    module = _load_parity_bug_loop_module()
    audit_output = tmp_path / "audit.json"
    loop_output = tmp_path / "loop.json"
    audit_payload = {
        "ok": False,
        "completed": True,
        "failed_workflows": ["full_validation"],
        "code_state": {"git_commit": "abc123", "git_dirty": False},
        "contract": {
            "audited_workflow_manifest_path": "pystamps/data/audited_workflow_manifest.json",
        },
        "audits": [
            {
                "workflow": "InSAR_dataset_test_audit",
                "dataset": "InSAR_dataset_test",
                "run_root": str(tmp_path / "run"),
                "golden_root": str(tmp_path / "golden"),
                "run_source": "generated_full_loop_run_copy",
                "run_generation": {"workflow_profile": "legacy_post"},
                "ok": False,
                "failed": 1,
                "failures": [
                    {
                        "path": "uw_grid.mat",
                        "message": "bad grid",
                        "stage_scope": "stage5_6",
                        "failure_class": "unwrap_smoothing",
                        "label": "Unwrap / smoothing",
                        "failing_key": "ph",
                        "guidance": "fix unwrap",
                    }
                ],
                "groups": [
                    {
                        "failure_class": "unwrap_smoothing",
                        "label": "Unwrap / smoothing",
                        "stage_scope": "stage5_6",
                        "guidance": "fix unwrap",
                        "paths": ["uw_grid.mat"],
                    }
                ],
            }
        ],
    }
    audit_output.write_text(json.dumps(audit_payload), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            datasets=None,
            golden_root=None,
            output=str(loop_output),
            audit_output=str(audit_output),
            allow_subset=True,
        ),
    )
    monkeypatch.setattr(module, "DEFAULT_REQUIRED_DATASETS", ("inputs_and_outputs/InSAR_dataset_test",))
    monkeypatch.setattr(module, "capture_code_state", lambda repo_root: audit_payload["code_state"])
    monkeypatch.setattr(
        module,
        "_run_validate_audit",
        lambda args, output: (_ for _ in ()).throw(AssertionError("existing matching audit should be reused")),
    )

    def fake_verify(run_root, golden_root, tolerance, patterns):
        return SimpleNamespace(ok=False, comparisons=[SimpleNamespace(relative_path="uw_grid.mat", ok=False, message="bad grid")])

    monkeypatch.setattr(module, "verify_run_against_golden", fake_verify)
    monkeypatch.setattr(
        module,
        "summarize_failures",
        lambda report: {
            "ok": False,
            "checked": 1,
            "failed": 1,
            "failures": [
                {
                    "path": "uw_grid.mat",
                    "message": "bad grid",
                    "stage_scope": "stage5_6",
                    "failure_class": "unwrap_smoothing",
                    "label": "Unwrap / smoothing",
                    "failing_key": "ph",
                    "guidance": "fix unwrap",
                }
            ],
            "groups": [],
            "trace": {},
        },
    )

    exit_code = module.main()

    assert exit_code == 1
    payload = json.loads(loop_output.read_text(encoding="utf-8"))
    assert payload["datasets"] == ["inputs_and_outputs/InSAR_dataset_test"]
    assert payload["audit_source"] == "reused_existing_output"
    assert payload["audit_output"] == str(audit_output)
    assert payload["code_state"] == audit_payload["code_state"]
    assert payload["contract_metadata"]["audited_workflow_manifest_path"] == "pystamps/data/audited_workflow_manifest.json"
    assert payload["audits"][0]["run_generation"]["workflow_profile"] == "legacy_post"
    assert payload["next_target"]["failure_class"] == "unwrap_smoothing"
