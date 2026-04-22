import importlib.resources
import json
from pathlib import Path

from pystamps.parity_contract import build_parity_contract


def test_build_parity_contract_without_datasets(tmp_path: Path) -> None:
    inputs_root = tmp_path / "inputs_and_outputs"
    inputs_root.mkdir()

    contract = build_parity_contract(inputs_root)

    assert contract["inputs_root"] == "inputs_and_outputs"
    assert contract["datasets"] == []
    assert contract["supported_audit"]["entrypoint"] == "scripts/validate_audit.py"
    assert contract["supported_audit"]["output_artifact"] == "inputs_and_outputs/validation_runs/latest_audit.json"
    assert contract["supported_audit"]["required_result_fields"] == [
        "generated_at_utc",
        "contract",
        "missing_datasets",
        "audits",
        "failed_workflows",
        "completed",
        "interrupted",
        "ok",
    ]
    assert contract["required_workflow_names"] == ["full_validation"]
    assert "full_validation" in contract["workflows"]
    assert contract["workflows"]["full_validation"]["driver"] == "scripts/validate_audit.py"
    assert contract["workflows"]["full_validation"]["output_artifact"] == "inputs_and_outputs/validation_runs/latest_audit.json"
    assert contract["oracle_contract_manifest_path"] == "pystamps/data/oracle_contract.json"
    assert contract["audited_workflow_manifest_path"] == "pystamps/data/audited_workflow_manifest.json"

    oracle = contract["oracle_contract"]
    assert oracle["matlab_source"]["upstream_repository_url"] == "https://github.com/dbekaert/StaMPS"
    assert oracle["matlab_source"]["pinned_revision"] == "c159eb81b16c446e0e8fdef7dd435eb22e0240ed"
    assert oracle["cpp_wrapper"]["repository_url"] == "https://github.com/dbekaert/StaMPS"
    assert oracle["cpp_wrapper"]["pinned_revision"] == "c159eb81b16c446e0e8fdef7dd435eb22e0240ed"
    assert oracle["precedence_rule"]["ordered_sources"] == ["cpp_wrapper", "matlab_source", "manual_references"]
    assert "must not claim oracle-backed completion" in oracle["negative_completion_rule"]

    workflow_manifest = contract["audited_workflow_manifest"]
    assert workflow_manifest["supported_audit"]["required_dataset_paths"] == [
        "inputs_and_outputs/InSAR_dataset_test_stage8diag",
        "inputs_and_outputs/InSAR_dataset_test",
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag",
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7",
    ]
    assert workflow_manifest["supported_audit"]["required_workflow_names"] == ["full_validation"]
    assert [target["id"] for target in workflow_manifest["workflow_targets"]] == [
        "single_master_diagnostic",
        "single_master_full",
        "small_baseline_diagnostic",
        "small_baseline_full",
    ]
    present_small_baseline = [
        target
        for target in workflow_manifest["workflow_targets"]
        if target["kind"] == "small_baseline"
    ]
    assert [target["status"] for target in present_small_baseline] == ["present", "present"]
    assert [target["audit_start_step"] for target in present_small_baseline] == [7, 7]
    assert [target["audit_end_step"] for target in present_small_baseline] == [7, 7]
    assert all(target["supports_validate_audit"] is True for target in present_small_baseline)
    assert all(target["oracle_reference_paths"] for target in present_small_baseline)


def test_packaged_parity_manifests_are_valid_json() -> None:
    oracle = json.loads(importlib.resources.files("pystamps.data").joinpath("oracle_contract.json").read_text(encoding="utf-8"))
    workflow_manifest = json.loads(
        importlib.resources.files("pystamps.data").joinpath("audited_workflow_manifest.json").read_text(encoding="utf-8")
    )

    assert oracle["cpp_wrapper"]["repository_url"]
    assert oracle["cpp_wrapper"]["pinned_revision"]
    assert workflow_manifest["supported_audit"]["required_dataset_paths"] == [
        "inputs_and_outputs/InSAR_dataset_test_stage8diag",
        "inputs_and_outputs/InSAR_dataset_test",
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7diag",
        "inputs_and_outputs/InSAR_dataset_small_baseline_stage7",
    ]
    assert all(target["required_for_done"] for target in workflow_manifest["workflow_targets"])
    assert all(target["status"] == "present" for target in workflow_manifest["workflow_targets"])
